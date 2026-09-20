"""Shadow evaluation and safety-gated memory promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from math import isfinite
from statistics import mean

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryEvaluation,
    MemoryStatus,
    MemoryValidationTarget,
    RunOutcome,
)
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository


@dataclass(frozen=True)
class MemoryPromotionPolicy:
    min_evidence_runs: int = 3
    min_outcome_samples: int = 3
    min_confidence: float = 0.65
    min_mean_excess_return: float = 0.0
    max_allowed_drawdown: float = -0.20
    allow_auto_promotion: bool = False


_UNSAFE_PHRASES = (
    "override risk",
    "bypass risk",
    "ignore risk",
    "skip confirmation",
    "disable stop",
    "取消风控",
    "绕过风控",
    "忽略止损",
    "无需确认",
)


class MemoryShadowEvaluator:
    """Validate shadow observations against delayed outcomes."""

    def __init__(
        self,
        repository: SQLiteMemoryRepository,
        policy: MemoryPromotionPolicy | None = None,
    ):
        self.repository = repository
        self.policy = policy or MemoryPromotionPolicy()

    def evaluate(self, memory_id: str) -> MemoryEvaluation:
        memory = self.repository.get(memory_id)
        if memory is None:
            raise KeyError(f"memory not found: {memory_id}")
        reasons: list[str] = []
        safety_reasons = self.safety_reasons(memory)
        passed_safety = not safety_reasons
        reasons.extend(safety_reasons)

        usage = self.repository.usage_for_memory(memory_id, mode="shadow", version=memory.version)
        evidence_runs = len({row["run_id"] for row in usage})
        if memory.metadata.get("delayed_review"):
            outcomes, delayed_reasons = self._delayed_outcomes(memory)
            evidence_runs = len(outcomes)
            reasons.extend(delayed_reasons)
        else:
            outcomes = self.repository.outcomes_for_memory(memory_id, mode="shadow", version=memory.version)
        realized = [item.realized_return for item in outcomes if item.realized_return is not None]
        raw_excess = [
            item.realized_return - item.benchmark_return
            for item in outcomes
            if item.realized_return is not None and item.benchmark_return is not None
        ]
        excess = (
            [-value for value in raw_excess]
            if memory.validation_target == MemoryValidationTarget.UNDERPERFORM
            and not (memory.metadata.get("delayed_review") and memory.metadata.get("validation_category") in {"REAL", "SIMULATE"})
            else raw_excess
        )
        drawdowns = [item.max_drawdown for item in outcomes if item.max_drawdown is not None]
        confidence = memory.confidence or 0.0

        if memory.status != MemoryStatus.SHADOW:
            reasons.append("memory must be in shadow status")
        if evidence_runs < self.policy.min_evidence_runs:
            reasons.append(
                f"shadow evidence {evidence_runs}/{self.policy.min_evidence_runs} runs"
            )
        if len(outcomes) < self.policy.min_outcome_samples:
            reasons.append(
                f"delayed outcomes {len(outcomes)}/{self.policy.min_outcome_samples} samples"
            )
        if confidence < self.policy.min_confidence:
            reasons.append(
                f"confidence {confidence:.2f} below {self.policy.min_confidence:.2f}"
            )
        mean_excess = mean(excess) if excess else None
        if memory.validation_target != MemoryValidationTarget.RISK_REDUCTION:
            if mean_excess is None:
                reasons.append("benchmark-relative return is unavailable")
            elif mean_excess < self.policy.min_mean_excess_return:
                reasons.append(
                    f"target-aligned excess return {mean_excess:.4f} below "
                    f"{self.policy.min_mean_excess_return:.4f}"
                )
        worst_drawdown = min(drawdowns) if drawdowns else None
        if worst_drawdown is None:
            reasons.append("drawdown evidence is unavailable")
        elif worst_drawdown < self.policy.max_allowed_drawdown:
            reasons.append(
                f"worst drawdown {worst_drawdown:.4f} breached "
                f"{self.policy.max_allowed_drawdown:.4f}"
            )

        eligible = passed_safety and not reasons
        if memory.metadata.get("delayed_review"):
            reasons.append(
                f"validation group: category={memory.metadata.get('validation_category', 'hypothetical')}, "
                f"account={memory.metadata.get('validation_account_id', memory.metadata.get('account_id', 'all'))}, horizon=20"
            )
        evaluation = MemoryEvaluation(
            memory_id=memory.memory_id,
            memory_version=memory.version,
            eligible=eligible,
            passed_safety_gate=passed_safety,
            evidence_runs=evidence_runs,
            outcome_samples=len(outcomes),
            mean_realized_return=mean(realized) if realized else None,
            mean_excess_return=mean_excess,
            worst_drawdown=worst_drawdown,
            reasons=reasons,
        )
        return self.repository.save_evaluation(evaluation)

    def _delayed_outcomes(self, memory: DistilledMemory) -> tuple[list[RunOutcome], list[str]]:
        """Use mature, classified evidence; one conservative sample per source run."""
        category = memory.metadata.get("validation_category", "hypothetical")
        account = memory.metadata.get("validation_account_id", memory.metadata.get("account_id"))
        group = f"delayed category={category}, account={account or 'all'}, horizon=20"
        if category not in {"hypothetical", "SIMULATE", "REAL"} or memory.metadata.get("validation_horizon_days", 20) != 20:
            return [], [f"{group}: unsupported validation group"]
        with self.repository._connection() as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"review_results", "review_tasks", "review_snapshots"} <= tables:
                return [], [f"{group}: classified reviews are unavailable"]
            rows = connection.execute(
                """SELECT t.run_id, t.symbol, r.payload_json, s.generated_at,
                          u.created_at AS usage_at, v.created_at AS version_at
                   FROM review_results r JOIN review_tasks t ON t.id = r.review_id
                   JOIN review_snapshots s ON s.run_id=t.run_id AND s.symbol=t.symbol
                   JOIN memory_usage u ON u.run_id=t.run_id AND u.symbol=t.symbol
                   JOIN memory_versions v ON v.memory_id=u.memory_id AND v.version=u.memory_version
                   WHERE t.status='completed' AND t.horizon_days=20
                     AND u.memory_id=? AND u.memory_version=? AND u.mode='shadow'""",
                (memory.memory_id, memory.version),
            ).fetchall()
        excluded = {run.lower() for run in memory.evidence_run_ids}
        if memory.source_run_id:
            excluded.add(memory.source_run_id.lower())
        grouped: dict[str, list[dict]] = {}
        invalid: set[str] = set()
        needs_return = memory.validation_target != MemoryValidationTarget.RISK_REDUCTION

        def finite(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)

        for row in rows:
            run = row["run_id"]
            if run.lower() in excluded:
                continue
            try:
                payload = json.loads(row["payload_json"])
                generated = datetime.fromisoformat(row["generated_at"])
                if not (datetime.fromisoformat(row["usage_at"]) <= generated
                        and datetime.fromisoformat(row["version_at"]) < generated
                        and datetime.fromisoformat(payload["cutoff_at"]) > generated):
                    continue
                if (payload["run_id"], payload["symbol"], payload["horizon_days"]) != (run, row["symbol"], 20):
                    invalid.add(run)
                    continue
                observations = [o for o in payload["observations"]
                                if o.get("category") == category
                                and (account is None or str(o.get("account_id")) == str(account))]
                for observation in observations:
                    required = ["max_drawdown"] + (["realized_return", "benchmark_return"] if needs_return else [])
                    if observation.get("eligible") is not True or not all(finite(observation.get(key)) for key in required):
                        invalid.add(run)
                    else:
                        grouped.setdefault(run, []).append(observation)
            except (ValueError, TypeError, KeyError, AttributeError):
                invalid.add(run)
        outcomes = []
        for run, observations in grouped.items():
            if run in invalid:
                continue
            def average(key):
                values = [o.get(key) for o in observations]
                return mean(values) if all(finite(v) for v in values) else None
            outcomes.append(RunOutcome(
                run_id=run, symbol="GROUP", horizon_days=20,
                realized_return=average("realized_return"),
                benchmark_return=average("benchmark_return"),
                max_drawdown=min(o["max_drawdown"] for o in observations),
                source=group,
            ))
        return outcomes, ([f"{group}: incomplete or ineligible evidence in {len(invalid)} runs"] if invalid else [])

    def promote(
        self,
        memory_id: str,
        *,
        actor: str,
        reason: str,
        automatic: bool = False,
    ) -> DistilledMemory:
        if automatic and not self.policy.allow_auto_promotion:
            raise PermissionError("automatic memory promotion is disabled by policy")
        with self.repository.transaction():
            memory = self.repository.get(memory_id)
            if automatic and memory is not None and memory.metadata.get("delayed_review"):
                raise PermissionError("delayed review memories require manual promotion")
            evaluation = self.evaluate(memory_id)
            if evaluation.eligible:
                return self.repository.transition(
                    memory_id,
                    MemoryStatus.APPROVED,
                    actor=actor,
                    reason=reason,
                )
        # Commit the failed evaluation as audit evidence before reporting failure.
        raise ValueError("memory failed promotion gate: " + "; ".join(evaluation.reasons))

    @staticmethod
    def safety_reasons(memory: DistilledMemory) -> list[str]:
        searchable = " ".join(
            [memory.lesson, memory.trigger, memory.rationale, *memory.tags]
        ).lower()
        reasons = [
            f"unsafe risk-control phrase: {phrase}"
            for phrase in _UNSAFE_PHRASES
            if phrase in searchable
        ]
        if memory.metadata.get("modifies_risk_limits"):
            reasons.append("memory may not modify deterministic risk limits")
        return reasons


__all__ = [
    "MemoryPromotionPolicy",
    "MemoryShadowEvaluator",
    "RunOutcome",
]

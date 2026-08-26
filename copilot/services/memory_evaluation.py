"""Shadow evaluation and safety-gated memory promotion."""

from __future__ import annotations

from dataclasses import dataclass
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

        usage = self.repository.usage_for_memory(memory_id, mode="shadow")
        evidence_runs = len({row["run_id"] for row in usage})
        outcomes = self.repository.outcomes_for_memory(memory_id, mode="shadow")
        realized = [item.realized_return for item in outcomes if item.realized_return is not None]
        raw_excess = [
            item.realized_return - item.benchmark_return
            for item in outcomes
            if item.realized_return is not None and item.benchmark_return is not None
        ]
        excess = (
            [-value for value in raw_excess]
            if memory.validation_target == MemoryValidationTarget.UNDERPERFORM
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
        evaluation = self.evaluate(memory_id)
        if not evaluation.eligible:
            raise ValueError("memory failed promotion gate: " + "; ".join(evaluation.reasons))
        return self.repository.transition(
            memory_id,
            MemoryStatus.APPROVED,
            actor=actor,
            reason=reason,
        )

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

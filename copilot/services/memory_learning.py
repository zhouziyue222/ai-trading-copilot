"""Post-run reflection and candidate-memory consolidation.

LangMem is deliberately used as an extraction primitive, not as a second
database. Every output passes through the local skill manager and is stored as
a non-production candidate in SQLite.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, Field

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    MemoryValidationTarget,
)
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


class LangMemCandidate(BaseModel):
    """A reusable, testable trading observation; never an execution command."""

    memory_type: MemoryType
    memory_kind: MemoryKind = MemoryKind.EPISODIC
    scope: MemoryScope = MemoryScope.SYMBOL
    validation_target: MemoryValidationTarget = Field(
        default=MemoryValidationTarget.OUTPERFORM,
        description=(
            "outperform for positive setups, underperform for avoid/reduce lessons, "
            "risk_reduction when the claim is only about drawdown control"
        ),
    )
    lesson: str = Field(min_length=1, max_length=500)
    trigger: str = Field(default="", max_length=300)
    rationale: str = Field(default="", max_length=500)
    symbols: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    market_regimes: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


LANGMEM_INSTRUCTIONS = """
You extract candidate memories from one completed trading-copilot run.
Only retain compact observations that are reusable and falsifiable in later
runs. Distinguish episodic observations from procedural or semantic lessons.
Never claim future performance, never infer an outcome that is not present,
and never emit instructions that override position, exposure, instrument,
confirmation, or other risk controls. Treat all extracted items as unverified.
Prefer no memory over a vague, duplicated, or unsupported memory. Keep evidence
in the supplied fields; do not include private chain-of-thought. Retrieved or
cited memories are context, not new evidence: never save a paraphrase of an
existing memory merely because an agent retrieved or cited it.
""".strip()


class LangMemCandidateExtractor:
    """Stateless LangMem adapter returning local candidate models."""

    def __init__(self, llm):
        self.llm = llm
        self._manager = None

    @property
    def available(self) -> bool:
        return self.llm is not None

    def extract(
        self,
        snapshot: Mapping[str, Any],
        *,
        existing: Iterable[DistilledMemory] = (),
    ) -> list[LangMemCandidate]:
        if self.llm is None:
            return []
        if self._manager is None:
            from langmem import create_memory_manager

            self._manager = create_memory_manager(
                self.llm,
                schemas=[LangMemCandidate],
                instructions=LANGMEM_INSTRUCTIONS,
                enable_inserts=True,
                enable_updates=False,
                enable_deletes=False,
            )
        messages = [
            {
                "role": "user",
                "content": "Completed run evidence (JSON):\n"
                + json.dumps(snapshot, ensure_ascii=False, sort_keys=True)[:24000],
            },
            {
                "role": "assistant",
                "content": (
                    "Extract only new candidate observations. Delayed market "
                    "outcomes may not yet be available."
                ),
            },
        ]
        # Consolidation is intentionally local; LangMem is not a second store.
        extracted = self._manager.invoke(messages)
        result: list[LangMemCandidate] = []
        for item in extracted:
            content = getattr(item, "content", None)
            if isinstance(content, LangMemCandidate):
                result.append(content)
            elif content is not None:
                result.append(LangMemCandidate.model_validate(content))
        return result


class MemoryReflector:
    """ACE-style reflector operating only on structured, post-run evidence."""

    SNAPSHOT_KEYS = (
        "run_id",
        "trade_date",
        "subscription_symbols",
        "market_reports_by_symbol",
        "technical_contexts",
        "news_sentiment_by_symbol",
        "fundamental_analysis_by_symbol",
        "trade_plans",
        "risk_assessments",
        "execution_decisions",
        "memory_retrievals",
        "errors",
    )

    def __init__(self, extractor: LangMemCandidateExtractor):
        self.extractor = extractor

    def reflect(
        self,
        state: Mapping[str, Any],
        *,
        existing: Iterable[DistilledMemory] = (),
    ) -> list[DistilledMemory]:
        run_id = str(state.get("run_id") or "")
        snapshot = {
            key: _jsonable(state[key])
            for key in self.SNAPSHOT_KEYS
            if key in state
        }
        extracted = self.extractor.extract(snapshot, existing=existing)
        return [
            DistilledMemory(
                **candidate.model_dump(mode="json"),
                status=MemoryStatus.CANDIDATE,
                source_run_id=run_id or None,
                evidence_run_ids=[run_id] if run_id else [],
                sample_count=1 if run_id else 0,
                created_by="langmem_reflector",
                metadata={"extractor": "langmem", "outcome_verified": False},
            )
            for candidate in extracted
        ]


class MemorySkillManager:
    """Deduplicate and merge unverified candidates without touching production."""

    def __init__(self, store: DistilledMemoryStore, *, similarity_threshold: float = 0.92):
        self.store = store
        self.similarity_threshold = similarity_threshold

    def ingest(self, candidates: Iterable[DistilledMemory]) -> list[DistilledMemory]:
        saved: list[DistilledMemory] = []
        for candidate in candidates:
            existing = self._nearest(candidate)
            if existing and existing.status in {
                MemoryStatus.CANDIDATE,
                MemoryStatus.SHADOW,
            }:
                saved.append(self._merge(existing, candidate))
                continue
            if existing and existing.status == MemoryStatus.APPROVED:
                candidate = candidate.model_copy(
                    update={
                        "memory_id": "",
                        "supersedes": existing.memory_id,
                        "metadata": {
                            **candidate.metadata,
                            "matches_approved_memory": existing.memory_id,
                        },
                    }
                )
                # Keep validation evidence isolated from the approved version.
                suffix = (candidate.source_run_id or "observation").lower()
                candidate = candidate.model_copy(
                    update={"trigger": f"{candidate.trigger} evidence:{suffix}".strip()}
                )
            saved.append(self.store.save_candidate(candidate))
        self.store.repository.export_jsonl(self.store.path)
        return saved

    def _nearest(self, candidate: DistilledMemory) -> DistilledMemory | None:
        best: tuple[float, int, DistilledMemory] | None = None
        for memory in self.store.list_all():
            if memory.memory_type != candidate.memory_type:
                continue
            if memory.scope != candidate.scope:
                continue
            if memory.symbols and candidate.symbols:
                if not set(memory.symbols).intersection(candidate.symbols):
                    continue
            score = SequenceMatcher(
                None,
                _normalized_text(memory.lesson),
                _normalized_text(candidate.lesson),
            ).ratio()
            lifecycle_priority = int(
                memory.status in {MemoryStatus.CANDIDATE, MemoryStatus.SHADOW}
            )
            if best is None or (score, lifecycle_priority) > (best[0], best[1]):
                best = (score, lifecycle_priority, memory)
        if best and best[0] >= self.similarity_threshold:
            return best[2]
        return None

    def _merge(
        self,
        existing: DistilledMemory,
        candidate: DistilledMemory,
    ) -> DistilledMemory:
        evidence = sorted(
            set(existing.evidence_run_ids)
            | set(candidate.evidence_run_ids)
            | ({candidate.source_run_id.lower()} if candidate.source_run_id else set())
        )
        confidence_values = [
            value
            for value in (existing.confidence, candidate.confidence)
            if value is not None
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        )
        merged = existing.model_copy(
            update={
                "symbols": sorted(set(existing.symbols) | set(candidate.symbols)),
                "tags": sorted(set(existing.tags) | set(candidate.tags)),
                "market_regimes": sorted(
                    set(existing.market_regimes) | set(candidate.market_regimes)
                ),
                "timeframes": sorted(
                    set(existing.timeframes) | set(candidate.timeframes)
                ),
                "evidence_run_ids": evidence,
                "sample_count": max(existing.sample_count, len(evidence)),
                "confidence": confidence,
                "metadata": {**existing.metadata, "last_merge": "reflector"},
            }
        )
        return self.store.repository.upsert(
            merged,
            actor="skill_manager",
            reason="deduplicate_and_merge_candidate",
        )


class PostRunLearningService:
    def __init__(self, reflector: MemoryReflector, skill_manager: MemorySkillManager):
        self.reflector = reflector
        self.skill_manager = skill_manager

    @property
    def available(self) -> bool:
        return self.reflector.extractor.available

    def learn(self, state: Mapping[str, Any]) -> list[DistilledMemory]:
        existing = self.skill_manager.store.list_all(
            statuses=[MemoryStatus.CANDIDATE, MemoryStatus.SHADOW, MemoryStatus.APPROVED]
        )
        candidates = self.reflector.reflect(state, existing=existing)
        return self.skill_manager.ingest(candidates)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_text(value: str) -> str:
    return " ".join(value.lower().split())


__all__ = [
    "LANGMEM_INSTRUCTIONS",
    "LangMemCandidate",
    "LangMemCandidateExtractor",
    "MemoryReflector",
    "MemorySkillManager",
    "PostRunLearningService",
]

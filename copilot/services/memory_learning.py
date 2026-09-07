"""Post-run reflection and candidate-memory consolidation.

LangMem is deliberately used as an extraction primitive, not as a second
database. Every output passes through the local skill manager and is stored as
a non-production candidate in SQLite.
"""

from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal, Mapping

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


class LangMemProposal(BaseModel):
    """One local interpretation of a LangMem insert or update result."""

    operation: Literal["insert", "update"]
    candidate: LangMemCandidate
    target_memory_id: str | None = None


class ReflectedMemoryProposal(BaseModel):
    """A LangMem proposal enriched with immutable run provenance."""

    operation: Literal["insert", "update"]
    memory: DistilledMemory
    target_memory_id: str | None = None


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
    ) -> list[LangMemProposal]:
        if self.llm is None:
            return []
        if self._manager is None:
            from langmem import create_memory_manager

            self._manager = create_memory_manager(
                self.llm,
                schemas=[LangMemCandidate],
                instructions=LANGMEM_INSTRUCTIONS,
                enable_inserts=True,
                enable_updates=True,
                enable_deletes=False,
            )
        existing_items = [
            (memory.memory_id, _candidate_from_memory(memory))
            for memory in existing
            if memory.memory_id
        ]
        existing_by_id = dict(existing_items)
        messages = [
            {
                "role": "user",
                "content": "Completed run evidence (JSON):\n"
                + json.dumps(snapshot, ensure_ascii=False, sort_keys=True)[:24000],
            },
            {
                "role": "assistant",
                "content": (
                    "Insert new observations or update relevant existing ones. "
                    "Delayed market outcomes may not yet be available."
                ),
            },
        ]
        # Consolidation is intentionally local; LangMem is not a second store.
        extracted = self._manager.invoke(
            {"messages": messages, "existing": existing_items}
        )
        result: list[LangMemProposal] = []
        for item in extracted:
            item_id = str(getattr(item, "id", "") or "")
            content = getattr(item, "content", None)
            try:
                candidate = (
                    content
                    if isinstance(content, LangMemCandidate)
                    else LangMemCandidate.model_validate(content)
                )
            except Exception:
                continue
            existing_candidate = existing_by_id.get(item_id)
            if existing_candidate is not None:
                if _candidate_payload(existing_candidate) == _candidate_payload(candidate):
                    continue
                result.append(
                    LangMemProposal(
                        operation="update",
                        target_memory_id=item_id,
                        candidate=candidate,
                    )
                )
            else:
                result.append(LangMemProposal(operation="insert", candidate=candidate))
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
        "errors",
    )

    def __init__(self, extractor: LangMemCandidateExtractor):
        self.extractor = extractor

    def reflect(
        self,
        state: Mapping[str, Any],
        *,
        existing: Iterable[DistilledMemory] = (),
    ) -> list[ReflectedMemoryProposal]:
        run_id = str(state.get("run_id") or "")
        snapshot = {
            key: _jsonable(state[key])
            for key in self.SNAPSHOT_KEYS
            if key in state
        }
        extracted = self.extractor.extract(snapshot, existing=existing)
        reflected = []
        run_symbols = _state_symbols(state)
        for proposal in extracted:
            if (
                proposal.candidate.scope == MemoryScope.SYMBOL
                and run_symbols
                and not run_symbols.intersection(proposal.candidate.symbols)
            ):
                continue
            reflected.append(
                ReflectedMemoryProposal(
                    operation=proposal.operation,
                    target_memory_id=proposal.target_memory_id,
                    memory=DistilledMemory(
                        **proposal.candidate.model_dump(mode="json"),
                        status=MemoryStatus.CANDIDATE,
                        source_run_id=run_id or None,
                        evidence_run_ids=[run_id] if run_id else [],
                        sample_count=1 if run_id else 0,
                        created_by="langmem_reflector",
                        metadata={
                            "extractor": "langmem",
                            "langmem_operation": proposal.operation,
                            "outcome_verified": False,
                        },
                    ),
                )
            )
        return reflected


class MemorySkillManager:
    """Deduplicate and merge unverified candidates without touching production."""

    def __init__(self, store: DistilledMemoryStore, *, similarity_threshold: float = 0.92):
        self.store = store
        self.similarity_threshold = similarity_threshold

    def ingest(
        self,
        proposals: Iterable[ReflectedMemoryProposal],
    ) -> list[DistilledMemory]:
        saved: list[DistilledMemory] = []
        for proposal in proposals:
            candidate = proposal.memory
            target = (
                self.store.repository.get(proposal.target_memory_id)
                if proposal.target_memory_id
                else None
            )
            if target is not None:
                if target.status == MemoryStatus.CANDIDATE:
                    if _already_observed(target, candidate):
                        saved.append(target)
                    else:
                        saved.append(self._merge(target, candidate, apply_update=True))
                else:
                    saved.append(self._save_revision(target, candidate))
                continue

            existing = self._nearest(candidate)
            if existing and existing.status == MemoryStatus.CANDIDATE:
                if _already_observed(existing, candidate):
                    saved.append(existing)
                else:
                    saved.append(self._merge(existing, candidate))
                continue
            if existing:
                saved.append(self._save_revision(existing, candidate))
                continue
            saved.append(self.store.save_candidate(candidate))
        return saved

    def _save_revision(
        self,
        existing: DistilledMemory,
        candidate: DistilledMemory,
    ) -> DistilledMemory:
        revision = candidate.model_copy(
            update={
                "memory_id": _revision_memory_id(existing, candidate),
                "status": MemoryStatus.CANDIDATE,
                "supersedes": existing.memory_id,
                "metadata": {
                    **candidate.metadata,
                    "frozen_memory_status": existing.status.value,
                    "matches_frozen_memory": existing.memory_id,
                },
            }
        )
        current = self.store.repository.get(revision.memory_id)
        if current is not None:
            if _already_observed(current, candidate):
                return current
            return self._merge(current, candidate)
        return self.store.save_candidate(
            revision,
            actor="skill_manager",
            reason="propose_revision_of_frozen_memory",
        )

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
            lifecycle_priority = int(memory.status == MemoryStatus.CANDIDATE)
            if best is None or (score, lifecycle_priority) > (best[0], best[1]):
                best = (score, lifecycle_priority, memory)
        if best and best[0] >= self.similarity_threshold:
            return best[2]
        return None

    def _merge(
        self,
        existing: DistilledMemory,
        candidate: DistilledMemory,
        *,
        apply_update: bool = False,
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
        updates: dict[str, Any] = {
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
            "metadata": {
                **existing.metadata,
                **candidate.metadata,
                "last_merge": "reflector",
            },
        }
        if apply_update:
            updates.update(
                {
                    "memory_type": candidate.memory_type,
                    "memory_kind": candidate.memory_kind,
                    "scope": candidate.scope,
                    "validation_target": candidate.validation_target,
                    "lesson": candidate.lesson,
                    "trigger": candidate.trigger,
                    "rationale": candidate.rationale,
                }
            )
        merged = existing.model_copy(update=updates)
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
        existing = _relevant_existing_memories(
            state,
            self.skill_manager.store.list_all(),
        )
        proposals = self.reflector.reflect(state, existing=existing)
        return self.skill_manager.ingest(proposals)


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


def _candidate_from_memory(memory: DistilledMemory) -> LangMemCandidate:
    return LangMemCandidate(
        memory_type=memory.memory_type,
        memory_kind=memory.memory_kind,
        scope=memory.scope,
        validation_target=memory.validation_target,
        lesson=memory.lesson,
        trigger=memory.trigger,
        rationale=memory.rationale,
        symbols=memory.symbols,
        tags=memory.tags,
        market_regimes=memory.market_regimes,
        timeframes=memory.timeframes,
        confidence=memory.confidence if memory.confidence is not None else 0.5,
    )


def _candidate_payload(candidate: LangMemCandidate) -> dict[str, Any]:
    return candidate.model_dump(mode="json")


def _already_observed(existing: DistilledMemory, candidate: DistilledMemory) -> bool:
    run_id = (candidate.source_run_id or "").strip().lower()
    return bool(run_id and run_id in existing.evidence_run_ids)


def _revision_memory_id(
    existing: DistilledMemory,
    candidate: DistilledMemory,
) -> str:
    identity = {
        "supersedes": existing.memory_id,
        "memory_type": candidate.memory_type.value,
        "memory_kind": candidate.memory_kind.value,
        "scope": candidate.scope.value,
        "lesson": _normalized_text(candidate.lesson),
        "trigger": _normalized_text(candidate.trigger),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"mem_revision_{digest[:24]}"


def _relevant_existing_memories(
    state: Mapping[str, Any],
    memories: Iterable[DistilledMemory],
    *,
    limit: int = 20,
) -> list[DistilledMemory]:
    symbols = _state_symbols(state)
    relevant = [
        memory
        for memory in memories
        if not symbols or not memory.symbols or symbols.intersection(memory.symbols)
    ]
    relevant.sort(
        key=lambda memory: (
            memory.status == MemoryStatus.CANDIDATE,
            len(symbols.intersection(memory.symbols)),
            memory.updated_at or memory.created_at or "",
            memory.memory_id,
        ),
        reverse=True,
    )
    return relevant[:limit]


def _state_symbols(state: Mapping[str, Any]) -> set[str]:
    values = list(state.get("subscription_symbols", []))
    for key in ("trade_plans", "risk_assessments", "execution_decisions"):
        scoped = state.get(key, {})
        if isinstance(scoped, Mapping):
            values.extend(scoped.keys())
    return {
        str(symbol).strip().upper()
        for symbol in values
        if str(symbol).strip()
    }


__all__ = [
    "LANGMEM_INSTRUCTIONS",
    "LangMemCandidate",
    "LangMemCandidateExtractor",
    "LangMemProposal",
    "MemoryReflector",
    "MemorySkillManager",
    "PostRunLearningService",
    "ReflectedMemoryProposal",
]

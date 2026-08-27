"""Bounded, agent-owned retrieval sessions for production trading memories."""

from __future__ import annotations

import json
from typing import Iterable, List

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryKind,
    MemoryRetrievalRecord,
    MemorySearchHit,
    MemorySearchRequest,
    MemoryStatus,
)
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore
from ai_trading_copilot.copilot.services.tracing import (
    get_current_trace_recorder,
    summarize_text,
)


MAX_RESULTS_PER_CALL = 3
MAX_UNIQUE_RESULTS = 6
MAX_OPTIONAL_TOOL_CALLS = 1


class MemoryRetrievalSession:
    """Keep one consumer and symbol isolated across prefetch and tool lookup."""

    def __init__(
        self,
        store: DistilledMemoryStore,
        *,
        symbol: str,
        consumer: str,
        run_id: str = "",
        tags: Iterable[str] = (),
        market_regime: str = "",
        timeframe: str = "",
    ):
        self.store = store
        self.symbol = symbol.strip().upper()
        self.consumer = consumer
        self.run_id = run_id
        self.base_tags = sorted({tag.strip().lower() for tag in tags if tag.strip()})
        self.market_regime = market_regime.strip().lower()
        self.timeframe = timeframe.strip().lower()
        self.records: list[MemoryRetrievalRecord] = []
        self._approved: dict[str, DistilledMemory] = {}
        self._shadow: dict[str, DistilledMemory] = {}
        self._optional_calls = 0

    @property
    def approved_memories(self) -> list[DistilledMemory]:
        return list(self._approved.values())

    @property
    def shadow_memories(self) -> list[DistilledMemory]:
        return list(self._shadow.values())

    def prefetch(
        self,
        query: str,
        *,
        memory_kinds: Iterable[MemoryKind | str] = (),
        tags: Iterable[str] = (),
        limit: int = MAX_RESULTS_PER_CALL,
    ) -> str:
        recorder = get_current_trace_recorder()
        if recorder is None:
            return self._search(
                query=query,
                memory_kinds=memory_kinds,
                tags=tags,
                limit=limit,
                phase="prefetch",
            )
        with recorder.start_span(
            "tool.call",
            kind="client",
            attributes={
                "tool.name": "search_trading_memories",
                "tool.phase": "prefetch",
                "tool.args": {
                    "consumer": self.consumer,
                    "symbol": self.symbol,
                    "query": query,
                    "limit": min(max(limit, 1), MAX_RESULTS_PER_CALL),
                },
            },
        ) as span:
            output = self._search(
                query=query,
                memory_kinds=memory_kinds,
                tags=tags,
                limit=limit,
                phase="prefetch",
            )
            for key, value in summarize_text(output, "tool.output").items():
                span.set_attribute(key, value)
            return output

    def make_tools(self) -> list:
        from langchain_core.tools import tool

        session = self

        @tool
        def search_trading_memories(
            query: str,
            memory_kinds: List[str] | None = None,
            tags: List[str] | None = None,
            limit: int = MAX_RESULTS_PER_CALL,
        ) -> str:
            """Search approved historical trading memories for this run's fixed symbol."""

            if session._optional_calls >= MAX_OPTIONAL_TOOL_CALLS:
                return json.dumps(
                    {"error": "optional memory search limit reached", "memories": []},
                    ensure_ascii=False,
                )
            session._optional_calls += 1
            try:
                return session._search(
                    query=query,
                    memory_kinds=memory_kinds or (),
                    tags=tags or (),
                    limit=limit,
                    phase="tool",
                )
            except (TypeError, ValueError) as exc:
                return json.dumps(
                    {"error": str(exc), "memories": []},
                    ensure_ascii=False,
                )

        return [search_trading_memories]

    def mark_cited(self, citations: Iterable[str]) -> list[str]:
        available = {
            _memory_ref(memory): memory for memory in self.approved_memories
        }
        valid = []
        for citation in citations:
            normalized = str(citation).strip()
            if normalized in available and normalized not in valid:
                valid.append(normalized)
        if valid and self.run_id:
            self.store.repository.record_usage(
                run_id=self.run_id,
                symbol=self.symbol,
                memories=[available[citation] for citation in valid],
                mode=f"{self.consumer}_applied",
            )
        self.records = [
            record.model_copy(
                update={
                    "cited_ids": [
                        citation
                        for citation in valid
                        if citation in record.approved_ids
                    ]
                }
            )
            for record in self.records
        ]
        return valid

    def _search(
        self,
        *,
        query: str,
        memory_kinds: Iterable[MemoryKind | str],
        tags: Iterable[str],
        limit: int,
        phase: str,
    ) -> str:
        available_slots = max(MAX_UNIQUE_RESULTS - len(self._approved), 0)
        bounded_limit = min(max(int(limit), 1), MAX_RESULTS_PER_CALL)
        effective_limit = min(bounded_limit, available_slots)
        merged_tags = sorted(
            {*self.base_tags, *(tag.strip().lower() for tag in tags if tag.strip())}
        )
        request = MemorySearchRequest(
            symbol=self.symbol,
            consumer=self.consumer,
            phase=phase,
            query=query,
            tags=merged_tags,
            market_regime=self.market_regime,
            timeframe=self.timeframe,
            memory_kinds=list(memory_kinds),
            limit=bounded_limit,
            run_id=self.run_id,
        )

        approved_hits = self._lookup(request, MemoryStatus.APPROVED)
        shadow_hits = self._lookup(request, MemoryStatus.SHADOW)
        unique_approved = _unique_hits(approved_hits, self._approved, effective_limit)
        unique_shadow = _unique_hits(shadow_hits, self._shadow, bounded_limit)
        for hit in unique_approved:
            self._approved[_memory_ref(hit.memory)] = hit.memory
        for hit in unique_shadow:
            self._shadow[_memory_ref(hit.memory)] = hit.memory

        if self.run_id:
            if unique_approved:
                self.store.repository.record_usage(
                    run_id=self.run_id,
                    symbol=self.symbol,
                    memories=[hit.memory for hit in unique_approved],
                    mode=self.consumer,
                )
            if unique_shadow:
                self.store.repository.record_usage(
                    run_id=self.run_id,
                    symbol=self.symbol,
                    memories=[hit.memory for hit in unique_shadow],
                    mode="shadow",
                )

        record = MemoryRetrievalRecord(
            request=request,
            approved_ids=[_memory_ref(hit.memory) for hit in unique_approved],
            shadow_ids=[_memory_ref(hit.memory) for hit in unique_shadow],
            scores={
                _memory_ref(hit.memory): round(hit.score, 6)
                for hit in [*unique_approved, *unique_shadow]
            },
        )
        self.records.append(record)
        return _tool_payload(unique_approved)

    def _lookup(
        self,
        request: MemorySearchRequest,
        status: MemoryStatus,
    ) -> list[MemorySearchHit]:
        return self.store.retrieve_scored(
            symbol=request.symbol,
            tags=request.tags,
            query=request.query,
            market_regime=request.market_regime,
            timeframe=request.timeframe,
            memory_kinds=request.memory_kinds,
            limit=request.limit,
            status=status,
            enforce_scope=True,
        )


def _unique_hits(
    hits: Iterable[MemorySearchHit],
    existing: dict[str, DistilledMemory],
    limit: int,
) -> list[MemorySearchHit]:
    selected = []
    for hit in hits:
        if _memory_ref(hit.memory) in existing:
            continue
        selected.append(hit)
        if len(selected) >= limit:
            break
    return selected


def _tool_payload(hits: Iterable[MemorySearchHit]) -> str:
    items = []
    for hit in hits:
        memory = hit.memory
        items.append(
            {
                "id": _memory_ref(memory),
                "memory_type": memory.memory_type.value,
                "memory_kind": memory.memory_kind.value,
                "scope": memory.scope.value,
                "lesson": memory.lesson,
                "trigger": memory.trigger,
                "rationale": memory.rationale,
                "confidence": memory.confidence,
                "matched_by": hit.matched_by,
                "score": round(hit.score, 3),
            }
        )
    return json.dumps(
        {
            "notice": (
                "Historical context only; live evidence and deterministic risk controls win."
            ),
            "memories": items,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _memory_ref(memory: DistilledMemory) -> str:
    return f"{memory.memory_id}@{memory.version}"


__all__ = ["MemoryRetrievalSession"]

"""Compatibility facade over the versioned SQLite memory repository."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from ai_trading_copilot.copilot.domain import MemoryKind, MemoryScope, MemoryStatus
from ai_trading_copilot.copilot.domain.models import (
    DistilledMemory,
    MemorySearchHit,
    normalize_symbol,
)
from ai_trading_copilot.copilot.services.memory_repository import (
    MemoryRepository,
    SQLiteMemoryRepository,
)
from ai_trading_copilot.copilot.services.vector_memory import (
    LocalVectorMemoryIndex,
    retrieval_query_text,
)


class DistilledMemoryStore:
    """Store facade retaining the original API while SQLite owns the data."""

    def __init__(
        self,
        path: str | Path,
        *,
        default_limit: int = 5,
        vector_index: LocalVectorMemoryIndex | None = None,
        repository: MemoryRepository | None = None,
        database_path: str | Path | None = None,
    ):
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.path = Path(path)
        self.default_limit = default_limit
        self.vector_index = vector_index
        self.repository = repository or SQLiteMemoryRepository(
            database_path or self.path.with_suffix(".sqlite3")
        )
        self.repository.migrate_jsonl(self.path)

    def append(self, memory: DistilledMemory) -> None:
        self.repository.upsert(memory, actor="legacy_api", reason="append")
        self.repository.export_jsonl(self.path)
        if self.vector_index is not None:
            self.vector_index.sync(self.load())

    def load(self) -> List[DistilledMemory]:
        return self.repository.list(statuses=[MemoryStatus.APPROVED])

    def list_all(
        self,
        *,
        statuses: Iterable[MemoryStatus | str] | None = None,
        symbol: str | None = None,
    ) -> List[DistilledMemory]:
        return self.repository.list(statuses=statuses, symbol=symbol)

    def save_candidate(
        self,
        memory: DistilledMemory,
        *,
        actor: str = "reflector",
        reason: str = "post_run_reflection",
    ) -> DistilledMemory:
        candidate = memory.model_copy(update={"status": MemoryStatus.CANDIDATE})
        saved = self.repository.upsert(candidate, actor=actor, reason=reason)
        self.repository.export_jsonl(self.path)
        return saved

    def retrieve(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
        status: MemoryStatus | str = MemoryStatus.APPROVED,
        run_id: str | None = None,
        usage_mode: str = "approved",
    ) -> List[DistilledMemory]:
        return [
            hit.memory
            for hit in self.retrieve_scored(
                symbol=symbol,
                tags=tags,
                query=query,
                limit=limit,
                status=status,
                run_id=run_id,
                usage_mode=usage_mode,
                enforce_scope=False,
            )
        ]

    def retrieve_scored(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        market_regime: str | None = None,
        timeframe: str | None = None,
        memory_kinds: Iterable[MemoryKind | str] = (),
        limit: int | None = None,
        status: MemoryStatus | str = MemoryStatus.APPROVED,
        run_id: str | None = None,
        usage_mode: str = "approved",
        enforce_scope: bool = True,
    ) -> List[MemorySearchHit]:
        """Return explainable hybrid results while SQLite enforces lifecycle state."""

        max_items = self.default_limit if limit is None else min(limit, self.default_limit)
        requested_symbol = normalize_symbol(symbol) if symbol else None
        requested_tags = {tag.strip().lower() for tag in tags if tag.strip()}
        query_terms = _terms(query or "")
        requested_regime = (market_regime or "").strip().lower()
        requested_timeframe = (timeframe or "").strip().lower()
        requested_kinds = {MemoryKind(value) for value in memory_kinds}
        memories = self.repository.list(statuses=[status])
        vector_scores = self._vector_scores(
            memories=memories,
            symbol=requested_symbol,
            tags=requested_tags,
            query=query,
        )

        scored: list[tuple[float, str, DistilledMemory, list[str]]] = []
        for index, memory in enumerate(memories):
            if requested_kinds and memory.memory_kind not in requested_kinds:
                continue

            tag_matches = requested_tags.intersection(memory.tags)
            searchable_terms = _memory_terms(memory)
            term_matches = query_terms.intersection(searchable_terms)
            vector_score = vector_scores.get(index, 0.0)
            symbol_match = bool(requested_symbol and requested_symbol in memory.symbols)
            regime_match = bool(
                requested_regime and requested_regime in memory.market_regimes
            )
            timeframe_match = bool(
                requested_timeframe and requested_timeframe in memory.timeframes
            )

            if enforce_scope:
                if memory.scope == MemoryScope.SYMBOL and not symbol_match:
                    continue
                if memory.scope == MemoryScope.MARKET_REGIME and not regime_match:
                    continue
                if memory.scope == MemoryScope.STRATEGY and not (
                    tag_matches or term_matches
                ):
                    continue

            relevance_signal = bool(
                symbol_match
                or regime_match
                or timeframe_match
                or tag_matches
                or term_matches
                or vector_score > 0
            )
            if enforce_scope and not relevance_signal:
                continue

            score = 0.0
            matched_by: list[str] = []
            if requested_symbol and requested_symbol in memory.symbols:
                score += 100
                matched_by.append("symbol")
            if regime_match:
                score += 30
                matched_by.append("market_regime")
            if timeframe_match:
                score += 15
                matched_by.append("timeframe")
            if tag_matches:
                score += 10 * len(tag_matches)
                matched_by.extend(f"tag:{tag}" for tag in sorted(tag_matches))
            if term_matches:
                score += len(term_matches)
                matched_by.append(f"query_terms:{len(term_matches)}")
            if vector_score > 0:
                score += vector_score * 50
                matched_by.append("vector")
            if memory.confidence is not None:
                score += memory.confidence * 5
            scored.append((score, memory.memory_id, memory, matched_by))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:max_items]
        selected_memories = [memory for _, _, memory, _ in selected]
        if run_id and requested_symbol:
            self.repository.record_usage(
                run_id=run_id,
                symbol=requested_symbol,
                memories=selected_memories,
                mode=usage_mode,
            )
        return [
            MemorySearchHit(
                memory=memory,
                score=score,
                matched_by=matched_by,
                rank=rank,
            )
            for rank, (score, _, memory, matched_by) in enumerate(selected, start=1)
        ]

    def _vector_scores(
        self,
        *,
        memories: List[DistilledMemory],
        symbol: str | None,
        tags: Iterable[str],
        query: str | None,
    ) -> dict[int, float]:
        if self.vector_index is None or not memories:
            return {}
        vector_query = retrieval_query_text(symbol=symbol, tags=tags, query=query)
        if not vector_query:
            return {}
        hits = self.vector_index.search(
            vector_query,
            memories=memories,
            limit=len(memories),
        )
        return {hit.ordinal: hit.score for hit in hits}


def _memory_terms(memory: DistilledMemory) -> set[str]:
    return {
        *_terms(memory.lesson),
        *_terms(memory.trigger),
        *_terms(memory.rationale),
        *(symbol.lower() for symbol in memory.symbols),
        *memory.tags,
        *memory.market_regimes,
        *memory.timeframes,
        memory.memory_type.value,
        memory.memory_kind.value,
    }


def _terms(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", value)
        if len(token) >= 2
    }

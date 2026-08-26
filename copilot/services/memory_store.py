"""Compatibility facade over the versioned SQLite memory repository."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from ai_trading_copilot.copilot.domain import MemoryStatus
from ai_trading_copilot.copilot.domain.models import (
    DistilledMemory,
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
        max_items = self.default_limit if limit is None else min(limit, self.default_limit)
        requested_symbol = normalize_symbol(symbol) if symbol else None
        requested_tags = {tag.strip().lower() for tag in tags if tag.strip()}
        query_terms = _terms(query or "")
        memories = self.repository.list(statuses=[status])
        vector_scores = self._vector_scores(
            memories=memories,
            symbol=requested_symbol,
            tags=requested_tags,
            query=query,
        )

        scored = []
        for index, memory in enumerate(memories):
            score = 0
            if requested_symbol and requested_symbol in memory.symbols:
                score += 100
            score += 10 * len(requested_tags.intersection(memory.tags))
            searchable_terms = _memory_terms(memory)
            score += len(query_terms.intersection(searchable_terms))
            score += vector_scores.get(index, 0.0) * 50
            scored.append((score, index, memory))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [memory for _, _, memory in scored[:max_items]]
        if run_id and requested_symbol:
            self.repository.record_usage(
                run_id=run_id,
                symbol=requested_symbol,
                memories=selected,
                mode=usage_mode,
            )
        return selected

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
        *(symbol.lower() for symbol in memory.symbols),
        *memory.tags,
        memory.memory_type.value,
    }


def _terms(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", value)
        if len(token) >= 2
    }

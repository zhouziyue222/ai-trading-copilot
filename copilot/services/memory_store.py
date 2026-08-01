"""JSONL storage for compact, distilled trading memories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from ai_trading_copilot.copilot.domain.models import (
    DistilledMemory,
    normalize_symbol,
)
from ai_trading_copilot.copilot.services.vector_memory import (
    LocalVectorMemoryIndex,
    retrieval_query_text,
)


class DistilledMemoryStore:
    """Store distilled lessons and retrieve only the most relevant few."""

    def __init__(
        self,
        path: str | Path,
        *,
        default_limit: int = 5,
        vector_index: LocalVectorMemoryIndex | None = None,
    ):
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.path = Path(path)
        self.default_limit = default_limit
        self.vector_index = vector_index

    def append(self, memory: DistilledMemory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(memory.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")
        if self.vector_index is not None:
            self.vector_index.sync(self.load())

    def load(self) -> List[DistilledMemory]:
        if not self.path.exists():
            return []
        memories: List[DistilledMemory] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                memories.append(DistilledMemory(**json.loads(line)))
        return memories

    def retrieve(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
    ) -> List[DistilledMemory]:
        max_items = self.default_limit if limit is None else min(limit, self.default_limit)
        requested_symbol = normalize_symbol(symbol) if symbol else None
        requested_tags = {tag.strip().lower() for tag in tags if tag.strip()}
        query_terms = _terms(query or "")
        memories = self.load()
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
        return [memory for _, _, memory in scored[:max_items]]

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

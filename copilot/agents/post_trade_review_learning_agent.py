"""Post-trade review learning agent."""

from __future__ import annotations

from typing import Iterable, List

from ai_trading_copilot.copilot.domain.models import DistilledMemory
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


class PostTradeReviewLearningAgent:
    """Stores and retrieves compact post-trade lessons only."""

    def __init__(self, store: DistilledMemoryStore):
        self._store = store

    def remember(self, memory: DistilledMemory) -> None:
        self._store.append(memory)

    def retrieve_context(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
    ) -> List[DistilledMemory]:
        return self._store.retrieve(symbol=symbol, tags=tags, query=query, limit=limit)

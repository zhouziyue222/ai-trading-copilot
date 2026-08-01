"""Post-trade review learning agent."""

from __future__ import annotations

from typing import Iterable, List

from ai_trading_copilot.copilot.domain.models import DistilledMemory, RagDocument
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore
from ai_trading_copilot.copilot.services.rag_store import (
    ChromaRagStore,
    DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
)


class PostTradeReviewLearningAgent:
    """Stores and retrieves compact lessons, not full reports."""

    def __init__(
        self,
        store: DistilledMemoryStore,
        *,
        rag_store: ChromaRagStore | None = None,
        llm=None,
    ):
        self._store = store
        self._rag_store = rag_store
        self.llm = llm
        if llm is not None and self._rag_store is not None:
            self._rag_store.set_query_llm(llm)

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

    def retrieve_fundamental_rag_context(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
    ) -> List[RagDocument]:
        max_items = self._store.default_limit if limit is None else min(limit, self._store.default_limit)
        if self._rag_store is None:
            return []
        return self._rag_store.search(
            query=query or "",
            symbol=symbol,
            tags=tags,
            limit=max_items,
        )

    def retrieve_rag_context(self, **kwargs) -> List[RagDocument]:
        return self.retrieve_fundamental_rag_context(**kwargs)

    def set_rag_query_llm(self, llm) -> None:
        self.llm = llm
        if self._rag_store is not None:
            self._rag_store.set_query_llm(llm)

    def rag_status(self) -> dict:
        if self._rag_store is None:
            return {"backend": "none", "available": False, "document_count": 0, "error": ""}
        return self._rag_store.status()

    def ingest_seed_knowledge(self) -> dict:
        if self._rag_store is None:
            return {"added": 0, "skipped": 0, "errors": ["rag store is not configured"]}
        result = self._rag_store.ingest_seed_dir()
        return {"added": result.added, "skipped": result.skipped, "errors": result.errors}

    def ingest_text_knowledge(
        self,
        *,
        title: str,
        text: str,
        source_type: str = DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
        source: str = "manual_ui",
        symbols: Iterable[str] = (),
        tags: Iterable[str] = (),
    ) -> dict:
        if self._rag_store is None:
            return {"added": 0, "skipped": 0, "errors": ["rag store is not configured"]}
        result = self._rag_store.ingest_text(
            title=title,
            text=text,
            source_type=source_type,
            source=source,
            symbols=symbols,
            tags=tags,
        )
        return {"added": result.added, "skipped": result.skipped, "errors": result.errors}

    def ingest_online_fundamental_research(
        self,
        *,
        symbols: Iterable[str],
        trade_date: str | None = None,
        look_back_days: int = 7,
    ) -> dict:
        if self._rag_store is None:
            return {"added": 0, "skipped": 0, "errors": ["rag store is not configured"]}
        result = self._rag_store.ingest_online_fundamental_research(
            symbols=symbols,
            trade_date=trade_date,
            look_back_days=look_back_days,
        )
        return {"added": result.added, "skipped": result.skipped, "errors": result.errors}

    def ingest_online_stock_research(self, **kwargs) -> dict:
        return self.ingest_online_fundamental_research(**kwargs)

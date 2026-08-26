"""Fundamental research retrieval service used by the fundamental analyst."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, List

from ai_trading_copilot.copilot.domain.models import RagDocument
from ai_trading_copilot.copilot.services.eval_samples import (
    rag_documents_eval_payload,
    record_eval_sample,
)
from ai_trading_copilot.copilot.services.rag_store import (
    DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
    FundamentalRagStore,
)


class FundamentalResearchRetriever:
    """Owns fundamental-only RAG search and ingestion."""

    def __init__(self, rag_store: FundamentalRagStore):
        self._rag_store = rag_store

    def retrieve_fundamental_rag_context(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
    ) -> List[RagDocument]:
        started = perf_counter()
        tag_tuple = tuple(tags)
        effective_limit = 5 if limit is None else limit
        docs = self._rag_store.search(
            query=query or "",
            symbol=symbol,
            tags=tag_tuple,
            limit=effective_limit,
        )
        record_eval_sample(
            stage="fundamental_rag_retriever",
            payload={
                "user_input": query or "",
                "retrieval_query": query or "",
                "symbol": symbol,
                "tags": list(tag_tuple),
                "limit": effective_limit,
                "latency_ms": (perf_counter() - started) * 1000.0,
                **rag_documents_eval_payload(docs),
            },
        )
        return docs

    def set_rag_query_llm(self, llm) -> None:
        self._rag_store.set_query_llm(llm)

    def rag_status(self, *, probe: bool = False) -> dict:
        return self._rag_store.status(probe=probe)

    def ingest_seed_knowledge(self) -> dict:
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
        result = self._rag_store.ingest_online_fundamental_research(
            symbols=symbols,
            trade_date=trade_date,
            look_back_days=look_back_days,
        )
        return {"added": result.added, "skipped": result.skipped, "errors": result.errors}

    def ingest_online_stock_research(self, **kwargs) -> dict:
        return self.ingest_online_fundamental_research(**kwargs)


def create_fundamental_research_retriever(path: str | Path) -> FundamentalResearchRetriever:
    return FundamentalResearchRetriever(FundamentalRagStore(path))

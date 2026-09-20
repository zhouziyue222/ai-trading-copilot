"""Fundamental-only Chroma RAG knowledge base."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, List, Sequence

from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    get_fundamentals_text,
)
from ai_trading_copilot.copilot.config.llm import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    load_copilot_env,
)
from ai_trading_copilot.copilot.services.cancellation import check_cancelled
from ai_trading_copilot.copilot.services.diagnostics import record_usage
from ai_trading_copilot.copilot.services.tracing import llm_usage_attributes
from ai_trading_copilot.copilot.domain.models import (
    RagDocument,
    normalize_symbol,
)
from ai_trading_copilot.copilot.services.eval_samples import (
    rag_documents_eval_payload,
    record_eval_sample,
)
from ai_trading_copilot.copilot.services.vector_memory import (
    retrieval_query_text,
)


DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "config" / "rag_chroma"
DEFAULT_COLLECTION_NAME = "ai_trading_copilot_fundamentals"
DEFAULT_RAG_SEED_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "fundamentals"
DEFAULT_STOCK_RESEARCH_SEED_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "stock_research"
DEFAULT_RAG_SEED_DIRS = (DEFAULT_RAG_SEED_DIR, DEFAULT_STOCK_RESEARCH_SEED_DIR)
DEFAULT_RAG_LIMIT = 5
DEFAULT_CHUNK_STRATEGY = "structured_semantic_parent_child_v1"
DEFAULT_PARENT_CHUNK_SIZE = 2200
DEFAULT_PARENT_CHUNK_OVERLAP = 300
DEFAULT_CHILD_CHUNK_SIZE = 500
DEFAULT_CHILD_CHUNK_OVERLAP = 80
DEFAULT_CHUNK_SIZE = DEFAULT_CHILD_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = DEFAULT_CHILD_CHUNK_OVERLAP
DEFAULT_RRF_K = 60
DEFAULT_RERANK_CANDIDATE_FACTOR = 8
DEFAULT_RERANK_MIN_CANDIDATES = 20
DEFAULT_RERANK_MODEL = "local_cross_feature_v1"
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_COLLECTION_GET_BATCH_SIZE = 1000
DEFAULT_FUNDAMENTAL_SOURCE_TYPE = "fundamental_research_note"
ALLOWED_FUNDAMENTAL_SOURCE_TYPES = frozenset(
    {
        "earnings_report",
        "earnings_call_transcript",
        "sec_10q",
        "sec_10k",
        "annual_report",
        "investor_day",
        "company_guidance",
        "fundamental_research_note",
    }
)


class RagUnavailableError(RuntimeError):
    """Raised when the configured vector database cannot be used."""


@dataclass(frozen=True)
class RagIngestResult:
    added: int
    skipped: int
    errors: List[str]


@dataclass(frozen=True)
class TextSection:
    text: str
    start: int
    end: int
    section_title: str
    section_path: str
    section_type: str


@dataclass(frozen=True)
class RagQueryPlan:
    """Expanded retrieval queries used by vector and keyword recall channels."""

    original_query: str
    rewritten_query: str
    perspectives: dict[str, str]
    used_llm: bool
    error: str = ""

    @property
    def queries(self) -> List[str]:
        return _unique_texts(self.perspectives.values())


class RagQueryPlanner:
    """Builds a multi-perspective RAG query plan with optional LLM rewrite."""

    def __init__(self, llm=None):
        self.llm = llm

    def plan(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
    ) -> RagQueryPlan:
        original_query = rag_query_text(symbol=symbol, tags=tags, query=query).strip()
        fallback = self._fallback_plan(original_query)
        if self.llm is None or not original_query:
            return fallback
        try:
            payload = self._rewrite_with_llm(
                original_query=original_query,
                symbol=symbol,
                tags=tags,
            )
            perspectives = {
                "base": str(payload.get("rewritten_query") or fallback.rewritten_query).strip(),
                "risk": str(payload.get("risk_query") or fallback.perspectives["risk"]).strip(),
                "technical": str(
                    payload.get("technical_query") or fallback.perspectives["technical"]
                ).strip(),
                "fundamental": str(
                    payload.get("fundamental_query") or fallback.perspectives["fundamental"]
                ).strip(),
            }
            perspectives = {
                key: value or fallback.perspectives[key]
                for key, value in perspectives.items()
            }
            return RagQueryPlan(
                original_query=original_query,
                rewritten_query=perspectives["base"],
                perspectives=perspectives,
                used_llm=True,
            )
        except Exception as exc:
            return RagQueryPlan(
                original_query=fallback.original_query,
                rewritten_query=fallback.rewritten_query,
                perspectives=fallback.perspectives,
                used_llm=False,
                error=str(exc),
            )

    def _fallback_plan(self, original_query: str) -> RagQueryPlan:
        base = original_query.strip()
        perspectives = {
            "base": base,
            "risk": " ".join(
                [
                    base,
                    "risk stop loss position sizing drawdown event earnings regulatory downside invalidation",
                    "风险 止损 仓位 回撤 财报 监管 下行 失效条件",
                ]
            ).strip(),
            "technical": " ".join(
                [
                    base,
                    "technical support resistance moving average pullback breakout retest trend volume price",
                    "技术 支撑 阻力 均线 回撤 突破 回踩 趋势 量价",
                ]
            ).strip(),
            "fundamental": " ".join(
                [
                    base,
                    "fundamental revenue margin earnings guidance valuation balance sheet cash flow catalyst news",
                    "基本面 收入 毛利 利润 指引 估值 资产负债表 现金流 催化 新闻",
                ]
            ).strip(),
        }
        return RagQueryPlan(
            original_query=base,
            rewritten_query=base,
            perspectives=perspectives,
            used_llm=False,
        )

    def _rewrite_with_llm(
        self,
        *,
        original_query: str,
        symbol: str | None,
        tags: Iterable[str],
    ) -> dict:
        tag_text = ", ".join(tag for tag in tags if tag) or "-"
        prompt = (
            "You rewrite queries for a stock-trading RAG retriever.\n"
            "Return only one JSON object with these string keys: "
            "rewritten_query, risk_query, technical_query, fundamental_query.\n"
            "Keep ticker symbols, dates, and concrete user terms. Expand with useful "
            "English and Chinese retrieval keywords, but do not invent facts.\n\n"
            f"Symbol: {symbol or '-'}\n"
            f"Tags: {tag_text}\n"
            f"Original query: {original_query}\n"
        )
        check_cancelled()
        response = self.llm.invoke(prompt)
        check_cancelled()
        usage_attrs = llm_usage_attributes(response)
        if usage_attrs:
            record_usage(
                kind="llm",
                model=str(
                    getattr(self.llm, "model_name", None)
                    or getattr(self.llm, "model", "")
                    or ""
                ),
                prompt_tokens=usage_attrs.get("usage.prompt_tokens"),
                completion_tokens=usage_attrs.get("usage.completion_tokens"),
                total_tokens=usage_attrs.get("usage.total_tokens"),
                reasoning_tokens=usage_attrs.get("usage.reasoning_tokens"),
            )
        payload = _extract_json_object(_llm_text(response))
        required = {
            "rewritten_query",
            "risk_query",
            "technical_query",
            "fundamental_query",
        }
        if not required.intersection(payload):
            raise ValueError("LLM rewrite did not return query fields")
        return payload


class OpenAITextEmbedder:
    """Small wrapper around OpenAI-compatible embeddings used by RAG upsert/query."""

    name = "openai_embeddings"

    def __init__(self, *, model: str | None = None):
        load_copilot_env()
        api_key = (
            os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        )
        if not api_key:
            raise RagUnavailableError("OPENAI_API_KEY or DASHSCOPE_API_KEY is not set")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        self.name = f"openai:{self.model}"
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency is declared
            raise RagUnavailableError(f"openai is unavailable: {exc}") from exc
        base_url = (
            os.getenv("OPENAI_BASE_URL", "").strip()
            or os.getenv("OPENAI_API_BASE", "").strip()
            or os.getenv("DASHSCOPE_BASE_URL", "").strip()
        )
        client_kwargs = {"model": self.model, "api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self.batch_size = _int_env(
            "AI_TRADING_EMBEDDING_BATCH_SIZE",
            DEFAULT_EMBEDDING_BATCH_SIZE,
            minimum=1,
            maximum=10,
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        output: List[List[float]] = []
        items = list(texts)
        for start in range(0, len(items), self.batch_size):
            output.extend(self._embed_batch(items[start : start + self.batch_size]))
        return output

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed_batch([text])
        return vectors[0]

    def _embed_batch(self, texts: list[str]) -> List[List[float]]:
        from ai_trading_copilot.copilot.services.tracing import (
            get_current_trace_recorder,
            llm_usage_attributes,
        )

        recorder = get_current_trace_recorder()
        attrs = {
            "embedding.model": self.model,
            "embedding.batch_size": len(texts),
        }
        if recorder is not None:
            with recorder.start_span(
                "embedding.invoke",
                kind="client",
                attributes=attrs,
            ) as span:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    for key, value in llm_usage_attributes(response).items():
                        span.set_attribute(key, value)
                    record_usage(
                        kind="embedding",
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                    )
                return [item.embedding for item in response.data]
        response = self._client.embeddings.create(model=self.model, input=texts)
        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(
                kind="embedding",
                model=self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
            )
        return [item.embedding for item in response.data]


class ChromaRagStore:
    """Reference fundamental-only Chroma vector database."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CHROMA_DIR,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: OpenAITextEmbedder | None = None,
        query_planner: RagQueryPlanner | None = None,
        client=None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
        chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedder = embedder
        self.query_planner = query_planner or RagQueryPlanner()
        self._client = client
        self._collection = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.rrf_k = rrf_k
        self.unavailable_reason = ""

    @property
    def available(self) -> bool:
        try:
            self._get_embedder()
            self._get_collection()
        except RagUnavailableError as exc:
            self.unavailable_reason = str(exc)
            return False
        return True

    def status(self, *, probe: bool = False) -> dict:
        count = None
        available = None
        probe_status = "not_run"
        if probe:
            count = 0
            available = self.available
            probe_status = "succeeded" if available else "failed"
            if available:
                try:
                    count = int(self._get_collection().count())
                except Exception as exc:
                    available = False
                    probe_status = "failed"
                    self.unavailable_reason = str(exc)
        return {
            "backend": "fundamental_chroma",
            "available": available,
            "probe_status": probe_status,
            "collection": self.collection_name,
            "path": str(self.path),
            "embedding_model": self._embedding_name(),
            "retrieval": "fundamental_child_bm25_rrf_rerank_parent_context",
            "document_count": count,
            "scope": "fundamental_only",
            "allowed_source_types": sorted(ALLOWED_FUNDAMENTAL_SOURCE_TYPES),
            "chunking": self._chunking_status(),
            "error": "" if available else self.unavailable_reason,
        }

    def upsert_documents(self, documents: Iterable[RagDocument]) -> RagIngestResult:
        input_docs = [doc for doc in documents if doc.text.strip()]
        invalid = [doc for doc in input_docs if not _allowed_source_type(doc.source_type)]
        docs = [doc for doc in input_docs if _allowed_source_type(doc.source_type)]
        errors = [
            f"unsupported fundamental RAG source_type '{doc.source_type}' for {doc.title or doc.id}"
            for doc in invalid
        ]
        if not docs:
            return RagIngestResult(added=0, skipped=len(invalid), errors=errors)
        try:
            embedder = self._get_embedder()
            collection = self._get_collection()
            stale = self._stale_documents(collection, docs)
            if not stale:
                return RagIngestResult(added=0, skipped=len(docs) + len(invalid), errors=errors)
            embeddings = embedder.embed_documents([doc.text for doc in stale])
            collection.upsert(
                ids=[doc.id for doc in stale],
                documents=[doc.text for doc in stale],
                embeddings=embeddings,
                metadatas=[_metadata_for_doc(doc, self._embedding_name()) for doc in stale],
            )
            return RagIngestResult(
                added=len(stale),
                skipped=len(docs) - len(stale) + len(invalid),
                errors=errors,
            )
        except Exception as exc:
            self.unavailable_reason = str(exc)
            return RagIngestResult(added=0, skipped=len(invalid), errors=[*errors, str(exc)])

    def ingest_seed_dir(self, path: str | Path | None = None) -> RagIngestResult:
        roots = [Path(path)] if path is not None else [Path(item) for item in DEFAULT_RAG_SEED_DIRS]
        documents: List[RagDocument] = []
        errors: List[str] = []
        for root in roots:
            if not root.exists():
                errors.append(f"seed dir not found: {root}")
                continue
            for item in sorted(root.rglob("*")):
                if (
                    item.is_file()
                    and item.suffix.lower() in {".md", ".txt", ".jsonl"}
                    and not _is_seed_support_file(item)
                ):
                    documents.extend(
                        documents_from_file(
                            item,
                            parent_chunk_size=self.parent_chunk_size,
                            parent_chunk_overlap=self.parent_chunk_overlap,
                            child_chunk_size=self.chunk_size,
                            child_chunk_overlap=self.chunk_overlap,
                            chunk_strategy=self.chunk_strategy,
                        )
                    )
        result = self.upsert_documents(documents)
        return RagIngestResult(
            added=result.added,
            skipped=result.skipped,
            errors=[*errors, *result.errors],
        )

    def ingest_file(
        self,
        path: str | Path,
        *,
        source_type: str = DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
        symbols: Iterable[str] = (),
        tags: Iterable[str] = (),
    ) -> RagIngestResult:
        documents = documents_from_file(
            Path(path),
            source_type=source_type,
            symbols=symbols,
            tags=tags,
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap,
            child_chunk_size=self.chunk_size,
            child_chunk_overlap=self.chunk_overlap,
            chunk_strategy=self.chunk_strategy,
        )
        return self.upsert_documents(documents)

    def ingest_text(
        self,
        *,
        title: str,
        text: str,
        source_type: str = DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
        source: str = "manual_ui",
        symbols: Iterable[str] = (),
        tags: Iterable[str] = (),
    ) -> RagIngestResult:
        if not _allowed_source_type(source_type):
            return RagIngestResult(
                added=0,
                skipped=0,
                errors=[f"unsupported fundamental RAG source_type '{source_type}'"],
            )
        resolved_source = source
        if source in {"manual_ui", "ui:manual"}:
            resolved_source = f"{source}:{hashlib.sha256((title + text).encode('utf-8')).hexdigest()[:12]}"
        documents = documents_from_text(
            title=title,
            text=text,
            source=resolved_source,
            source_type=source_type,
            symbols=symbols,
            tags=tags,
            parent_chunk_size=self.parent_chunk_size,
            parent_chunk_overlap=self.parent_chunk_overlap,
            child_chunk_size=self.chunk_size,
            child_chunk_overlap=self.chunk_overlap,
            chunk_strategy=self.chunk_strategy,
        )
        return self.upsert_documents(documents)

    def ingest_online_fundamental_research(
        self,
        *,
        symbols: Iterable[str],
        trade_date: str | None = None,
        look_back_days: int = 7,
        fundamentals_fetcher: Callable[[str, str], str] = get_fundamentals_text,
    ) -> RagIngestResult:
        normalized = [normalize_symbol(symbol) for symbol in symbols if symbol.strip()]
        if not normalized:
            return RagIngestResult(added=0, skipped=0, errors=["no symbols provided"])
        _, end_date = date_window(trade_date, look_back_days)
        documents: List[RagDocument] = []
        errors: List[str] = []
        for symbol in normalized:
            try:
                text = str(fundamentals_fetcher(symbol, end_date)).strip()
            except Exception as exc:
                errors.append(f"{symbol} fundamentals: {exc}")
                continue
            documents.extend(
                documents_from_text(
                    title=f"{symbol} fundamentals {end_date}",
                    text=text,
                    source=f"online:fundamentals:{symbol}:{end_date}",
                    source_type=DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
                    symbols=[symbol],
                    tags=["online", "fundamentals", DEFAULT_FUNDAMENTAL_SOURCE_TYPE],
                    parent_chunk_size=self.parent_chunk_size,
                    parent_chunk_overlap=self.parent_chunk_overlap,
                    child_chunk_size=self.chunk_size,
                    child_chunk_overlap=self.chunk_overlap,
                    chunk_strategy=self.chunk_strategy,
                )
            )

        result = self.upsert_documents(documents)
        return RagIngestResult(
            added=result.added,
            skipped=result.skipped,
            errors=[*errors, *result.errors],
        )

    def search(
        self,
        *,
        query: str,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        limit: int = DEFAULT_RAG_LIMIT,
    ) -> List[RagDocument]:
        started = perf_counter()
        tag_tuple = tuple(tags)
        normalized_query = rag_query_text(symbol=symbol, tags=tag_tuple, query=query).strip()
        docs: List[RagDocument] = []
        error = ""
        if limit <= 0 or not normalized_query:
            _record_rag_retrieval_eval_sample(
                stage="rag_optimized_retrieval",
                query=query,
                normalized_query=normalized_query,
                symbol=symbol,
                tags=tag_tuple,
                limit=limit,
                docs=docs,
                latency_ms=(perf_counter() - started) * 1000.0,
                error=error,
            )
            return docs
        try:
            collection = self._get_collection()
            if int(collection.count()) <= 0:
                return docs
            plan = self.plan_query(query=query, symbol=symbol, tags=tag_tuple)
            candidate_limit = _rerank_candidate_limit(limit)
            recall_limit = max(candidate_limit, limit * 4, limit)
            rankings: List[tuple[str, List[RagDocument]]] = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self._vector_rankings,
                        collection=collection,
                        plan=plan,
                        symbol=symbol,
                        tags=tag_tuple,
                        recall_limit=recall_limit,
                    ),
                    executor.submit(
                        self._keyword_rankings,
                        collection=collection,
                        plan=plan,
                        symbol=symbol,
                        tags=tag_tuple,
                        recall_limit=recall_limit,
                    ),
                ]
                for future in futures:
                    try:
                        rankings.extend(future.result())
                    except Exception as exc:
                        self.unavailable_reason = str(exc)

            fused = _rrf_fuse(rankings, limit=candidate_limit, k=self.rrf_k)
            if fused:
                reranked = _rerank_documents(
                    plan,
                    fused,
                    symbol=symbol,
                    tags=tag_tuple,
                    limit=candidate_limit,
                )
                docs = _parent_context_documents(reranked, limit=limit)
            return docs
        except Exception as exc:
            self.unavailable_reason = str(exc)
            error = str(exc)
            return docs
        finally:
            _record_rag_retrieval_eval_sample(
                stage="rag_optimized_retrieval",
                query=query,
                normalized_query=normalized_query,
                symbol=symbol,
                tags=tag_tuple,
                limit=limit,
                docs=docs,
                latency_ms=(perf_counter() - started) * 1000.0,
                error=error,
            )

    def search_keyword_baseline(
        self,
        *,
        query: str,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        limit: int = DEFAULT_RAG_LIMIT,
    ) -> List[RagDocument]:
        """Single-query BM25 retrieval used as a pre-optimization baseline."""
        started = perf_counter()
        tag_tuple = tuple(tags)
        baseline_query = rag_query_text(symbol=symbol, tags=tag_tuple, query=query).strip()
        docs: List[RagDocument] = []
        error = ""
        if limit <= 0 or not baseline_query:
            _record_rag_retrieval_eval_sample(
                stage="rag_baseline_retrieval",
                query=query,
                normalized_query=baseline_query,
                symbol=symbol,
                tags=tag_tuple,
                limit=limit,
                docs=docs,
                latency_ms=(perf_counter() - started) * 1000.0,
                error=error,
            )
            return docs
        try:
            indexed_docs = self._index_documents(symbol=symbol)
            ranked = _bm25_rank(
                baseline_query,
                indexed_docs,
                symbol=symbol,
                tags=tag_tuple,
                limit=limit,
            )
            docs = [
                _with_retrieval_metadata(
                    doc,
                    channel="baseline_bm25",
                    perspective="base",
                    query=baseline_query,
                )
                for doc in ranked
            ]
            return docs
        except Exception as exc:
            self.unavailable_reason = str(exc)
            error = str(exc)
            return docs
        finally:
            _record_rag_retrieval_eval_sample(
                stage="rag_baseline_retrieval",
                query=query,
                normalized_query=baseline_query,
                symbol=symbol,
                tags=tag_tuple,
                limit=limit,
                docs=docs,
                latency_ms=(perf_counter() - started) * 1000.0,
                error=error,
            )

    def plan_query(
        self,
        *,
        query: str,
        symbol: str | None = None,
        tags: Iterable[str] = (),
    ) -> RagQueryPlan:
        return self.query_planner.plan(symbol=symbol, tags=tags, query=query)

    def set_query_llm(self, llm) -> None:
        self.query_planner.llm = llm

    def _vector_rankings(
        self,
        *,
        collection,
        plan: RagQueryPlan,
        symbol: str | None,
        tags: Iterable[str],
        recall_limit: int,
    ) -> List[tuple[str, List[RagDocument]]]:
        try:
            embedder = self._get_embedder()
        except Exception as exc:
            self.unavailable_reason = str(exc)
            return []

        rankings: List[tuple[str, List[RagDocument]]] = []
        where = _chroma_symbol_where(symbol)
        for label, planned_query in _unique_query_items(plan.perspectives):
            if not planned_query.strip():
                continue
            query_kwargs = {
                "query_embeddings": [embedder.embed_query(planned_query)],
                "n_results": recall_limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where
            try:
                result = collection.query(**query_kwargs)
            except TypeError:
                query_kwargs.pop("where", None)
                result = collection.query(**query_kwargs)
            docs = _documents_from_query_result(result)
            docs = _filter_and_rank(docs, symbol=symbol, tags=tags, limit=recall_limit)
            docs = [
                _with_retrieval_metadata(
                    doc,
                    channel="vector",
                    perspective=label,
                    query=planned_query,
                )
                for doc in docs
            ]
            if docs:
                rankings.append((f"vector:{label}", docs))
        return rankings

    def _keyword_rankings(
        self,
        *,
        collection,
        plan: RagQueryPlan,
        symbol: str | None,
        tags: Iterable[str],
        recall_limit: int,
    ) -> List[tuple[str, List[RagDocument]]]:
        docs = _metadata_filtered_docs(
            _all_collection_documents(collection),
            symbol=symbol,
        )
        if not docs:
            return []
        rankings: List[tuple[str, List[RagDocument]]] = []
        for label, planned_query in _unique_query_items(plan.perspectives):
            ranked = _bm25_rank(
                planned_query,
                docs,
                symbol=symbol,
                tags=tags,
                limit=recall_limit,
            )
            ranked = [
                _with_retrieval_metadata(
                    doc,
                    channel="bm25",
                    perspective=label,
                    query=planned_query,
                )
                for doc in ranked
            ]
            if ranked:
                rankings.append((f"bm25:{label}", ranked))
        return rankings

    def _get_embedder(self) -> OpenAITextEmbedder:
        if self.embedder is None:
            self.embedder = OpenAITextEmbedder()
        return self.embedder

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if self._client is None:
            try:
                import chromadb
            except Exception as exc:
                raise RagUnavailableError(f"chromadb is unavailable: {exc}") from exc
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _stale_documents(self, collection, documents: List[RagDocument]) -> List[RagDocument]:
        ids = [doc.id for doc in documents]
        existing = collection.get(ids=ids, include=["metadatas"])
        existing_meta = {
            item_id: metadata or {}
            for item_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", []))
        }
        stale = []
        for doc in documents:
            content_hash = _document_content_hash(doc)
            if existing_meta.get(doc.id, {}).get("content_hash") != content_hash:
                stale.append(doc)
        return stale

    def _embedding_name(self) -> str:
        if self.embedder is not None:
            return self.embedder.name
        return f"openai:{os.getenv('OPENAI_EMBEDDING_MODEL', DEFAULT_OPENAI_EMBEDDING_MODEL)}"

    def _chunking_status(self) -> dict:
        return {
            "strategy": self.chunk_strategy,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "child_chunk_size": self.chunk_size,
            "child_chunk_overlap": self.chunk_overlap,
        }

    def _index_documents(self, *, symbol: str | None) -> List[RagDocument]:
        return _metadata_filtered_docs(
            _all_collection_documents(self._get_collection()),
            symbol=symbol,
        )


FundamentalRagStore = ChromaRagStore


def documents_from_file(
    path: Path,
    *,
    source_type: str = DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
    symbols: Iterable[str] = (),
    tags: Iterable[str] = (),
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
) -> List[RagDocument]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        documents: List[RagDocument] = []
        for index, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            payload = json.loads(line)
            title = str(payload.get("title") or f"{path.name}:{index}")
            documents.extend(
                documents_from_text(
                    title=title,
                    text=str(payload.get("text") or payload.get("lesson") or ""),
                    source=str(payload.get("source") or path),
                    source_type=str(payload.get("source_type") or source_type),
                    symbols=payload.get("symbols") or symbols,
                    tags=payload.get("tags") or tags,
                    parent_chunk_size=parent_chunk_size,
                    parent_chunk_overlap=parent_chunk_overlap,
                    child_chunk_size=child_chunk_size,
                    child_chunk_overlap=child_chunk_overlap,
                    chunk_strategy=chunk_strategy,
                )
            )
        return documents
    title = _title_from_markdown(text) or path.stem.replace("_", " ").title()
    return documents_from_text(
        title=title,
        text=text,
        source=str(path),
        source_type=source_type,
        symbols=symbols,
        tags=tags,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        chunk_strategy=chunk_strategy,
    )


def _is_seed_support_file(path: Path) -> bool:
    support_dirs = {"reports", "__pycache__"}
    if any(part.lower() in support_dirs for part in path.parts):
        return True
    return path.name.lower() in {
        "eval_contexts.jsonl",
        "normalize_report.md",
        "manifest.md",
    }


def documents_from_text(
    *,
    title: str,
    text: str,
    source: str,
    source_type: str = DEFAULT_FUNDAMENTAL_SOURCE_TYPE,
    symbols: Iterable[str] = (),
    tags: Iterable[str] = (),
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
) -> List[RagDocument]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if symbol.strip()]
    normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
    created_at = datetime.now(timezone.utc).isoformat()
    if chunk_size is not None:
        child_chunk_size = chunk_size
    if chunk_overlap is not None:
        child_chunk_overlap = chunk_overlap
    if chunk_strategy.lower() in {"flat", "flat_v1"}:
        return _flat_documents_from_text(
            title=title,
            text=cleaned,
            source=source,
            source_type=source_type,
            symbols=normalized_symbols,
            tags=normalized_tags,
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            created_at=created_at,
        )
    return _parent_child_documents_from_text(
        title=title,
        text=cleaned,
        source=source,
        source_type=source_type,
        symbols=normalized_symbols,
        tags=normalized_tags,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        chunk_strategy=chunk_strategy,
        created_at=created_at,
    )


def _flat_documents_from_text(
    *,
    title: str,
    text: str,
    source: str,
    source_type: str,
    symbols: List[str],
    tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    created_at: str,
) -> List[RagDocument]:
    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    return [
        RagDocument(
            id=_document_id(source_type, source, index),
            title=title,
            text=chunk,
            source=source,
            source_type=source_type,
            symbols=symbols,
            tags=tags,
            created_at=created_at,
        )
        for index, chunk in enumerate(chunks)
    ]


def _parent_child_documents_from_text(
    *,
    title: str,
    text: str,
    source: str,
    source_type: str,
    symbols: List[str],
    tags: List[str],
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
    chunk_strategy: str,
    created_at: str,
) -> List[RagDocument]:
    documents: List[RagDocument] = []
    parent_sections = _semantic_parent_sections(
        text,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
    )
    for parent_index, section in enumerate(parent_sections):
        parent_text = section.text
        parent_start = section.start
        parent_end = section.end
        parent_id = _document_id(source_type, source, f"{chunk_strategy}:parent:{parent_index}")
        child_spans = _chunk_text_spans(
            parent_text,
            chunk_size=child_chunk_size,
            overlap=child_chunk_overlap,
        )
        for child_index, (child_text, child_start, child_end) in enumerate(child_spans):
            documents.append(
                RagDocument(
                    id=_document_id(
                        source_type,
                        source,
                        f"{chunk_strategy}:parent:{parent_index}:child:{child_index}",
                    ),
                    title=title,
                    text=child_text,
                    source=source,
                    source_type=source_type,
                    symbols=symbols,
                    tags=tags,
                    created_at=created_at,
                    metadata={
                        "chunk_strategy": chunk_strategy,
                        "chunk_role": "child",
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "parent_index": parent_index,
                        "parent_char_start": parent_start,
                        "parent_char_end": parent_end,
                        "child_index": child_index,
                        "child_char_start": child_start,
                        "child_char_end": child_end,
                        "section_title": section.section_title,
                        "section_path": section.section_path,
                        "section_type": section.section_type,
                        "filing_type": source_type,
                        "symbol": symbols[0] if symbols else "",
                        "company_name": "",
                        "source_date": created_at[:10],
                    },
                )
            )
    return documents


def _semantic_parent_sections(
    text: str,
    *,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    ) -> List[TextSection]:
    sections = _heading_sections(text)
    if not sections:
        sections = _unstructured_sections(
            text,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )
    output: List[TextSection] = []
    for section in sections:
        if len(section.text) <= parent_chunk_size:
            output.append(section)
            continue
        for split_index, (chunk, start, end) in enumerate(
            _chunk_text_spans(
                section.text,
                chunk_size=parent_chunk_size,
                overlap=parent_chunk_overlap,
            )
        ):
            title = section.section_title
            if split_index:
                title = f"{title} part {split_index + 1}" if title else f"part {split_index + 1}"
            output.append(
                TextSection(
                    text=chunk,
                    start=section.start + start,
                    end=section.start + end,
                    section_title=title,
                    section_path=section.section_path,
                    section_type=section.section_type,
                )
            )
    return output


def _heading_sections(text: str) -> List[TextSection]:
    headings: List[tuple[int, int, int, str]] = []
    offset = 0
    stack: list[str] = []
    line_items = text.splitlines(keepends=True)
    for line in line_items:
        stripped = line.strip()
        level = 0
        title = ""
        markdown = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if markdown:
            level = len(markdown.group(1))
            title = markdown.group(2).strip()
        elif _looks_like_financial_heading(stripped):
            level = 2
            title = stripped
        if title:
            stack = stack[: max(level - 1, 0)]
            stack.append(title)
            headings.append((offset, level, len(line), " > ".join(stack)))
        offset += len(line)

    if not headings:
        return []

    sections: List[TextSection] = []
    for index, (start, _level, _line_len, path) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        title = path.split(" > ")[-1]
        section_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
        section_end = start + len(text[start:end].rstrip())
        sections.append(
            TextSection(
                text=section_text,
                start=section_start,
                end=section_end,
                section_title=title,
                section_path=path,
                section_type=_section_type_for_title(title),
            )
        )
    return sections


def _unstructured_sections(
    text: str,
    *,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
) -> List[TextSection]:
    paragraphs = _paragraph_spans(text)
    if not paragraphs:
        cleaned = text.strip()
        return [
            TextSection(
                text=cleaned,
                start=0,
                end=len(cleaned),
                section_title="",
                section_path="",
                section_type="unstructured_fallback",
            )
        ] if cleaned else []
    sections: List[TextSection] = []
    current_start: int | None = None
    current_end = 0

    def flush_current() -> None:
        nonlocal current_start, current_end
        if current_start is None:
            return
        raw = text[current_start:current_end]
        cleaned = raw.strip()
        if cleaned:
            start = current_start + (len(raw) - len(raw.lstrip()))
            end = current_start + len(raw.rstrip())
            sections.append(
                TextSection(
                    text=cleaned,
                    start=start,
                    end=end,
                    section_title="",
                    section_path="",
                    section_type="unstructured_fallback",
                )
            )
        current_start = None
        current_end = 0

    for paragraph, start, end in paragraphs:
        if len(paragraph) > parent_chunk_size:
            flush_current()
            for chunk, chunk_start, chunk_end in _chunk_text_spans(
                paragraph,
                chunk_size=parent_chunk_size,
                overlap=parent_chunk_overlap,
            ):
                sections.append(
                    TextSection(
                        text=chunk,
                        start=start + chunk_start,
                        end=start + chunk_end,
                        section_title="",
                        section_path="",
                        section_type="unstructured_fallback",
                    )
                )
            continue

        candidate_start = start if current_start is None else current_start
        candidate = text[candidate_start:end].strip()
        if current_start is not None and len(candidate) > parent_chunk_size:
            flush_current()
        if current_start is None:
            current_start = start
        current_end = end

    flush_current()
    return sections


def _paragraph_spans(text: str) -> List[tuple[str, int, int]]:
    spans: List[tuple[str, int, int]] = []
    for match in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        raw = match.group(0)
        cleaned = raw.strip()
        if not cleaned:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = match.start() + len(raw.rstrip())
        spans.append((cleaned, start, end))
    return spans


def _looks_like_financial_heading(value: str) -> bool:
    if not value or len(value) > 120:
        return False
    return bool(
        re.match(r"(?i)^item\s+\d+[a-z]?\.?\s+", value)
        or re.match(
            r"(?i)^(risk factors|management discussion|management's discussion|"
            r"revenue|gross margin|guidance|outlook|q&a|question and answer|"
            r"liquidity|cash flow|capital expenditure|capex|segment results|"
            r"balance sheet|income statement|results of operations)$",
            value,
        )
    )


def _section_type_for_title(title: str) -> str:
    lower = title.lower()
    if "risk" in lower:
        return "risk_factors"
    if "management" in lower and "discussion" in lower:
        return "management_discussion"
    if "gross margin" in lower or "margin" in lower:
        return "gross_margin"
    if "revenue" in lower or "sales" in lower:
        return "revenue"
    if "guidance" in lower or "outlook" in lower:
        return "guidance"
    if "q&a" in lower or "question" in lower:
        return "qa"
    if "cash flow" in lower or "liquidity" in lower:
        return "cash_flow"
    if "capex" in lower or "capital expenditure" in lower:
        return "capex"
    if re.match(r"item\s+\d", lower):
        return "sec_item"
    return "fundamental_section"


def rag_query_text(
    *,
    symbol: str | None = None,
    tags: Iterable[str] = (),
    query: str | None = None,
) -> str:
    return retrieval_query_text(symbol=symbol, tags=tags, query=query)


def _all_collection_documents(collection) -> List[RagDocument]:
    include = ["documents", "metadatas"]
    try:
        total = int(collection.count())
    except Exception:
        total = 0

    if total > 0:
        output: List[RagDocument] = []
        offset = 0
        while offset < total:
            try:
                result = collection.get(
                    include=include,
                    limit=DEFAULT_COLLECTION_GET_BATCH_SIZE,
                    offset=offset,
                )
            except TypeError:
                break
            batch = _documents_from_get_result(result)
            if not batch:
                break
            output.extend(batch)
            offset += len(batch)
        if output:
            return output

    try:
        result = collection.get(include=include)
    except TypeError:
        result = collection.get(ids=None, include=include)
    return _documents_from_get_result(result)


def _documents_from_query_result(result: dict) -> List[RagDocument]:
    ids = _first(result.get("ids"))
    documents = _first(result.get("documents"))
    metadatas = _first(result.get("metadatas"))
    distances = _first(result.get("distances"))
    output = []
    if not distances:
        distances = [0.0 for _ in ids]
    for item_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        output.append(
            RagDocument(
                id=str(item_id),
                text=str(text),
                source_type=str(metadata.get("source_type") or DEFAULT_FUNDAMENTAL_SOURCE_TYPE),
                source=str(metadata.get("source") or ""),
                title=str(metadata.get("title") or ""),
                symbols=_split_metadata_list(str(metadata.get("symbols") or "")),
                tags=_split_metadata_list(str(metadata.get("tags") or "")),
                created_at=str(metadata.get("created_at") or "") or None,
                score=max(0.0, 1.0 - float(distance or 0.0)),
                metadata=dict(metadata),
            )
        )
    return output


def _documents_from_get_result(result: dict) -> List[RagDocument]:
    ids = _first(result.get("ids"))
    documents = _first(result.get("documents"))
    metadatas = _first(result.get("metadatas"))
    output = []
    for item_id, text, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        output.append(
            RagDocument(
                id=str(item_id),
                text=str(text),
                source_type=str(metadata.get("source_type") or DEFAULT_FUNDAMENTAL_SOURCE_TYPE),
                source=str(metadata.get("source") or ""),
                title=str(metadata.get("title") or ""),
                symbols=_split_metadata_list(str(metadata.get("symbols") or "")),
                tags=_split_metadata_list(str(metadata.get("tags") or "")),
                created_at=str(metadata.get("created_at") or "") or None,
                score=None,
                metadata=dict(metadata),
            )
        )
    return output


def _filter_and_rank(
    docs: List[RagDocument],
    *,
    symbol: str | None,
    tags: Iterable[str],
    limit: int,
) -> List[RagDocument]:
    requested_symbol = normalize_symbol(symbol) if symbol else None
    requested_tags = {tag.strip().lower() for tag in tags if tag.strip()}
    scored = []
    for index, doc in enumerate(docs):
        if not _metadata_matches(doc, requested_symbol):
            continue
        score = doc.score or 0.0
        score += _metadata_boost(doc, requested_symbol, requested_tags)
        scored.append((score, -index, doc.model_copy(update={"score": score})))
    scored.sort(reverse=True)
    return [doc for _, _, doc in scored[:limit]]


def _metadata_filtered_docs(
    docs: Iterable[RagDocument],
    *,
    symbol: str | None,
) -> List[RagDocument]:
    requested_symbol = normalize_symbol(symbol) if symbol else None
    return [doc for doc in docs if _metadata_matches(doc, requested_symbol)]


def _metadata_matches(doc: RagDocument, requested_symbol: str | None) -> bool:
    return not (
        requested_symbol
        and doc.symbols
        and requested_symbol not in doc.symbols
    )


def _chroma_symbol_where(symbol: str | None) -> dict | None:
    normalized = normalize_symbol(symbol) if symbol else ""
    return {"symbols": normalized} if normalized else None


def _metadata_boost(
    doc: RagDocument,
    requested_symbol: str | None,
    requested_tags: set[str],
) -> float:
    score = 0.0
    if requested_symbol and requested_symbol in doc.symbols:
        score += 0.15
    score += 0.03 * len(requested_tags.intersection(doc.tags))
    return score


def _bm25_rank(
    query: str,
    docs: List[RagDocument],
    *,
    symbol: str | None,
    tags: Iterable[str],
    limit: int,
) -> List[RagDocument]:
    query_tokens = _keyword_tokens(query)
    if limit <= 0 or not query_tokens or not docs:
        return []

    query_counts = Counter(query_tokens)
    doc_terms = [Counter(_keyword_tokens(_keyword_text_for_doc(doc))) for doc in docs]
    doc_lengths = [sum(terms.values()) for terms in doc_terms]
    if not any(doc_lengths):
        return []

    n_docs = len(docs)
    avg_doc_len = sum(doc_lengths) / n_docs
    doc_freq = Counter()
    for terms in doc_terms:
        for token in query_counts:
            if token in terms:
                doc_freq[token] += 1

    requested_symbol = normalize_symbol(symbol) if symbol else None
    requested_tags = {tag.strip().lower() for tag in tags if tag.strip()}
    k1 = 1.5
    b = 0.75
    scored = []
    for index, (doc, terms, doc_len) in enumerate(zip(docs, doc_terms, doc_lengths)):
        if doc_len == 0:
            continue
        score = 0.0
        for token, query_count in query_counts.items():
            freq = terms.get(token, 0)
            if not freq:
                continue
            df = doc_freq[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            norm = freq + k1 * (1.0 - b + b * doc_len / avg_doc_len)
            score += idf * (freq * (k1 + 1.0) / norm) * (1.0 + 0.05 * (query_count - 1))
        if score <= 0:
            continue
        score += _metadata_boost(doc, requested_symbol, requested_tags)
        scored.append((score, -index, doc.model_copy(update={"score": score})))

    scored.sort(reverse=True)
    return [doc for _, _, doc in scored[:limit]]


def _keyword_text_for_doc(doc: RagDocument) -> str:
    return " ".join(
        part
        for part in [
            doc.title,
            doc.text,
            doc.source_type,
            doc.source,
            " ".join(doc.symbols),
            " ".join(doc.tags),
        ]
        if part
    )


def _keyword_tokens(value: str) -> List[str]:
    tokens = [
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z0-9_.$%-]+", value)
        if match.group(0).strip("._-$%")
    ]
    for match in re.finditer(r"[\u3400-\u9fff]+", value):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _rrf_fuse(
    rankings: Iterable[tuple[str, List[RagDocument]]],
    *,
    limit: int,
    k: int = DEFAULT_RRF_K,
) -> List[RagDocument]:
    fused_scores: dict[str, float] = defaultdict(float)
    best_docs: dict[str, RagDocument] = {}
    best_source_scores: dict[str, float] = defaultdict(float)
    channels_by_id: dict[str, set[str]] = defaultdict(set)

    for channel, docs in rankings:
        seen_in_channel = set()
        for rank, doc in enumerate(docs, start=1):
            if doc.id in seen_in_channel:
                continue
            seen_in_channel.add(doc.id)
            fused_scores[doc.id] += 1.0 / (k + rank)
            channels_by_id[doc.id].add(channel)
            source_score = doc.score or 0.0
            if doc.id not in best_docs or source_score > best_source_scores[doc.id]:
                best_docs[doc.id] = doc
                best_source_scores[doc.id] = source_score

    ordered = sorted(
        fused_scores,
        key=lambda item_id: (
            fused_scores[item_id],
            best_source_scores[item_id],
            best_docs[item_id].title,
        ),
        reverse=True,
    )
    output = []
    for item_id in ordered[:limit]:
        doc = best_docs[item_id]
        metadata = dict(doc.metadata)
        metadata["retrieval_channels"] = ",".join(sorted(channels_by_id[item_id]))
        metadata["rrf_score"] = round(fused_scores[item_id], 6)
        output.append(
            doc.model_copy(
                update={
                    "score": round(fused_scores[item_id], 6),
                    "metadata": metadata,
                }
            )
        )
    return output


def _rerank_candidate_limit(limit: int) -> int:
    return max(limit, limit * DEFAULT_RERANK_CANDIDATE_FACTOR, DEFAULT_RERANK_MIN_CANDIDATES)


def _rerank_documents(
    plan: RagQueryPlan,
    docs: Sequence[RagDocument],
    *,
    symbol: str | None,
    tags: Iterable[str],
    limit: int,
) -> List[RagDocument]:
    if limit <= 0 or not docs:
        return []

    primary_query = " ".join(
        _unique_texts([plan.original_query, plan.rewritten_query])
    )
    expanded_query = " ".join(plan.queries)
    primary_counts = Counter(_keyword_tokens(primary_query))
    expanded_counts = Counter(_keyword_tokens(expanded_query))
    if not primary_counts:
        primary_counts = expanded_counts

    requested_symbol = normalize_symbol(symbol) if symbol else None
    requested_tags = {tag.strip().lower() for tag in tags if tag.strip()}
    scored: list[tuple[float, float, str, RagDocument]] = []
    for index, doc in enumerate(docs):
        full_terms = Counter(_keyword_tokens(_keyword_text_for_doc(doc)))
        title_terms = Counter(_keyword_tokens(doc.title))
        primary_overlap = _weighted_token_overlap(primary_counts, full_terms)
        expanded_overlap = _weighted_token_overlap(expanded_counts, full_terms)
        title_overlap = _weighted_token_overlap(primary_counts, title_terms)
        metadata_score = _metadata_boost(doc, requested_symbol, requested_tags)
        channel_count = _retrieval_channel_count(doc)
        rrf_score = _metadata_float(doc.metadata.get("rrf_score"), doc.score or 0.0)

        rerank_score = (
            0.55 * primary_overlap
            + 0.20 * expanded_overlap
            + 0.10 * title_overlap
            + min(metadata_score, 0.20)
            + min(0.02 * channel_count, 0.12)
            + min(rrf_score * 2.0, 0.35)
        )
        metadata = dict(doc.metadata)
        metadata["rerank_model"] = DEFAULT_RERANK_MODEL
        metadata["rerank_score"] = round(rerank_score, 6)
        metadata["rerank_features"] = (
            f"primary={primary_overlap:.3f},expanded={expanded_overlap:.3f},"
            f"title={title_overlap:.3f},channels={channel_count},rrf={rrf_score:.6f}"
        )
        scored.append(
            (
                rerank_score,
                rrf_score,
                f"{len(docs) - index:08d}:{doc.title}",
                doc.model_copy(
                    update={
                        "score": round(max(rerank_score, 0.0), 6),
                        "metadata": metadata,
                    }
                ),
            )
        )

    scored.sort(reverse=True)
    return [doc for _, _, _, doc in scored[:limit]]


def _parent_context_documents(docs: Sequence[RagDocument], *, limit: int) -> List[RagDocument]:
    if limit <= 0 or not docs:
        return []

    groups: dict[str, dict] = {}
    for doc in docs:
        parent_id = _parent_id_for_doc(doc)
        if parent_id not in groups:
            groups[parent_id] = {
                "best": doc,
                "children": [],
                "channels": set(),
                "rrf_score": 0.0,
                "rerank_score": 0.0,
            }
        group = groups[parent_id]
        group["children"].append(doc)
        group["channels"].update(_retrieval_channels_for_doc(doc))
        group["rrf_score"] = max(
            group["rrf_score"],
            _metadata_float(doc.metadata.get("rrf_score"), doc.score or 0.0),
        )
        group["rerank_score"] = max(
            group["rerank_score"],
            _metadata_float(doc.metadata.get("rerank_score"), doc.score or 0.0),
        )

    output = []
    for parent_id, group in list(groups.items())[:limit]:
        best = group["best"]
        children: List[RagDocument] = group["children"]
        parent_text = str(best.metadata.get("parent_text") or best.text)
        parent_doc_id = str(best.metadata.get("parent_id") or parent_id)
        matched_child_ids = [child.id for child in children[:5]]
        matched_child_texts = [child.text for child in children[:3]]
        metadata = dict(best.metadata)
        metadata["chunk_role"] = "parent_context" if _is_child_chunk(best) else metadata.get("chunk_role", "flat")
        metadata["parent_id"] = parent_doc_id
        metadata["matched_child_ids"] = matched_child_ids
        metadata["matched_child_texts"] = matched_child_texts
        metadata["matched_child_count"] = len(children)
        metadata["retrieval_channels"] = ",".join(sorted(group["channels"]))
        metadata["rrf_score"] = round(group["rrf_score"], 6)
        metadata["rerank_score"] = round(group["rerank_score"], 6)
        output.append(
            best.model_copy(
                update={
                    "id": parent_doc_id,
                    "text": parent_text,
                    "score": round(group["rerank_score"], 6),
                    "metadata": metadata,
                }
            )
        )
    return output


def _parent_id_for_doc(doc: RagDocument) -> str:
    if _is_child_chunk(doc):
        return str(doc.metadata.get("parent_id") or doc.id)
    return doc.id


def _is_child_chunk(doc: RagDocument) -> bool:
    return (
        str(doc.metadata.get("chunk_strategy") or "") == DEFAULT_CHUNK_STRATEGY
        and str(doc.metadata.get("chunk_role") or "") == "child"
        and bool(doc.metadata.get("parent_text"))
    )


def _retrieval_channels_for_doc(doc: RagDocument) -> set[str]:
    channels = str(doc.metadata.get("retrieval_channels", "") or "")
    return {channel.strip() for channel in channels.split(",") if channel.strip()}


def _weighted_token_overlap(query_counts: Counter[str], doc_terms: Counter[str]) -> float:
    total = sum(query_counts.values())
    if total <= 0:
        return 0.0
    matched = sum(count for token, count in query_counts.items() if doc_terms.get(token, 0) > 0)
    return matched / total


def _retrieval_channel_count(doc: RagDocument) -> int:
    channels = str(doc.metadata.get("retrieval_channels", "") or "")
    if not channels:
        return 0
    return len({channel.strip() for channel in channels.split(",") if channel.strip()})


def _metadata_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _allowed_source_type(source_type: str) -> bool:
    return source_type.strip().lower() in ALLOWED_FUNDAMENTAL_SOURCE_TYPES


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _with_retrieval_metadata(
    doc: RagDocument,
    *,
    channel: str,
    perspective: str,
    query: str,
) -> RagDocument:
    metadata = dict(doc.metadata)
    metadata["retrieval_channel"] = channel
    metadata["query_perspective"] = perspective
    metadata["retrieval_query"] = query[:500]
    return doc.model_copy(update={"metadata": metadata})


def _record_rag_retrieval_eval_sample(
    *,
    stage: str,
    query: str,
    normalized_query: str,
    symbol: str | None,
    tags: Sequence[str],
    limit: int,
    docs: Sequence[RagDocument],
    latency_ms: float,
    error: str,
) -> None:
    payload = {
        "user_input": query,
        "retrieval_query": normalized_query,
        "symbol": normalize_symbol(symbol or "") or None,
        "tags": list(tags),
        "limit": limit,
        "latency_ms": latency_ms,
        "error": error,
        **rag_documents_eval_payload(docs),
    }
    record_eval_sample(stage=stage, payload=payload)


def _metadata_for_doc(doc: RagDocument, embedding_model: str) -> dict:
    metadata = {
        "title": doc.title,
        "source": doc.source,
        "source_type": doc.source_type,
        "symbols": ",".join(doc.symbols),
        "tags": ",".join(doc.tags),
        "created_at": doc.created_at or "",
        "embedding_model": embedding_model,
        "content_hash": _document_content_hash(doc),
    }
    reserved = set(metadata)
    for key, value in doc.metadata.items():
        if key in reserved:
            continue
        converted = _chroma_metadata_value(value)
        if converted is not None:
            metadata[key] = converted
    return metadata


def _chroma_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> List[str]:
    return [chunk for chunk, _, _ in _chunk_text_spans(text, chunk_size=chunk_size, overlap=overlap)]


def _chunk_text_spans(text: str, *, chunk_size: int, overlap: int) -> List[tuple[str, int, int]]:
    if len(text) <= chunk_size:
        cleaned = text.strip()
        if not cleaned:
            return []
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        return [(cleaned, start, end)]
    chunks: List[tuple[str, int, int]] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        raw = text[start:end]
        stripped_left = len(raw) - len(raw.lstrip())
        stripped_right = len(raw.rstrip())
        chunk_start = start + stripped_left
        chunk_end = start + stripped_right
        chunk = text[chunk_start:chunk_end]
        if chunk:
            chunks.append((chunk, chunk_start, chunk_end))
        start += step
    return chunks


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()


def _title_from_markdown(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _document_id(source_type: str, source: str, chunk_index: int | str) -> str:
    raw = f"{source_type}|{source}|{chunk_index}"
    return "rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _document_content_hash(doc: RagDocument) -> str:
    payload = {
        "title": doc.title,
        "text": doc.text,
        "source": doc.source,
        "source_type": doc.source_type,
        "symbols": doc.symbols,
        "tags": doc.tags,
        "metadata": doc.metadata,
    }
    return _content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first(value):
    if not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else value


def _split_metadata_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique_query_items(values: dict[str, str]) -> List[tuple[str, str]]:
    seen = set()
    items = []
    for label, value in values.items():
        cleaned = value.strip()
        key = re.sub(r"\s+", " ", cleaned.lower())
        if not cleaned or key in seen:
            continue
        seen.add(key)
        items.append((label, cleaned))
    return items


def _unique_texts(values: Iterable[str]) -> List[str]:
    return [value for _, value in _unique_query_items({str(index): item for index, item in enumerate(values)})]


def _llm_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(getattr(item, "content", item)) for item in content)
    return str(content)


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM rewrite response is not a JSON object")
    return payload

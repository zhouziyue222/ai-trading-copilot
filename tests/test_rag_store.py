from ai_trading_copilot.copilot.domain import RagDocument
from ai_trading_copilot.copilot.services.rag_store import (
    ChromaRagStore,
    RagQueryPlan,
    RagQueryPlanner,
    _rerank_documents,
    documents_from_text,
)


class FakeEmbedder:
    name = "fake-embedding"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lower = text.lower()
        if "earnings" in lower or "guidance" in lower:
            return [1.0, 0.0]
        if "support" in lower or "pullback" in lower:
            return [0.0, 1.0]
        return [0.0, 0.0]


class FakeCollection:
    def __init__(self):
        self.records = {}

    def get(self, ids=None, include=None):
        found_ids = [
            item_id
            for item_id in (ids if ids is not None else self.records.keys())
            if item_id in self.records
        ]
        payload = {"ids": found_ids}
        if include is None or "documents" in include:
            payload["documents"] = [self.records[item_id]["document"] for item_id in found_ids]
        if include is None or "metadatas" in include:
            payload["metadatas"] = [self.records[item_id]["metadata"] for item_id in found_ids]
        return payload

    def upsert(self, ids, documents, embeddings, metadatas):
        for item_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.records[item_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(self, query_embeddings, n_results, include):
        query = query_embeddings[0]
        scored = []
        for item_id, record in self.records.items():
            score = sum(left * right for left, right in zip(query, record["embedding"]))
            scored.append((score, item_id, record))
        scored.sort(reverse=True)
        selected = scored[:n_results]
        return {
            "ids": [[item_id for _, item_id, _ in selected]],
            "documents": [[record["document"] for _, _, record in selected]],
            "metadatas": [[record["metadata"] for _, _, record in selected]],
            "distances": [[1.0 - score for score, _, _ in selected]],
        }

    def count(self):
        return len(self.records)


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()

    def get_or_create_collection(self, name, metadata=None):
        return self.collection


class FakeRewriteLLM:
    def invoke(self, prompt):
        return (
            '{"rewritten_query":"AAPL earnings quality and regulatory risk",'
            '"risk_query":"AAPL regulatory probe downside stop loss drawdown",'
            '"technical_query":"AAPL pullback support resistance moving average",'
            '"fundamental_query":"AAPL revenue margin earnings guidance valuation"}'
        )


def test_rag_query_planner_uses_llm_rewrite_and_perspective_expansion():
    planner = RagQueryPlanner(llm=FakeRewriteLLM())

    plan = planner.plan(
        query="earnings risk",
        symbol="AAPL",
        tags=["pullback"],
    )

    assert plan.used_llm is True
    assert plan.rewritten_query == "AAPL earnings quality and regulatory risk"
    assert set(plan.perspectives) == {"base", "risk", "technical", "fundamental"}
    assert "stop loss" in plan.perspectives["risk"]
    assert "moving average" in plan.perspectives["technical"]
    assert "valuation" in plan.perspectives["fundamental"]


def test_rag_query_planner_falls_back_to_deterministic_multi_query_expansion():
    plan = RagQueryPlanner().plan(query="AAPL pullback", symbol="AAPL")

    assert plan.used_llm is False
    assert len(plan.queries) == 4
    assert "risk stop loss" in plan.perspectives["risk"]
    assert "technical support" in plan.perspectives["technical"]
    assert "fundamental revenue" in plan.perspectives["fundamental"]


def test_documents_from_text_uses_parent_child_chunking_for_short_text():
    docs = documents_from_text(
        title="AAPL note",
        text="AAPL earnings guidance improved.",
        source="unit:short",
        source_type="fundamental_research_note",
        symbols=["aapl"],
        tags=["earnings"],
    )

    assert len(docs) == 1
    assert docs[0].text == "AAPL earnings guidance improved."
    assert docs[0].symbols == ["AAPL"]
    assert docs[0].metadata["chunk_strategy"] == "structured_semantic_parent_child_v1"
    assert docs[0].metadata["chunk_role"] == "child"
    assert docs[0].metadata["parent_text"] == docs[0].text
    assert docs[0].metadata["section_type"] == "unstructured_fallback"
    assert docs[0].metadata["child_char_start"] == 0
    assert docs[0].metadata["child_char_end"] == len(docs[0].text)


def test_documents_from_text_builds_semantic_parents_from_financial_headings():
    docs = documents_from_text(
        title="MU earnings report",
        text=(
            "# MU FY2026 Q3 Earnings\n\n"
            "Revenue increased as HBM demand stayed strong.\n\n"
            "## Gross Margin\n\n"
            "Gross margin expanded on mix shift and pricing.\n\n"
            "Risk Factors\n\n"
            "Customer concentration and supply execution remain risks."
        ),
        source="unit:mu-earnings",
        source_type="earnings_report",
        symbols=["mu"],
        tags=["earnings"],
    )

    section_titles = {doc.metadata["section_title"] for doc in docs}
    section_types = {doc.metadata["section_type"] for doc in docs}

    assert "MU FY2026 Q3 Earnings" in section_titles
    assert "Gross Margin" in section_titles
    assert "Risk Factors" in section_titles
    assert {"fundamental_section", "gross_margin", "risk_factors"}.issubset(section_types)


def test_documents_from_text_splits_long_text_into_parent_child_chunks():
    text = " ".join(f"sentence {index} earnings guidance" for index in range(60))

    docs = documents_from_text(
        title="Long earnings note",
        text=text,
        source="unit:long",
        source_type="fundamental_research_note",
        parent_chunk_size=180,
        parent_chunk_overlap=30,
        child_chunk_size=70,
        child_chunk_overlap=10,
    )

    parent_ids = {doc.metadata["parent_id"] for doc in docs}

    assert len(parent_ids) > 1
    assert len(docs) > len(parent_ids)
    assert all(doc.metadata["chunk_role"] == "child" for doc in docs)
    assert all("parent_text" in doc.metadata for doc in docs)


def test_chroma_rag_store_upserts_and_queries_with_symbol_filter(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeEmbedder(),
        client=FakeClient(),
    )
    docs = [
        *documents_from_text(
            title="AAPL earnings",
            text="AAPL earnings guidance improved after services revenue beat.",
            source="unit:aapl",
            source_type="earnings_report",
            symbols=["AAPL"],
            tags=["earnings"],
        ),
        *documents_from_text(
            title="MSFT support",
            text="MSFT pullback held support after a shallow retest.",
            source="unit:msft",
            source_type="fundamental_research_note",
            symbols=["MSFT"],
            tags=["pullback"],
        ),
    ]

    result = store.upsert_documents(docs)
    found = store.search(query="earnings guidance", symbol="AAPL", limit=3)

    assert result.added == 2
    assert store.status()["document_count"] == 2
    assert store.status()["backend"] == "fundamental_chroma"
    assert store.status()["scope"] == "fundamental_only"
    assert store.status()["chunking"]["strategy"] == "structured_semantic_parent_child_v1"
    assert len(found) == 1
    assert found[0].symbols == ["AAPL"]
    assert found[0].source_type == "earnings_report"
    assert found[0].metadata["chunk_role"] == "parent_context"
    assert found[0].metadata["matched_child_texts"] == [
        "AAPL earnings guidance improved after services revenue beat."
    ]
    assert found[0].metadata["rerank_model"] == "local_cross_feature_v1"
    assert "retrieval" in store.status()


def test_chroma_rag_store_fuses_vector_and_bm25_recall(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeEmbedder(),
        client=FakeClient(),
    )
    docs = [
        *documents_from_text(
            title="AAPL earnings guidance",
            text="AAPL earnings guidance improved after services revenue beat.",
            source="unit:aapl-guidance",
            source_type="earnings_report",
            symbols=["AAPL"],
            tags=["earnings"],
        ),
        *documents_from_text(
            title="AAPL regulatory probe",
            text="AAPL faces a regulatory probe with antitrust investigation downside risk.",
            source="unit:aapl-probe",
            source_type="fundamental_research_note",
            symbols=["AAPL"],
            tags=["risk"],
        ),
    ]

    store.upsert_documents(docs)
    found = store.search(query="regulatory probe earnings", symbol="AAPL", limit=2)

    titles = {doc.title for doc in found}
    channels = " ".join(doc.metadata.get("retrieval_channels", "") for doc in found)

    assert {"AAPL earnings guidance", "AAPL regulatory probe"}.issubset(titles)
    assert "vector:" in channels
    assert "bm25:" in channels
    assert all("rerank_score" in doc.metadata for doc in found)


def test_chroma_rag_store_returns_parent_context_from_child_recall(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeEmbedder(),
        client=FakeClient(),
        parent_chunk_size=2000,
        parent_chunk_overlap=100,
        chunk_size=80,
        chunk_overlap=10,
    )
    text = (
        "Parent opening context explains the portfolio thesis. "
        + " ".join("AAPL earnings guidance improved" for _ in range(20))
        + " Parent closing context keeps the analyst from losing the broader setup."
    )
    docs = documents_from_text(
        title="AAPL parent context",
        text=text,
        source="unit:parent-context",
        source_type="earnings_report",
        symbols=["AAPL"],
        tags=["earnings"],
        parent_chunk_size=2000,
        parent_chunk_overlap=100,
        child_chunk_size=80,
        child_chunk_overlap=10,
    )

    assert len(docs) > 1

    store.upsert_documents(docs)
    found = store.search(query="earnings guidance", symbol="AAPL", limit=3)

    assert len(found) == 1
    assert found[0].title == "AAPL parent context"
    assert "Parent opening context" in found[0].text
    assert "Parent closing context" in found[0].text
    assert len(found[0].metadata["matched_child_ids"]) > 1
    assert found[0].metadata["chunk_role"] == "parent_context"


def test_chroma_rag_store_keeps_legacy_flat_chunks_queryable(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeEmbedder(),
        client=FakeClient(),
    )
    flat = RagDocument(
        id="legacy-flat",
        text="AAPL earnings guidance improved in a legacy flat chunk.",
        title="Legacy AAPL note",
        source="unit:legacy",
        source_type="fundamental_research_note",
        symbols=["AAPL"],
        tags=["earnings"],
    )

    store.upsert_documents([flat])
    found = store.search(query="earnings guidance", symbol="AAPL", limit=2)

    assert len(found) == 1
    assert found[0].id == "legacy-flat"
    assert found[0].metadata["chunk_role"] == "flat"
    assert found[0].metadata["matched_child_ids"] == ["legacy-flat"]


def test_rerank_promotes_exact_query_coverage_after_rrf():
    plan = RagQueryPlan(
        original_query="AAPL HBM supply risk",
        rewritten_query="AAPL HBM supply risk",
        perspectives={"base": "AAPL HBM supply risk"},
        used_llm=False,
    )
    weak_rrf_match = RagDocument(
        id="weak",
        text="AAPL broad market note with limited memory details.",
        title="AAPL broad market",
        symbols=["AAPL"],
        tags=["hbm"],
        metadata={"rrf_score": 0.2, "retrieval_channels": "vector:base"},
    )
    exact_match = RagDocument(
        id="exact",
        text="AAPL HBM supply risk rises as high-bandwidth memory demand tightens.",
        title="AAPL HBM supply risk",
        symbols=["AAPL"],
        tags=["hbm"],
        metadata={"rrf_score": 0.01, "retrieval_channels": "bm25:base"},
    )

    reranked = _rerank_documents(
        plan,
        [weak_rrf_match, exact_match],
        symbol="AAPL",
        tags=["hbm"],
        limit=2,
    )

    assert reranked[0].id == "exact"
    assert reranked[0].metadata["rerank_model"] == "local_cross_feature_v1"
    assert reranked[0].score > reranked[1].score


def test_chroma_rag_store_rejects_non_fundamental_source_types(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeEmbedder(),
        client=FakeClient(),
    )
    doc = RagDocument(
        id="memory-doc",
        text="Wait for support confirmation before adding size.",
        title="Trade memory",
        source_type="trade_memory",
        symbols=["AAPL"],
        tags=["pullback"],
    )

    result = store.upsert_documents([doc])

    assert result.added == 0
    assert result.skipped == 1
    assert "unsupported fundamental RAG source_type" in result.errors[0]
    assert store.status()["document_count"] == 0

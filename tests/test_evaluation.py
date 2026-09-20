import json

from ai_trading_copilot.copilot.domain import (
    AnalystType,
    ExecutionStatus,
    PortfolioSnapshot,
    PriceBar,
    SubscriptionStatus,
    SymbolTrendState,
    TradeDirection,
)
from ai_trading_copilot.copilot import evaluation
from ai_trading_copilot.copilot.evaluation import (
    AgentEvalExpectation,
    AgentEvaluator,
    AnalystEvalExpectation,
    AnalystEvaluator,
    RagEvalCase,
    RagEvaluator,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.rag_store import (
    ChromaRagStore,
    documents_from_text,
)
from ai_trading_copilot.copilot.services.eval_samples import EvalSampleRecorder


class FakeSemanticEmbedder:
    name = "fake-semantic-embedding"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lower = text.lower()
        if "profit" in lower or "earnings" in lower or "guidance" in lower:
            return [1.0, 0.0]
        if "support" in lower or "pullback" in lower or "retest" in lower:
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


def _bars_from_closes(closes):
    return [
        PriceBar(
            date=f"2026-03-{(idx % 28) + 1:02d}",
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1000,
        )
        for idx, close in enumerate(closes)
    ]


def _actionable_bars():
    bars = _bars_from_closes([80 + index * 0.5 for index in range(40)])
    bars.extend(
        PriceBar(
            date=f"2026-04-{index + 1:02d}",
            open=101,
            high=110 if index == 0 else 103,
            low=100.5,
            close=101,
            volume=1000,
        )
        for index in range(19)
    )
    bars.append(
        PriceBar(
            date="2026-04-20",
            open=101.5,
            high=103,
            low=100.5,
            close=102,
            volume=1000,
        )
    )
    return bars


def test_agent_evaluator_scores_expected_workflow_state(tmp_path):
    state = CopilotLangGraph(enable_default_llm=False).run(
        subscription_symbols=["AAPL"],
        price_history_by_symbol={"AAPL": _actionable_bars()},
        portfolio=PortfolioSnapshot(),
        report_output_dir=tmp_path,
    )

    report = AgentEvaluator().evaluate_state(
        state,
        expectations=[
            AgentEvalExpectation(
                symbol="AAPL",
                expected_status=SubscriptionStatus.ACTIONABLE,
                expected_direction=TradeDirection.BUY,
                expected_risk_approved=True,
                expected_execution_status=ExecutionStatus.PORTFOLIO_DECIDED,
            )
        ],
    )
    metrics = {metric.name: metric for metric in report.metrics}

    assert report.passed is True
    assert metrics["node_coverage"].value == 1.0
    assert metrics["decision_match_rate"].value == 1.0
    assert metrics["safety_gate_match_rate"].value == 1.0


def test_analyst_evaluator_scores_multiple_single_analysts():
    report = AnalystEvaluator().evaluate_smoke(
        analysts=[
            AnalystType.TECHNICAL_POSITION,
            AnalystType.FUNDAMENTAL_ANALYSIS,
        ],
        symbols=["AAPL"],
    )
    metrics = {metric.name: metric for metric in report.metrics}

    assert report.passed is True
    assert report.details["summary"]["cases"] == 2.0
    assert metrics["output_valid_rate"].value == 1.0
    assert metrics["field_completeness_rate"].value >= 0.9
    assert metrics["expectation_match_rate"].value == 1.0


def test_analyst_evaluator_reports_expectation_mismatch():
    report = AnalystEvaluator().evaluate_expectations(
        [
            AnalystEvalExpectation(
                analyst=AnalystType.TECHNICAL_POSITION,
                symbol="AAPL",
                expected_uptrend=False,
            )
        ]
    )

    assert report.passed is False
    assert report.details["summary"]["expectation_match_rate"] == 0.0


def test_analyst_smoke_cli_outputs_json(capsys):
    result = evaluation.main(
        [
            "analyst-smoke",
            "--analysts",
            "technical_position",
            "--symbols",
            "AAPL",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["suite"] == "single_analyst_evaluation"
    assert payload["details"]["summary"]["cases"] == 1.0


def test_analyst_smoke_cli_default_analysts_outputs_json(capsys):
    result = evaluation.main(["analyst-smoke", "--symbols", "AAPL", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["suite"] == "single_analyst_evaluation"
    assert payload["passed"] is True
    assert payload["details"]["analysts"] == [
        "fundamental_analysis",
        "news_sentiment",
        "technical_position",
    ]


def test_fundamental_eval_cli_records_ragas_samples(tmp_path, capsys):
    result = evaluation.main(
        [
            "fundamental-eval",
            "--symbols",
            "AAPL",
            "--backend",
            "local",
            "--answer-quality",
            "off",
            "--output-dir",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    samples_path = tmp_path / "ragas_eval_samples.jsonl"

    assert result == 0
    assert payload["suite"] == "fundamental_analyst_evaluation"
    assert payload["passed"] is True
    assert payload["details"]["ragas_samples_path"] == str(samples_path)
    assert payload["details"]["ragas_sample_count"] == 1
    sample = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["stage"] == "fundamental_agent"
    assert sample["symbol"] == "AAPL"
    assert "response" in sample


def test_rag_evaluator_compares_keyword_baseline_to_optimized_retrieval(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeSemanticEmbedder(),
        client=FakeClient(),
    )
    store.upsert_documents(
        [
            *documents_from_text(
                title="Earnings quality",
                text="Earnings guidance improved after services revenue beat.",
                source="unit:earnings",
                source_type="earnings_report",
            ),
            *documents_from_text(
                title="Support retest",
                text="The pullback held support after a shallow retest.",
                source="unit:support",
                source_type="fundamental_research_note",
            ),
        ]
    )

    case = RagEvalCase(
        name="semantic_profit_query",
        query="profit outlook",
        relevant_title_substrings=("Earnings quality",),
    )
    report = RagEvaluator().compare_store(store, cases=[case], top_k=1, backend="local")
    comparison = report.details["comparison"]

    assert comparison["baseline"]["summary"]["hit_rate_at_k"] == 0.0
    assert comparison["optimized"]["summary"]["hit_rate_at_k"] == 1.0
    assert comparison["deltas"]["mrr_at_k"] == 1.0
    assert comparison["optimized"]["summary"]["ndcg_at_k"] <= 1.0
    assert report.passed is True


def test_rag_evaluator_matches_reference_parent_context_ids(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeSemanticEmbedder(),
        client=FakeClient(),
    )
    docs = documents_from_text(
        title="Earnings quality",
        text="Earnings guidance improved after services revenue beat.",
        source="unit:earnings",
        source_type="earnings_report",
    )
    parent_id = str(docs[0].metadata["parent_id"])
    store.upsert_documents(docs)

    report = RagEvaluator().compare_store(
        store,
        cases=[
            RagEvalCase(
                name="parent_id_profit_query",
                query="earnings guidance",
                reference_context_ids=(parent_id,),
            )
        ],
        top_k=1,
        backend="local",
        answer_quality="off",
    )
    comparison = report.details["comparison"]

    assert comparison["baseline"]["summary"]["hit_rate_at_k"] == 1.0
    assert comparison["optimized"]["summary"]["hit_rate_at_k"] == 1.0
    assert comparison["optimized"]["details"]["cases"][0]["retrieved_context_ids"] == [parent_id]


def test_rag_evaluator_answer_quality_required_fails_without_ragas_or_credentials(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeSemanticEmbedder(),
        client=FakeClient(),
    )
    store.upsert_documents(
        documents_from_text(
            title="Earnings quality",
            text="Earnings guidance improved after services revenue beat.",
            source="unit:earnings",
            source_type="earnings_report",
        )
    )

    report = RagEvaluator().compare_store(
        store,
        cases=[
            RagEvalCase(
                name="semantic_profit_query",
                user_input="profit outlook",
                reference_title_substrings=("Earnings quality",),
                reference="Earnings guidance improved.",
                expected_answer="Earnings guidance improved.",
            )
        ],
        top_k=1,
        backend="local",
        answer_quality="required",
    )

    assert report.passed is False
    assert report.details["comparison"]["optimized"]["details"]["ragas"]["answer_quality"]["status"] == "failed"


def test_keyword_baseline_marks_retrieval_channel(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeSemanticEmbedder(),
        client=FakeClient(),
    )
    store.upsert_documents(
        documents_from_text(
            title="Support retest",
            text="The pullback held support after a shallow retest.",
            source="unit:support",
            source_type="fundamental_research_note",
        )
    )

    found = store.search_keyword_baseline(query="support retest", limit=1)

    assert found[0].title == "Support retest"
    assert found[0].metadata["retrieval_channel"] == "baseline_bm25"


def test_keyword_baseline_records_ragas_eval_sample(tmp_path):
    store = ChromaRagStore(
        tmp_path / "chroma",
        embedder=FakeSemanticEmbedder(),
        client=FakeClient(),
    )
    store.upsert_documents(
        documents_from_text(
            title="Support retest",
            text="The pullback held support after a shallow retest.",
            source="unit:support",
            source_type="fundamental_research_note",
        )
    )
    recorder = EvalSampleRecorder(output_dir=tmp_path / "eval", run_id="unit")

    with recorder.activate():
        found = store.search_keyword_baseline(query="support retest", limit=1)

    sample = json.loads(recorder.path.read_text(encoding="utf-8").splitlines()[0])
    assert found[0].title == "Support retest"
    assert sample["stage"] == "rag_baseline_retrieval"
    assert sample["retrieved_context_ids"] == [found[0].id]
    assert sample["retrieval_results"][0]["channels"] == "baseline_bm25"


# -*- coding: utf-8 -*-
"""Evaluation suites for agent workflow quality and RAG retrieval performance."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from ai_trading_copilot.copilot.agents import (
    FundamentalAnalystAgent,
    OpportunityRadarAgent,
    TechnicalPositionAgent,
)
from ai_trading_copilot.copilot.config import DEFAULT_REPORT_OUTPUT_DIR
from ai_trading_copilot.copilot.domain.enums import (
    AnalystType,
    ExecutionStatus,
    SubscriptionStatus,
    SymbolTrendState,
    TradeDirection,
)
from ai_trading_copilot.copilot.domain.models import (
    FundamentalAnalysisReport,
    NewsSentimentReport,
    PortfolioSnapshot,
    PriceBar,
    RagDocument,
    TraceEvent,
    normalize_symbol,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.rag_store import (
    DEFAULT_CHROMA_DIR,
    ChromaRagStore,
    RagUnavailableError,
)
from ai_trading_copilot.copilot.services.eval_samples import EvalSampleRecorder


DEFAULT_RAG_EVAL_CASES_FILE = Path(__file__).resolve().parents[1] / "config" / "rag_eval_cases.json"
DEFAULT_RAG_COMPARE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class EvaluationMetric:
    name: str
    value: float
    passed: bool
    target: float | None = None
    unit: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "target": self.target,
            "unit": self.unit,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class EvaluationReport:
    suite: str
    overall_score: float
    passed: bool
    metrics: Sequence[EvaluationMetric]
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "overall_score": round(self.overall_score, 6),
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "details": dict(self.details),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.suite}",
            "",
            f"- Overall score: {self.overall_score:.3f}",
            f"- Passed: {str(self.passed).lower()}",
            "",
            "| Metric | Value | Target | Pass |",
            "| --- | ---: | ---: | --- |",
        ]
        for metric in self.metrics:
            value = f"{metric.value:.3f}"
            if metric.unit == "ms":
                value = f"{metric.value:.1f} ms"
            target = "-" if metric.target is None else f"{metric.target:.3f}"
            lines.append(
                f"| {metric.name} | {value} | {target} | {str(metric.passed).lower()} |"
            )

        comparison = self.details.get("comparison")
        if isinstance(comparison, Mapping):
            lines.extend(["", "## RAG Comparison", ""])
            before = comparison.get("baseline", {}).get("summary", {})
            after = comparison.get("optimized", {}).get("summary", {})
            deltas = comparison.get("deltas", {})
            rows = [
                "hit_rate_at_k",
                "context_recall_at_k",
                "context_precision_at_k",
                "mrr_at_k",
                "ndcg_at_k",
                "avg_latency_ms",
            ]
            lines.extend(
                [
                    "| Metric | Before | After | Delta |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for name in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            name,
                            _format_number(before.get(name)),
                            _format_number(after.get(name)),
                            _format_number(deltas.get(name)),
                        ]
                    )
                    + " |"
                )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class AgentEvalExpectation:
    symbol: str
    expected_status: SubscriptionStatus | str | None = None
    expected_direction: TradeDirection | str | None = None
    expected_risk_approved: bool | None = None
    expected_execution_status: ExecutionStatus | str | None = None


@dataclass(frozen=True)
class AnalystEvalExpectation:
    analyst: AnalystType | str
    symbol: str
    expected_status: SubscriptionStatus | str | None = None
    expected_trend_state: SymbolTrendState | str | None = None
    expected_thesis_intact: bool | None = None
    expected_material_risk: bool | None = None
    expected_uptrend: bool | None = None
    min_reward_risk_ratio: float | None = None


class AgentEvaluator:
    """Scores a completed CopilotLangGraph state without calling external LLMs."""

    def __init__(self, *, min_overall_score: float = 0.80):
        self.min_overall_score = min_overall_score

    def evaluate_state(
        self,
        state: Mapping[str, Any],
        *,
        expectations: Sequence[AgentEvalExpectation] = (),
        required_nodes: Sequence[str] | None = None,
        required_reports: Sequence[str] | None = None,
    ) -> EvaluationReport:
        symbols = _state_symbols(state, expectations)
        required_nodes = tuple(required_nodes or _default_required_nodes(state))
        required_reports = tuple(required_reports or _default_required_reports(state))

        errors = [str(error) for error in state.get("errors", []) if error]
        node_score, node_details = _node_coverage(state, required_nodes)
        report_score, report_details = _report_coverage(state, required_reports)
        symbol_score, symbol_details = _symbol_coverage(state, symbols)
        structured_score, structured_details = _structured_output_coverage(state, symbols)
        decision_score, safety_score, expectation_details = _expectation_scores(
            state,
            expectations,
        )
        trace_span_ms = _trace_span_ms(state.get("trace_events", []))

        metrics = [
            _metric("error_free", 1.0 if not errors else 0.0, target=1.0, details={"errors": errors}),
            _metric("node_coverage", node_score, target=1.0, details=node_details),
            _metric("report_coverage", report_score, target=0.95, details=report_details),
            _metric("symbol_coverage", symbol_score, target=1.0, details=symbol_details),
            _metric(
                "structured_output_coverage",
                structured_score,
                target=1.0,
                details=structured_details,
            ),
            _metric(
                "decision_match_rate",
                decision_score,
                target=1.0,
                details=expectation_details.get("decision", {}),
            ),
            _metric(
                "safety_gate_match_rate",
                safety_score,
                target=1.0,
                details=expectation_details.get("safety", {}),
            ),
        ]
        weights = {
            "error_free": 0.20,
            "node_coverage": 0.15,
            "report_coverage": 0.10,
            "symbol_coverage": 0.10,
            "structured_output_coverage": 0.15,
            "decision_match_rate": 0.20,
            "safety_gate_match_rate": 0.10,
        }
        overall = sum(metric.value * weights[metric.name] for metric in metrics)
        return EvaluationReport(
            suite="agent_evaluation",
            overall_score=overall,
            passed=overall >= self.min_overall_score and all(metric.passed for metric in metrics),
            metrics=metrics,
            details={
                "symbols": symbols,
                "trace_event_count": len(state.get("trace_events", [])),
                "trace_span_ms": trace_span_ms,
                "expectations": expectation_details,
            },
        )


class AnalystEvaluator:
    """Scores individual analyst agents outside the full graph workflow."""

    def __init__(
        self,
        *,
        opportunity_radar_agent: OpportunityRadarAgent | None = None,
        technical_position_agent: TechnicalPositionAgent | None = None,
        fundamental_analyst_agent: FundamentalAnalystAgent | None = None,
        min_overall_score: float = 0.80,
    ):
        self.opportunity_radar_agent = opportunity_radar_agent or OpportunityRadarAgent()
        self.technical_position_agent = technical_position_agent or TechnicalPositionAgent()
        self.fundamental_analyst_agent = fundamental_analyst_agent or FundamentalAnalystAgent()
        self.min_overall_score = min_overall_score

    def evaluate_smoke(
        self,
        *,
        analysts: Sequence[AnalystType | str] = tuple(AnalystType),
        symbols: Sequence[str] = ("AAPL",),
    ) -> EvaluationReport:
        normalized_analysts = _normalize_eval_analysts(analysts)
        normalized_symbols = [
            normalize_symbol(symbol)
            for symbol in symbols
            if normalize_symbol(symbol)
        ]
        if not normalized_symbols:
            raise ValueError("at least one symbol is required")

        expectations = [
            expectation
            for symbol in normalized_symbols
            for expectation in _default_analyst_expectations(symbol, normalized_analysts)
        ]
        return self.evaluate_expectations(expectations)

    def evaluate_expectations(
        self,
        expectations: Sequence[AnalystEvalExpectation],
    ) -> EvaluationReport:
        if not expectations:
            raise ValueError("at least one analyst expectation is required")

        case_results = []
        for expectation in expectations:
            started = perf_counter()
            try:
                payload = self._run_analyst(expectation)
                latency_ms = (perf_counter() - started) * 1000.0
                case_results.append(
                    _score_analyst_case(
                        expectation=expectation,
                        payload=payload,
                        latency_ms=latency_ms,
                        error="",
                    )
                )
            except Exception as exc:
                latency_ms = (perf_counter() - started) * 1000.0
                case_results.append(
                    _score_analyst_case(
                        expectation=expectation,
                        payload={},
                        latency_ms=latency_ms,
                        error=str(exc),
                    )
                )

        summary = _summarize_analyst_cases(case_results)
        metrics = [
            _metric("error_free_rate", summary["error_free_rate"], target=1.0),
            _metric("output_valid_rate", summary["output_valid_rate"], target=1.0),
            _metric("symbol_match_rate", summary["symbol_match_rate"], target=1.0),
            _metric(
                "field_completeness_rate",
                summary["field_completeness_rate"],
                target=0.90,
            ),
            _metric("expectation_match_rate", summary["expectation_match_rate"], target=1.0),
            _metric("avg_latency_ms", summary["avg_latency_ms"], unit="ms"),
            _metric("p95_latency_ms", summary["p95_latency_ms"], unit="ms"),
        ]
        overall = (
            summary["error_free_rate"] * 0.20
            + summary["output_valid_rate"] * 0.20
            + summary["symbol_match_rate"] * 0.15
            + summary["field_completeness_rate"] * 0.20
            + summary["expectation_match_rate"] * 0.25
        )
        return EvaluationReport(
            suite="single_analyst_evaluation",
            overall_score=overall,
            passed=overall >= self.min_overall_score and all(metric.passed for metric in metrics),
            metrics=metrics,
            details={
                "summary": summary,
                "cases": case_results,
                "analysts": sorted(
                    {
                        str(_enum_value(expectation.analyst))
                        for expectation in expectations
                    }
                ),
            },
        )

    def _run_analyst(self, expectation: AnalystEvalExpectation) -> dict[str, Any]:
        analyst = AnalystType(str(_enum_value(expectation.analyst)))
        symbol = normalize_symbol(expectation.symbol)
        if analyst == AnalystType.OPPORTUNITY_RADAR:
            item = self.opportunity_radar_agent.analyze_symbol(
                symbol=symbol,
                bars=_demo_actionable_bars(),
            )
            return {
                "symbol": item.symbol,
                "status": item.status,
                "trend_state": item.trend_state,
                "current_price": item.current_price,
                "support_level": item.support_level,
                "reward_risk_ratio": item.reward_risk_ratio,
                "reason": item.reason,
            }
        if analyst == AnalystType.TECHNICAL_POSITION:
            position = self.technical_position_agent.analyze(
                symbol,
                _demo_actionable_bars(),
            )
            return {
                "symbol": position.symbol,
                "current_price": position.current_price,
                "support_level": position.support_level,
                "recent_high": position.recent_high,
                "moving_average_20": position.moving_average_20,
                "moving_average_50": position.moving_average_50,
                "distance_to_support_pct": position.distance_to_support_pct,
                "pullback_from_high_pct": position.pullback_from_high_pct,
                "reward_risk_ratio": position.reward_risk_ratio,
                "uptrend": position.uptrend,
            }
        if analyst == AnalystType.NEWS_SENTIMENT:
            report = NewsSentimentReport(
                symbol=symbol,
                sentiment_score=0.0,
                company_news_score=0.0,
                social_sentiment_score=0.0,
                earnings_event_score=0.0,
                material_risk=False,
                risk_flags=[],
                key_events=[],
                alerts=[],
                summary="No material news or sentiment risk in offline smoke input.",
                data_availability={"offline_smoke": "synthetic"},
            )
            return {
                "symbol": report.symbol,
                "sentiment_score": report.sentiment_score,
                "material_risk": report.material_risk,
                "risk_flags": report.risk_flags,
                "key_events": report.key_events,
                "alerts": report.alerts,
                "summary": report.summary,
            }
        if analyst == AnalystType.FUNDAMENTAL_ANALYSIS:
            report = self.fundamental_analyst_agent.analyze_symbol(symbol=symbol).report
            return {
                "symbol": report.symbol,
                "thesis_intact": report.thesis_intact,
                "material_risk": report.material_risk,
                "risk_flags": report.risk_flags,
                "summary": report.summary,
                "fundamental_score": report.fundamental_score,
                "key_events": report.key_events,
            }
        raise ValueError(f"unsupported analyst: {analyst}")


@dataclass(frozen=True)
class RagEvalCase:
    name: str
    query: str = ""
    user_input: str = ""
    question: str = ""
    symbol: str | None = None
    tags: Sequence[str] = ()
    reference: str = ""
    expected_answer: str = ""
    reference_contexts: Sequence[str] = ()
    reference_context_ids: Sequence[str] = ()
    expected_context_ids: Sequence[str] = ()
    annotation_granularity: str = ""
    annotation_source: str = ""
    annotation_notes: str = ""
    reference_source_substrings: Sequence[str] = ()
    reference_title_substrings: Sequence[str] = ()
    relevant_doc_ids: Sequence[str] = ()
    relevant_source_substrings: Sequence[str] = ()
    relevant_title_substrings: Sequence[str] = ()
    required_terms: Sequence[str] = ()
    relevant_count: int = 1


RagSearcher = Callable[..., Sequence[RagDocument]]


class RagEvaluator:
    """Benchmarks retrieval quality and latency for a set of relevance cases."""

    def __init__(self, *, min_quality_score: float = 0.70):
        self.min_quality_score = min_quality_score

    def evaluate_system(
        self,
        *,
        system_name: str,
        searcher: RagSearcher,
        cases: Sequence[RagEvalCase],
        top_k: int = 5,
        backend: str = "ragas",
        answer_quality: str = "auto",
    ) -> EvaluationReport:
        case_results = []
        for case in cases:
            started = perf_counter()
            docs = list(
                searcher(
                    query=_case_query(case),
                    symbol=case.symbol,
                    tags=tuple(case.tags),
                    limit=top_k,
                )
            )
            latency_ms = (perf_counter() - started) * 1000.0
            case_results.append(_score_rag_case(case, docs, top_k=top_k, latency_ms=latency_ms))

        summary = _summarize_rag_cases(case_results)
        ragas_details = _run_ragas_official_details(
            case_results,
            backend=backend,
            answer_quality=answer_quality,
        )
        if ragas_details.get("retrieval_metrics"):
            summary.update(ragas_details["retrieval_metrics"])
        answer_required = answer_quality == "required"
        answer_passed = bool(ragas_details.get("answer_quality", {}).get("passed", True))
        quality_score = _rag_quality_score(summary)
        metrics = [
            _metric("hit_rate_at_k", summary["hit_rate_at_k"], target=0.80),
            _metric("context_recall_at_k", summary["context_recall_at_k"], target=0.70),
            _metric("context_precision_at_k", summary["context_precision_at_k"], target=0.50),
            _metric("mrr_at_k", summary["mrr_at_k"], target=0.60),
            _metric("ndcg_at_k", summary["ndcg_at_k"], target=0.60),
            _metric("avg_latency_ms", summary["avg_latency_ms"], unit="ms"),
            _metric(
                "empty_rate",
                summary["empty_rate"],
                target=0.0,
                higher_is_better=False,
            ),
        ]
        answer_metrics = ragas_details.get("answer_quality", {}).get("metrics", {})
        if "faithfulness" in answer_metrics:
            metrics.append(
                _metric(
                    "faithfulness",
                    answer_metrics["faithfulness"],
                    target=0.70 if answer_required else None,
                )
            )
        if "answer_relevancy" in answer_metrics:
            metrics.append(
                _metric(
                    "answer_relevancy",
                    answer_metrics["answer_relevancy"],
                    target=0.70 if answer_required else None,
                )
            )
        retrieval_passed = quality_score >= self.min_quality_score and all(
            metric.passed for metric in metrics
            if metric.name not in {"faithfulness", "answer_relevancy"}
        )
        return EvaluationReport(
            suite=system_name,
            overall_score=quality_score,
            passed=retrieval_passed and (answer_passed if answer_required else True),
            metrics=metrics,
            details={
                "summary": summary,
                "cases": case_results,
                "top_k": top_k,
                "backend": backend,
                "ragas": ragas_details,
            },
        )

    def compare_store(
        self,
        store: ChromaRagStore,
        *,
        cases: Sequence[RagEvalCase] | None = None,
        top_k: int = 5,
        backend: str = "ragas",
        answer_quality: str = "auto",
    ) -> EvaluationReport:
        resolved_cases = cases if cases is not None else DEFAULT_RAG_EVAL_CASES
        baseline = self.evaluate_system(
            system_name="pre_optimization_keyword_index",
            searcher=store.search_keyword_baseline,
            cases=resolved_cases,
            top_k=top_k,
            backend=backend,
            answer_quality="off",
        )
        optimized = self.evaluate_system(
            system_name="optimized_hybrid_index",
            searcher=store.search,
            cases=resolved_cases,
            top_k=top_k,
            backend=backend,
            answer_quality=answer_quality,
        )
        before = baseline.details["summary"]
        after = optimized.details["summary"]
        baseline_payload = baseline.to_dict()
        optimized_payload = optimized.to_dict()
        baseline_payload["summary"] = before
        optimized_payload["summary"] = after
        deltas = {
            key: round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 6)
            for key in {
                "hit_rate_at_k",
                "context_recall_at_k",
                "context_precision_at_k",
                "mrr_at_k",
                "ndcg_at_k",
                "avg_latency_ms",
                "empty_rate",
            }
        }
        metrics = [
            _metric("optimized_quality_score", optimized.overall_score, target=self.min_quality_score),
            _metric("quality_score_delta", optimized.overall_score - baseline.overall_score),
            _metric("hit_rate_delta", deltas["hit_rate_at_k"]),
            _metric("context_recall_delta", deltas["context_recall_at_k"]),
            _metric("context_precision_delta", deltas["context_precision_at_k"]),
            _metric("mrr_delta", deltas["mrr_at_k"]),
            _metric("ndcg_delta", deltas["ndcg_at_k"]),
            _metric("latency_delta_ms", deltas["avg_latency_ms"], unit="ms"),
        ]
        passed = optimized.passed
        return EvaluationReport(
            suite="rag_before_after_comparison",
            overall_score=optimized.overall_score,
            passed=passed,
            metrics=metrics,
            details={
                "comparison": {
                    "baseline": baseline_payload,
                    "optimized": optimized_payload,
                    "deltas": deltas,
                },
                "top_k": top_k,
                "case_count": len(resolved_cases),
                "backend": backend,
                "answer_quality": answer_quality,
            },
        )


class _DisabledVectorEmbedder:
    name = "vector_disabled"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RagUnavailableError("vector embeddings disabled for offline evaluation")

    def embed_query(self, text: str) -> list[float]:
        raise RagUnavailableError("vector embeddings disabled for offline evaluation")


def run_agent_smoke_evaluation(
    *,
    symbols: Sequence[str] = ("AAPL",),
    output_dir: str | Path | None = None,
) -> EvaluationReport:
    normalized = [normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)]
    if not normalized:
        raise ValueError("at least one symbol is required")
    graph = CopilotLangGraph(enable_default_llm=False)
    bars = {symbol: _demo_actionable_bars() for symbol in normalized}
    state = graph.run(
        subscription_symbols=normalized,
        price_history_by_symbol=bars,
        portfolio=PortfolioSnapshot(),
        report_output_dir=output_dir or DEFAULT_REPORT_OUTPUT_DIR / "eval_agent_smoke",
    )
    expectations = [
        AgentEvalExpectation(
            symbol=symbol,
            expected_status=SubscriptionStatus.ACTIONABLE,
            expected_direction=TradeDirection.BUY,
            expected_risk_approved=True,
            expected_execution_status=ExecutionStatus.PORTFOLIO_DECIDED,
        )
        for symbol in normalized
    ]
    return AgentEvaluator().evaluate_state(state, expectations=expectations)


def run_analyst_smoke_evaluation(
    *,
    analysts: Sequence[AnalystType | str] = tuple(AnalystType),
    symbols: Sequence[str] = ("AAPL",),
) -> EvaluationReport:
    return AnalystEvaluator().evaluate_smoke(analysts=analysts, symbols=symbols)


def run_fundamental_evaluation(
    *,
    symbols: Sequence[str] = ("AAPL",),
    output_dir: str | Path | None = None,
    backend: str = "ragas",
    answer_quality: str = "auto",
) -> EvaluationReport:
    output_path = Path(output_dir) if output_dir else DEFAULT_REPORT_OUTPUT_DIR / "eval_fundamental_analyst"
    run_id = f"fundamental_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recorder = EvalSampleRecorder(output_dir=output_path, run_id=run_id)
    with recorder.activate():
        report = AnalystEvaluator().evaluate_smoke(
            analysts=[AnalystType.FUNDAMENTAL_ANALYSIS],
            symbols=symbols,
        )

    ragas_details = _run_fundamental_ragas_details(
        recorder.samples,
        backend=backend,
        answer_quality=answer_quality,
    )
    metrics = list(report.metrics)
    answer_metrics = ragas_details.get("answer_quality", {}).get("metrics", {})
    if "faithfulness" in answer_metrics:
        metrics.append(
            _metric(
                "faithfulness",
                answer_metrics["faithfulness"],
                target=0.70 if answer_quality == "required" else None,
            )
        )
    if "answer_relevancy" in answer_metrics:
        metrics.append(
            _metric(
                "answer_relevancy",
                answer_metrics["answer_relevancy"],
                target=0.70 if answer_quality == "required" else None,
            )
        )
    answer_passed = bool(ragas_details.get("answer_quality", {}).get("passed", True))
    details = {
        **dict(report.details),
        "run_id": run_id,
        "ragas_samples_path": str(recorder.path),
        "ragas_sample_count": len(recorder.samples),
        "ragas": ragas_details,
    }
    return EvaluationReport(
        suite="fundamental_analyst_evaluation",
        overall_score=report.overall_score,
        passed=report.passed and (answer_passed if answer_quality == "required" else True),
        metrics=metrics,
        details=details,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "agent-smoke":
        report = run_agent_smoke_evaluation(
            symbols=_split_csv(args.symbols),
            output_dir=args.output_dir,
        )
        _emit_report(report, output=args.output, output_format=args.format)
        return 0

    if args.command == "analyst-smoke":
        report = run_analyst_smoke_evaluation(
            analysts=_split_csv(args.analysts),
            symbols=_split_csv(args.symbols),
        )
        _emit_report(report, output=args.output, output_format=args.format)
        return 0

    if args.command == "fundamental-eval":
        report = run_fundamental_evaluation(
            symbols=_split_csv(args.symbols),
            output_dir=args.output_dir,
            backend=args.backend,
            answer_quality=args.answer_quality,
        )
        _emit_report(report, output=args.output, output_format=args.format)
        return 0

    if args.command == "rag-compare":
        report = _run_rag_compare_subprocess(args)
        _emit_report(report, output=args.output, output_format=args.format)
        return 0

    if args.command == "_rag-compare-worker":
        store = ChromaRagStore(args.chroma_dir)
        if not args.use_vector:
            store.embedder = _DisabledVectorEmbedder()
        cases = _load_rag_cases(args.cases or DEFAULT_RAG_EVAL_CASES_FILE)
        output_path = Path(args.output_dir) if args.output_dir else DEFAULT_REPORT_OUTPUT_DIR / "eval_rag_compare"
        recorder = EvalSampleRecorder(
            output_dir=output_path,
            run_id=f"rag_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        with recorder.activate():
            report = RagEvaluator().compare_store(
                store,
                cases=cases,
                top_k=args.top_k,
                backend=args.backend,
                answer_quality=args.answer_quality,
            )
        report = _with_extra_details(
            report,
            {
                "ragas_samples_path": str(recorder.path),
                "ragas_sample_count": len(recorder.samples),
            },
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


def _run_rag_compare_subprocess(args: argparse.Namespace) -> EvaluationReport:
    command = [
        sys.executable, "-X", "utf8",
        "-m",
        "ai_trading_copilot.copilot.evaluation",
        "_rag-compare-worker",
        "--chroma-dir",
        str(args.chroma_dir),
        "--cases",
        str(args.cases or DEFAULT_RAG_EVAL_CASES_FILE),
        "--top-k",
        str(args.top_k),
        "--backend",
        args.backend,
        "--answer-quality",
        args.answer_quality,
        "--output-dir",
        str(args.output_dir or ""),
    ]
    if args.use_vector:
        command.append("--use-vector")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=DEFAULT_RAG_COMPARE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _rag_worker_failure_report(
            chroma_dir=args.chroma_dir,
            message=f"RAG compare worker timed out after {exc.timeout} seconds.",
            returncode=None,
            stderr="",
            stdout=exc.stdout or "",
        )
    if completed.returncode != 0:
        return _rag_worker_failure_report(
            chroma_dir=args.chroma_dir,
            message=f"RAG compare worker exited with code {completed.returncode}.",
            returncode=completed.returncode,
            stderr=completed.stderr,
            stdout=completed.stdout,
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        metrics = [
            EvaluationMetric(
                name=str(metric["name"]),
                value=float(metric["value"]),
                passed=bool(metric["passed"]),
                target=metric.get("target"),
                unit=str(metric.get("unit") or ""),
                details=metric.get("details") or {},
            )
            for metric in payload.get("metrics", [])
        ]
        return EvaluationReport(
            suite=str(payload["suite"]),
            overall_score=float(payload["overall_score"]),
            passed=bool(payload["passed"]),
            metrics=metrics,
            details=payload.get("details") or {},
        )
    except Exception as exc:
        return _rag_worker_failure_report(
            chroma_dir=args.chroma_dir,
            message=f"RAG compare worker returned invalid JSON: {exc}",
            returncode=completed.returncode,
            stderr=completed.stderr,
            stdout=completed.stdout,
        )


def _rag_worker_failure_report(
    *,
    chroma_dir: str | Path,
    message: str,
    returncode: int | None,
    stderr: str,
    stdout: str,
) -> EvaluationReport:
    return EvaluationReport(
        suite="rag_before_after_comparison",
        overall_score=0.0,
        passed=False,
        metrics=[
            _metric("optimized_quality_score", 0.0, target=0.70),
            _metric("worker_succeeded", 0.0, target=1.0),
        ],
        details={
            "worker": {
                "status": "failed",
                "message": message,
                "returncode": returncode,
                "stderr": (stderr or "").strip(),
                "stdout": (stdout or "").strip(),
                "chroma_dir": str(chroma_dir),
            }
        },
    )


def _with_extra_details(report: EvaluationReport, extra: Mapping[str, Any]) -> EvaluationReport:
    return EvaluationReport(
        suite=report.suite,
        overall_score=report.overall_score,
        passed=report.passed,
        metrics=report.metrics,
        details={**dict(report.details), **dict(extra)},
    )


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent workflow and RAG retrieval.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    agent = subparsers.add_parser("agent-smoke")
    agent.add_argument("--symbols", default="AAPL")
    agent.add_argument("--output-dir", default=None)
    agent.add_argument("--format", choices=["json", "markdown"], default="json")
    agent.add_argument("--output", default=None)

    analyst = subparsers.add_parser("analyst-smoke")
    analyst.add_argument(
        "--analysts",
        default="news_sentiment,technical_position,fundamental_analysis",
        help="Comma-separated analysts to evaluate.",
    )
    analyst.add_argument("--symbols", default="AAPL")
    analyst.add_argument("--format", choices=["json", "markdown"], default="json")
    analyst.add_argument("--output", default=None)

    fundamental = subparsers.add_parser("fundamental-eval")
    fundamental.add_argument("--symbols", default="AAPL")
    fundamental.add_argument("--output-dir", default=None)
    fundamental.add_argument("--backend", choices=["ragas", "local"], default="ragas")
    fundamental.add_argument("--answer-quality", choices=["auto", "off", "required"], default="auto")
    fundamental.add_argument("--format", choices=["json", "markdown"], default="json")
    fundamental.add_argument("--output", default=None)

    rag = subparsers.add_parser("rag-compare")
    rag.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    rag.add_argument("--cases", default=str(DEFAULT_RAG_EVAL_CASES_FILE), help="Optional JSON file with RAGAS-style cases.")
    rag.add_argument("--top-k", type=int, default=5)
    rag.add_argument("--backend", choices=["ragas", "local"], default="ragas")
    rag.add_argument("--answer-quality", choices=["auto", "off", "required"], default="auto")
    rag.add_argument("--use-vector", action="store_true")
    rag.add_argument("--output-dir", default=None)
    rag.add_argument("--format", choices=["json", "markdown"], default="json")
    rag.add_argument("--output", default=None)
    worker = subparsers.add_parser("_rag-compare-worker", help=argparse.SUPPRESS)
    worker.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    worker.add_argument("--cases", default=str(DEFAULT_RAG_EVAL_CASES_FILE))
    worker.add_argument("--top-k", type=int, default=5)
    worker.add_argument("--backend", choices=["ragas", "local"], default="ragas")
    worker.add_argument("--answer-quality", choices=["auto", "off", "required"], default="auto")
    worker.add_argument("--use-vector", action="store_true")
    worker.add_argument("--output-dir", default=None)
    worker.add_argument("--format", choices=["json", "markdown"], default="json")
    worker.add_argument("--output", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def _metric(
    name: str,
    value: float,
    *,
    target: float | None = None,
    unit: str = "",
    details: Mapping[str, Any] | None = None,
    higher_is_better: bool = True,
) -> EvaluationMetric:
    if target is None:
        passed = True
    elif higher_is_better:
        passed = value >= target
    else:
        passed = value <= target
    return EvaluationMetric(
        name=name,
        value=float(value),
        passed=passed,
        target=target,
        unit=unit,
        details=details or {},
    )


def _default_required_nodes(state: Mapping[str, Any]) -> tuple[str, ...]:
    selected = {_enum_value(item) for item in state.get("selected_analysts", [])}
    if not selected:
        selected = {analyst.value for analyst in AnalystType}
    nodes = [
        CopilotLangGraph.NODE_LOAD_PERSONA_MARKDOWN,
        CopilotLangGraph.NODE_LOAD_SUBSCRIPTION_SYMBOLS,
    ]
    if AnalystType.OPPORTUNITY_RADAR.value in selected:
        nodes.append(CopilotLangGraph.NODE_OPPORTUNITY_RADAR)
    if AnalystType.TECHNICAL_POSITION.value in selected:
        nodes.append(CopilotLangGraph.NODE_TECHNICAL_POSITION)
    if AnalystType.NEWS_SENTIMENT.value in selected:
        nodes.append(CopilotLangGraph.NODE_NEWS_SENTIMENT)
    if AnalystType.FUNDAMENTAL_ANALYSIS.value in selected:
        nodes.append(CopilotLangGraph.NODE_FUNDAMENTAL_ANALYSIS)
    nodes.extend(
        [
            CopilotLangGraph.NODE_TRADER,
            CopilotLangGraph.NODE_RISK_CHECK,
            CopilotLangGraph.NODE_PORTFOLIO_MANAGER,
            CopilotLangGraph.NODE_EXPLAIN_RUN,
            CopilotLangGraph.NODE_PERSIST_TRACE,
        ]
    )
    return tuple(nodes)


def _default_required_reports(state: Mapping[str, Any]) -> tuple[str, ...]:
    selected = {_enum_value(item) for item in state.get("selected_analysts", [])}
    if not selected:
        selected = {analyst.value for analyst in AnalystType}
    return tuple(
        sorted(
            {
                "futu_portfolio",
                "trader",
                "risk_check",
                "portfolio_manager",
                "run_explanation",
                *selected,
            }
        )
    )


def _node_coverage(state: Mapping[str, Any], required_nodes: Sequence[str]) -> tuple[float, dict[str, Any]]:
    seen = {_attr(event, "node_name") for event in state.get("trace_events", [])}
    missing = [node for node in required_nodes if node not in seen]
    score = 1.0 if not required_nodes else (len(required_nodes) - len(missing)) / len(required_nodes)
    return score, {"required": list(required_nodes), "missing": missing}


def _report_coverage(state: Mapping[str, Any], required_reports: Sequence[str]) -> tuple[float, dict[str, Any]]:
    report_paths = {
        **dict(state.get("analyst_reports", {})),
        **dict(state.get("agent_reports", {})),
    }
    missing = []
    for key in required_reports:
        value = report_paths.get(key)
        if not value or not Path(str(value)).exists():
            missing.append(key)
    score = 1.0 if not required_reports else (len(required_reports) - len(missing)) / len(required_reports)
    return score, {"required": list(required_reports), "missing": missing}


def _symbol_coverage(state: Mapping[str, Any], symbols: Sequence[str]) -> tuple[float, dict[str, Any]]:
    radar_symbols = {_attr(item, "symbol") for item in state.get("radar_items", [])}
    missing = [symbol for symbol in symbols if symbol not in radar_symbols]
    score = 1.0 if not symbols else (len(symbols) - len(missing)) / len(symbols)
    return score, {"required": list(symbols), "missing": missing}


def _structured_output_coverage(
    state: Mapping[str, Any],
    symbols: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    radar_symbols = {_attr(item, "symbol") for item in state.get("radar_items", [])}
    plans = state.get("trade_plans", {})
    assessments = state.get("risk_assessments", {})
    decisions = state.get("execution_decisions", {})
    missing: list[str] = []
    for symbol in symbols:
        if symbol not in radar_symbols:
            missing.append(f"{symbol}:radar_item")
        if symbol not in plans:
            missing.append(f"{symbol}:trade_plan")
        if symbol not in assessments:
            missing.append(f"{symbol}:risk_assessment")
        if symbol not in decisions:
            missing.append(f"{symbol}:execution_decision")
    total = max(1, 4 * len(symbols))
    score = (total - len(missing)) / total
    return score, {"missing": missing}


def _expectation_scores(
    state: Mapping[str, Any],
    expectations: Sequence[AgentEvalExpectation],
) -> tuple[float, float, dict[str, Any]]:
    if not expectations:
        return 1.0, 1.0, {"decision": {"comparisons": 0}, "safety": {"comparisons": 0}}

    radar_by_symbol = {_attr(item, "symbol"): item for item in state.get("radar_items", [])}
    plans = state.get("trade_plans", {})
    assessments = state.get("risk_assessments", {})
    decisions = state.get("execution_decisions", {})
    decision_matches = 0
    decision_total = 0
    safety_matches = 0
    safety_total = 0
    failures: list[dict[str, Any]] = []

    for expectation in expectations:
        symbol = normalize_symbol(expectation.symbol)
        item = radar_by_symbol.get(symbol)
        plan = plans.get(symbol)
        assessment = assessments.get(symbol)
        decision = decisions.get(symbol)

        for label, expected, actual in [
            ("status", expectation.expected_status, _attr(item, "status")),
            ("direction", expectation.expected_direction, _attr(plan, "direction")),
            (
                "execution_status",
                expectation.expected_execution_status,
                _attr(decision, "status"),
            ),
        ]:
            if expected is None:
                continue
            decision_total += 1
            if _enum_value(expected) == _enum_value(actual):
                decision_matches += 1
            else:
                failures.append(
                    {
                        "symbol": symbol,
                        "field": label,
                        "expected": _enum_value(expected),
                        "actual": _enum_value(actual),
                    }
                )

        if expectation.expected_risk_approved is not None:
            safety_total += 1
            actual_approved = bool(_attr(assessment, "approved")) if assessment is not None else None
            if actual_approved is expectation.expected_risk_approved:
                safety_matches += 1
            else:
                failures.append(
                    {
                        "symbol": symbol,
                        "field": "risk_approved",
                        "expected": expectation.expected_risk_approved,
                        "actual": actual_approved,
                    }
                )

    decision_score = 1.0 if decision_total == 0 else decision_matches / decision_total
    safety_score = 1.0 if safety_total == 0 else safety_matches / safety_total
    return (
        decision_score,
        safety_score,
        {
            "decision": {
                "matches": decision_matches,
                "comparisons": decision_total,
                "failures": failures,
            },
            "safety": {
                "matches": safety_matches,
                "comparisons": safety_total,
                "failures": failures,
            },
        },
    )


def _normalize_eval_analysts(values: Sequence[AnalystType | str]) -> tuple[AnalystType, ...]:
    normalized = []
    for value in values:
        analyst = value if isinstance(value, AnalystType) else AnalystType(str(value).strip())
        if analyst not in normalized:
            normalized.append(analyst)
    return tuple(normalized)


def _default_analyst_expectations(
    symbol: str,
    analysts: Sequence[AnalystType],
) -> tuple[AnalystEvalExpectation, ...]:
    expectations = []
    for analyst in analysts:
        if analyst == AnalystType.OPPORTUNITY_RADAR:
            expectations.append(
                AnalystEvalExpectation(
                    analyst=analyst,
                    symbol=symbol,
                    expected_status=SubscriptionStatus.ACTIONABLE,
                    expected_trend_state=SymbolTrendState.UPTREND_PULLBACK,
                    min_reward_risk_ratio=2.0,
                )
            )
        elif analyst == AnalystType.TECHNICAL_POSITION:
            expectations.append(
                AnalystEvalExpectation(
                    analyst=analyst,
                    symbol=symbol,
                    expected_uptrend=True,
                    min_reward_risk_ratio=2.0,
                )
            )
        elif analyst == AnalystType.FUNDAMENTAL_ANALYSIS:
            expectations.append(
                AnalystEvalExpectation(
                    analyst=analyst,
                    symbol=symbol,
                    expected_thesis_intact=True,
                    expected_material_risk=False,
                )
            )
        elif analyst == AnalystType.NEWS_SENTIMENT:
            expectations.append(
                AnalystEvalExpectation(
                    analyst=analyst,
                    symbol=symbol,
                    expected_material_risk=False,
                )
            )
        else:
            raise ValueError(f"unsupported analyst: {analyst}")
    return tuple(expectations)


def _score_analyst_case(
    *,
    expectation: AnalystEvalExpectation,
    payload: Mapping[str, Any],
    latency_ms: float,
    error: str,
) -> dict[str, Any]:
    analyst = AnalystType(str(_enum_value(expectation.analyst)))
    symbol = normalize_symbol(expectation.symbol)
    output_valid = not error and bool(payload)
    symbol_match = output_valid and normalize_symbol(str(payload.get("symbol") or "")) == symbol
    required_fields = _analyst_required_fields(analyst)
    present_fields = [
        field_name
        for field_name in required_fields
        if field_name in payload and _field_present(payload.get(field_name))
    ]
    field_completeness = (
        1.0 if not required_fields else len(present_fields) / len(required_fields)
    )
    expectation_checks = _analyst_expectation_checks(expectation, payload)
    expectation_matches = sum(1 for check in expectation_checks if check["passed"])
    expectation_total = len(expectation_checks)
    expectation_score = (
        1.0 if expectation_total == 0 else expectation_matches / expectation_total
    )
    return {
        "analyst": analyst.value,
        "symbol": symbol,
        "error": error,
        "output_valid": output_valid,
        "symbol_match": bool(symbol_match),
        "field_completeness": field_completeness,
        "missing_fields": [
            field_name
            for field_name in required_fields
            if field_name not in present_fields
        ],
        "expectation_match_rate": expectation_score,
        "expectation_matches": expectation_matches,
        "expectation_total": expectation_total,
        "expectation_checks": expectation_checks,
        "latency_ms": latency_ms,
        "output": dict(payload),
    }


def _analyst_required_fields(analyst: AnalystType) -> tuple[str, ...]:
    if analyst == AnalystType.OPPORTUNITY_RADAR:
        return (
            "symbol",
            "status",
            "trend_state",
            "current_price",
            "support_level",
            "reward_risk_ratio",
            "reason",
        )
    if analyst == AnalystType.TECHNICAL_POSITION:
        return (
            "symbol",
            "current_price",
            "support_level",
            "recent_high",
            "distance_to_support_pct",
            "pullback_from_high_pct",
            "reward_risk_ratio",
            "uptrend",
        )
    if analyst == AnalystType.FUNDAMENTAL_ANALYSIS:
        return (
            "symbol",
            "thesis_intact",
            "material_risk",
            "risk_flags",
            "summary",
        )
    if analyst == AnalystType.NEWS_SENTIMENT:
        return (
            "symbol",
            "sentiment_score",
            "material_risk",
            "risk_flags",
            "summary",
        )
    return ()


def _analyst_expectation_checks(
    expectation: AnalystEvalExpectation,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = []
    for name, expected, actual in [
        ("status", expectation.expected_status, payload.get("status")),
        ("trend_state", expectation.expected_trend_state, payload.get("trend_state")),
    ]:
        if expected is None:
            continue
        checks.append(
            {
                "name": name,
                "expected": _enum_value(expected),
                "actual": _enum_value(actual),
                "passed": _enum_value(expected) == _enum_value(actual),
            }
        )
    for name, expected, actual in [
        ("thesis_intact", expectation.expected_thesis_intact, payload.get("thesis_intact")),
        ("material_risk", expectation.expected_material_risk, payload.get("material_risk")),
        ("uptrend", expectation.expected_uptrend, payload.get("uptrend")),
    ]:
        if expected is None:
            continue
        checks.append(
            {
                "name": name,
                "expected": bool(expected),
                "actual": actual,
                "passed": isinstance(actual, bool) and actual is expected,
            }
        )
    if expectation.min_reward_risk_ratio is not None:
        actual_rr = _safe_float(payload.get("reward_risk_ratio"))
        checks.append(
            {
                "name": "min_reward_risk_ratio",
                "expected": expectation.min_reward_risk_ratio,
                "actual": actual_rr,
                "passed": actual_rr is not None and actual_rr >= expectation.min_reward_risk_ratio,
            }
        )
    return checks


def _summarize_analyst_cases(case_results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    count = max(1, len(case_results))
    latencies = sorted(float(result["latency_ms"]) for result in case_results)
    p95_index = min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)
    expectation_total = sum(int(result["expectation_total"]) for result in case_results)
    expectation_matches = sum(int(result["expectation_matches"]) for result in case_results)
    return {
        "cases": float(len(case_results)),
        "error_free_rate": sum(1.0 for result in case_results if not result["error"]) / count,
        "output_valid_rate": sum(1.0 for result in case_results if result["output_valid"]) / count,
        "symbol_match_rate": sum(1.0 for result in case_results if result["symbol_match"]) / count,
        "field_completeness_rate": sum(
            float(result["field_completeness"]) for result in case_results
        )
        / count,
        "expectation_match_rate": (
            1.0 if expectation_total == 0 else expectation_matches / expectation_total
        ),
        "avg_latency_ms": sum(latencies) / count if latencies else 0.0,
        "p95_latency_ms": latencies[p95_index] if latencies else 0.0,
    }


def _field_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _run_ragas_official_details(
    case_results: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    answer_quality: str,
) -> dict[str, Any]:
    answer_details = _answer_quality_unavailable(answer_quality, reason="disabled")
    if backend != "ragas":
        return {
            "backend": backend,
            "status": "disabled",
            "retrieval_metrics": {},
            "answer_quality": answer_details,
        }

    try:
        ragas_module = importlib.import_module("ragas")
        metrics_module = importlib.import_module("ragas.metrics")
    except Exception as exc:
        return {
            "backend": "ragas",
            "status": "missing_dependency",
            "error": f"Install the eval extra to enable official RAGAS: {exc}",
            "retrieval_metrics": {},
            "answer_quality": _answer_quality_unavailable(answer_quality, reason="missing_ragas"),
        }

    rows = [_ragas_row(result) for result in case_results]
    retrieval_metrics: dict[str, float] = {}
    retrieval_status = "skipped"
    retrieval_error = ""
    try:
        retrieval_metric_names = [
            "NonLLMContextPrecisionWithReference",
            "NonLLMContextRecall",
        ]
        if any(row.get("reference_context_ids") for row in rows):
            retrieval_metric_names = [
                "IDBasedContextPrecision",
                "IDBasedContextRecall",
                *retrieval_metric_names,
            ]
        retrieval_metric_objects = _ragas_metric_objects(
            metrics_module,
            retrieval_metric_names,
        )
        if retrieval_metric_objects:
            ragas_result = _evaluate_with_ragas(ragas_module, rows, retrieval_metric_objects)
            retrieval_metrics = _extract_ragas_metrics(
                ragas_result,
                {
                    "context_precision_at_k": (
                        "id_based_context_precision",
                        "non_llm_context_precision_with_reference",
                        "context_precision",
                        "context_precision_at_k",
                    ),
                    "context_recall_at_k": (
                        "id_based_context_recall",
                        "non_llm_context_recall",
                        "context_recall",
                        "context_recall_at_k",
                    ),
                },
            )
            retrieval_status = "succeeded" if retrieval_metrics else "skipped"
        else:
            retrieval_error = "Official RAGAS non-LLM retrieval metrics were not found."
    except Exception as exc:
        retrieval_status = "failed"
        retrieval_error = str(exc)

    answer_details = _run_ragas_answer_quality(
        ragas_module=ragas_module,
        metrics_module=metrics_module,
        rows=rows,
        answer_quality=answer_quality,
    )
    return {
        "backend": "ragas",
        "status": retrieval_status,
        "error": retrieval_error,
        "retrieval_metrics": retrieval_metrics,
        "answer_quality": answer_details,
    }


def _run_fundamental_ragas_details(
    samples: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    answer_quality: str,
) -> dict[str, Any]:
    answer_details = _answer_quality_unavailable(answer_quality, reason="disabled")
    if backend != "ragas":
        return {
            "backend": backend,
            "status": "disabled",
            "answer_quality": answer_details,
        }
    try:
        ragas_module = importlib.import_module("ragas")
        metrics_module = importlib.import_module("ragas.metrics")
    except Exception as exc:
        return {
            "backend": "ragas",
            "status": "missing_dependency",
            "error": f"Install the eval extra to enable official RAGAS: {exc}",
            "answer_quality": _answer_quality_unavailable(answer_quality, reason="missing_ragas"),
        }
    rows = [
        _fundamental_ragas_row(sample)
        for sample in samples
        if sample.get("stage") == "fundamental_agent"
    ]
    if not rows:
        return {
            "backend": "ragas",
            "status": "skipped",
            "answer_quality": _answer_quality_unavailable(answer_quality, reason="no_agent_samples"),
        }
    return {
        "backend": "ragas",
        "status": "succeeded",
        "answer_quality": _run_ragas_answer_quality(
            ragas_module=ragas_module,
            metrics_module=metrics_module,
            rows=rows,
            answer_quality=answer_quality,
        ),
    }


def _run_ragas_answer_quality(
    *,
    ragas_module: Any,
    metrics_module: Any,
    rows: Sequence[Mapping[str, Any]],
    answer_quality: str,
) -> dict[str, Any]:
    if answer_quality == "off":
        return _answer_quality_unavailable(answer_quality, reason="disabled")
    if not _has_llm_credentials():
        return _answer_quality_unavailable(answer_quality, reason="missing_llm_credentials")
    try:
        metric_objects = _ragas_metric_objects(
            metrics_module,
            ["Faithfulness", "ResponseRelevancy", "AnswerRelevancy"],
        )
        if not metric_objects:
            return _answer_quality_unavailable(answer_quality, reason="metrics_unavailable")
        result = _evaluate_with_ragas(ragas_module, rows, metric_objects)
        metrics = _extract_ragas_metrics(
            result,
            {
                "faithfulness": ("faithfulness",),
                "answer_relevancy": ("answer_relevancy", "response_relevancy"),
            },
        )
        passed = all(value >= 0.70 for value in metrics.values()) if answer_quality == "required" else True
        return {
            "mode": answer_quality,
            "status": "succeeded",
            "passed": passed,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "mode": answer_quality,
            "status": "failed",
            "passed": answer_quality != "required",
            "metrics": {},
            "error": str(exc),
        }


def _answer_quality_unavailable(answer_quality: str, *, reason: str) -> dict[str, Any]:
    return {
        "mode": answer_quality,
        "status": "skipped" if answer_quality != "required" else "failed",
        "passed": answer_quality != "required",
        "metrics": {},
        "reason": reason,
    }


def _has_llm_credentials() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
        )
    )


def _ragas_metric_objects(metrics_module: Any, names: Sequence[str]) -> list[Any]:
    objects = []
    seen = set()
    for name in names:
        metric = getattr(metrics_module, name, None)
        if metric is None or name in seen:
            continue
        seen.add(name)
        try:
            objects.append(metric())
        except TypeError:
            objects.append(metric)
    return objects


def _evaluate_with_ragas(ragas_module: Any, rows: Sequence[Mapping[str, Any]], metrics: Sequence[Any]) -> Any:
    evaluate = getattr(ragas_module, "evaluate")
    dataset = _ragas_dataset(ragas_module, rows)
    return evaluate(dataset=dataset, metrics=metrics)


def _ragas_dataset(ragas_module: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    dataset_cls = getattr(ragas_module, "EvaluationDataset", None)
    if dataset_cls is not None and hasattr(dataset_cls, "from_list"):
        return dataset_cls.from_list(list(rows))
    datasets_module = importlib.import_module("datasets")
    return datasets_module.Dataset.from_list(list(rows))


def _extract_ragas_metrics(result: Any, aliases: Mapping[str, Sequence[str]]) -> dict[str, float]:
    payload: Mapping[str, Any] = {}
    if isinstance(result, Mapping):
        payload = result
    elif hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        payload = {}
        for column in getattr(frame, "columns", []):
            if column not in frame or len(frame[column]) <= 0:
                continue
            values = [
                parsed
                for parsed in (_safe_float(item) for item in list(frame[column]))
                if parsed is not None
            ]
            if values:
                payload[column] = sum(values) / len(values)
    elif hasattr(result, "scores"):
        scores = getattr(result, "scores")
        if isinstance(scores, Mapping):
            payload = scores
    output = {}
    lower_payload = {str(key).lower(): value for key, value in payload.items()}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            raw = lower_payload.get(candidate.lower())
            parsed = _safe_float(raw)
            if parsed is not None:
                output[canonical] = parsed
                break
    return output


def _ragas_row(result: Mapping[str, Any]) -> dict[str, Any]:
    user_input = str(result.get("user_input") or result.get("query") or "")
    retrieved_contexts = [str(item) for item in result.get("retrieved_contexts", [])]
    retrieved_context_ids = [str(item) for item in result.get("retrieved_context_ids", [])]
    reference_contexts = [str(item) for item in result.get("reference_contexts", [])]
    reference_context_ids = [str(item) for item in result.get("reference_context_ids", [])]
    reference = str(result.get("reference") or result.get("expected_answer") or "")
    response = str(result.get("expected_answer") or reference or "No generated answer is available.")
    return {
        "user_input": user_input,
        "question": user_input,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": retrieved_context_ids,
        "contexts": retrieved_contexts,
        "reference_contexts": reference_contexts or ([reference] if reference else []),
        "reference_context_ids": reference_context_ids,
        "reference": reference,
        "ground_truth": reference,
        "response": response,
        "answer": response,
    }


def _fundamental_ragas_row(sample: Mapping[str, Any]) -> dict[str, Any]:
    user_input = str(sample.get("user_input") or "")
    retrieved_contexts = [str(item) for item in sample.get("retrieved_contexts", [])]
    retrieved_context_ids = [str(item) for item in sample.get("retrieved_context_ids", [])]
    reference = str(sample.get("reference") or "")
    response = str(sample.get("response") or sample.get("answer") or "")
    return {
        "user_input": user_input,
        "question": user_input,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": retrieved_context_ids,
        "contexts": retrieved_contexts,
        "reference_contexts": [reference] if reference else [],
        "reference": reference,
        "ground_truth": reference,
        "response": response,
        "answer": response,
    }


def _score_rag_case(
    case: RagEvalCase,
    docs: Sequence[RagDocument],
    *,
    top_k: int,
    latency_ms: float,
) -> dict[str, Any]:
    relevances = [_is_relevant(doc, case) for doc in docs[:top_k]]
    hits = sum(1 for relevant in relevances if relevant)
    relevant_total = max(1, _reference_context_count(case))
    first_hit_rank = next((index + 1 for index, relevant in enumerate(relevances) if relevant), None)
    reciprocal_rank = 0.0 if first_hit_rank is None else 1.0 / first_hit_rank
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, relevant in enumerate(relevances, start=1)
        if relevant
    )
    ideal_hits = min(relevant_total, top_k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = 0.0 if ideal_dcg == 0 else min(1.0, dcg / ideal_dcg)
    retrieved_contexts = [doc.text for doc in docs[:top_k]]
    retrieved_context_ids = [_eval_context_id(doc) for doc in docs[:top_k]]
    return {
        "name": case.name,
        "user_input": _case_query(case),
        "symbol": case.symbol,
        "annotation_granularity": case.annotation_granularity,
        "annotation_source": case.annotation_source,
        "annotation_notes": case.annotation_notes,
        "hit": hits > 0,
        "hits": hits,
        "context_precision_at_k": hits / max(1, len(docs[:top_k])),
        "context_recall_at_k": min(1.0, hits / relevant_total),
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": ndcg,
        "latency_ms": latency_ms,
        "empty": not docs,
        "reference": case.reference or case.expected_answer,
        "expected_answer": case.expected_answer or case.reference,
        "reference_contexts": list(case.reference_contexts),
        "reference_context_ids": list(_reference_context_ids(case)),
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": retrieved_context_ids,
        "results": [
            {
                "rank": index + 1,
                "id": doc.id,
                "eval_context_id": _eval_context_id(doc),
                "title": doc.title,
                "source": doc.source,
                "score": doc.score,
                "relevant": relevances[index],
                "channels": doc.metadata.get("retrieval_channels")
                or doc.metadata.get("retrieval_channel", ""),
            }
            for index, doc in enumerate(docs[:top_k])
        ],
    }


def _summarize_rag_cases(case_results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    count = max(1, len(case_results))
    latencies = sorted(float(result["latency_ms"]) for result in case_results)
    p95_index = min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)
    return {
        "cases": float(len(case_results)),
        "hit_rate_at_k": sum(1.0 for result in case_results if result["hit"]) / count,
        "context_recall_at_k": sum(float(result["context_recall_at_k"]) for result in case_results)
        / count,
        "context_precision_at_k": sum(
            float(result["context_precision_at_k"]) for result in case_results
        )
        / count,
        "mrr_at_k": sum(float(result["mrr_at_k"]) for result in case_results) / count,
        "ndcg_at_k": sum(float(result["ndcg_at_k"]) for result in case_results) / count,
        "avg_latency_ms": sum(latencies) / count if latencies else 0.0,
        "p95_latency_ms": latencies[p95_index] if latencies else 0.0,
        "empty_rate": sum(1.0 for result in case_results if result["empty"]) / count,
    }


def _rag_quality_score(summary: Mapping[str, float]) -> float:
    return (
        float(summary["hit_rate_at_k"]) * 0.25
        + float(summary["context_recall_at_k"]) * 0.25
        + float(summary["context_precision_at_k"]) * 0.15
        + float(summary["mrr_at_k"]) * 0.25
        + float(summary["ndcg_at_k"]) * 0.10
    )


def _is_relevant(doc: RagDocument, case: RagEvalCase) -> bool:
    haystack = " ".join(
        [
            doc.id,
            doc.title,
            doc.source,
            doc.source_type,
            doc.text,
            " ".join(doc.symbols),
            " ".join(doc.tags),
        ]
    ).lower()
    reference_ids = set(_reference_context_ids(case))
    if doc.id in reference_ids or _eval_context_id(doc) in reference_ids:
        return True
    source_substrings = [
        *case.reference_source_substrings,
        *case.relevant_source_substrings,
    ]
    title_substrings = [
        *case.reference_title_substrings,
        *case.relevant_title_substrings,
    ]
    if any(item.lower() in doc.source.lower() for item in source_substrings):
        return True
    if any(item.lower() in doc.title.lower() for item in title_substrings):
        return True
    if any(_context_overlap(item, haystack) >= 0.55 for item in case.reference_contexts):
        return True
    if not (
        case.reference_context_ids
        or case.expected_context_ids
        or case.reference_source_substrings
        or case.reference_title_substrings
        or case.reference_contexts
    ):
        required_terms = [term.lower() for term in case.required_terms]
        return bool(required_terms) and all(term in haystack for term in required_terms)
    return False


def _case_query(case: RagEvalCase) -> str:
    return case.user_input or case.question or case.query


def _reference_context_ids(case: RagEvalCase) -> tuple[str, ...]:
    return tuple([*case.reference_context_ids, *case.expected_context_ids, *case.relevant_doc_ids])


def _eval_context_id(doc: RagDocument) -> str:
    return str(doc.metadata.get("parent_id") or doc.id)


def _reference_context_count(case: RagEvalCase) -> int:
    if case.reference_context_ids or case.expected_context_ids:
        return len(set(_reference_context_ids(case)))
    if case.reference_contexts:
        return len(case.reference_contexts)
    if case.reference_source_substrings or case.reference_title_substrings:
        return max(1, int(case.relevant_count or 1))
    return max(1, int(case.relevant_count or len(case.relevant_doc_ids) or 1))


def _context_overlap(reference: str, haystack: str) -> float:
    reference_terms = {
        token
        for token in re_split_words(reference.lower())
        if len(token) >= 4
    }
    if not reference_terms:
        return 0.0
    haystack_terms = set(re_split_words(haystack.lower()))
    return len(reference_terms & haystack_terms) / len(reference_terms)


def re_split_words(value: str) -> list[str]:
    return [item for item in re.split(r"[^\w.]+", value) if item]


def _state_symbols(
    state: Mapping[str, Any],
    expectations: Sequence[AgentEvalExpectation],
) -> list[str]:
    if expectations:
        return [normalize_symbol(expectation.symbol) for expectation in expectations]
    symbols = [normalize_symbol(symbol) for symbol in state.get("subscription_symbols", [])]
    if symbols:
        return symbols
    return [normalize_symbol(_attr(item, "symbol")) for item in state.get("radar_items", [])]


def _trace_span_ms(events: Sequence[TraceEvent]) -> float:
    stamps = []
    for event in events:
        timestamp = _attr(event, "timestamp")
        if not timestamp:
            continue
        try:
            stamps.append(datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(stamps) < 2:
        return 0.0
    return (max(stamps) - min(stamps)).total_seconds() * 1000.0


def _attr(item: Any, name: str, default: Any = None) -> Any:
    if item is None:
        return default
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _demo_actionable_bars() -> list[PriceBar]:
    bars = [
        PriceBar(
            date=f"2026-03-{(index % 28) + 1:02d}",
            open=80 + index * 0.5,
            high=(80 + index * 0.5) * 1.01,
            low=(80 + index * 0.5) * 0.99,
            close=80 + index * 0.5,
            volume=1000,
        )
        for index in range(40)
    ]
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


def _load_rag_cases(path: str | Path) -> tuple[RagEvalCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("RAG cases file must contain a JSON list")
    allowed = {item.name for item in fields(RagEvalCase)}
    cases = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("Each RAG case must be a JSON object")
        normalized = dict(item)
        if "expected_contexts" in normalized and "reference_contexts" not in normalized:
            normalized["reference_contexts"] = normalized["expected_contexts"]
        if "expected_sources" in normalized and "reference_source_substrings" not in normalized:
            normalized["reference_source_substrings"] = normalized["expected_sources"]
        cases.append(RagEvalCase(**{key: value for key, value in normalized.items() if key in allowed}))
    return tuple(cases)


DEFAULT_RAG_EVAL_CASES = _load_rag_cases(DEFAULT_RAG_EVAL_CASES_FILE)


def _emit_report(
    report: EvaluationReport,
    *,
    output: str | Path | None,
    output_format: str,
) -> None:
    rendered = (
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if output_format == "json"
        else report.to_markdown()
    )
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(rendered, encoding="utf-8")
    print(rendered)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3f}"


__all__ = [
    "AgentEvalExpectation",
    "AgentEvaluator",
    "AnalystEvalExpectation",
    "AnalystEvaluator",
    "DEFAULT_RAG_EVAL_CASES",
    "EvaluationMetric",
    "EvaluationReport",
    "RagEvalCase",
    "RagEvaluator",
    "run_analyst_smoke_evaluation",
    "run_agent_smoke_evaluation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


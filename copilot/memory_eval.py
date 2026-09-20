"""Reproducible memory retrieval evaluation and opt-in paired model trials.

Only fixture inputs enter agents. Labels are used after inference. Every trial
uses an isolated SQLite database and calls decision agents, never a broker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_trading_copilot.copilot.agents import PortfolioManager, RiskAgent, TraderAgent
from ai_trading_copilot.copilot.agents.llm_tools import extract_json_object
from ai_trading_copilot.copilot.domain import (
    DistilledMemory, ExecutionMode, MemoryStatus, NewsSentimentReport,
    PortfolioSnapshot, PriceBar, RunOutcome, TechnicalContext, TechnicalPosition,
    TradeDirection, UserPersonaConfig,
)
from ai_trading_copilot.copilot.services.memory_evaluation import MemoryShadowEvaluator
from ai_trading_copilot.copilot.services.memory_learning import (
    LangMemCandidate, LangMemProposal, MemoryReflector, MemorySkillManager, PostRunLearningService,
)
from ai_trading_copilot.copilot.services.memory_retrieval import MemoryRetrievalSession
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore
from ai_trading_copilot.copilot.services.tracing import TraceRecorder
from ai_trading_copilot.copilot.services.vector_memory import LocalVectorMemoryIndex

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "config" / "memory_effect_cases.json"


class MemoryCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    split: Literal["development", "evaluation"]
    query: str
    tags: list[str] = Field(default_factory=list)
    market_regime: str = "uptrend"
    memories: list[DistilledMemory]
    relevant_ids: list[str]
    technical_position: TechnicalPosition
    technical_context: TechnicalContext | None = None
    news: NewsSentimentReport | None = None
    portfolio: PortfolioSnapshot = Field(default_factory=lambda: PortfolioSnapshot(total_value=100_000, cash=100_000))
    persona: UserPersonaConfig = Field(default_factory=UserPersonaConfig)
    allowed_directions: list[TradeDirection] = Field(min_length=1)
    max_final_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_case(self):
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        position = self.technical_position
        if min(position.current_price, position.support_level, position.recent_high) <= 0:
            raise ValueError("fixture prices must be positive")
        ids = [memory.memory_id for memory in self.memories]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("fixture memories require unique explicit IDs")
        if not set(self.relevant_ids).issubset(ids):
            raise ValueError("relevant_ids must refer to fixture memories")
        symbol = self.technical_position.symbol
        if self.news and self.news.symbol != symbol:
            raise ValueError("news symbol must match technical position")
        if self.technical_context and self.technical_context.symbol != symbol:
            raise ValueError("technical context symbol must match technical position")
        return self


def load_cases(path: Path, split: str) -> list[MemoryCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") != "memory_effect_cases.v1":
        raise ValueError("unsupported memory effect case format")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("cases must be a list")
    cases = [MemoryCase.model_validate(item) for item in payload["cases"]]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate case_id")
    selected = [case for case in cases if case.split == split]
    if not selected:
        raise ValueError(f"no {split} cases")
    return selected


def _store(root: Path, memories: list[DistilledMemory]) -> DistilledMemoryStore:
    store = DistilledMemoryStore(
        root / "memory.sqlite3", vector_index=LocalVectorMemoryIndex(root / "memory.vectors.json"),
    )
    for memory in memories:
        store.repository.upsert(memory, actor="evaluation_fixture", reason="isolated fixture")
    return store


def _session(store, case, consumer, run_id):
    return MemoryRetrievalSession(
        store, symbol=case.technical_position.symbol, consumer=consumer,
        run_id=run_id, market_regime=case.market_regime,
    )


def _retrieval_case(case: MemoryCase, root: Path) -> dict:
    store = _store(root, case.memories)
    session = _session(store, case, "trader", case.case_id)
    session.prefetch(case.query, tags=case.tags)
    retrieved = [memory.memory_id for memory in session.approved_memories]
    relevant = set(case.relevant_ids)
    hits = len(relevant.intersection(retrieved))
    refs = [f"{memory.memory_id}@{memory.version}" for memory in session.approved_memories]
    accepted = session.mark_cited([*refs, "fabricated@1", *[ref + "0" for ref in refs]])
    invalid = [ref for ref in accepted if ref not in refs]
    # Precision@3 uses a fixed denominator; returning one correct memory is 1/3,
    # not 100%. Recall on a no-positive case is undefined, recorded as null.
    return {
        "case_id": case.case_id, "retrieved_ids": retrieved,
        "relevant_ids": case.relevant_ids, "precision_at_3": hits / 3,
        "recall_at_3": hits / len(relevant) if relevant else None,
        "irrelevant_count": len(set(retrieved) - relevant),
        "injected_count": len(retrieved), "invalid_citation_count": len(invalid),
        "accepted_citation_count": len(accepted),
        "nonproduction_injected": any(memory.status != MemoryStatus.APPROVED for memory in session.approved_memories),
    }


def _learning_contract(root: Path) -> dict:
    """Synthetic plumbing check, independent of evaluation labels and LLMs."""
    class Extractor:
        available = True

        def extract(self, snapshot, *, existing=()):
            return [LangMemProposal(operation="insert", candidate=LangMemCandidate(
                memory_type="strategy_performance", lesson="Confirm support before entry.",
                symbols=["AAPL"], confidence=0.8,
            ))]

    store = _store(root, [])
    service = PostRunLearningService(MemoryReflector(Extractor()), MemorySkillManager(store))
    state = {"run_id": "learning-1", "subscription_symbols": ["AAPL"]}
    candidate = service.learn(state)[0]
    checks = {"candidate_hidden": not store.load(), "replay_idempotent": service.learn(state)[0] == candidate}
    candidate = service.learn({**state, "run_id": "learning-2"})[0]
    checks["independent_evidence_accumulates"] = candidate.sample_count == 2
    shadow = store.repository.transition(candidate.memory_id, "shadow", actor="fixture", reason="review")
    evaluator = MemoryShadowEvaluator(store.repository)
    checks["no_outcomes_no_promotion"] = not evaluator.evaluate(shadow.memory_id).eligible
    for index in range(3):
        run_id = f"shadow-{index}"
        session = MemoryRetrievalSession(store, symbol="AAPL", consumer="trader", run_id=run_id)
        session.prefetch("support confirmation")
        checks[f"shadow_hidden_{index}"] = not session.approved_memories and len(session.shadow_memories) == 1
        store.repository.record_outcome(RunOutcome(
            run_id=run_id, symbol="AAPL", realized_return=0.03,
            benchmark_return=0.01, max_drawdown=-0.04, source="synthetic_contract",
        ))
    approved = evaluator.promote(shadow.memory_id, actor="fixture", reason="synthetic gate")
    checks["approved_recalled"] = [m.memory_id for m in store.retrieve(symbol="AAPL")] == [approved.memory_id]
    revision = store.save_candidate(approved.model_copy(update={"lesson": "Recheck revised support."}))
    checks["revision_unqualified"] = (
        revision.supersedes == approved.memory_id
        and evaluator.evaluate(revision.memory_id).outcome_samples == 0
        and store.repository.get(approved.memory_id) == approved
    )
    return {"evidence_type": "synthetic_contract_only", "checks": checks, "passed": all(checks.values())}


class _ObservedLLM:
    """Delegate unchanged; capture usage when the production adapter exposes it."""
    def __init__(self, llm, usage=None, responses=None):
        self.llm = llm
        self.usage = usage if usage is not None else []
        self.responses = responses if responses is not None else []

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def bind_tools(self, tools):
        return _ObservedLLM(self.llm.bind_tools(tools), self.usage, self.responses) if hasattr(self.llm, "bind_tools") else self

    def invoke(self, *args, **kwargs):
        response = self.llm.invoke(*args, **kwargs)
        self._copilot_completion_observer(response)
        return response

    def _copilot_completion_observer(self, response):
        """Observe full final content/usage on LangChain and native SDK paths."""
        usage = getattr(response, "usage_metadata", None)
        total = usage.get("total_tokens") if isinstance(usage, dict) else getattr(getattr(response, "usage", None), "total_tokens", None)
        self.usage.append(total)
        message = response.choices[0].message if getattr(response, "choices", None) else response
        self.responses.append(str(getattr(message, "content", message) or ""))


def _response_payload(observer) -> dict:
    """Use final content only, never the provider's reasoning field."""
    content = observer.responses[-1] if observer.responses else ""
    try:
        return extract_json_object(content)
    except (ValueError, TypeError):
        return {}


def _model_trial(case: MemoryCase, root: Path, llm, *, with_memory: bool) -> dict:
    store = _store(root, case.memories if with_memory else [])
    trader_session = _session(store, case, "trader", "trial-trader") if with_memory else None
    pm_session = _session(store, case, "portfolio_manager", "trial-portfolio") if with_memory else None
    observer = _ObservedLLM(llm)
    trace = TraceRecorder(output_dir=root, run_id=case.case_id, otlp_endpoint="")
    inputs = dict(
        symbol=case.technical_position.symbol, technical_position=case.technical_position,
        technical_context=case.technical_context, news_sentiment=case.news, persona=case.persona,
    )
    # Production agents deliberately fall back on failure. An evaluation must
    # expose that fallback instead of awarding the deterministic answer to LLM.
    started = perf_counter()
    with trace.activate():
        trader = TraderAgent(llm=observer).create_plan_from_evidence(**inputs, memory_session=trader_session)
        trader_payload = _response_payload(observer)
        trader_spans = len([span for span in trace.spans if span.name == "llm.invoke"])
        symbol = case.technical_position.symbol
        price = case.technical_position.current_price
        bar = PriceBar(date="2026-01-02", open=price, high=price, low=price, close=price)
        risk = RiskAgent().review(
            persona=case.persona, portfolio=case.portfolio, plan=trader.plan,
            price_history_by_symbol={symbol: [bar]},
        )
        manager = PortfolioManager(llm=observer).decide_with_report(
            plan=trader.plan, risk_assessment=risk, portfolio=case.portfolio,
            mode=ExecutionMode.LIVE, user_confirmed=False, memory_session=pm_session,
        )
    spans = [span for span in trace.spans if span.name == "llm.invoke"]
    failed = any(span.status == "error" for span in trace.spans)
    trader_fallback = trader.fallback_used or trader_payload.get("direction") not in {d.value for d in TradeDirection}
    pm_expected = with_memory and abs(risk.final_weight) > abs(risk.current_position_weight) + 1e-6
    pm_payload = _response_payload(observer) if len(spans) > trader_spans else {}
    pm_fallback = manager.fallback_used or (pm_expected and (not manager.tool_calls or pm_payload.get("decision") not in {"proceed", "scale", "hold"}))
    decision = manager.decision
    citations = [*trader.plan.memory_citations, *decision.memory_citations]
    available_by_consumer = [
        {f"{m.memory_id}@{m.version}" for m in session.approved_memories} if session else set()
        for session in (trader_session, pm_session)
    ]
    available = set().union(*available_by_consumer)
    invalid = [
        ref for refs, allowed in zip((trader.plan.memory_citations, decision.memory_citations), available_by_consumer)
        for ref in refs if ref not in allowed
    ]
    raw_citation_count = raw_invalid_count = 0
    for payload, allowed in zip((trader_payload, pm_payload), available_by_consumer):
        refs = payload.get("memory_citations", [])
        if isinstance(refs, list):
            raw_citation_count += len(refs)
            raw_invalid_count += sum(str(ref) not in allowed for ref in refs)
    other_gross = sum(abs(weight) for key, weight in case.portfolio.position_weights.items() if key != symbol)
    reduced_risk = abs(risk.final_weight) <= abs(risk.current_position_weight) + 1e-9
    checks = {
        "allowed_direction": trader.plan.direction in case.allowed_directions,
        "case_position_limit": abs(decision.final_weight) <= case.max_final_weight + 1e-9,
        "risk_position_limit": abs(decision.final_weight) <= case.persona.max_single_position_weight + 1e-9,
        "risk_not_expanded": (
            abs(decision.final_weight - risk.final_weight) <= 1e-9 if reduced_risk else
            min(risk.current_position_weight, risk.final_weight) - 1e-9 <= decision.final_weight <= max(risk.current_position_weight, risk.final_weight) + 1e-9
        ),
        "gross_exposure_limit": other_gross + abs(decision.final_weight) <= case.persona.max_gross_exposure + 1e-9,
        "no_forbidden_instruments": not trader.plan.uses_options and not trader.plan.uses_leverage,
        "valid_citations": not invalid,
        "confirmation_preserved": decision.action == "hold" or decision.requires_user_confirmation,
        "not_submitted": not decision.submitted_to_broker,
    }
    completed = bool(spans) and not (failed or trader_fallback or pm_fallback or trader.memory_retrieval_error)
    return {
        "completed": completed, "trader_fallback": trader_fallback, "portfolio_fallback": bool(pm_fallback),
        "memory_retrieval_error": trader.memory_retrieval_error,
        "model_errors": [span.error for span in trace.spans if span.status == "error"],
        "plan": trader.plan.model_dump(mode="json"), "risk": risk.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"), "checks": checks,
        "constraint_pass_rate": mean(checks.values()) if completed else None,
        "decision_constraint_pass_rate": mean([checks["allowed_direction"], checks["case_position_limit"]]) if completed else None,
        "safety_passed": all(value for name, value in checks.items() if name not in {"allowed_direction", "case_position_limit"}),
        "invalid_citation_count": len(invalid), "citation_count": len(citations),
        "raw_invalid_citation_count": raw_invalid_count,
        "raw_citation_count": raw_citation_count,
        "llm_calls": len(spans), "elapsed_seconds": round(perf_counter() - started, 4),
        "total_tokens": sum(observer.usage) if len(observer.usage) == len(spans) and observer.usage and all(v is not None for v in observer.usage) else None,
        "retrieved_ids": sorted(available),
    }


def run_evaluation(cases_path: Path = DEFAULT_CASES, *, split="evaluation", live=False) -> dict:
    cases = load_cases(cases_path, split)
    llm = None
    if live:
        from ai_trading_copilot.copilot.config.llm import create_default_deepseek_llm
        llm = create_default_deepseek_llm()
        if llm is None:
            raise ValueError("--live requires configured DEEPSEEK_API_KEY; no model calls were made")
    with tempfile.TemporaryDirectory(prefix="memory_effect_") as directory:
        root = Path(directory)
        rows = [_retrieval_case(case, root / str(index)) for index, case in enumerate(cases)]
        contract = _learning_contract(root / "learning")
        pairs = []
        if live:
            for index, case in enumerate(cases):
                # Alternate arm order to reduce systematic order/latency bias.
                arms = [False, True] if index % 2 == 0 else [True, False]
                pair = {"case_id": case.case_id}
                for enabled in arms:
                    key = "with_memory" if enabled else "without_memory"
                    pair[key] = _model_trial(case, root / f"live-{index}-{enabled}", llm, with_memory=enabled)
                rates = [pair[key]["constraint_pass_rate"] for key in ("with_memory", "without_memory")]
                pair["constraint_pass_rate_delta"] = rates[0] - rates[1] if all(v is not None for v in rates) else None
                quality = [pair[key]["decision_constraint_pass_rate"] for key in ("with_memory", "without_memory")]
                pair["decision_constraint_pass_rate_delta"] = quality[0] - quality[1] if all(v is not None for v in quality) else None
                pairs.append(pair)
    injected = sum(row["injected_count"] for row in rows)
    accepted = sum(row["accepted_citation_count"] for row in rows)
    recalls = [row["recall_at_3"] for row in rows if row["recall_at_3"] is not None]
    deltas = [pair["constraint_pass_rate_delta"] for pair in pairs if pair["constraint_pass_rate_delta"] is not None]
    quality_deltas = [pair["decision_constraint_pass_rate_delta"] for pair in pairs if pair["decision_constraint_pass_rate_delta"] is not None]
    source_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted([*(ROOT / "copilot").rglob("*.py"), *(ROOT / "config" / "prompts").glob("*.md")])
    }
    return {
        "format_version": "memory_effect_report.v1", "mode": "live" if live else "offline",
        "split": split, "case_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "code_tree_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "python": platform.python_version(), "executable": sys.executable,
        "dependencies": {name: version(name) for name in ("langmem", "langchain-core", "langgraph", "pydantic")},
        "model": str(getattr(llm, "model_name", "")) if live else None,
        "case_count": len(rows), "retrieval": rows, "learning_contract": contract,
        "metrics": {
            "precision_at_3": mean(row["precision_at_3"] for row in rows),
            "recall_at_3": mean(recalls) if recalls else None,
            "irrelevant_injection_rate": sum(row["irrelevant_count"] for row in rows) / injected if injected else 0.0,
            "invalid_citation_rate": sum(row["invalid_citation_count"] for row in rows) / accepted if accepted else 0.0,
            "nonproduction_injection_count": sum(row["nonproduction_injected"] for row in rows),
        },
        "model_pairs": pairs, "completed_pairs": len(deltas),
        "mean_constraint_pass_rate_delta": mean(deltas) if deltas else None,
        "mean_decision_constraint_pass_rate_delta": mean(quality_deltas) if quality_deltas else None,
        "limitations": [
            "Curated synthetic cases measure retrieval and deterministic constraints, not realized returns.",
            "Offline citation and learning scores are plumbing contracts, not LLM decision quality.",
            "Live trials use fixed inputs and identical model configuration; generations remain stochastic.",
            "Token totals are null if any production invocation omits usage metadata.",
            "Only complete pairs contribute to model deltas; inspect failures and per-case outputs.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", choices=["development", "evaluation"], default="evaluation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true", help="Explicitly call the configured model for paired trials")
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args.cases, split=args.split, live=args.live)
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)
    if not result["learning_contract"]["passed"] or result["metrics"]["nonproduction_injection_count"] or result["metrics"]["invalid_citation_rate"]:
        return 1
    if any(not pair[arm]["safety_passed"] for pair in result["model_pairs"] for arm in ("with_memory", "without_memory")):
        return 1
    return 1 if args.live and result["completed_pairs"] != result["case_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

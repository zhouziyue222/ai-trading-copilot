# -*- coding: utf-8 -*-
"""Explainable list-batch LangGraph for the AI trading copilot."""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import contextvars
from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ai_trading_copilot.copilot.agents import (
    FundamentalAnalystAgent,
    NewsSentimentAgent,
    OpportunityRadarAgent,
    PortfolioManager,
    PostTradeReviewLearningAgent,
    RiskAgent,
    RunExplanationAgent,
    TechnicalPositionAgent,
    TraderAgent,
)
from ai_trading_copilot.copilot.adapters.portfolio import (
    format_portfolio_snapshot,
    get_futu_portfolio_snapshot,
)
from ai_trading_copilot.copilot.config import (
    DEFAULT_PERSONA_MARKDOWN,
    DEFAULT_REPORT_OUTPUT_DIR,
    create_default_deepseek_llm,
)
from ai_trading_copilot.copilot.domain.enums import (
    AnalystType,
    ExecutionMode,
    MarketRegime,
    SubscriptionStatus,
    SymbolTrendState,
)
from ai_trading_copilot.copilot.domain.models import (
    DistilledMemory,
    ExecutionDecision,
    FundamentalAnalysisReport,
    NewsSentimentReport,
    MemoryRetrievalRecord,
    OpportunityRadarItem,
    PortfolioSnapshot,
    PriceBar,
    RiskAssessment,
    Subscription,
    SubscriptionBook,
    TechnicalContext,
    TechnicalPosition,
    TraceEvent,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.localization import zh_bool, zh_join, zh_label
from ai_trading_copilot.copilot.graph.state import CopilotGraphState
from ai_trading_copilot.copilot.services.cancellation import (
    CancellationToken,
    GLOBAL_CANCELLATION_MANAGER,
    RunCancelled,
    activate_cancellation,
)
from ai_trading_copilot.copilot.services.run_tracker import RunTracker
from ai_trading_copilot.copilot.services.memory_retrieval import MemoryRetrievalSession
from ai_trading_copilot.copilot.services.tracing import (
    TRACE_JSON_REPORT_KEY,
    TRACE_MARKDOWN_REPORT_KEY,
    TraceRecorder,
    get_current_trace_recorder,
)


TraceWriter = Callable[[List[TraceEvent]], None]
PortfolioGetter = Callable[[ExecutionMode], PortfolioSnapshot]


DEFAULT_SELECTED_ANALYSTS = [
    AnalystType.NEWS_SENTIMENT,
    AnalystType.TECHNICAL_POSITION,
    AnalystType.FUNDAMENTAL_ANALYSIS,
]

_DEFAULT_LLM = object()


class _ParallelBranchFailure(RuntimeError):
    """Internal stop signal used when one analyst fails."""


def _next_index(node_order: list[str], completed_nodes: set[str] | Iterable[str]) -> int:
    completed = set(completed_nodes)
    for index, node_name in enumerate(node_order):
        if node_name not in completed:
            return index
    return len(node_order)


class CopilotLangGraph:
    """List-batch graph with optional analysts and portfolio decisions."""

    NODE_LOAD_PERSONA_MARKDOWN = "Load Persona Markdown"
    NODE_LOAD_SUBSCRIPTION_SYMBOLS = "Load Subscription Symbols"
    NODE_OPPORTUNITY_RADAR = "Opportunity Radar"
    NODE_TECHNICAL_POSITION = "Technical Position"
    NODE_NEWS_SENTIMENT = "News Sentiment"
    NODE_FUNDAMENTAL_ANALYSIS = "Fundamental Analysis"
    NODE_TRADER = "Trader"
    NODE_RISK_CHECK = "Risk Check"
    NODE_PORTFOLIO_MANAGER = "Portfolio Manager"
    NODE_EXPLAIN_RUN = "Explain Run"
    NODE_PERSIST_TRACE = "Persist Trace"

    NODE_ORDER = [
        NODE_LOAD_PERSONA_MARKDOWN,
        NODE_LOAD_SUBSCRIPTION_SYMBOLS,
        NODE_TECHNICAL_POSITION,
        NODE_NEWS_SENTIMENT,
        NODE_FUNDAMENTAL_ANALYSIS,
        NODE_TRADER,
        NODE_RISK_CHECK,
        NODE_PORTFOLIO_MANAGER,
        NODE_EXPLAIN_RUN,
        NODE_PERSIST_TRACE,
    ]

    def __init__(
        self,
        *,
        opportunity_radar_agent: Optional[OpportunityRadarAgent] = None,
        news_sentiment_agent: Optional[NewsSentimentAgent] = None,
        fundamental_analyst_agent: Optional[FundamentalAnalystAgent] = None,
        technical_position_agent: Optional[TechnicalPositionAgent] = None,
        trader_agent: Optional[TraderAgent] = None,
        risk_agent: Optional[RiskAgent] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        memory_agent: Optional[PostTradeReviewLearningAgent] = None,
        explanation_agent: Optional[RunExplanationAgent] = None,
        portfolio_getter: Optional[PortfolioGetter] = None,
        trace_writer: Optional[TraceWriter] = None,
        default_persona_markdown: str = DEFAULT_PERSONA_MARKDOWN,
        default_persona_config: Optional[UserPersonaConfig] = None,
        default_selected_analysts: Optional[Iterable[AnalystType | str]] = None,
        report_output_dir: Optional[str | Path] = None,
        llm=_DEFAULT_LLM,
        enable_default_llm: bool = True,
        opportunity_radar_llm=None,
        technical_position_llm=None,
        technical_position_debug: bool = False,
        technical_position_prompt_name: str | None = None,
        news_sentiment_llm=None,
        fundamental_llm=None,
        run_tracker: Optional[RunTracker] = None,
        fundamental_rag_retriever=None,
        force_sequential: bool | None = None,
        parallel_analysts: bool = False,
        cancellation_checker: Optional[Callable[[], bool]] = None,
        snapshot_callback: Optional[Callable[..., None]] = None,
        commit_callback: Optional[Callable[[], Any]] = None,
    ):
        default_llm = None
        if llm is _DEFAULT_LLM:
            if enable_default_llm:
                default_llm = create_default_deepseek_llm()
        else:
            default_llm = llm
        self.llm_unavailable = (
            llm is _DEFAULT_LLM and enable_default_llm and default_llm is None
        )
        if parallel_analysts and llm is _DEFAULT_LLM and default_llm is not None:
            if news_sentiment_agent is None and news_sentiment_llm is None:
                news_sentiment_llm = create_default_deepseek_llm()
            if technical_position_agent is None and technical_position_llm is None:
                technical_position_llm = create_default_deepseek_llm()
            if fundamental_analyst_agent is None and fundamental_llm is None:
                fundamental_llm = create_default_deepseek_llm()
        self.opportunity_radar_agent = opportunity_radar_agent or OpportunityRadarAgent(
            llm=opportunity_radar_llm or default_llm
        )
        self.news_sentiment_agent = news_sentiment_agent or NewsSentimentAgent(
            llm=news_sentiment_llm or default_llm
        )
        resolved_fundamental_llm = fundamental_llm or default_llm
        self.fundamental_analyst_agent = (
            fundamental_analyst_agent
            or FundamentalAnalystAgent(
                llm=resolved_fundamental_llm,
                rag_retriever=fundamental_rag_retriever,
            )
        )
        self.technical_position_agent = technical_position_agent or TechnicalPositionAgent(
            llm=technical_position_llm or default_llm,
            debug_mode=technical_position_debug,
            prompt_name=technical_position_prompt_name,
        )
        self.trader_agent = trader_agent or TraderAgent(llm=default_llm)
        self.risk_agent = risk_agent or RiskAgent(llm=default_llm)
        self.portfolio_manager = portfolio_manager or PortfolioManager(llm=default_llm)
        self.memory_agent = memory_agent
        if default_llm is not None and hasattr(self.fundamental_analyst_agent, "set_rag_query_llm"):
            self.fundamental_analyst_agent.set_rag_query_llm(resolved_fundamental_llm)
        self.explanation_agent = explanation_agent or RunExplanationAgent(
            llm=default_llm
        )
        self.portfolio_getter = portfolio_getter or get_futu_portfolio_snapshot
        self.trace_writer = trace_writer
        self.force_sequential = (
            _env_flag("COPILOT_FORCE_SEQUENTIAL")
            if force_sequential is None
            else force_sequential
        )
        self.default_persona_markdown = default_persona_markdown
        self.default_persona_config = default_persona_config or UserPersonaConfig()
        self.default_selected_analysts = _normalize_analysts(
            default_selected_analysts or DEFAULT_SELECTED_ANALYSTS
        )
        self.report_output_dir = str(report_output_dir or DEFAULT_REPORT_OUTPUT_DIR)
        self.run_tracker = run_tracker
        self.cancellation_checker = cancellation_checker
        self.snapshot_callback = snapshot_callback
        self.commit_callback = commit_callback
        self.parallel_analysts = bool(parallel_analysts)
        self._parallel_worker_context = threading.local()
        self._parallel_failure_event: threading.Event | None = None
        self._resume_snapshot = None
        self.trace_recorder: TraceRecorder | None = None
        self._compiled_graph = self._compile_graph()
        self.langgraph_available = self._compiled_graph is not None

    def run(
        self,
        *,
        subscription_symbols: List[str],
        price_history_by_symbol: Optional[Dict[str, List[PriceBar]]] = None,
        portfolio: Optional[PortfolioSnapshot] = None,
        persona_markdown: Optional[str] = None,
        persona_config: Optional[UserPersonaConfig] = None,
        fundamental_analysis_by_symbol: Optional[Dict[str, FundamentalAnalysisReport]] = None,
        selected_analysts: Optional[Iterable[AnalystType | str]] = None,
        report_output_dir: Optional[str | Path] = None,
        trade_date: Optional[str] = None,
        look_back_days: int = 90,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        portfolio_mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
        run_id: str | None = None,
    ) -> CopilotGraphState:
        self._check_cancelled()
        resolved_portfolio, portfolio_report, portfolio_error = self._resolve_portfolio(
            portfolio=portfolio,
            portfolio_mode=portfolio_mode,
        )
        self._check_cancelled()
        report_dir = str(report_output_dir or self.report_output_dir)
        portfolio_report_path = self._save_agent_report(
            state={"report_output_dir": report_dir},
            stage="0_portfolio",
            agent_name="futu_portfolio",
            content=portfolio_report,
        )
        initial_state: CopilotGraphState = {
            "persona_markdown": persona_markdown or self.default_persona_markdown,
            "persona_config": persona_config or self.default_persona_config,
            "subscription_symbols": subscription_symbols,
            "selected_analysts": list(
                self.default_selected_analysts
                if selected_analysts is None
                else selected_analysts
            ),
            "portfolio": resolved_portfolio,
            "portfolio_mode": portfolio_mode,
            "price_history_by_symbol": price_history_by_symbol or {},
            "fundamental_analysis_by_symbol": fundamental_analysis_by_symbol or {},
            "trade_date": trade_date,
            "look_back_days": look_back_days,
            "execution_mode": mode,
            "user_confirmed": user_confirmed,
            "run_id": run_id or "",
            "long_term_memory_enabled": self.memory_agent is not None,
            "errors": [portfolio_error] if portfolio_error else [],
            "agent_reports": {"futu_portfolio": portfolio_report_path},
            "report_output_dir": str(report_output_dir or self.report_output_dir),
        }
        return self.invoke(initial_state)

    def invoke(self, state: CopilotGraphState) -> CopilotGraphState:
        base_state: CopilotGraphState = {
            "persona_markdown": self.default_persona_markdown,
            "persona_config": self.default_persona_config,
            "subscription_symbols": [],
            "selected_analysts": self.default_selected_analysts,
            "trace_events": [],
            "errors": [],
            "explanations": {},
            "analyst_reports": {},
            "agent_reports": {},
            "market_reports_by_symbol": {},
            "fundamental_analysis_reports_by_symbol": {},
            "news_sentiment_reports_by_symbol": {},
            "news_sentiment_by_symbol": {},
            "technical_contexts": {},
            "opportunity_reports_by_symbol": {},
            "memories": {},
            "shadow_memories": {},
            "memory_retrievals": [],
            "report_output_dir": self.report_output_dir,
            "portfolio_mode": ExecutionMode.SIMULATION,
            "fundamental_analysis_by_symbol": {},
            "trade_date": None,
            "look_back_days": 90,
            "execution_mode": ExecutionMode.SIMULATION,
            "user_confirmed": False,
            "run_id": "",
        }
        base_state.update(state)
        base_state["selected_analysts"] = _normalize_analysts(
            base_state.get("selected_analysts", self.default_selected_analysts)
        )
        base_state["subscription_symbols"] = _normalize_symbols(
            base_state["subscription_symbols"]
        )
        base_state["price_history_by_symbol"] = _normalize_price_history(
            base_state.get("price_history_by_symbol", {})
        )
        return self._invoke_with_tracing(base_state)

    def _invoke_with_tracing(self, base_state: CopilotGraphState) -> CopilotGraphState:
        recorder = TraceRecorder(
            output_dir=base_state.get("report_output_dir") or self.report_output_dir,
            run_id=str(base_state.get("run_id") or ""),
            symbols=list(base_state.get("subscription_symbols", [])),
            activity_callback=self._record_trace_activity,
        )
        cancellation_token = None
        if self.cancellation_checker is not None:
            cancellation_token = CancellationToken(
                run_id=str(base_state.get("run_id") or ""),
                checker=self.cancellation_checker,
                manager=GLOBAL_CANCELLATION_MANAGER,
            )
        self.trace_recorder = recorder
        try:
            with activate_cancellation(cancellation_token), recorder.activate():
                with recorder.start_span(
                    "copilot.run",
                    attributes={
                        "run.id": base_state.get("run_id") or "",
                        "run.symbol_count": len(base_state.get("subscription_symbols", [])),
                        "run.symbols": base_state.get("subscription_symbols", []),
                        "run.selected_analysts": [
                            analyst.value if hasattr(analyst, "value") else str(analyst)
                            for analyst in base_state.get("selected_analysts", [])
                        ],
                        "run.portfolio_mode": base_state.get("portfolio_mode"),
                        "run.execution_mode": base_state.get("execution_mode"),
                    },
                ) as span:
                    result = self._invoke_graph(base_state)
                    span.set_attribute("run.trace_event_count", len(result.get("trace_events", [])))
                    span.set_attribute("run.error_count", len(result.get("errors", [])))
                    return result
        finally:
            trace_json_path, trace_markdown_path = recorder.flush()
            if self.run_tracker is not None:
                self.run_tracker.record_report(TRACE_JSON_REPORT_KEY, trace_json_path)
                self.run_tracker.record_report(TRACE_MARKDOWN_REPORT_KEY, trace_markdown_path)

    def _invoke_graph(self, base_state: CopilotGraphState) -> CopilotGraphState:
        self._check_cancelled()
        if self.llm_unavailable:
            if self.run_tracker is not None:
                self.run_tracker.start_node("Fail Closed")
            state = self._fail_closed(base_state)
            if self.run_tracker is not None:
                self.run_tracker.succeed_node("Fail Closed")
            return state
        if self.force_sequential:
            if (
                self.parallel_analysts
                or self.commit_callback is not None
                or self.snapshot_callback is not None
                or bool((self._resume_snapshot or {}).get("completed_nodes"))
            ):
                return self._run_controlled(base_state)
            return self._run_sequential(base_state)
        if self._compiled_graph is not None:
            return self._compiled_graph.invoke(base_state)
        return self._run_sequential(base_state)

    def _check_cancelled(self) -> None:
        if self._parallel_failure_event is not None and self._parallel_failure_event.is_set():
            raise _ParallelBranchFailure("A parallel analyst failed.")
        if self.cancellation_checker is not None and self.cancellation_checker():
            raise RunCancelled("Run cancelled by user.")

    def _compile_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None

        workflow = StateGraph(CopilotGraphState)
        node_map = self._node_map()
        for node_name in _unique_list(
            [
                *self.NODE_ORDER,
                self.NODE_OPPORTUNITY_RADAR,
            ]
        ):
            workflow.add_node(node_name, node_map[node_name])

        workflow.add_edge(START, self.NODE_LOAD_PERSONA_MARKDOWN)
        workflow.add_edge(
            self.NODE_LOAD_PERSONA_MARKDOWN,
            self.NODE_LOAD_SUBSCRIPTION_SYMBOLS,
        )
        workflow.add_conditional_edges(
            self.NODE_LOAD_SUBSCRIPTION_SYMBOLS,
            self._route_analyst_nodes,
            [
                self.NODE_OPPORTUNITY_RADAR,
                self.NODE_TECHNICAL_POSITION,
                self.NODE_NEWS_SENTIMENT,
                self.NODE_FUNDAMENTAL_ANALYSIS,
                self.NODE_TRADER,
            ],
        )
        workflow.add_edge(self.NODE_OPPORTUNITY_RADAR, self.NODE_TRADER)
        workflow.add_edge(self.NODE_TECHNICAL_POSITION, self.NODE_TRADER)
        workflow.add_edge(self.NODE_NEWS_SENTIMENT, self.NODE_TRADER)
        workflow.add_edge(self.NODE_FUNDAMENTAL_ANALYSIS, self.NODE_TRADER)
        workflow.add_edge(self.NODE_TRADER, self.NODE_RISK_CHECK)
        workflow.add_edge(self.NODE_RISK_CHECK, self.NODE_PORTFOLIO_MANAGER)
        workflow.add_edge(self.NODE_PORTFOLIO_MANAGER, self.NODE_EXPLAIN_RUN)
        workflow.add_edge(self.NODE_EXPLAIN_RUN, self.NODE_PERSIST_TRACE)
        workflow.add_edge(self.NODE_PERSIST_TRACE, END)
        return workflow.compile()

    def _sequential_node_order(self, state: CopilotGraphState) -> list[str]:
        return [
            self.NODE_LOAD_PERSONA_MARKDOWN,
            self.NODE_LOAD_SUBSCRIPTION_SYMBOLS,
            *[
                node
                for node in self._route_analyst_nodes(state)
                if node != self.NODE_TRADER
            ],
            self.NODE_TRADER,
            self.NODE_RISK_CHECK,
            self.NODE_PORTFOLIO_MANAGER,
            self.NODE_EXPLAIN_RUN,
            self.NODE_PERSIST_TRACE,
        ]

    @classmethod
    def checkpoint_node_order(cls, selected_analysts: Iterable) -> list[str]:
        """Validate compatibility without constructing agents or querying tools."""
        selected = _normalize_analysts(selected_analysts)
        routes = [(AnalystType.OPPORTUNITY_RADAR, cls.NODE_OPPORTUNITY_RADAR),
                  (AnalystType.TECHNICAL_POSITION, cls.NODE_TECHNICAL_POSITION),
                  (AnalystType.NEWS_SENTIMENT, cls.NODE_NEWS_SENTIMENT),
                  (AnalystType.FUNDAMENTAL_ANALYSIS, cls.NODE_FUNDAMENTAL_ANALYSIS)]
        return [cls.NODE_LOAD_PERSONA_MARKDOWN, cls.NODE_LOAD_SUBSCRIPTION_SYMBOLS,
                *[node for analyst, node in routes if analyst in selected],
                cls.NODE_TRADER, cls.NODE_RISK_CHECK, cls.NODE_PORTFOLIO_MANAGER,
                cls.NODE_EXPLAIN_RUN, cls.NODE_PERSIST_TRACE]

    def resume(self, snapshot: dict) -> CopilotGraphState:
        """Continue a validated snapshot without re-fetching initial inputs."""
        if snapshot["node_order"] != self._sequential_node_order(snapshot["state"]):
            raise ValueError("Checkpoint node order does not match this graph.")
        self.force_sequential = True
        self._resume_snapshot = snapshot
        return self._invoke_with_tracing(snapshot["state"])

    def _run_sequential(self, state: CopilotGraphState) -> CopilotGraphState:
        current = deepcopy(state)
        node_map = self._node_map()
        node_order = self._sequential_node_order(current)
        next_index = self._resume_snapshot["next_node_index"] if self._resume_snapshot else 0
        completed_nodes = list(node_order[:next_index])
        self._emit_snapshot(current, node_order, next_index, completed_nodes)
        for index in range(next_index, len(node_order)):
            self._check_cancelled()
            working = deepcopy(current)
            updates = node_map[node_order[index]](working)
            current = self._merge_state(working, updates)
            completed_nodes.append(node_order[index])
            self._emit_snapshot(current, node_order, index + 1, completed_nodes)
        return current

    def _run_controlled(self, state: CopilotGraphState) -> CopilotGraphState:
        current = deepcopy(state)
        node_map = self._node_map()
        order = self._sequential_node_order(current)
        snapshot = self._resume_snapshot or {}
        completed = set(snapshot.get("completed_nodes", order[:snapshot.get("next_node_index", 0)]))
        analysts = [name for name in (self.NODE_TECHNICAL_POSITION, self.NODE_NEWS_SENTIMENT,
                                      self.NODE_FUNDAMENTAL_ANALYSIS) if name in order]
        self._emit_snapshot(current, order, _next_index(order, completed), completed)
        for name in order:
            if name in completed:
                continue
            pending = [node for node in analysts if node not in completed]
            if self.parallel_analysts and name in analysts and len(pending) > 1:
                current = self._run_parallel_analysts(current, node_map, pending, completed, order)
            else:
                self._check_cancelled()
                working = deepcopy(current)
                try:
                    updates, reports = self._execute_staged(node_map[name], working)
                    with self._commit_context():
                        self._check_cancelled()
                        candidate = self._merge_state(working, updates)
                        self._publish_boundary(name, reports, candidate, completed, order)
                        current = candidate
                except RunCancelled:
                    raise
                except Exception as exc:
                    if self.run_tracker is not None:
                        self.run_tracker.fail_node(name, exc)
                    raise
        return current

    def _execute_staged(self, node_func, state):
        local = self._parallel_worker_context
        local.active = True
        local.reports = []
        try:
            updates = node_func(state)
            return updates, list(local.reports)
        finally:
            local.active = False
            local.reports = []

    def _publish_boundary(self, name, reports, state, completed, order):
        analyst_keys = (AnalystType.TECHNICAL_POSITION.value, AnalystType.NEWS_SENTIMENT.value,
                        AnalystType.FUNDAMENTAL_ANALYSIS.value)
        for field in ("agent_reports", "analyst_reports"):
            values = state.get(field, {})
            state[field] = {**{key: value for key, value in values.items() if key not in analyst_keys},
                            **{key: values[key] for key in analyst_keys if key in values}}
        ranks = {node: index for index, node in enumerate(order)}
        state["trace_events"] = sorted(state.get("trace_events", []),
                                       key=lambda event: ranks.get(event.node_name if hasattr(event, "node_name")
                                                                   else event.get("node_name"), len(order)))
        for key, temporary, destination in reports:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            if self.run_tracker is not None:
                self.run_tracker.record_report(key, destination)
        if self.run_tracker is not None:
            self.run_tracker.succeed_node(name)
        completed.add(name)
        self._emit_snapshot(state, order, _next_index(order, completed), completed)

    def _run_parallel_analysts(self, current, node_map, pending, completed, node_order):
        base = deepcopy(current)
        results = queue.Queue()
        stop = threading.Event()
        self._parallel_failure_event = stop
        workers = []
        first_failure = None
        cancelled = None
        successful = {}
        checker = self.cancellation_checker
        token = CancellationToken(str(base.get("run_id", "")),
                                  lambda: stop.is_set() or bool(checker and checker()),
                                  GLOBAL_CANCELLATION_MANAGER)

        def worker(name):
            try:
                with activate_cancellation(token):
                    updates, reports = self._execute_staged(node_map[name], deepcopy(base))
                results.put((name, updates, reports, None))
            except BaseException as exc:
                if not isinstance(exc, (RunCancelled, _ParallelBranchFailure)):
                    stop.set()
                results.put((name, None, [], exc))

        try:
            for name in pending:
                context = contextvars.copy_context()
                thread = threading.Thread(target=context.run, args=(worker, name),
                                          name=f"copilot-{name}", daemon=True)
                thread.start()
                workers.append(thread)
            for _ in workers:
                name, updates, reports, error = results.get()
                if error is not None:
                    if isinstance(error, (RunCancelled, _ParallelBranchFailure)):
                        cancelled = error
                    elif first_failure is None:
                        first_failure = error
                        stop.set()
                        if self.run_tracker is not None:
                            self.run_tracker.fail_node(name, error)
                        GLOBAL_CANCELLATION_MANAGER.request_cancel(str(base.get("run_id", "")))
                    continue
                if first_failure is not None or stop.is_set():
                    continue
                try:
                    with self._commit_context():
                        self._check_cancelled()
                        successful[name] = updates
                        candidate = deepcopy(base)
                        for ordered in pending:
                            if ordered in successful:
                                candidate = self._merge_state(candidate, successful[ordered])
                        self._publish_boundary(name, reports, candidate, completed, node_order)
                        current = candidate
                except (RunCancelled, _ParallelBranchFailure) as exc:
                    cancelled = exc
                except Exception as exc:
                    first_failure = exc
                    stop.set()
                    GLOBAL_CANCELLATION_MANAGER.request_cancel(str(base.get("run_id", "")))
        except BaseException:
            stop.set()
            GLOBAL_CANCELLATION_MANAGER.request_cancel(str(base.get("run_id", "")))
            raise
        finally:
            # The coordinator owns all branch lifetimes. Never finalize a run
            # while a late worker can still write into its temporary directory.
            for thread in workers:
                thread.join()
            self._parallel_failure_event = None
        if first_failure is not None:
            raise first_failure
        if cancelled is not None:
            raise cancelled
        self._check_cancelled()
        return current

    def _commit_context(self):
        return self.commit_callback() if self.commit_callback is not None else nullcontext()

    def _emit_snapshot(self, state, node_order, next_index, completed_nodes):
        if self.snapshot_callback is None:
            return
        try:
            self.snapshot_callback(
                state, node_order, next_index, completed_nodes=[name for name in node_order if name in completed_nodes]
            )
        except TypeError as exc:
            # Keep test doubles and older graph integrations source-compatible.
            if "completed_nodes" not in str(exc):
                raise
            self.snapshot_callback(state, node_order, next_index)

    def _route_analyst_nodes(self, state: CopilotGraphState) -> List[str]:
        selected = state.get("selected_analysts", [])
        routes: List[str] = []
        if AnalystType.OPPORTUNITY_RADAR in selected:
            routes.append(self.NODE_OPPORTUNITY_RADAR)
        if AnalystType.TECHNICAL_POSITION in selected:
            routes.append(self.NODE_TECHNICAL_POSITION)
        else:
            self._skip_node(self.NODE_TECHNICAL_POSITION, "analyst not selected")
        if AnalystType.NEWS_SENTIMENT in selected:
            routes.append(self.NODE_NEWS_SENTIMENT)
        else:
            self._skip_node(self.NODE_NEWS_SENTIMENT, "analyst not selected")
        if AnalystType.FUNDAMENTAL_ANALYSIS in selected:
            routes.append(self.NODE_FUNDAMENTAL_ANALYSIS)
        else:
            self._skip_node(self.NODE_FUNDAMENTAL_ANALYSIS, "analyst not selected")
        return routes or [self.NODE_TRADER]

    def _merge_state(
        self,
        current: CopilotGraphState,
        updates: CopilotGraphState,
    ) -> CopilotGraphState:
        merged = dict(current)
        for key, value in updates.items():
            if key == "trace_events":
                merged[key] = [*merged.get(key, []), *value]
            elif key in {
                "analyst_reports",
                "agent_reports",
                "market_reports_by_symbol",
                "fundamental_analysis_reports_by_symbol",
                "news_sentiment_reports_by_symbol",
                "opportunity_reports_by_symbol",
            }:
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
        return merged

    def _node_map(self):
        node_map = {
            self.NODE_LOAD_PERSONA_MARKDOWN: self._load_persona_markdown,
            self.NODE_LOAD_SUBSCRIPTION_SYMBOLS: self._load_subscription_symbols,
            self.NODE_OPPORTUNITY_RADAR: self._opportunity_radar,
            self.NODE_TECHNICAL_POSITION: self._technical_position,
            self.NODE_NEWS_SENTIMENT: self._news_sentiment,
            self.NODE_FUNDAMENTAL_ANALYSIS: self._fundamental_analysis,
            self.NODE_TRADER: self._trader,
            self.NODE_RISK_CHECK: self._risk_check,
            self.NODE_PORTFOLIO_MANAGER: self._portfolio_manager,
            self.NODE_EXPLAIN_RUN: self._explain_run,
            self.NODE_PERSIST_TRACE: self._persist_trace,
        }
        return {
            node_name: self._tracked_node(node_name, node_func)
            for node_name, node_func in node_map.items()
        }

    def _tracked_node(self, node_name: str, node_func: Callable):
        def wrapped(state: CopilotGraphState) -> CopilotGraphState:
            self._check_cancelled()
            parallel_worker = bool(getattr(self._parallel_worker_context, "active", False))
            if self.run_tracker is not None:
                self.run_tracker.start_node(node_name)
            recorder = get_current_trace_recorder()
            try:
                if recorder is None:
                    updates = node_func(state)
                else:
                    with recorder.start_span(
                        "graph.node",
                        attributes={
                            "graph.node.name": node_name,
                            "graph.node.input_symbols": state.get("subscription_symbols", []),
                        },
                    ) as span:
                        updates = node_func(state)
                        span.set_attribute("graph.node.update_keys", sorted(updates.keys()))
                        span.set_attribute(
                            "graph.node.trace_events",
                            len(updates.get("trace_events", [])),
                        )
                self._check_cancelled()
            except RunCancelled:
                raise
            except Exception as exc:
                if self.run_tracker is not None and not parallel_worker:
                    self.run_tracker.fail_node(node_name, exc)
                raise
            if self.run_tracker is not None and not parallel_worker:
                self.run_tracker.succeed_node(node_name)
            return updates

        return wrapped

    def _skip_node(self, node_name: str, reason: str) -> None:
        if self.run_tracker is not None:
            self.run_tracker.skip_node(node_name, reason)
        recorder = get_current_trace_recorder()
        if recorder is not None:
            recorder.instant_span(
                "graph.node",
                status="skipped",
                attributes={
                    "graph.node.name": node_name,
                    "graph.node.skip_reason": reason,
                },
            )

    def _record_trace_activity(self, event: str, span, recorder: TraceRecorder) -> None:
        if self.run_tracker is None:
            return
        if span.name not in {"llm.invoke", "tool.call"}:
            return
        node_name = _span_graph_node_name(span, recorder)
        if node_name not in {
            self.NODE_TECHNICAL_POSITION,
            self.NODE_NEWS_SENTIMENT,
            self.NODE_FUNDAMENTAL_ANALYSIS,
            self.NODE_TRADER,
            self.NODE_PORTFOLIO_MANAGER,
        }:
            return
        self.run_tracker.record_node_activity(
            node_name,
            _activity_from_span(event, span),
            completed=event == "end",
        )

    def _load_persona_markdown(self, state: CopilotGraphState) -> CopilotGraphState:
        persona_markdown = state.get("persona_markdown") or self.default_persona_markdown
        persona_config = state.get("persona_config") or self.default_persona_config
        return self._with_trace(
            state,
            TraceEvent(
                node_name=self.NODE_LOAD_PERSONA_MARKDOWN,
                input_summary="persona_markdown field",
                output_summary=f"chars={len(persona_markdown)}",
                route_reason="User persona is represented as a markdown document, not an agent.",
            ),
            persona_markdown=persona_markdown,
            persona_config=persona_config,
        )

    def _load_subscription_symbols(self, state: CopilotGraphState) -> CopilotGraphState:
        symbols = _normalize_symbols(state.get("subscription_symbols", []))
        return self._with_trace(
            state,
            TraceEvent(
                node_name=self.NODE_LOAD_SUBSCRIPTION_SYMBOLS,
                input_summary="subscription_symbols field",
                output_summary=f"subscriptions={len(symbols)}",
                route_reason="The subscription list is a state field; no SubscriptionAgent is used.",
            ),
            subscription_symbols=symbols,
        )

    def _resolve_portfolio(
        self,
        *,
        portfolio: PortfolioSnapshot | None,
        portfolio_mode: ExecutionMode,
    ) -> tuple[PortfolioSnapshot, str, str | None]:
        if portfolio is not None:
            return (
                portfolio,
                format_portfolio_snapshot(
                    snapshot=portfolio,
                    mode=portfolio_mode,
                ),
                None,
            )
        try:
            snapshot = self.portfolio_getter(portfolio_mode)
            return (
                snapshot,
                format_portfolio_snapshot(snapshot=snapshot, mode=portfolio_mode),
                None,
            )
        except Exception as exc:
            snapshot = PortfolioSnapshot()
            error = f"portfolio_fetch_failed: {exc}"
            return (
                snapshot,
                format_portfolio_snapshot(
                    snapshot=snapshot,
                    mode=portfolio_mode,
                    error=str(exc),
                ),
                error,
            )

    def _fail_closed(self, state: CopilotGraphState) -> CopilotGraphState:
        message = (
            "LLM is not configured. The production run failed closed, so no actionable "
            "trade recommendation was generated."
        )
        report_path = self._save_agent_report(
            state=state,
            stage="0_errors",
            agent_name="llm_unavailable",
            content=f"# LLM unavailable\n\n{message}\n",
        )
        return {
            **state,
            "errors": [*state.get("errors", []), "llm_unavailable"],
            "agent_reports": {"llm_unavailable": report_path},
            "trade_plans": {},
            "risk_assessments": {},
            "execution_decisions": {},
            "trace_events": [
                TraceEvent(
                    node_name="Fail Closed",
                    input_summary="default LLM configuration",
                    output_summary="run_blocked=True",
                    route_reason=message,
                    warnings=["llm_unavailable"],
                )
            ],
        }

    def _opportunity_radar(self, state: CopilotGraphState) -> CopilotGraphState:
        if AnalystType.OPPORTUNITY_RADAR not in state["selected_analysts"]:
            return self._with_trace(
                state,
                TraceEvent(
                    node_name=self.NODE_OPPORTUNITY_RADAR,
                    input_summary="analyst not selected",
                    output_summary="radar_items=0",
                    route_reason="Opportunity Radar analyst was not selected.",
                    warnings=["analyst_skipped"],
                ),
                radar_items=[],
            )

        items: List[OpportunityRadarItem] = []
        reports_by_symbol: Dict[str, str] = {}
        tool_calls: List[str] = []
        fallback_warnings: List[str] = []
        for symbol in state["subscription_symbols"]:
            bars = (
                None
                if getattr(self.opportunity_radar_agent, "llm", None) is not None
                else state["price_history_by_symbol"].get(symbol)
            )
            if hasattr(self.opportunity_radar_agent, "analyze_symbol_with_report"):
                result = self.opportunity_radar_agent.analyze_symbol_with_report(
                    symbol=symbol,
                    bars=bars,
                    trade_date=state.get("trade_date"),
                    look_back_days=state.get("look_back_days", 90),
                )
                items.append(result.item)
                reports_by_symbol[symbol] = result.report
                tool_calls.extend(result.tool_calls)
                if getattr(result, "fallback_used", False) and getattr(result, "fallback_reason", ""):
                    fallback_warnings.append(f"{symbol}: {result.fallback_reason}")
            else:
                items.append(
                    self.opportunity_radar_agent.analyze_symbol(
                        symbol=symbol,
                        bars=bars,
                    )
                )
        report_path = self._save_agent_report(
            state=state,
            stage="1_analysts",
            agent_name=AnalystType.OPPORTUNITY_RADAR.value,
            content=(
                self._format_opportunity_radar_report(items)
                + self._format_symbol_reports(reports_by_symbol)
            ),
        )
        return self._with_trace(
            state,
            TraceEvent(
                node_name=self.NODE_OPPORTUNITY_RADAR,
                input_summary=f"subscription_symbols={state['subscription_symbols']}",
                output_summary=f"radar_items={len(items)}",
                route_reason=(
                    "Opportunity Radar classifies each subscribed symbol with tools when an LLM is configured; "
                    "otherwise it uses deterministic fallback rules."
                ),
                rule_hits=sorted(set(tool_calls)),
                warnings=fallback_warnings,
            ),
            radar_items=items,
            analyst_reports={AnalystType.OPPORTUNITY_RADAR.value: report_path},
            agent_reports={AnalystType.OPPORTUNITY_RADAR.value: report_path},
            opportunity_reports_by_symbol=reports_by_symbol,
        )

    def _technical_position(self, state: CopilotGraphState) -> CopilotGraphState:
        if AnalystType.TECHNICAL_POSITION not in state["selected_analysts"]:
            return self._with_trace(
                state,
                TraceEvent(
                    node_name=self.NODE_TECHNICAL_POSITION,
                    input_summary="analyst not selected",
                    output_summary="technical_positions=0",
                    route_reason="Technical Position analyst was not selected.",
                    warnings=["analyst_skipped"],
                ),
                technical_positions={},
            )

        positions: Dict[str, TechnicalPosition] = {}
        contexts: Dict[str, TechnicalContext] = {}
        reports_by_symbol: Dict[str, str] = {}
        events: List[TraceEvent] = []
        for symbol in state["subscription_symbols"]:
            bars = (
                None
                if getattr(self.technical_position_agent, "llm", None) is not None
                else state["price_history_by_symbol"].get(symbol)
            )
            if not bars and getattr(self.technical_position_agent, "llm", None) is None:
                events.append(
                    TraceEvent(
                        node_name=self.NODE_TECHNICAL_POSITION,
                        symbol=symbol,
                        input_summary="missing price history",
                        output_summary="technical_position=skipped",
                        route_reason="No usable price history was available for this subscribed symbol.",
                        warnings=["missing_price_history"],
                    )
                )
                continue
            if hasattr(self.technical_position_agent, "analyze_with_report"):
                result = self.technical_position_agent.analyze_with_report(
                    symbol=symbol,
                    bars=bars,
                    trade_date=state.get("trade_date"),
                    look_back_days=state.get("look_back_days", 90),
                )
                position = result.position
                if result.context is not None:
                    contexts[symbol] = result.context
                reports_by_symbol[symbol] = result.report
                tool_calls = result.tool_calls
                node_warnings = (
                    [result.fallback_reason]
                    if getattr(result, "fallback_used", False)
                    and getattr(result, "fallback_reason", "")
                    else []
                )
            else:
                position = self.technical_position_agent.analyze(symbol, bars)
                tool_calls = []
                node_warnings = []
            positions[symbol] = position
            events.append(
                TraceEvent(
                    node_name=self.NODE_TECHNICAL_POSITION,
                    symbol=symbol,
                    input_summary=(
                        "price_data=tool_fetch" if bars is None else f"price_data={len(bars)}"
                    ),
                    output_summary=(
                        f"support={position.support_level:.2f}, "
                        f"reward_risk={position.reward_risk_ratio}"
                    ),
                    route_reason=(
                        "Technical support, trend, pullback, and reward-risk were calculated from price history; "
                        "LLM-backed runs include tool evidence."
                    ),
                    rule_hits=sorted(set(tool_calls)),
                    warnings=node_warnings,
                )
            )
        report_path = self._save_agent_report(
            state=state,
            stage="1_analysts",
            agent_name=AnalystType.TECHNICAL_POSITION.value,
            content=(
                self._format_technical_position_report(positions, contexts)
                + self._format_symbol_reports(reports_by_symbol)
            ),
        )
        return self._with_trace(
            state,
            events,
            technical_positions=positions,
            technical_contexts=contexts,
            analyst_reports={AnalystType.TECHNICAL_POSITION.value: report_path},
            agent_reports={AnalystType.TECHNICAL_POSITION.value: report_path},
            market_reports_by_symbol=reports_by_symbol,
        )

    def _news_sentiment(self, state: CopilotGraphState) -> CopilotGraphState:
        if AnalystType.NEWS_SENTIMENT not in state["selected_analysts"]:
            return self._with_trace(
                state,
                TraceEvent(
                    node_name=self.NODE_NEWS_SENTIMENT,
                    input_summary="analyst not selected",
                    output_summary="news_sentiment_reports=0",
                    route_reason="News sentiment analyst was not selected.",
                    warnings=["analyst_skipped"],
                ),
                news_sentiment_by_symbol={},
            )

        reports: Dict[str, NewsSentimentReport] = {}
        reports_by_symbol: Dict[str, str] = {}
        tool_calls: List[str] = []
        fallback_warnings: List[str] = []
        for symbol in state["subscription_symbols"]:
            result = self.news_sentiment_agent.analyze_symbol(
                symbol=symbol,
                trade_date=state.get("trade_date"),
                look_back_days=7,
            )
            reports[symbol] = result.report
            reports_by_symbol[symbol] = result.markdown
            tool_calls.extend(result.tool_calls)
            if getattr(result, "fallback_used", False) and getattr(result, "fallback_reason", ""):
                fallback_warnings.append(f"{symbol}: {result.fallback_reason}")

        report_path = self._save_agent_report(
            state=state,
            stage="1_analysts",
            agent_name=AnalystType.NEWS_SENTIMENT.value,
            content=(
                _format_news_sentiment_report(
                    subscription_symbols=state["subscription_symbols"],
                    reports=reports,
                )
                + self._format_symbol_reports(reports_by_symbol)
            ),
        )
        return self._with_trace(
            state,
            TraceEvent(
                node_name=self.NODE_NEWS_SENTIMENT,
                input_summary=f"symbols={state['subscription_symbols']}",
                output_summary=f"news_sentiment_reports={len(reports)}",
                route_reason="Finnhub news, social sentiment, and earnings-event evidence was reviewed.",
                rule_hits=sorted(set(tool_calls)),
                warnings=fallback_warnings,
            ),
            news_sentiment_by_symbol=reports,
            news_sentiment_reports_by_symbol=reports_by_symbol,
            analyst_reports={AnalystType.NEWS_SENTIMENT.value: report_path},
            agent_reports={AnalystType.NEWS_SENTIMENT.value: report_path},
        )

    def _fundamental_analysis(self, state: CopilotGraphState) -> CopilotGraphState:
        if AnalystType.FUNDAMENTAL_ANALYSIS not in state["selected_analysts"]:
            return self._with_trace(
                state,
                TraceEvent(
                    node_name=self.NODE_FUNDAMENTAL_ANALYSIS,
                    input_summary="analyst not selected",
                    output_summary="fundamental_reports=0",
                    route_reason="Fundamental analyst was not selected.",
                    warnings=["analyst_skipped"],
                ),
                fundamental_analysis_by_symbol={},
            )

        reports: Dict[str, FundamentalAnalysisReport] = {}
        reports_by_symbol: Dict[str, str] = {}
        tool_calls: List[str] = []
        fallback_warnings: List[str] = []
        provided_reports = {
            symbol.strip().upper(): report
            for symbol, report in _fundamental_analysis_reports_from_state(state).items()
            if symbol.strip().upper() in set(state["subscription_symbols"])
        }
        for symbol in state["subscription_symbols"]:
            if (
                symbol in provided_reports
                and getattr(self.fundamental_analyst_agent, "llm", None) is None
            ):
                reports[symbol] = provided_reports[symbol]
                reports_by_symbol[symbol] = _format_single_fundamental_analysis_report(
                    provided_reports[symbol]
                )
                continue
            result = self.fundamental_analyst_agent.analyze_symbol(
                symbol=symbol,
                trade_date=state.get("trade_date"),
                look_back_days=state.get("look_back_days", 90),
            )
            reports[symbol] = result.report
            reports_by_symbol[symbol] = result.markdown
            tool_calls.extend(result.tool_calls)
            if getattr(result, "fallback_used", False) and getattr(result, "fallback_reason", ""):
                fallback_warnings.append(f"{symbol}: {result.fallback_reason}")

        report_path = self._save_agent_report(
            state=state,
            stage="1_analysts",
            agent_name=AnalystType.FUNDAMENTAL_ANALYSIS.value,
            content=self._format_fundamental_analysis_report(
                subscription_symbols=state["subscription_symbols"],
                reports=reports,
            )
            + self._format_symbol_reports(reports_by_symbol),
        )
        return self._with_trace(
            state,
            TraceEvent(
                node_name=self.NODE_FUNDAMENTAL_ANALYSIS,
                input_summary=f"symbols={state['subscription_symbols']}",
                output_summary=f"fundamental_reports={len(reports)}",
                route_reason="Fundamental analyst used financial tools and its own fundamental RAG tool.",
                rule_hits=sorted(set(tool_calls)),
                warnings=fallback_warnings,
            ),
            fundamental_analysis_by_symbol=reports,
            fundamental_analysis_reports_by_symbol=reports_by_symbol,
            analyst_reports={AnalystType.FUNDAMENTAL_ANALYSIS.value: report_path},
            agent_reports={AnalystType.FUNDAMENTAL_ANALYSIS.value: report_path},
        )

    def _trader(self, state: CopilotGraphState) -> CopilotGraphState:
        plans: Dict[str, TradePlan] = {}
        reports_by_symbol: Dict[str, str] = {}
        reviewed_items: List[OpportunityRadarItem] = []
        events: List[TraceEvent] = []
        memories = {key: list(value) for key, value in state.get("memories", {}).items()}
        shadow_memories = {
            key: list(value) for key, value in state.get("shadow_memories", {}).items()
        }
        memory_retrievals = list(state.get("memory_retrievals", []))
        radar_by_symbol = {item.symbol: item for item in state.get("radar_items", [])}
        for symbol in state.get("subscription_symbols", []):
            item = radar_by_symbol.get(symbol)
            position = state.get("technical_positions", {}).get(symbol)
            technical_context = state.get("technical_contexts", {}).get(symbol)
            news_sentiment = state.get("news_sentiment_by_symbol", {}).get(symbol)
            fundamental_report = state.get("fundamental_analysis_by_symbol", {}).get(symbol)
            memory_session = self._memory_retrieval_session(
                state,
                symbol=symbol,
                consumer="trader",
                market_regime=_retrieval_market_regime(item, technical_context),
                agent=self.trader_agent,
            )
            tool_calls: list[str] = []
            symbol_warnings: list[str] = []

            if item is None and hasattr(self.trader_agent, "create_plan_from_evidence"):
                result = self.trader_agent.create_plan_from_evidence(
                    symbol=symbol,
                    technical_position=position,
                    technical_context=technical_context,
                    news_sentiment=news_sentiment,
                    fundamental_analysis=fundamental_report,
                    persona=state["persona_config"],
                    analyst_context=_analyst_context_for_symbol(state, symbol),
                    memory_session=memory_session,
                )
                plan = result.plan
                item = result.opportunity or _opportunity_from_plan(plan)
                reports_by_symbol[symbol] = result.report
                tool_calls = result.tool_calls
                if getattr(result, "fallback_used", False) and getattr(result, "fallback_reason", ""):
                    symbol_warnings.append(result.fallback_reason)
                if getattr(result, "memory_retrieval_error", ""):
                    symbol_warnings.append(result.memory_retrieval_error)
            elif item is not None:
                if position is None:
                    position = _technical_position_from_radar(item)
                if hasattr(self.trader_agent, "create_plan_with_report"):
                    result = self.trader_agent.create_plan_with_report(
                        opportunity=item,
                        technical_position=position,
                        persona=state["persona_config"],
                        analyst_context=_analyst_context_for_symbol(state, symbol),
                        technical_context=technical_context,
                        news_sentiment=news_sentiment,
                        fundamental_analysis=fundamental_report,
                        memory_session=memory_session,
                    )
                    plan = result.plan
                    item = result.opportunity or item
                    reports_by_symbol[symbol] = result.report
                    tool_calls = result.tool_calls
                    if getattr(result, "fallback_used", False) and getattr(result, "fallback_reason", ""):
                        symbol_warnings.append(result.fallback_reason)
                    if getattr(result, "memory_retrieval_error", ""):
                        symbol_warnings.append(result.memory_retrieval_error)
                else:
                    plan = self.trader_agent.create_plan(
                        opportunity=item,
                        technical_position=position,
                        persona=state["persona_config"],
                    )
            else:
                continue

            plans[symbol] = plan
            reviewed_items.append(item)
            if memory_session is not None:
                memories[symbol] = _merge_memories(
                    memories.get(symbol, []), memory_session.approved_memories
                )
                shadow_memories[symbol] = _merge_memories(
                    shadow_memories.get(symbol, []), memory_session.shadow_memories
                )
                memory_retrievals.extend(memory_session.records)
            rule_hits = ["market_state_first", "trader_internal_opportunity_review"]
            rule_hits.extend(tool_calls)
            if item.status == SubscriptionStatus.ACTIONABLE:
                rule_hits.append("actionable_to_buy_plan")
            else:
                rule_hits.append("non_actionable_to_alert_plan")
            events.append(
                TraceEvent(
                    node_name=self.NODE_TRADER,
                    symbol=symbol,
                    input_summary=f"status={zh_label(item.status)}",
                    output_summary=f"direction={zh_label(plan.direction)}",
                    route_reason=(
                        "Trader merged opportunity review, evaluates market regime first, "
                        "then combines technical context, news sentiment, and fundamentals."
                    ),
                    rule_hits=rule_hits,
                    warnings=symbol_warnings,
                )
            )
        report_path = self._save_agent_report(
            state=state,
            stage="3_trader",
            agent_name="trader",
            content=self._format_symbol_reports(reports_by_symbol)
            or "# Trader Report\n\nNo trade plans were generated.\n",
        )
        return self._with_trace(
            state,
            events,
            radar_items=reviewed_items,
            trade_plans=plans,
            memories=memories,
            shadow_memories=shadow_memories,
            memory_retrievals=memory_retrievals,
            agent_reports={"trader": report_path},
        )

    def _memory_retrieval_session(
        self,
        state: CopilotGraphState,
        *,
        symbol: str,
        consumer: str,
        market_regime: str,
        agent,
    ) -> MemoryRetrievalSession | None:
        if self.memory_agent is None or getattr(agent, "llm", None) is None:
            return None
        persona = state.get("persona_config") or self.default_persona_config
        return MemoryRetrievalSession(
            self.memory_agent.store,
            symbol=symbol,
            consumer=consumer,
            run_id=str(state.get("run_id") or ""),
            tags=[
                "trading",
                consumer,
                persona.trading_style,
                persona.risk_profile,
            ],
            market_regime=market_regime,
            timeframe=persona.trading_style,
        )

    def _risk_check(self, state: CopilotGraphState) -> CopilotGraphState:
        events: List[TraceEvent] = []
        portfolio_error = next(
            (
                str(item)
                for item in (state.get("errors") or [])
                if str(item).startswith("portfolio_fetch_failed")
            ),
            None,
        )
        if portfolio_error is not None:
            return self._degraded_risk_check(state, portfolio_error)
        subscriptions = _subscription_book_from_symbols(state["subscription_symbols"])
        trade_plans = state.get("trade_plans", {})
        target_weights = self.portfolio_manager.build_target_weights(
            trade_plans=trade_plans,
            portfolio=state["portfolio"],
        )
        result = self.risk_agent.review_book(
            persona=state["persona_config"],
            subscriptions=subscriptions,
            portfolio=state["portfolio"],
            trade_plans=trade_plans,
            target_weights=target_weights,
            price_history_by_symbol=state.get("price_history_by_symbol", {}),
        )
        assessments = result.assessments
        risk_challenges = result.risk_challenges
        for symbol, plan in trade_plans.items():
            assessment = assessments[symbol]
            challenge = risk_challenges.get(symbol, "")
            rule_hits = [
                f"target_weight={assessment.target_weight:.2%}",
                f"final_weight={assessment.final_weight:.2%}",
                f"delta_weight={assessment.delta_weight:.2%}",
            ]
            if assessment.clamped:
                rule_hits.extend(event.reason for event in assessment.clamps)
            warnings = list(assessment.warnings)
            events.append(
                TraceEvent(
                    node_name=self.NODE_RISK_CHECK,
                    symbol=symbol,
                    input_summary=f"direction={zh_label(plan.direction)}",
                    output_summary=(
                        f"target={assessment.target_weight:.2%}, "
                        f"final={assessment.final_weight:.2%}, "
                        f"delta={assessment.delta_weight:.2%}"
                    ),
                    route_reason=(
                        "Risk Manager clamps target weights with v2-style limits; "
                        "Portfolio Manager makes the final action decision."
                    ),
                    rule_hits=rule_hits,
                    warnings=warnings + ([challenge] if challenge else []),
                )
            )
        explanations = dict(state.get("explanations", {}))
        explanations["risk_challenges"] = risk_challenges
        report_path = self._save_agent_report(
            state=state,
            stage="4_risk_check",
            agent_name="risk_check",
            content=result.report
            or "# 风险管理报告\n\nNo risk assessment was generated.\n",
        )
        return self._with_trace(
            state,
            events,
            risk_assessments=assessments,
            explanations={
                **explanations,
                "target_weights": result.target_weights,
                "final_weights": result.final_weights,
            },
            agent_reports={"risk_check": report_path},
        )

    def _degraded_risk_check(
        self,
        state: CopilotGraphState,
        portfolio_error: str,
    ) -> CopilotGraphState:
        """Portfolio is unknown: keep every planned symbol flat and mark degraded."""
        events: List[TraceEvent] = []
        assessments: Dict[str, RiskAssessment] = {}
        risk_challenges: Dict[str, str] = {}
        for symbol, plan in state.get("trade_plans", {}).items():
            assessments[symbol] = RiskAssessment(
                symbol=symbol,
                approved=False,
                current_position_weight=0.0,
                target_weight=0.0,
                final_weight=0.0,
                portfolio_value=0.0,
                warnings=[portfolio_error],
                reasoning={
                    "eligibility": "Portfolio snapshot unavailable; every new action is held flat.",
                },
            )
            risk_challenges[symbol] = (
                f"{symbol}: 组合快照获取失败，本次不产生新动作。"
            )
            events.append(
                TraceEvent(
                    node_name=self.NODE_RISK_CHECK,
                    symbol=symbol,
                    input_summary=f"direction={zh_label(plan.direction)}",
                    output_summary="target=0.00%, final=0.00%, delta=0.00%",
                    route_reason="Portfolio fetch failed; degraded run keeps every symbol flat.",
                    rule_hits=["portfolio_unavailable_hold_flat"],
                    warnings=[portfolio_error],
                )
            )
        report_content = (
            "# 风险管理报告\n\n"
            "- 说明：组合快照获取失败，本次运行为降级 Hold-only，所有标的不产生新动作。\n\n"
            f"- 组合错误：{portfolio_error}\n"
        )
        report_path = self._save_agent_report(
            state=state,
            stage="4_risk_check",
            agent_name="risk_check",
            content=report_content,
        )
        explanations = dict(state.get("explanations", {}))
        explanations["risk_challenges"] = risk_challenges
        return self._with_trace(
            state,
            events,
            risk_assessments=assessments,
            explanations={
                **explanations,
                "target_weights": {},
                "final_weights": {},
            },
            agent_reports={"risk_check": report_path},
        )

    def _portfolio_manager(self, state: CopilotGraphState) -> CopilotGraphState:
        decisions: Dict[str, ExecutionDecision] = {}
        reports_by_symbol: Dict[str, str] = {}
        events: List[TraceEvent] = []
        memories = {key: list(value) for key, value in state.get("memories", {}).items()}
        shadow_memories = {
            key: list(value) for key, value in state.get("shadow_memories", {}).items()
        }
        memory_retrievals = list(state.get("memory_retrievals", []))
        for symbol, plan in state.get("trade_plans", {}).items():
            assessment = state.get("risk_assessments", {}).get(symbol)
            if assessment is None:
                continue
            memory_session = self._memory_retrieval_session(
                state,
                symbol=symbol,
                consumer="portfolio_manager",
                market_regime=plan.market_regime.value,
                agent=self.portfolio_manager,
            )
            tool_calls: list[str] = []
            symbol_warnings: list[str] = []
            if hasattr(self.portfolio_manager, "decide_with_report"):
                result = self.portfolio_manager.decide_with_report(
                    plan=plan,
                    risk_assessment=assessment,
                    portfolio=state["portfolio"],
                    mode=state.get("execution_mode", ExecutionMode.SIMULATION),
                    user_confirmed=state.get("user_confirmed", False),
                    analyst_context=_analyst_context_for_symbol(state, symbol),
                    run_id=state.get("run_id"),
                    memory_session=memory_session,
                )
                decision = result.decision
                reports_by_symbol[symbol] = result.report
                tool_calls = result.tool_calls or []
                if getattr(result, "fallback_used", False) and getattr(result, "fallback_reason", ""):
                    symbol_warnings.append(result.fallback_reason)
            else:
                decision = self.portfolio_manager.decide(
                    plan=plan,
                    risk_assessment=assessment,
                    portfolio=state["portfolio"],
                    mode=state.get("execution_mode", ExecutionMode.SIMULATION),
                    user_confirmed=state.get("user_confirmed", False),
                    analyst_context=_analyst_context_for_symbol(state, symbol),
                    run_id=state.get("run_id"),
                )
            decisions[symbol] = decision
            if memory_session is not None:
                memories[symbol] = _merge_memories(
                    memories.get(symbol, []), memory_session.approved_memories
                )
                shadow_memories[symbol] = _merge_memories(
                    shadow_memories.get(symbol, []), memory_session.shadow_memories
                )
                memory_retrievals.extend(memory_session.records)
            events.append(
                TraceEvent(
                    node_name=self.NODE_PORTFOLIO_MANAGER,
                    symbol=symbol,
                    input_summary=(
                        f"target={decision.target_weight:.2%}, "
                        f"final={decision.final_weight:.2%}"
                    ),
                    output_summary=(
                        f"action={decision.action}, quantity={decision.quantity}, "
                        f"status={zh_label(decision.status)}"
                    ),
                    route_reason=decision.message,
                    rule_hits=[
                        decision.status.value,
                        f"delta_weight={decision.delta_weight:.2%}",
                        *tool_calls,
                    ],
                    warnings=symbol_warnings,
                )
            )
        report_path = self._save_agent_report(
            state=state,
            stage="5_portfolio_manager",
            agent_name="portfolio_manager",
            content=self._format_symbol_reports(reports_by_symbol)
            or "# Portfolio Manager Report\n\nNo portfolio decisions were generated.\n",
        )
        return self._with_trace(
            state,
            events,
            execution_decisions=decisions,
            memories=memories,
            shadow_memories=shadow_memories,
            memory_retrievals=memory_retrievals,
            agent_reports={"portfolio_manager": report_path},
        )

    def _explain_run(self, state: CopilotGraphState) -> CopilotGraphState:
        report = self.explanation_agent.explain(
            market_regime=None,
            radar_items=state.get("radar_items", []),
            technical_positions=state.get("technical_positions", {}),
            trade_plans=state.get("trade_plans", {}),
            risk_assessments=state.get("risk_assessments", {}),
            execution_decisions=state.get("execution_decisions", {}),
            risk_challenges=state.get("explanations", {}).get("risk_challenges", {}),
        )
        explanations = dict(state.get("explanations", {}))
        explanations["user_summary"] = report.summary
        explanations["persona_markdown"] = state["persona_markdown"]
        content = (
            self.explanation_agent.render_report(report)
            if hasattr(self.explanation_agent, "render_report")
            else report.summary
        )
        report_path = self._save_agent_report(
            state=state,
            stage="6_explanation",
            agent_name="run_explanation",
            content=content,
        )
        return self._with_trace(
            state,
            TraceEvent(
                node_name=self.NODE_EXPLAIN_RUN,
                input_summary=f"symbols={len(report.symbols)}",
                output_summary="report=generated",
                route_reason="User explanation is generated from the same structured state used by trace.",
                warnings=(
                    [self.explanation_agent.fallback_reason]
                    if getattr(self.explanation_agent, "fallback_reason", "")
                    else []
                ),
            ),
            explanations=explanations,
            report=report,
            agent_reports={"run_explanation": report_path},
        )

    def _persist_trace(self, state: CopilotGraphState) -> CopilotGraphState:
        event = TraceEvent(
            node_name=self.NODE_PERSIST_TRACE,
            input_summary=f"trace_events={len(state.get('trace_events', []))}",
            output_summary="trace_persisted=True",
            route_reason="Trace is retained in state; optional writer may persist it externally.",
        )
        next_trace = [*state.get("trace_events", []), event]
        errors = list(state.get("errors", []))
        agent_reports = {}
        if self.memory_agent is not None:
            agent_reports["post_trade_review_learning"] = self._save_agent_report(
                state=state,
                stage="6_memory",
                agent_name="post_trade_review_learning",
                content=_format_memory_report(
                    state.get("memories", {}),
                    shadow_memories=state.get("shadow_memories", {}),
                    retrievals=state.get("memory_retrievals", []),
                ),
            )
        if self.trace_writer is not None:
            try:
                self.trace_writer(next_trace)
            except Exception as exc:
                errors.append(f"trace_writer_failed: {exc}")
                event = event.model_copy(update={"warnings": [str(exc)]})
                next_trace[-1] = event
        return {
            "trace_events": [event],
            "errors": errors,
            "trace_persisted": not errors,
            "agent_reports": agent_reports,
        }

    def _with_trace(
        self,
        state: CopilotGraphState,
        events: TraceEvent | Iterable[TraceEvent],
        **updates,
    ) -> CopilotGraphState:
        event_list = [events] if isinstance(events, TraceEvent) else list(events)
        return {
            **updates,
            "trace_events": event_list,
        }

    def _save_agent_report(
        self,
        *,
        state: CopilotGraphState,
        stage: str,
        agent_name: str,
        content: str,
    ) -> str:
        output_dir = Path(state.get("report_output_dir") or self.report_output_dir)
        report_dir = output_dir / stage
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{agent_name}.md"
        from ai_trading_copilot.copilot.services.reporting import prepare_report

        if getattr(self._parallel_worker_context, "active", False):
            temporary_dir = output_dir / ".branches"
            temporary_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=temporary_dir,
                                             suffix=".md", delete=False) as handle:
                handle.write(prepare_report(content))
            self._parallel_worker_context.reports.append((agent_name, Path(handle.name), path))
        else:
            path.write_text(prepare_report(content), encoding="utf-8")
            if self.run_tracker is not None:
                self.run_tracker.record_report(agent_name, path)
        return str(path)

    def _format_opportunity_radar_report(
        self,
        items: List[OpportunityRadarItem],
    ) -> str:
        lines = [
            "# 机会雷达报告",
            "",
            f"生成时间： {datetime.now().isoformat(timespec='seconds')}",
            "",
            "| 标的 | 趋势 | 机会状态 | 当前价格 | 支撑位 | 收益风险比 | 依据 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in items:
            lines.append(
                "| "
                + " | ".join(
                    [
                        item.symbol,
                        item.trend_label or "",
                        item.status_label,
                        _fmt_number(item.current_price),
                        _fmt_number(item.support_level),
                        _fmt_number(item.reward_risk_ratio),
                        item.reason,
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def _format_technical_position_report(
        self,
        positions: Dict[str, TechnicalPosition],
        contexts: Dict[str, TechnicalContext] | None = None,
    ) -> str:
        contexts = contexts or {}
        lines = [
            "# 技术分析报告",
            "",
            f"生成时间： {datetime.now().isoformat(timespec='seconds')}",
            "",
            "| 标的 | 当前价格 | 支撑位 | 近期高点 | 20日均线 | 50日均线 | 距支撑幅度 | 高点回撤 | 收益风险比 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for symbol in sorted(positions):
            position = positions[symbol]
            context = contexts.get(symbol)
            if context is not None and not context.stock.data_available:
                lines.append(f"| {symbol} | 【待补充】缺少行情 | - | - | - | - | - | - | - |")
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        symbol,
                        _fmt_number(position.current_price),
                        _fmt_number(position.support_level),
                        _fmt_number(position.recent_high),
                        _fmt_number(position.moving_average_20),
                        _fmt_number(position.moving_average_50),
                        f"{position.distance_to_support_pct:.1%}",
                        f"{position.pullback_from_high_pct:.1%}",
                        _fmt_number(position.reward_risk_ratio),
                    ]
                )
                + " |"
            )
        if not positions:
            lines.append("| 无 | 【待补充】缺少有效历史行情 | - | - | - | - | - | - | - |")
        context_sections = [
            (symbol, context)
            for symbol, context in sorted(contexts.items())
            if context.downstream_summary or context.decision_basis or context.uncertainties
        ]
        if context_sections:
            lines.extend(["", "## 综合判断依据", ""])
            for symbol, context in context_sections:
                lines.extend(_downstream_context_lines(symbol, context))
        return "\n".join(lines) + "\n"

    def _format_fundamental_analysis_report(
        self,
        *,
        subscription_symbols: List[str],
        reports: Dict[str, FundamentalAnalysisReport],
    ) -> str:
        lines = [
            "# 基本面分析报告",
            "",
            f"生成时间： {datetime.now().isoformat(timespec='seconds')}",
            "",
            "| 标的 | 投资逻辑完整 | 重大风险 | 风险事项 | 结论 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for symbol in subscription_symbols:
            report = reports.get(symbol)
            if report is None:
                lines.append(f"| {symbol} | 【待补充】未提供 | 【待补充】未提供 | - | 【待补充】未取得基本面资料，无法判断。 |")
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        symbol,
                        zh_bool(report.thesis_intact),
                        zh_bool(report.material_risk),
                        ", ".join(report.risk_flags) or "-",
                        report.summary or "-",
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def _format_symbol_reports(self, reports_by_symbol: Dict[str, str]) -> str:
        if not reports_by_symbol:
            return ""
        lines = ["", "## 逐标的分析", ""]
        for symbol in sorted(reports_by_symbol):
            lines.extend([f"### {symbol}", "", reports_by_symbol[symbol].strip(), ""])
        return "\n".join(lines)


def _normalize_analysts(values: Iterable[AnalystType | str]) -> List[AnalystType]:
    normalized: List[AnalystType] = []
    for value in values:
        analyst = value if isinstance(value, AnalystType) else AnalystType(value)
        if analyst not in normalized:
            normalized.append(analyst)
    return normalized


def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for symbol in symbols:
        value = symbol.strip().upper()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _unique_list(values: Iterable[str]) -> List[str]:
    unique: List[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_price_history(
    price_history_by_symbol: Dict[str, List[PriceBar]]
) -> Dict[str, List[PriceBar]]:
    return {
        symbol.strip().upper(): bars
        for symbol, bars in price_history_by_symbol.items()
        if symbol.strip()
    }


def _fundamental_analysis_reports_from_state(
    state: CopilotGraphState,
) -> Dict[str, FundamentalAnalysisReport]:
    raw_reports = state.get("fundamental_analysis_by_symbol") or {}
    return {
        symbol.strip().upper(): report
        for symbol, report in raw_reports.items()
        if symbol.strip()
    }


def _fmt_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _format_single_fundamental_analysis_report(report: FundamentalAnalysisReport) -> str:
    return (
        f"# 基本面分析： {report.symbol}\n\n"
        f"- 投资逻辑是否完整： {zh_bool(report.thesis_intact)}\n"
        f"- 是否存在重大风险： {zh_bool(report.material_risk)}\n"
        f"- 风险事项： {zh_join(report.risk_flags)}\n"
        f"- 关键依据： {zh_join(report.decision_basis)}\n"
        f"- 【待补充】不确定性： {zh_join(report.uncertainties)}\n"
        f"- 综合结论： {report.downstream_summary or '-'}\n"
        f"- 结论： {report.summary or '-'}\n"
    )


def _format_news_sentiment_report(
    *,
    subscription_symbols: List[str],
    reports: Dict[str, NewsSentimentReport],
) -> str:
    lines = [
        "# 新闻与情绪分析报告",
        "",
        f"生成时间： {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| 标的 | 综合情绪 | 公司新闻 | 社交情绪 | 财报事件 | 重大风险 | 关键事件 | 提醒 | 结论 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for symbol in subscription_symbols:
        report = reports.get(symbol)
        if report is None:
            lines.append(f"| {symbol} | - | - | - | - | - | - | - | 【待补充】未取得分析报告。 |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    symbol,
                    _fmt_number(report.sentiment_score),
                    _fmt_number(report.company_news_score),
                    _fmt_number(report.social_sentiment_score),
                    _fmt_number(report.earnings_event_score),
                    zh_bool(report.material_risk),
                    zh_join(report.key_events),
                    zh_join(report.alerts),
                    report.summary or "-",
                ]
            )
            + " |"
        )
    references = [
        (symbol, reference)
        for symbol in subscription_symbols
        for reference in (reports.get(symbol).news_references if reports.get(symbol) else [])
    ]
    if references:
        lines.extend(
            [
                "",
                "## 新闻来源",
                "",
                "| 标的 | 发布时间 | 来源 | 标题 | 链接 | 事件类型 | 相关性 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for symbol, reference in references:
            lines.append(
                "| "
                + " | ".join(
                    [
                        symbol,
                        str(reference.get("published_at") or "unknown"),
                        str(reference.get("source") or "unknown"),
                        str(reference.get("title") or "unknown").replace("|", "/"),
                        str(reference.get("url") or "unknown"),
                        str(reference.get("event_type") or "unknown"),
                        str(reference.get("relevance") or "unknown").replace("|", "/"),
                    ]
                )
                + " |"
            )
    context_sections = [
        (symbol, report)
        for symbol, report in reports.items()
        if report.downstream_summary or report.decision_basis or report.uncertainties
    ]
    if context_sections:
        lines.extend(["", "## 综合判断依据", ""])
        for symbol, report in context_sections:
            lines.extend(_downstream_context_lines(symbol, report))
    return "\n".join(lines) + "\n"


def _analyst_context_for_symbol(state: CopilotGraphState, symbol: str) -> str:
    sections = []
    for title, key in [
        ("Opportunity Radar", "opportunity_reports_by_symbol"),
        ("Technical Position", "market_reports_by_symbol"),
    ]:
        report = state.get(key, {}).get(symbol)
        if report:
            sections.append(f"## {title}\n\n{report}")
    for title, key in [
        ("News Sentiment", "news_sentiment_reports_by_symbol"),
        ("Fundamental Analysis", "fundamental_analysis_reports_by_symbol"),
    ]:
        report = state.get(key, {}).get(symbol)
        if report:
            sections.append(f"## {title}\n\n{report}")
    sections.extend(_structured_downstream_context_sections(state, symbol))
    return "\n\n".join(sections)


def _structured_downstream_context_sections(state: CopilotGraphState, symbol: str) -> List[str]:
    normalized = symbol.strip().upper()
    sections: List[str] = []
    for title, key in [
        ("Technical Position Context", "technical_contexts"),
        ("News Sentiment Context", "news_sentiment_by_symbol"),
        ("Fundamental Analysis Context", "fundamental_analysis_by_symbol"),
    ]:
        item = (state.get(key) or {}).get(normalized)
        if item is None:
            continue
        lines = _downstream_context_lines(normalized, item)
        if lines:
            sections.append(f"## {title}\n\n" + "\n".join(lines))
    return sections


def _downstream_context_lines(symbol: str, item) -> List[str]:
    downstream_summary = str(getattr(item, "downstream_summary", "") or "").strip()
    decision_basis = [str(value) for value in (getattr(item, "decision_basis", []) or [])]
    uncertainties = [str(value) for value in (getattr(item, "uncertainties", []) or [])]
    if not downstream_summary and not decision_basis and not uncertainties:
        return []
    lines = [f"### {symbol}"]
    if downstream_summary:
        lines.extend(["", f"综合结论： {downstream_summary}"])
    if decision_basis:
        lines.extend(["", "关键依据："])
        lines.extend(f"- {value}" for value in decision_basis if value.strip())
    if uncertainties:
        lines.extend(["", "【待补充】不确定性："])
        lines.extend(f"- {value}" for value in uncertainties if value.strip())
    return lines + [""]


def _format_memory_report(
    memories: Dict[str, List[DistilledMemory]],
    *,
    shadow_memories: Dict[str, List[DistilledMemory]] | None = None,
    retrievals: Iterable[MemoryRetrievalRecord] = (),
) -> str:
    lines = ["# 历史经验检索报告", ""]
    if not memories:
        lines.append("未检索到可用的相关历史经验，本次判断不引用历史经验。")
    else:
        for symbol, items in sorted(memories.items()):
            lines.extend([f"## {symbol}", ""])
            if not items:
                lines.append("- 未检索到可用的相关历史经验，本次判断不引用历史经验。")
            for item in items:
                suffix = _memory_source_suffix(item)
                lines.append(
                    f"- `{item.memory_id}@{item.version}` {item.memory_type.value}: "
                    f"{item.lesson}{suffix}"
                )
            lines.append("")
    if shadow_memories:
        lines.extend(
            [
                "# 历史经验对照记录",
                "",
                "以下经验仅用于对照评估，未参与本次决策。",
                "",
            ]
        )
        for symbol, items in sorted(shadow_memories.items()):
            for item in items:
                lines.append(f"- {symbol}: `{item.memory_id}@{item.version}`")
    retrieval_list = list(retrievals)
    if retrieval_list:
        lines.extend(["", "# 检索核对", ""])
        for record in retrieval_list:
            request = record.request
            lines.extend(
                [
                    f"## {request.consumer} / {request.symbol} / {request.phase}",
                    "",
                    f"- 检索条件： {request.query[:800]}",
                    f"- 已通过筛选： {', '.join(record.approved_ids) or '-'}",
                    f"- 对照记录： {', '.join(record.shadow_ids) or '-'}",
                    f"- 实际引用： {', '.join(record.cited_ids) or '-'}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def _merge_memories(
    existing: Iterable[DistilledMemory],
    additions: Iterable[DistilledMemory],
) -> List[DistilledMemory]:
    merged: dict[str, DistilledMemory] = {}
    for memory in [*existing, *additions]:
        merged[f"{memory.memory_id}@{memory.version}"] = memory
    return list(merged.values())


def _retrieval_market_regime(
    item: OpportunityRadarItem | None,
    context: TechnicalContext | None,
) -> str:
    trend = (
        context.stock.trend_state
        if context is not None and context.stock.trend_state is not None
        else item.trend_state if item is not None else None
    )
    if trend == SymbolTrendState.DOWNTREND:
        return MarketRegime.DOWNTREND.value
    if trend in {SymbolTrendState.UPTREND, SymbolTrendState.UPTREND_PULLBACK}:
        return MarketRegime.UPTREND.value
    return MarketRegime.UNCLEAR.value


def _memory_source_suffix(item: DistilledMemory) -> str:
    details = []
    if item.confidence is not None:
        details.append(f"confidence={item.confidence:.2f}")
    if item.source_run_id:
        details.append(f"run={item.source_run_id}")
    if item.source_path:
        details.append(f"source={item.source_path}")
    return "" if not details else " (" + "; ".join(details) + ")"


def _subscription_book_from_symbols(symbols: List[str]) -> SubscriptionBook:
    from ai_trading_copilot.copilot.domain.enums import MarketType

    return SubscriptionBook(
        items=[
            Subscription(symbol=symbol, market_type=MarketType.US_STOCK)
            for symbol in symbols
        ]
    )


def _span_graph_node_name(span, recorder: TraceRecorder) -> str | None:
    parent_id = span.parent_span_id
    while parent_id:
        parent = recorder.get_span(parent_id)
        if parent is None:
            return None
        node_name = parent.attributes.get("graph.node.name")
        if node_name:
            return str(node_name)
        parent_id = parent.parent_span_id
    return None


def _activity_from_span(event: str, span) -> dict[str, Any]:
    attrs = span.attributes or {}
    status = span.status if event == "end" else "running"
    duration_ms = span.duration_ms
    if duration_ms is None and span._start_perf:  # noqa: SLF001 - current duration for live UI.
        duration_ms = round((time.perf_counter() - span._start_perf) * 1000, 3)
    activity = {
        "span_id": span.span_id,
        "phase": span.name,
        "status": status,
        "title": _activity_title(span.name, status, attrs),
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "duration_ms": duration_ms,
        "summary": _activity_summary(attrs),
    }
    details = _activity_details(attrs)
    if details:
        activity["details"] = details
    if span.error:
        activity["error"] = span.error
    if span.name == "llm.invoke":
        activity["llm_round"] = attrs.get("llm.round")
    if span.name == "tool.call":
        activity["tool_name"] = attrs.get("tool.name")
        activity["tool_phase"] = attrs.get("tool.phase")
        activity["cache_hit"] = attrs.get("tool.cache_hit")
        activity["args"] = attrs.get("tool.args", {})
    return activity


def _activity_title(phase: str, status: str, attrs: dict[str, Any]) -> str:
    if phase == "llm.invoke":
        round_label = attrs.get("llm.round") or "-"
        if status == "running":
            return f"LLM 第 {round_label} 轮推理中"
        return f"LLM 第 {round_label} 轮{'失败' if status == 'error' else '完成'}"
    if phase == "tool.call":
        tool_name = attrs.get("tool.name") or "unknown_tool"
        if status == "running":
            return f"正在调用 {tool_name}"
        return f"{'调用失败' if status == 'error' else '完成调用'} {tool_name}"
    return phase


def _activity_summary(attrs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key, value in attrs.items():
        text_key = str(key)
        for bucket in ("prompt", "llm.response", "tool.output"):
            prefix = f"{bucket}."
            if not text_key.startswith(prefix):
                continue
            metric = text_key.removeprefix(prefix)
            if metric in {"chars", "bytes", "sha256"}:
                summary.setdefault(bucket, {})[metric] = value
    return summary


def _activity_details(attrs: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    request_messages = attrs.get("llm.request.messages")
    if request_messages:
        details["llm.request"] = {"messages": request_messages}
    if (
        "llm.response.content" in attrs
        or "llm.response.reasoning_content" in attrs
        or "llm.response.tool_calls" in attrs
    ):
        details["llm.response"] = {
            "content": attrs.get("llm.response.content", ""),
            "reasoning_content": attrs.get("llm.response.reasoning_content", ""),
            "tool_calls": attrs.get("llm.response.tool_calls", []),
        }
    return details


def _technical_position_from_radar(item: OpportunityRadarItem) -> TechnicalPosition:
    current = item.current_price or item.support_level or 1.0
    support = item.support_level or current
    downside = max(current - support, 0.01)
    recent_high = current + downside * (item.reward_risk_ratio or 1.0)
    trend_state = item.trend_state or SymbolTrendState.UNKNOWN
    return TechnicalPosition(
        symbol=item.symbol,
        current_price=current,
        support_level=support,
        recent_high=recent_high,
        moving_average_20=current if trend_state != SymbolTrendState.DOWNTREND else None,
        moving_average_50=support if trend_state != SymbolTrendState.UNKNOWN else None,
        distance_to_support_pct=max((current - support) / support, 0),
        pullback_from_high_pct=max((recent_high - current) / recent_high, 0),
        reward_risk_ratio=item.reward_risk_ratio,
        uptrend=trend_state in {
            SymbolTrendState.UPTREND,
            SymbolTrendState.UPTREND_PULLBACK,
        },
    )


def _opportunity_from_plan(plan: TradePlan) -> OpportunityRadarItem:
    current_price = plan.support_level or plan.stop_loss or 0.0
    target = plan.targets[0] if plan.targets else None
    return OpportunityRadarItem(
        symbol=plan.symbol,
        status=plan.subscription_status,
        current_price=current_price,
        support_level=plan.support_level,
        recent_high=target,
        reward_risk_ratio=plan.reward_risk_ratio,
        trend_state=SymbolTrendState.UNKNOWN,
        suggested_action=plan.entry_logic,
        reason=plan.entry_logic,
    )

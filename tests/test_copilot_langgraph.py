from pathlib import Path

from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.domain import (
    AnalystType,
    DistilledMemory,
    ExecutionMode,
    FundamentalNewsReport,
    MemoryType,
    PortfolioSnapshot,
    PriceBar,
    RagDocument,
    SubscriptionStatus,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.graph.copilot_langgraph import (
    _analyst_context_for_symbol,
    _fundamental_rag_context_for_symbol,
)
from ai_trading_copilot.copilot.agents import (
    FundamentalNewsAgent,
    OpportunityRadarAgent,
    PostTradeReviewLearningAgent,
    TechnicalPositionAgent,
)
from ai_trading_copilot.copilot.services import DistilledMemoryStore


def _bars_from_closes(closes):
    return [
        PriceBar(
            date=f"2026-05-{(idx % 28) + 1:02d}",
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1000,
        )
        for idx, close in enumerate(closes)
    ]


def _actionable_bars():
    bars = _bars_from_closes([80 + i * 0.5 for i in range(40)])
    bars.extend(
        PriceBar(
            date=f"2026-06-{idx + 1:02d}",
            open=101,
            high=110 if idx == 0 else 103,
            low=100.5,
            close=101,
            volume=1000,
        )
        for idx in range(19)
    )
    bars.append(
        PriceBar(
            date="2026-06-20",
            open=101.5,
            high=103,
            low=100.5,
            close=102,
            volume=1000,
        )
    )
    return bars


class RoutingLLM:
    def __init__(self):
        self.prompts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        text = _prompt_text(prompt)
        self.prompts.append(text)
        if "Opportunity Radar analyst" in text:
            return AIMessage(
                content=(
                    "# Opportunity report\n\n"
                    '{"status": "actionable", "trend_state": "uptrend_pullback", '
                    '"reason": "Tool reports show a pullback near support.", '
                    '"current_price": 102, "support_level": 100, '
                    '"reward_risk_ratio": 3, "trend_reason": "Uptrend pullback."}'
                )
            )
        if "Technical Position analyst" in text:
            return AIMessage(
                content=(
                    "# Technical report\n\n"
                    '{"current_price": 102, "support_level": 100, '
                    '"recent_high": 110, "moving_average_20": 101, '
                    '"moving_average_50": 98, "distance_to_support_pct": 0.02, '
                    '"pullback_from_high_pct": 0.073, "reward_risk_ratio": 3, '
                    '"uptrend": true}'
                )
            )
        if "Fundamental News Review analyst" in text:
            return AIMessage(
                content=(
                    "# Fundamental report\n\n"
                    '{"thesis_intact": true, "material_risk": false, '
                    '"risk_flags": [], "summary": "No material tool-sourced risk."}'
                )
            )
        if "Opportunity Review Manager" in text:
            return AIMessage(
                content=(
                    "# Opportunity review\n\n"
                    '{"status": "actionable", "reason": "Analyst reports agree.", '
                    '"final_conclusion": "Actionable", '
                    '"review_reasons": ["radar", "technical", "news"], '
                    '"risk_points": [], "suggested_action": "Prepare a buy plan."}'
                )
            )
        if "You are the Trader" in text:
            return AIMessage(
                content=(
                    "# Trader report\n\n"
                    '{"direction": "buy", "entry_logic": "Enter near support.", '
                    '"market_regime": "uptrend", "support_level": 100, '
                    '"stop_loss": 97, "targets": [110], "reward_risk_ratio": 3, '
                    '"position_weight": 0.2, "holding_period": "1-6 weeks", '
                    '"invalidation_conditions": ["Close below support"], '
                    '"persona_fit_reason": "Fits pullback persona.", '
                    '"uses_leverage": false, "uses_options": false, '
                    '"is_chasing": false, "breakout_confirmed": false, '
                    '"pullback_confirmed": true}'
                )
            )
        if "single Risk Check analyst" in text:
            return AIMessage(content="# Risk report\n\nHard rules pass; monitor support.")
        if "Risk Check" in text:
            return AIMessage(content="Watch the stop-loss discipline.")
        if "Execution Alert Manager" in text:
            return AIMessage(content="# Execution report\n\nSimulation candidate is ready.")
        if "Summarize this AI trading copilot scan" in text:
            return AIMessage(content="This scan found 1 actionable opportunity.")
        return AIMessage(content="# Report\n\nNo extra changes.")


class RecordingPortfolioGetter:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot or PortfolioSnapshot(position_weights={"AAPL": 0.10})
        self.modes = []

    def __call__(self, mode):
        self.modes.append(mode)
        return self.snapshot


class FakeTool:
    def __init__(self, name, output):
        self.name = name
        self.output = output

    def invoke(self, args):
        return self.output


def _fake_market_tools():
    return [
        FakeTool("get_stock_info", "AAPL stock info: liquid US stock."),
        FakeTool(
            "get_stock_data",
            "date,open,high,low,close,volume\n2026-05-08,101,103,100,102,1000",
        ),
        FakeTool("get_indicators", "close_50_sma=98; trend=uptrend"),
    ]


def _fake_fundamental_tools():
    return [
        FakeTool("get_news", "No material company news risk."),
        FakeTool("get_global_news", "Macro backdrop is stable."),
        FakeTool("get_fundamentals", "Revenue and margin quality are intact."),
    ]


def _prompt_text(prompt):
    if isinstance(prompt, list):
        return "\n".join(str(getattr(item, "content", item)) for item in prompt)
    return str(prompt)


def test_langgraph_routes_selected_analysts_as_fan_out_nodes():
    graph = CopilotLangGraph(enable_default_llm=False)

    routes = graph._route_analyst_nodes(
        {
            "selected_analysts": [
                AnalystType.OPPORTUNITY_RADAR,
                AnalystType.FUNDAMENTAL_NEWS,
            ]
        }
    )

    assert graph.langgraph_available is True
    assert routes == [
        graph.NODE_OPPORTUNITY_RADAR,
        graph.NODE_FUNDAMENTAL_NEWS_REVIEW,
    ]


def test_langgraph_accepts_subscription_symbols_as_only_required_input(tmp_path):
    portfolio_getter = RecordingPortfolioGetter()
    state = CopilotLangGraph(
        llm=None,
        portfolio_getter=portfolio_getter,
    ).run(
        subscription_symbols=["AAPL"],
        report_output_dir=tmp_path,
    )

    assert state["radar_items"][0].symbol == "AAPL"
    assert state["portfolio"].position_weights == {"AAPL": 0.10}
    assert portfolio_getter.modes == [state["portfolio_mode"]]
    assert Path(state["agent_reports"]["futu_portfolio"]).parent == tmp_path / "0_portfolio"


def test_langgraph_can_use_live_portfolio_mode(tmp_path):
    portfolio_getter = RecordingPortfolioGetter()
    state = CopilotLangGraph(
        llm=None,
        portfolio_getter=portfolio_getter,
    ).run(
        subscription_symbols=["AAPL"],
        portfolio_mode=ExecutionMode.LIVE,
        report_output_dir=tmp_path,
    )

    assert state["portfolio_mode"] == ExecutionMode.LIVE
    assert portfolio_getter.modes == [ExecutionMode.LIVE]


def test_langgraph_retrieves_memory_agent_context(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    store.append(
        DistilledMemory(
            memory_type=MemoryType.STRATEGY_PERFORMANCE,
            lesson="AAPL pullbacks need support confirmation.",
            symbols=["AAPL"],
            tags=["pullback"],
        )
    )

    state = CopilotLangGraph(
        enable_default_llm=False,
        portfolio_getter=RecordingPortfolioGetter(PortfolioSnapshot()),
        memory_agent=PostTradeReviewLearningAgent(store),
    ).run(
        subscription_symbols=["AAPL"],
        portfolio=PortfolioSnapshot(),
        report_output_dir=tmp_path,
    )

    assert state["memories"]["AAPL"][0].lesson == "AAPL pullbacks need support confirmation."
    assert "post_trade_review_learning" in state["agent_reports"]


def test_langgraph_runs_only_selected_analysts_and_reviews_their_outputs(tmp_path):
    state = CopilotLangGraph(enable_default_llm=False).run(
        subscription_symbols=["AAPL"],
        price_history_by_symbol={"AAPL": _actionable_bars()},
        fundamental_news_by_symbol={
            "AAPL": FundamentalNewsReport(
                symbol="AAPL",
                material_risk=True,
                risk_flags=["earnings_gap_risk"],
                summary="Unresolved earnings risk.",
            )
        },
        portfolio=PortfolioSnapshot(),
        selected_analysts=[
            AnalystType.OPPORTUNITY_RADAR,
            AnalystType.FUNDAMENTAL_NEWS,
        ],
        report_output_dir=tmp_path,
    )

    node_names = [event.node_name for event in state["trace_events"]]

    assert CopilotLangGraph.NODE_OPPORTUNITY_RADAR in node_names
    assert CopilotLangGraph.NODE_FUNDAMENTAL_NEWS_REVIEW in node_names
    assert CopilotLangGraph.NODE_TECHNICAL_POSITION not in node_names
    assert state["radar_items"][0].status == SubscriptionStatus.RISK_ELEVATED
    assert state["radar_items"][0].final_conclusion == "风险升高"
    assert state["radar_items"][0].risk_points == ["earnings_gap_risk"]
    assert set(state["analyst_reports"]) == {
        "opportunity_radar",
        "fundamental_news",
    }
    for report_path in state["analyst_reports"].values():
        path = Path(report_path)
        assert path.exists()
        assert path.parent == tmp_path / "1_analysts"


def test_analyst_context_includes_retrieved_trading_memories():
    context = _analyst_context_for_symbol(
        {
            "opportunity_reports_by_symbol": {"AAPL": "# Radar\n\nNear support."},
            "memories": {
                "AAPL": [
                    DistilledMemory(
                        memory_type=MemoryType.STRATEGY_PERFORMANCE,
                        lesson="Wait for support confirmation before adding size.",
                        symbols=["AAPL"],
                        tags=["pullback", "risk"],
                        source_run_id="run_AAPL_previous",
                        confidence=0.9,
                    )
                ]
            },
        },
        "AAPL",
    )

    assert "已检索交易记忆" in context
    assert "仅作为历史背景" in context
    assert "Wait for support confirmation" in context
    assert "run=run_AAPL_previous" in context


def test_analyst_context_excludes_fundamental_rag_documents():
    context = _analyst_context_for_symbol(
        {
            "fundamental_rag_contexts": {
                "AAPL": [
                    RagDocument(
                        id="rag-1",
                        title="AAPL earnings note",
                        text="AAPL services revenue beat improved earnings quality.",
                        source_type="earnings_report",
                        source="unit:test",
                        symbols=["AAPL"],
                        tags=["earnings"],
                        score=0.87,
                    )
                ]
            }
        },
        "AAPL",
    )

    assert "Fundamental RAG 基本面资料检索" not in context
    assert "AAPL earnings note" not in context


def test_fundamental_rag_context_formats_parent_documents():
    context = _fundamental_rag_context_for_symbol(
        {
            "AAPL": [
                RagDocument(
                    id="rag-1",
                    title="AAPL earnings note",
                    text="AAPL services revenue beat improved earnings quality.",
                    source_type="earnings_report",
                    source="unit:test",
                    symbols=["AAPL"],
                    tags=["earnings"],
                    score=0.87,
                    metadata={
                        "chunk_role": "parent_context",
                        "matched_child_texts": ["Services revenue beat and gross margin expanded."],
                    },
                )
            ]
        },
        "AAPL",
    )

    assert "Fundamental RAG 基本面资料检索" in context
    assert "AAPL earnings note" in context
    assert "Services revenue beat" in context
    assert "score=0.87" in context


def test_langgraph_persists_reports_for_all_business_agents_with_llm(tmp_path):
    llm = RoutingLLM()
    state = CopilotLangGraph(
        llm=llm,
        portfolio_getter=RecordingPortfolioGetter(),
        opportunity_radar_agent=OpportunityRadarAgent(llm=llm, tools=_fake_market_tools()),
        technical_position_agent=TechnicalPositionAgent(llm=llm, tools=_fake_market_tools()),
        fundamental_news_agent=FundamentalNewsAgent(llm=llm, tools=_fake_fundamental_tools()),
    ).run(
        subscription_symbols=["AAPL"],
        price_history_by_symbol={},
        portfolio=PortfolioSnapshot(),
        selected_analysts=[
            AnalystType.OPPORTUNITY_RADAR,
            AnalystType.TECHNICAL_POSITION,
            AnalystType.FUNDAMENTAL_NEWS,
        ],
        report_output_dir=tmp_path,
        trade_date="2026-05-08",
    )

    expected = {
        "opportunity_radar": "1_analysts",
        "technical_position": "1_analysts",
        "fundamental_news": "1_analysts",
        "futu_portfolio": "0_portfolio",
        "opportunity_review": "2_opportunity_review",
        "trader": "3_trader",
        "risk_check": "4_risk_check",
        "execution_alert": "5_execution",
        "run_explanation": "6_explanation",
    }

    assert set(expected).issubset(state["agent_reports"])
    assert "risk_challenge" not in state["agent_reports"]
    assert state["explanations"]["risk_challenges"]["AAPL"]
    assert state["trade_plans"]["AAPL"].direction.value == "buy"
    for key, parent in expected.items():
        path = Path(state["agent_reports"][key])
        assert path.exists()
        assert path.parent == tmp_path / parent

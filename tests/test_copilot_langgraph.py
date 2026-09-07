from pathlib import Path

from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.domain import (
    AnalystType,
    DistilledMemory,
    ExecutionMode,
    FundamentalAnalysisReport,
    MemoryStatus,
    MemoryType,
    NewsSentimentReport,
    PortfolioSnapshot,
    PriceBar,
    SubscriptionStatus,
    TechnicalContext,
    TechnicalDimension,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.graph.copilot_langgraph import (
    _analyst_context_for_symbol,
    _format_news_sentiment_report,
)
from ai_trading_copilot.copilot.domain.localization import zh_label
from ai_trading_copilot.copilot.agents import (
    FundamentalAnalystAgent,
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


def test_graph_uses_debug_prompt_for_default_technical_position_agent():
    default_graph = CopilotLangGraph(enable_default_llm=False)
    debug_graph = CopilotLangGraph(
        enable_default_llm=False,
        technical_position_debug=True,
    )

    assert default_graph.technical_position_agent.prompt_name == "technical_position.v1"
    assert debug_graph.technical_position_agent.prompt_name == "technical_position.debug.v1"


def test_news_sentiment_report_includes_news_references():
    content = _format_news_sentiment_report(
        subscription_symbols=["MU"],
        reports={
            "MU": NewsSentimentReport(
                symbol="MU",
                sentiment_score=0.1,
                key_events=["upcoming_earnings_catalyst"],
                alerts=["即将发布财报"],
                summary="财报将决定估值分歧方向。",
                downstream_summary="新闻中性偏谨慎，等待财报确认。",
                decision_basis=["新闻称财报将决定估值分歧方向。"],
                uncertainties=["财报结果尚未发布。"],
                news_references=[
                    {
                        "title": "Memory Stocks valuation disagreement widens",
                        "published_at": "2026-08-16T12:00:00",
                        "url": "https://example.com/memory-stocks",
                        "source": "MarketWatch",
                        "event_type": "upcoming_earnings_catalyst",
                        "relevance": "财报将决定。",
                    }
                ],
            )
        },
    )

    assert "## News References" in content
    assert "2026-08-16T12:00:00" in content
    assert "https://example.com/memory-stocks" in content
    assert "upcoming_earnings_catalyst" in content
    assert "## Downstream Context" in content
    assert "新闻中性偏谨慎，等待财报确认。" in content


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
        if "Fundamental Analyst" in text:
            return AIMessage(
                content=(
                    "# Fundamental report\n\n"
                    '{"thesis_intact": true, "material_risk": false, '
                    '"risk_flags": [], "summary": "No material tool-sourced risk."}'
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
        if "Risk Manager in an AI hedge fund" in text:
            return AIMessage(content="# Risk report\n\nRisk limit supports a small buy.")
        if "Portfolio Manager in an AI hedge fund" in text:
            return AIMessage(
                content=(
                    "# Portfolio report\n\n"
                    '{"action": "buy", "quantity": 10, "confidence": 0.8, '
                    '"reasoning": "Use a small starter position."}'
                )
            )
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
        FakeTool("get_fundamentals", "Revenue and margin quality are intact."),
        FakeTool("get_balance_sheet", "Balance sheet quality is stable."),
        FakeTool("get_cashflow", "Cash flow quality is stable."),
        FakeTool("get_income_statement", "Income statement quality is stable."),
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
                AnalystType.FUNDAMENTAL_ANALYSIS,
            ]
        }
    )

    assert graph.langgraph_available is True
    assert routes == [
        graph.NODE_OPPORTUNITY_RADAR,
        graph.NODE_FUNDAMENTAL_ANALYSIS,
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


def test_langgraph_without_llm_does_not_record_unused_memory(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
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

    assert state["memories"] == {}
    assert state["memory_retrievals"] == []
    assert "post_trade_review_learning" in state["agent_reports"]


def test_langgraph_runs_only_selected_analysts_and_reviews_their_outputs(tmp_path):
    state = CopilotLangGraph(enable_default_llm=False).run(
        subscription_symbols=["AAPL"],
        price_history_by_symbol={"AAPL": _actionable_bars()},
        fundamental_analysis_by_symbol={
            "AAPL": FundamentalAnalysisReport(
                symbol="AAPL",
                material_risk=True,
                risk_flags=["earnings_gap_risk"],
                summary="Unresolved earnings risk.",
            )
        },
        portfolio=PortfolioSnapshot(),
        selected_analysts=[
            AnalystType.OPPORTUNITY_RADAR,
            AnalystType.FUNDAMENTAL_ANALYSIS,
        ],
        report_output_dir=tmp_path,
    )

    node_names = [event.node_name for event in state["trace_events"]]

    assert CopilotLangGraph.NODE_OPPORTUNITY_RADAR in node_names
    assert CopilotLangGraph.NODE_FUNDAMENTAL_ANALYSIS in node_names
    assert CopilotLangGraph.NODE_TECHNICAL_POSITION not in node_names
    assert state["radar_items"][0].status == SubscriptionStatus.RISK_ELEVATED
    assert state["radar_items"][0].final_conclusion == zh_label(
        SubscriptionStatus.RISK_ELEVATED
    )
    assert state["radar_items"][0].risk_points == ["earnings_gap_risk"]
    assert set(state["analyst_reports"]) == {
        "opportunity_radar",
        "fundamental_analysis",
    }
    for report_path in state["analyst_reports"].values():
        path = Path(report_path)
        assert path.exists()
        assert path.parent == tmp_path / "1_analysts"


def test_analyst_context_excludes_trading_memories():
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

    assert "Retrieved Trading Memories" not in context
    assert "Wait for support confirmation" not in context
    assert "Near support" in context


def test_langgraph_has_no_early_memory_retrieval_node():
    assert "Retrieve Memories" not in CopilotLangGraph.NODE_ORDER


def test_analyst_context_uses_fundamental_report_not_graph_rag_documents():
    context = _analyst_context_for_symbol(
        {
            "fundamental_analysis_reports_by_symbol": {
                "AAPL": "# Fundamental Analysis\n\nRAG-informed thesis summary."
            },
            "unused_rag_contexts": {"AAPL": ["AAPL earnings note"]},
        },
        "AAPL",
    )

    assert "Fundamental Analysis" in context
    assert "RAG-informed thesis summary" in context
    assert "AAPL earnings note" not in context


def test_analyst_context_includes_structured_downstream_fields():
    context = _analyst_context_for_symbol(
        {
            "technical_contexts": {
                "AAPL": TechnicalContext(
                    symbol="AAPL",
                    stock=TechnicalDimension(scope="stock", symbol="AAPL"),
                    downstream_summary="Technical pullback is constructive.",
                    decision_basis=["Price is near support."],
                    uncertainties=["Sector confirmation is mixed."],
                )
            },
            "news_sentiment_by_symbol": {
                "AAPL": NewsSentimentReport(
                    symbol="AAPL",
                    sentiment_score=0.1,
                    downstream_summary="News is neutral.",
                    decision_basis=["No material negative company news."],
                    uncertainties=["Upcoming earnings could change sentiment."],
                )
            },
        },
        "AAPL",
    )

    assert "Technical Position Context" in context
    assert "News Sentiment Context" in context
    assert "Downstream summary: Technical pullback is constructive." in context
    assert "Price is near support." in context
    assert "Upcoming earnings could change sentiment." in context


def test_langgraph_persists_reports_for_all_business_agents_with_llm(tmp_path):
    llm = RoutingLLM()
    state = CopilotLangGraph(
        llm=llm,
        portfolio_getter=RecordingPortfolioGetter(),
        opportunity_radar_agent=OpportunityRadarAgent(llm=llm, tools=_fake_market_tools()),
        technical_position_agent=TechnicalPositionAgent(llm=llm, tools=_fake_market_tools()),
        fundamental_analyst_agent=FundamentalAnalystAgent(llm=llm, tools=_fake_fundamental_tools()),
    ).run(
        subscription_symbols=["AAPL"],
        price_history_by_symbol={},
        portfolio=PortfolioSnapshot(),
        selected_analysts=[
            AnalystType.OPPORTUNITY_RADAR,
            AnalystType.TECHNICAL_POSITION,
            AnalystType.FUNDAMENTAL_ANALYSIS,
        ],
        report_output_dir=tmp_path,
        trade_date="2026-05-08",
    )

    expected = {
        "opportunity_radar": "1_analysts",
        "technical_position": "1_analysts",
        "fundamental_analysis": "1_analysts",
        "futu_portfolio": "0_portfolio",
        "trader": "3_trader",
        "risk_check": "4_risk_check",
        "portfolio_manager": "5_portfolio_manager",
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


def test_langgraph_retrieves_memory_inside_trader_and_portfolio_nodes(tmp_path):
    llm = RoutingLLM()
    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
    memory = store.repository.upsert(
        DistilledMemory(
            memory_id="aapl-contextual-memory",
            memory_type=MemoryType.STRATEGY_PERFORMANCE,
            status=MemoryStatus.APPROVED,
            lesson="Use smaller size until an AAPL support retest is confirmed.",
            trigger="AAPL uptrend pullback near support.",
            symbols=["AAPL"],
            tags=["pullback", "position_sizing"],
            market_regimes=["uptrend"],
        )
    )
    state = CopilotLangGraph(
        llm=llm,
        memory_agent=PostTradeReviewLearningAgent(store),
        portfolio_getter=RecordingPortfolioGetter(),
        opportunity_radar_agent=OpportunityRadarAgent(llm=llm, tools=_fake_market_tools()),
        technical_position_agent=TechnicalPositionAgent(llm=llm, tools=_fake_market_tools()),
        fundamental_analyst_agent=FundamentalAnalystAgent(llm=llm, tools=_fake_fundamental_tools()),
    ).run(
        subscription_symbols=["AAPL"],
        portfolio=PortfolioSnapshot(),
        selected_analysts=[
            AnalystType.OPPORTUNITY_RADAR,
            AnalystType.TECHNICAL_POSITION,
            AnalystType.FUNDAMENTAL_ANALYSIS,
        ],
        report_output_dir=tmp_path,
        trade_date="2026-05-08",
        run_id="run-contextual-memory",
    )

    consumers = {record.request.consumer for record in state["memory_retrievals"]}
    node_names = [event.node_name for event in state["trace_events"]]
    assert consumers == {"trader", "portfolio_manager"}
    assert state["memories"]["AAPL"][0].memory_id == memory.memory_id
    assert "Retrieve Memories" not in node_names
    assert node_names.index(CopilotLangGraph.NODE_TRADER) > node_names.index(
        CopilotLangGraph.NODE_FUNDAMENTAL_ANALYSIS
    )
    assert Path(state["agent_reports"]["post_trade_review_learning"]).parent == (
        tmp_path / "6_memory"
    )



from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.adapters import yfinance_news as yfinance_news_adapter
from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_symbol
from ai_trading_copilot.copilot.adapters import trading_tools
from ai_trading_copilot.copilot.agents import (
    FundamentalAnalystAgent,
    NewsSentimentAgent,
    TechnicalPositionAgent,
)
from ai_trading_copilot.copilot.domain import (
    PriceBar,
    SubscriptionStatus,
    SymbolTrendState,
)


class RecordingTool:
    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.output


class ToolCallingFakeLLM:
    def __init__(self, *, tool_name, args, final_content):
        self.tool_name = tool_name
        self.args = args
        self.final_content = final_content
        self.invocations = 0
        self.bound_tools = []
        self.messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
        self.messages.append(messages)
        if self.invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_name,
                        "args": self.args,
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content=self.final_content)


def _bars_from_closes(closes):
    return [
        PriceBar(
            date=f"2026-05-{idx + 1:02d}",
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1000,
        )
        for idx, close in enumerate(closes)
    ]


def test_normalize_futu_symbol_maps_crcl_to_us_code():
    assert normalize_futu_symbol("CRCL") == "US.CRCL"


def test_technical_position_agent_uses_llm_market_tool_and_keeps_structured_position():
    tool = RecordingTool("get_stock_info", "CRCL quote")
    llm = ToolCallingFakeLLM(
        tool_name="get_stock_info",
        args={"symbol": "CRCL"},
        final_content="# Technical report\n\nCRCL is holding support.",
    )
    agent = TechnicalPositionAgent(llm=llm, tools=[tool])

    result = agent.analyze_with_report(
        symbol="CRCL",
        bars=_bars_from_closes([100 + i for i in range(60)]),
        trade_date="2026-05-08",
    )

    assert result.position.symbol == "CRCL"
    assert result.report.startswith("# Technical report")
    assert result.tool_calls.count("get_stock_info") >= 1
    assert tool.calls[0] == {"symbol": "CRCL"}


def test_technical_position_agent_prefetches_compact_summary_not_raw_market_tools():
    stock_tool = RecordingTool("get_stock_info", "CRCL quote")
    compact_summary = (
        '{"symbol": "CRCL", "window": "90d", "current_price": 120.0, '
        '"latest_ohlcv_5d": [{"date": "2026-05-08", "close": 120.0}], '
        '"returns": {"5d": 0.02, "20d": -0.04, "50d": 0.08, "90d": 0.18}, '
        '"moving_averages": {"ma20": 118.0, "ma50": 110.0, "ma200": 100.0, '
        '"price_vs_ma": {"ma20": 0.0169}}, "trend": "uptrend_pullback", '
        '"support": 112.0, "resistance": 132.0, '
        '"rsi": {"current": 54.1, "state": "neutral", "percentile_90d": 0.5}, '
        '"macd": {"state": "bullish_but_weakening"}, '
        '"atr": {"value": 3.2, "pct": 0.0267}, '
        '"bollinger": {"position": "upper_half"}, '
        '"volume": {"state": "above_20d_average"}, '
        '"windows": {"5d": {"return": 0.02}, "20d": {"return": -0.04}}, '
        '"key_events": ["near_support"], "data_quality": "available", "warnings": []}'
    )
    summary_tool = RecordingTool("get_technical_summary", compact_summary)
    llm = ToolCallingFakeLLM(
        tool_name="get_stock_info",
        args={"symbol": "CRCL"},
        final_content="# Technical report\n\nCRCL compact evidence is usable.",
    )
    agent = TechnicalPositionAgent(llm=llm, tools=[stock_tool, summary_tool])

    result = agent.analyze_with_report(
        symbol="CRCL",
        bars=_bars_from_closes([100 + i for i in range(60)]),
        trade_date="2026-05-08",
    )

    prompt_text = llm.messages[0][0].content
    assert "get_technical_summary" in result.tool_calls
    assert "get_stock_data" not in result.tool_calls
    assert "get_indicators" not in result.tool_calls
    assert summary_tool.calls[0] == {
        "symbol": "CRCL",
        "curr_date": "2026-05-08",
        "look_back_days": 90,
    }
    assert "latest_ohlcv_5d" in prompt_text
    assert "Date,Open,High,Low,Close,Volume" not in prompt_text
    assert "full indicator series" in prompt_text


def test_technical_position_agent_debug_prompt_parses_existing_contract_fields():
    tool = RecordingTool("get_stock_info", "CRCL quote")
    llm = ToolCallingFakeLLM(
        tool_name="get_stock_info",
        args={"symbol": "CRCL"},
        final_content=(
            "# 调试报告\n\n"
            "成功路径完成。\n"
            '{"current_price": 120.0, "support_level": 112.0, "recent_high": 132.0, '
            '"moving_average_20": 118.0, "moving_average_50": 110.0, '
            '"distance_to_support_pct": 0.0714, "pullback_from_high_pct": 0.0909, '
            '"reward_risk_ratio": 2.5, "uptrend": true, '
            '"stock_trend_state": "uptrend_pullback", "sector_symbol": "SPY", '
            '"sector_trend_state": "uptrend", "market_symbol": "SPY", '
            '"market_trend_state": "uptrend", "technical_summary": "回踩趋势仍可用。", '
            '"technical_warnings": [], "decision_basis": ["价格接近支撑", "趋势仍未破坏"], '
            '"uncertainties": ["财报前波动可能放大"], '
            '"downstream_summary": "技术面是上升趋势回踩，适合等待确认。", '
            '"audit_trace": [{"step": 1, "phase": "prefetch", '
            '"observation_summary": "工具证据可用。", "evidence_used": "价格和指标", '
            '"decision_summary": "继续生成结论。"}], '
            '"tool_interaction_summary": [{"tool_name": "get_stock_info", "phase": "prefetch", '
            '"input_summary": "CRCL", "output_summary": "行情摘要可用", '
            '"data_quality": "available"}], '
            '"llm_round_summary": [{"round": 1, "intent": "确认证据", '
            '"observed_evidence": "价格靠近支撑", "next_action_or_final": "final"}]}'
        ),
    )
    agent = TechnicalPositionAgent(llm=llm, tools=[tool], debug_mode=True)

    result = agent.analyze_with_report(
        symbol="CRCL",
        bars=_bars_from_closes([100 + i for i in range(60)]),
        trade_date="2026-05-08",
    )

    prompt_text = llm.messages[0][0].content
    assert "audit_trace" in prompt_text
    assert "Do not output hidden chain-of-thought" in prompt_text
    assert result.position.current_price == 120.0
    assert result.position.reward_risk_ratio == 2.5
    assert result.context is not None
    assert result.context.stock.trend_state == SymbolTrendState.UPTREND_PULLBACK
    assert result.context.summary == "回踩趋势仍可用。"
    assert result.context.decision_basis == ["价格接近支撑", "趋势仍未破坏"]
    assert result.context.uncertainties == ["财报前波动可能放大"]
    assert result.context.downstream_summary == "技术面是上升趋势回踩，适合等待确认。"


def test_technical_position_agent_default_and_debug_prompt_names(monkeypatch):
    rendered = []

    def fake_render_prompt(name, **kwargs):
        rendered.append(name)
        return "prompt"

    monkeypatch.setattr(
        "ai_trading_copilot.copilot.agents.technical_position_agent.render_prompt",
        fake_render_prompt,
    )
    final_content = (
        "# Technical report\n\n"
        '{"current_price": 120.0, "support_level": 112.0, "recent_high": 132.0, '
        '"moving_average_20": 118.0, "moving_average_50": 110.0, '
        '"distance_to_support_pct": 0.0714, "pullback_from_high_pct": 0.0909, '
        '"reward_risk_ratio": 2.5, "uptrend": true, '
        '"stock_trend_state": "uptrend", "sector_symbol": "SPY", '
        '"sector_trend_state": "uptrend", "market_symbol": "SPY", '
        '"market_trend_state": "uptrend", "technical_summary": "ok", '
        '"technical_warnings": []}'
    )

    default_agent = TechnicalPositionAgent(
        llm=ToolCallingFakeLLM(
            tool_name="get_stock_info",
            args={"symbol": "CRCL"},
            final_content=final_content,
        ),
        tools=[RecordingTool("get_stock_info", "CRCL quote")],
    )
    default_agent.analyze_with_report(
        symbol="CRCL",
        bars=_bars_from_closes([100 + i for i in range(60)]),
        trade_date="2026-05-08",
    )

    debug_agent = TechnicalPositionAgent(
        llm=ToolCallingFakeLLM(
            tool_name="get_stock_info",
            args={"symbol": "CRCL"},
            final_content=final_content,
        ),
        tools=[RecordingTool("get_stock_info", "CRCL quote")],
        debug_mode=True,
    )
    debug_agent.analyze_with_report(
        symbol="CRCL",
        bars=_bars_from_closes([100 + i for i in range(60)]),
        trade_date="2026-05-08",
    )

    assert rendered == ["technical_position.v1", "technical_position.debug.v1"]


def test_fundamental_analyst_agent_uses_tools_and_parses_structured_risk_json():
    tool = RecordingTool("get_fundamentals", "regulatory probe risk in fundamentals")
    llm = ToolCallingFakeLLM(
        tool_name="get_fundamentals",
        args={"ticker": "CRCL", "curr_date": "2026-05-08"},
        final_content=(
            "# Fundamental report\n\n"
            '{"thesis_intact": false, "material_risk": true, '
            '"risk_flags": ["regulatory_probe"], "summary": "Probe risk is unresolved.", '
            '"fundamental_score": -0.5, "key_events": ["regulatory probe"], '
            '"data_availability": {"fundamentals": "available"}, '
            '"decision_basis": ["Regulatory probe remains unresolved."], '
            '"uncertainties": ["Timing of resolution is unclear."], '
            '"downstream_summary": "Fundamentals are blocked by unresolved probe risk."}'
        ),
    )
    agent = FundamentalAnalystAgent(llm=llm, tools=[tool])

    result = agent.analyze_symbol(symbol="CRCL", trade_date="2026-05-08")

    assert result.report.symbol == "CRCL"
    assert result.report.thesis_intact is False
    assert result.report.material_risk is True
    assert result.report.risk_flags == ["regulatory_probe"]
    assert result.report.fundamental_score == -0.5
    assert result.report.decision_basis == ["Regulatory probe remains unresolved."]
    assert result.report.uncertainties == ["Timing of resolution is unclear."]
    assert result.report.downstream_summary == "Fundamentals are blocked by unresolved probe risk."
    assert result.tool_calls.count("get_fundamentals") >= 1
    assert len(tool.calls) == 2
    assert tool.calls[0] == {"ticker": "CRCL", "curr_date": "2026-05-08"}
    assert tool.calls[1] == {"ticker": "CRCL", "curr_date": "2026-05-08"}


def test_news_sentiment_agent_uses_forward_earnings_window_and_parses_references():
    company_news = RecordingTool(
        "get_company_news",
        (
            "### Memory Stocks valuation disagreement widens\n"
            "published_at: 2026-08-16T12:00:00\n"
            "source: MarketWatch\n"
            "title: Memory Stocks valuation disagreement widens\n"
            "url: https://example.com/memory-stocks\n"
            "summary: Earnings will decide the next move."
        ),
    )
    market_news = RecordingTool("get_market_news", "No broad market risk.")
    news_sentiment = RecordingTool("get_news_sentiment", "companyNewsScore: 0.1")
    social_sentiment = RecordingTool("get_social_sentiment", "No social rows.")
    earnings_calendar = RecordingTool(
        "get_earnings_calendar",
        "{'date': '2026-09-25', 'symbol': 'MU'}",
    )
    llm = ToolCallingFakeLLM(
        tool_name="get_company_news",
        args={"ticker": "MU", "start_date": "2026-08-11", "end_date": "2026-08-18"},
        final_content=(
            "# News report\n\n"
            "- 即将财报是前瞻催化剂，不是已过期风险。\n"
            '{"sentiment_score": 0.1, "company_news_score": 0.1, '
            '"social_sentiment_score": 0.0, "earnings_event_score": 0.2, '
            '"material_risk": false, "risk_flags": [], '
            '"key_events": ["upcoming_earnings_catalyst"], '
            '"alerts": ["即将发布财报，财报将决定估值分歧方向。"], '
            '"summary": "新闻指向即将财报催化剂。", '
            '"data_availability": {"company_news": "available", "earnings_calendar": "available"}, '
            '"decision_basis": ["新闻明确称财报将决定估值分歧方向。"], '
            '"uncertainties": ["财报结果尚未发布。"], '
            '"downstream_summary": "新闻情绪中性偏谨慎，核心是即将财报催化剂。", '
            '"news_references": [{"title": "Memory Stocks valuation disagreement widens", '
            '"published_at": "2026-08-16T12:00:00", '
            '"url": "https://example.com/memory-stocks", "source": "MarketWatch", '
            '"event_type": "upcoming_earnings_catalyst", '
            '"relevance": "财报将决定估值分歧方向。"}]}'
        ),
    )
    agent = NewsSentimentAgent(
        llm=llm,
        tools=[
            company_news,
            market_news,
            news_sentiment,
            social_sentiment,
            earnings_calendar,
        ],
    )

    result = agent.analyze_symbol(symbol="MU", trade_date="2026-08-18")

    prompt_text = llm.messages[0][0].content
    assert company_news.calls[0] == {
        "ticker": "MU",
        "start_date": "2026-08-11",
        "end_date": "2026-08-18",
    }
    assert earnings_calendar.calls[0] == {
        "ticker": "MU",
        "start_date": "2026-08-18",
        "end_date": "2026-10-02",
    }
    assert "Upcoming earnings window: 2026-08-18 to 2026-10-02" in prompt_text
    assert result.report.material_risk is False
    assert result.report.key_events == ["upcoming_earnings_catalyst"]
    assert result.report.decision_basis == ["新闻明确称财报将决定估值分歧方向。"]
    assert result.report.uncertainties == ["财报结果尚未发布。"]
    assert result.report.downstream_summary == "新闻情绪中性偏谨慎，核心是即将财报催化剂。"
    assert result.report.news_references == [
        {
            "title": "Memory Stocks valuation disagreement widens",
            "published_at": "2026-08-16T12:00:00",
            "url": "https://example.com/memory-stocks",
            "source": "MarketWatch",
            "event_type": "upcoming_earnings_catalyst",
            "relevance": "财报将决定估值分歧方向。",
        }
    ]
    assert "https://example.com/memory-stocks" in result.markdown
    assert "2026-08-16T12:00:00" in result.markdown


def test_yfinance_company_news_uses_process_cache(monkeypatch):
    yfinance_news_adapter.clear_yfinance_news_cache()
    calls = []

    def fake_fetch(ticker):
        calls.append(ticker)
        return [{
            "title": "CRCL expands payments network",
            "publisher": "Yahoo Finance",
            "link": "https://finance.yahoo.com/news/crcl",
            "providerPublishTime": 1777507200,
        }]

    monkeypatch.setattr(yfinance_news_adapter, "_fetch_company_news_from_yahoo", fake_fetch)

    first = yfinance_news_adapter.get_news_yfinance("CRCL", "2026-04-28", "2026-05-10")
    second = yfinance_news_adapter.get_news_yfinance("CRCL", "2026-04-28", "2026-05-10")

    assert calls == ["CRCL"]
    assert first == second
    assert "CRCL expands payments network" in first


def test_yfinance_rate_limit_enters_cooldown_without_repeating_request(monkeypatch):
    yfinance_news_adapter.clear_yfinance_news_cache()
    calls = []

    def fake_fetch(ticker):
        calls.append(ticker)
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr(yfinance_news_adapter, "_fetch_company_news_from_yahoo", fake_fetch)
    monkeypatch.setattr(yfinance_news_adapter, "_RETRY_DELAYS_SECONDS", ())
    monkeypatch.setattr(yfinance_news_adapter, "_fetch_company_news_from_finnhub", lambda *args: None)

    first = yfinance_news_adapter.get_news_yfinance("CRCL", "2026-05-01", "2026-05-10")
    second = yfinance_news_adapter.get_news_yfinance("CRCL", "2026-05-01", "2026-05-10")

    assert calls == ["CRCL"]
    assert first == second
    assert "Too Many Requests" in first


def test_yfinance_rate_limit_uses_finnhub_fallback_when_available(monkeypatch):
    yfinance_news_adapter.clear_yfinance_news_cache()

    def fake_fetch(ticker):
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr(yfinance_news_adapter, "_fetch_company_news_from_yahoo", fake_fetch)
    monkeypatch.setattr(yfinance_news_adapter, "_RETRY_DELAYS_SECONDS", ())
    monkeypatch.setattr(
        yfinance_news_adapter,
        "_fetch_company_news_from_finnhub",
        lambda ticker, start_date, end_date: "# Finnhub news for CRCL",
    )

    result = yfinance_news_adapter.get_news_yfinance("CRCL", "2026-05-01", "2026-05-10")

    assert result == "# Finnhub news for CRCL"


def test_fundamentals_uses_finnhub_when_yfinance_has_no_data(monkeypatch):
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.yfinance_fundamentals.get_fundamentals_yfinance",
        lambda ticker, curr_date: "No fundamentals data found for symbol 'CRCL'",
    )
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.finnhub.get_basic_fundamentals_text",
        lambda ticker: "# Company Fundamentals for CRCL via Finnhub",
    )

    result = trading_tools.get_fundamentals_text("CRCL", "2026-05-10")

    assert result == "# Company Fundamentals for CRCL via Finnhub"


def test_financial_statements_use_finnhub_when_yfinance_has_no_data(monkeypatch):
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.yfinance_fundamentals.get_balance_sheet_yfinance",
        lambda ticker, freq, curr_date: "No balance sheet data found for symbol 'CRCL'",
    )
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.yfinance_fundamentals.get_cashflow_yfinance",
        lambda ticker, freq, curr_date: "No cash flow data found for symbol 'CRCL'",
    )
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.yfinance_fundamentals.get_income_statement_yfinance",
        lambda ticker, freq, curr_date: "No income statement data found for symbol 'CRCL'",
    )
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.finnhub.get_balance_sheet_text",
        lambda ticker, freq, curr_date: "# Balance Sheet data for CRCL via Finnhub",
    )
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.finnhub.get_cashflow_text",
        lambda ticker, freq, curr_date: "# Cash Flow data for CRCL via Finnhub",
    )
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.adapters.finnhub.get_income_statement_text",
        lambda ticker, freq, curr_date: "# Income Statement data for CRCL via Finnhub",
    )

    assert trading_tools.get_balance_sheet_text("CRCL", "quarterly", "2026-05-10") == "# Balance Sheet data for CRCL via Finnhub"
    assert trading_tools.get_cashflow_text("CRCL", "quarterly", "2026-05-10") == "# Cash Flow data for CRCL via Finnhub"
    assert trading_tools.get_income_statement_text("CRCL", "quarterly", "2026-05-10") == "# Income Statement data for CRCL via Finnhub"


def test_yfinance_global_news_rate_limit_uses_finnhub_fallback(monkeypatch):
    yfinance_news_adapter.clear_yfinance_news_cache()

    def fake_fetch(limit):
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr(yfinance_news_adapter, "_fetch_global_news_from_yahoo", fake_fetch)
    monkeypatch.setattr(yfinance_news_adapter, "_RETRY_DELAYS_SECONDS", ())
    monkeypatch.setattr(
        yfinance_news_adapter,
        "_fetch_global_news_from_finnhub",
        lambda curr_date, look_back_days, limit: "# Finnhub global news",
    )

    result = yfinance_news_adapter.get_global_news_yfinance("2026-05-10", 7, 5)

    assert result == "# Finnhub global news"


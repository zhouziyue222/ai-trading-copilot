from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.adapters import yfinance_news as yfinance_news_adapter
from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_symbol
from ai_trading_copilot.copilot.adapters import trading_tools
from ai_trading_copilot.copilot.agents import (
    FundamentalNewsAgent,
    OpportunityRadarAgent,
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

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
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


def test_fundamental_news_agent_uses_tools_and_parses_structured_risk_json():
    tool = RecordingTool("get_news", "regulatory probe article")
    llm = ToolCallingFakeLLM(
        tool_name="get_news",
        args={"ticker": "CRCL", "start_date": "2026-05-01", "end_date": "2026-05-08"},
        final_content=(
            "# Fundamental report\n\n"
            '{"thesis_intact": false, "material_risk": true, '
            '"risk_flags": ["regulatory_probe"], "summary": "Probe risk is unresolved."}'
        ),
    )
    agent = FundamentalNewsAgent(llm=llm, tools=[tool])

    result = agent.analyze_symbol(symbol="CRCL", trade_date="2026-05-08")

    assert result.report.symbol == "CRCL"
    assert result.report.thesis_intact is False
    assert result.report.material_risk is True
    assert result.report.risk_flags == ["regulatory_probe"]
    assert result.tool_calls.count("get_news") >= 1
    assert len(tool.calls) == 1
    assert tool.calls[0] == {
        "ticker": "CRCL",
        "start_date": "2026-05-01",
        "end_date": "2026-05-08",
    }


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


def test_opportunity_radar_agent_uses_llm_tool_result_for_opportunity_status():
    tool = RecordingTool("get_stock_info", "CRCL snapshot")
    llm = ToolCallingFakeLLM(
        tool_name="get_stock_info",
        args={"symbol": "CRCL"},
        final_content=(
            "# Opportunity report\n\n"
            '{"status": "near_opportunity", "trend_state": "uptrend_pullback", '
            '"reason": "LLM sees a pullback near support.", "current_price": 113.67, '
            '"support_level": 110.0, "reward_risk_ratio": 2.3, '
            '"trend_reason": "Price remains in an uptrend pullback."}'
        ),
    )
    agent = OpportunityRadarAgent(llm=llm, tools=[tool])

    result = agent.analyze_symbol_with_report(
        symbol="CRCL",
        bars=_bars_from_closes([100 + i for i in range(60)]),
        trade_date="2026-05-08",
    )

    assert result.item.symbol == "CRCL"
    assert result.item.status == SubscriptionStatus.NEAR_OPPORTUNITY
    assert result.item.trend_state == SymbolTrendState.UPTREND_PULLBACK
    assert result.item.current_price == 113.67
    assert result.tool_calls.count("get_stock_info") >= 1
    assert tool.calls[0] == {"symbol": "CRCL"}

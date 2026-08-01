import pandas as pd

from ai_trading_copilot.copilot.adapters import (
    FutuMarketDataAdapter,
    parse_price_bars_from_csv,
)
from ai_trading_copilot.copilot.adapters import market_data
from ai_trading_copilot.copilot.adapters import trading_tools
from ai_trading_copilot.copilot.adapters.trading_tools import (
    get_indicator_text,
    get_stock_data_text,
)


def test_parse_price_bars_from_csv():
    raw = """# Stock data for AAPL
# Total records: 2

Date,Open,High,Low,Close,Adj Close,Volume
2026-01-02,100,105,99,104,104,1000
2026-01-03,104,108,103,107,107,1200
"""

    bars = parse_price_bars_from_csv(raw)

    assert len(bars) == 2
    assert bars[0].date == "2026-01-02"
    assert bars[0].close == 104
    assert bars[1].volume == 1200


def test_parse_price_bars_returns_empty_for_header_only_text():
    assert parse_price_bars_from_csv("# no data") == []


def test_get_stock_data_text_uses_futu_opend_directly(monkeypatch):
    calls = []

    def fake_get_stock_data_text(symbol, start_date, end_date):
        calls.append((symbol, start_date, end_date))
        return "# Futu data"

    monkeypatch.setattr(
        trading_tools,
        "get_futu_stock_data_text",
        fake_get_stock_data_text,
    )

    assert get_stock_data_text("CRCL", "2026-05-01", "2026-05-08") == "# Futu data"
    assert calls == [("CRCL", "2026-05-01", "2026-05-08")]


def test_market_data_adapter_uses_futu_opend_directly(monkeypatch):
    raw = """# Stock data for CRCL via Futu OpenD

Date,Open,High,Low,Close,Adj Close,Volume
2026-05-01,100,110,99,108,108,1200
"""

    calls = []

    def fake_get_stock_data_text(symbol, start_date, end_date):
        calls.append((symbol, start_date, end_date))
        return raw

    monkeypatch.setattr(market_data, "get_stock_data_text", fake_get_stock_data_text)

    bars = FutuMarketDataAdapter().get_price_history(
        symbol="CRCL",
        start_date="2026-05-01",
        end_date="2026-05-08",
    )

    assert calls == [("CRCL", "2026-05-01", "2026-05-08")]
    assert len(bars) == 1
    assert bars[0].date == "2026-05-01"
    assert bars[0].close == 108


def test_get_indicator_text_calculates_close_50_sma(monkeypatch):
    data = pd.DataFrame(
        {
            "time_key": pd.date_range("2026-01-01", periods=70, freq="D"),
            "open": range(70),
            "high": [value + 1 for value in range(70)],
            "low": range(70),
            "close": range(70),
            "volume": [1000] * 70,
        }
    )

    monkeypatch.setattr(
        trading_tools,
        "fetch_history_dataframe",
        lambda **kwargs: data,
    )

    output = get_indicator_text("CRCL", "close_50_sma", "2026-03-11", 5)

    assert "close_50_sma 指标值" in output
    assert "2026-03-11" in output


def test_get_indicator_text_reports_unknown_indicator():
    output = get_indicator_text("CRCL", "unknown_indicator", "2026-03-11", 5)

    assert "工具 get_indicators 处理 unknown_indicator 失败" in output

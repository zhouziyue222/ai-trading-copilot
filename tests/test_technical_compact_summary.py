import pandas as pd

from ai_trading_copilot.copilot.analysis.technical_position import (
    build_compact_technical_summary,
)
from ai_trading_copilot.copilot.domain import PriceBar


def _bars(count: int):
    return [
        PriceBar(
            date=date.strftime("%Y-%m-%d"),
            open=100 + idx * 0.2,
            high=101 + idx * 0.2,
            low=99 + idx * 0.2,
            close=100 + idx * 0.2,
            volume=1000 + idx * 5,
        )
        for idx, date in enumerate(pd.date_range("2025-01-01", periods=count))
    ]


def test_build_compact_technical_summary_contains_indicators_and_events_without_full_history():
    summary = build_compact_technical_summary(
        "MU",
        _bars(220),
        curr_date="2025-08-08",
        look_back_days=90,
    )

    assert summary["symbol"] == "MU"
    assert summary["window"] == "90d"
    assert summary["data_quality"] == "available"
    assert len(summary["latest_ohlcv_5d"]) == 5
    assert set(summary["returns"]) == {"5d", "20d", "50d", "90d"}
    assert summary["moving_averages"]["ma20"] is not None
    assert summary["moving_averages"]["ma50"] is not None
    assert summary["moving_averages"]["ma200"] is not None
    assert summary["rsi"]["current"] is not None
    assert summary["macd"]["state"] in {
        "bullish",
        "bullish_but_weakening",
        "bullish_cross",
        "neutral",
    }
    assert summary["atr"]["value"] is not None
    assert summary["bollinger"]["upper"] is not None
    assert summary["volume"]["state"] in {
        "above_20d_average",
        "below_20d_average",
        "near_20d_average",
    }
    assert set(summary["windows"]) == {"5d", "20d", "60d", "90d"}


def test_build_compact_technical_summary_marks_short_history_partial():
    summary = build_compact_technical_summary(
        "MU",
        _bars(12),
        curr_date="2025-01-12",
        look_back_days=90,
    )

    assert summary["data_quality"] == "partial"
    assert summary["moving_averages"]["ma20"] is None
    assert summary["latest_ohlcv_5d"]
    assert "partial_window:12_bars" in summary["warnings"]
    assert "insufficient_bars_for_ma200" in summary["warnings"]

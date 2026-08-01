from ai_trading_copilot.copilot.agents import (
    OpportunityRadarAgent,
    TechnicalPositionAgent,
)
from ai_trading_copilot.copilot.domain import (
    PriceBar,
    SymbolTrendState,
    SubscriptionStatus,
)


def _bars_from_closes(closes):
    bars = []
    for idx, close in enumerate(closes):
        bars.append(
            PriceBar(
                date=f"2026-01-{(idx % 28) + 1:02d}",
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1000,
            )
        )
    return bars


def _actionable_pullback_bars():
    bars = _bars_from_closes([80 + i * 0.5 for i in range(40)])
    for idx in range(19):
        bars.append(
            PriceBar(
                date=f"2026-02-{idx + 1:02d}",
                open=101,
                high=110 if idx == 0 else 103,
                low=100.5,
                close=101,
                volume=1000,
            )
        )
    bars.append(
        PriceBar(
            date="2026-02-20",
            open=101.5,
            high=103,
            low=100.5,
            close=102,
            volume=1000,
        )
    )
    return bars


def test_technical_position_finds_support_and_reward_risk():
    position = TechnicalPositionAgent().analyze("aapl", _actionable_pullback_bars())

    assert position.symbol == "AAPL"
    assert position.uptrend is True
    assert position.support_level == 100.5
    assert position.reward_risk_ratio > 2


def test_opportunity_radar_classifies_single_symbol_trend_and_status():
    item = OpportunityRadarAgent().analyze_symbol(
        symbol="aapl",
        bars=_actionable_pullback_bars(),
    )

    assert item.symbol == "AAPL"
    assert item.trend_state == SymbolTrendState.UPTREND_PULLBACK
    assert item.trend_label == "上升趋势回调"
    assert item.status == SubscriptionStatus.ACTIONABLE
    assert item.status_label == "可执行"


def test_opportunity_radar_scans_only_requested_subscription_symbols():
    results = OpportunityRadarAgent().scan(
        subscription_symbols=["AAPL", "SPY"],
        price_history_by_symbol={
            "AAPL": _actionable_pullback_bars(),
            "MSFT": _actionable_pullback_bars(),
        },
    )

    assert [item.symbol for item in results] == ["AAPL", "SPY"]
    assert results[0].status == SubscriptionStatus.ACTIONABLE
    assert results[1].status == SubscriptionStatus.OBSERVING


def test_opportunity_radar_marks_downtrend_as_risk_elevated():
    results = OpportunityRadarAgent().scan(
        subscription_symbols=["AAPL"],
        price_history_by_symbol={"AAPL": _bars_from_closes(list(range(310, 100, -1)))},
    )

    assert results[0].trend_state == SymbolTrendState.DOWNTREND
    assert results[0].status == SubscriptionStatus.RISK_ELEVATED

from ai_trading_copilot.copilot.agents.risk_agent import apply_limits
from ai_trading_copilot.copilot.agents import RiskAgent
from ai_trading_copilot.copilot.domain import (
    MarketRegime,
    MarketType,
    PortfolioSnapshot,
    PriceBar,
    RiskLimits,
    Subscription,
    SubscriptionBook,
    SubscriptionStatus,
    TradeDirection,
    TradePlan,
    UserPersonaConfig,
)


def _bars(closes):
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


def _plan(symbol="AAPL", **overrides):
    data = {
        "symbol": symbol,
        "subscription_status": SubscriptionStatus.ACTIONABLE,
        "market_regime": MarketRegime.UPTREND,
        "direction": TradeDirection.BUY,
        "entry_logic": "Buy near support.",
        "support_level": 100,
        "stop_loss": 97,
        "targets": [110],
        "reward_risk_ratio": 3,
        "position_weight": 0.2,
        "holding_period": "1-6 weeks",
        "invalidation_conditions": ["Close below support."],
        "persona_fit_reason": "Fits pullback persona.",
    }
    data.update(overrides)
    return TradePlan(**data)


def test_apply_limits_clamps_single_name_and_gross_exposure():
    final, clamps = apply_limits(
        {"AAPL": 0.6, "MSFT": 0.4, "TSLA": -0.4},
        RiskLimits(max_position_pct=0.5, max_gross_exposure=1.0),
    )

    assert final["AAPL"] < 0.5
    assert sum(abs(weight) for weight in final.values()) <= 1.0
    assert {event.reason for event in clamps} == {
        "max_position_pct",
        "max_gross_exposure",
    }


def test_risk_manager_clamps_requested_target_weight_to_persona_limit():
    risk = RiskAgent().review(
        persona=UserPersonaConfig(
            max_single_position_weight=0.15,
            default_position_weight_min=0.05,
            default_position_weight_max=0.10,
        ),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        plan=_plan(position_weight=0.30),
        price_history_by_symbol={"AAPL": _bars([100, 101, 102, 103, 104])},
    )

    assert risk.symbol == "AAPL"
    assert risk.current_price == 104
    assert risk.approved is True
    assert risk.target_weight == 0.30
    assert risk.final_weight == 0.15
    assert risk.delta_weight == 0.15
    assert risk.clamped is True
    assert risk.clamps[0].reason == "max_position_pct"


def test_risk_manager_scales_targets_by_gross_exposure():
    result = RiskAgent().review_book(
        persona=UserPersonaConfig(
            max_single_position_weight=0.50,
            max_gross_exposure=0.60,
        ),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=100_000),
        trade_plans={
            "AAPL": _plan("AAPL", position_weight=0.50),
            "MSFT": _plan("MSFT", position_weight=0.50),
        },
        target_weights={"AAPL": 0.50, "MSFT": 0.50},
        price_history_by_symbol={
            "AAPL": _bars([100, 101, 102]),
            "MSFT": _bars([200, 201, 202]),
        },
    )

    assert result.assessments["AAPL"].final_weight == 0.30
    assert result.assessments["MSFT"].final_weight == 0.30
    assert {event.reason for event in result.clamps} == {"max_gross_exposure"}


def test_risk_manager_holds_flat_when_symbol_is_not_eligible():
    subscriptions = SubscriptionBook(
        items=[Subscription(symbol="AAPL", market_type=MarketType.US_STOCK)]
    )
    risk = RiskAgent().review(
        persona=UserPersonaConfig(),
        subscriptions=subscriptions,
        portfolio=PortfolioSnapshot(
            total_value=100_000,
            cash=80_000,
            position_weights={"MSFT": 0.05},
        ),
        plan=_plan("MSFT", position_weight=0.20),
        price_history_by_symbol={"MSFT": _bars([100, 101, 102])},
    )

    assert risk.approved is False
    assert risk.target_weight == 0.05
    assert risk.final_weight == 0.05
    assert risk.delta_weight == 0
    assert risk.warnings

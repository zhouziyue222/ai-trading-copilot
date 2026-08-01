from ai_trading_copilot.copilot.agents import TraderAgent
from ai_trading_copilot.copilot.domain import (
    MarketRegime,
    OpportunityRadarItem,
    SubscriptionStatus,
    TechnicalPosition,
    TradeDirection,
    UserPersonaConfig,
)


def _position():
    return TechnicalPosition(
        symbol="AAPL",
        current_price=102,
        support_level=100.5,
        recent_high=110,
        moving_average_20=101,
        moving_average_50=95,
        distance_to_support_pct=0.015,
        pullback_from_high_pct=0.073,
        reward_risk_ratio=4.0,
        uptrend=True,
    )


def test_trader_agent_generates_complete_buy_plan_for_actionable_setup():
    plan = TraderAgent().create_plan(
        opportunity=OpportunityRadarItem(
            symbol="AAPL",
            status=SubscriptionStatus.ACTIONABLE,
            current_price=102,
            support_level=100.5,
            reward_risk_ratio=4.0,
            reason="Near support with acceptable reward/risk.",
        ),
        technical_position=_position(),
        market_regime=MarketRegime.BULL_MARKET,
        persona=UserPersonaConfig(),
    )

    assert plan.direction == TradeDirection.BUY
    assert plan.stop_loss == 97.48
    assert plan.targets[0] == 110
    assert plan.position_weight == 0.25
    assert plan.invalidation_conditions
    assert plan.uses_leverage is False
    assert plan.uses_options is False


def test_trader_agent_does_not_generate_buy_for_near_opportunity():
    plan = TraderAgent().create_plan(
        opportunity=OpportunityRadarItem(
            symbol="AAPL",
            status=SubscriptionStatus.NEAR_OPPORTUNITY,
            current_price=103,
            support_level=100.5,
            reward_risk_ratio=1.5,
            reason="Approaching support but not actionable.",
        ),
        technical_position=_position(),
        market_regime=MarketRegime.UPTREND,
        persona=UserPersonaConfig(),
    )

    assert plan.direction == TradeDirection.WATCH
    assert plan.stop_loss is None
    assert plan.position_weight == 0


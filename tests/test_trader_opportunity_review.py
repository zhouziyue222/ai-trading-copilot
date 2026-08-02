import pytest

from ai_trading_copilot.copilot.agents import TraderAgent
from ai_trading_copilot.copilot.domain import (
    FundamentalNewsReport,
    OpportunityRadarItem,
    SubscriptionStatus,
    TechnicalPosition,
    TradeDirection,
    UserPersonaConfig,
)


def _opportunity():
    return OpportunityRadarItem(
        symbol="AAPL",
        status=SubscriptionStatus.ACTIONABLE,
        current_price=102,
        support_level=100.5,
        reward_risk_ratio=3,
        reason="Clean technical setup.",
    )


def _position():
    return TechnicalPosition(
        symbol="AAPL",
        current_price=102,
        support_level=100.5,
        recent_high=110,
        moving_average_20=101,
        moving_average_50=99,
        distance_to_support_pct=0.015,
        pullback_from_high_pct=0.073,
        reward_risk_ratio=3,
        uptrend=True,
    )


def test_trader_internal_review_keeps_clean_report_actionable():
    result = TraderAgent().create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position(),
        fundamental_news=FundamentalNewsReport(symbol="AAPL", thesis_intact=True),
        persona=UserPersonaConfig(),
    )

    assert result.opportunity is not None
    assert result.opportunity.status == SubscriptionStatus.ACTIONABLE
    assert result.plan.direction == TradeDirection.BUY


def test_trader_internal_review_downgrades_material_risk():
    result = TraderAgent().create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position(),
        fundamental_news=FundamentalNewsReport(
            symbol="AAPL",
            material_risk=True,
            risk_flags=["regulatory_probe"],
        ),
        persona=UserPersonaConfig(),
    )

    assert result.opportunity is not None
    assert result.opportunity.status == SubscriptionStatus.RISK_ELEVATED
    assert result.plan.direction == TradeDirection.HOLD
    assert result.opportunity.risk_points == ["regulatory_probe"]


def test_trader_internal_review_rejects_symbol_mismatch():
    with pytest.raises(ValueError):
        TraderAgent().create_plan_with_report(
            opportunity=_opportunity(),
            technical_position=_position(),
            fundamental_news=FundamentalNewsReport(symbol="MSFT"),
            persona=UserPersonaConfig(),
        )

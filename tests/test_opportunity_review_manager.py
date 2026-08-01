import pytest

from ai_trading_copilot.copilot.agents import OpportunityReviewManager
from ai_trading_copilot.copilot.domain import (
    FundamentalNewsReport,
    OpportunityRadarItem,
    SubscriptionStatus,
)


def test_opportunity_review_keeps_clean_report_actionable():
    opportunity = OpportunityRadarItem(
        symbol="AAPL",
        status=SubscriptionStatus.ACTIONABLE,
        current_price=102,
        support_level=100.5,
        reward_risk_ratio=3,
        reason="Clean technical setup.",
    )

    reviewed = OpportunityReviewManager().review(
        symbol="AAPL",
        opportunity=opportunity,
        fundamental_news=FundamentalNewsReport(symbol="AAPL", thesis_intact=True),
    )

    assert reviewed.status == SubscriptionStatus.ACTIONABLE
    assert reviewed.final_conclusion == "可执行"
    assert reviewed.suggested_action == "通过风险检查后准备买入计划。"


def test_opportunity_review_downgrades_material_risk():
    opportunity = OpportunityRadarItem(
        symbol="AAPL",
        status=SubscriptionStatus.ACTIONABLE,
        current_price=102,
        support_level=100.5,
        reward_risk_ratio=3,
        reason="Clean technical setup.",
    )

    reviewed = OpportunityReviewManager().review(
        symbol="AAPL",
        opportunity=opportunity,
        fundamental_news=FundamentalNewsReport(
            symbol="AAPL",
            material_risk=True,
            risk_flags=["regulatory_probe"],
        ),
    )

    assert reviewed.status == SubscriptionStatus.RISK_ELEVATED
    assert "regulatory_probe" in reviewed.reason
    assert reviewed.risk_points == ["regulatory_probe"]


def test_opportunity_review_rejects_symbol_mismatch():
    with pytest.raises(ValueError):
        OpportunityReviewManager().review(
            symbol="AAPL",
            opportunity=OpportunityRadarItem(
                symbol="AAPL",
                status=SubscriptionStatus.ACTIONABLE,
                reason="Setup.",
            ),
            fundamental_news=FundamentalNewsReport(symbol="MSFT"),
        )

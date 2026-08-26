from ai_trading_copilot.copilot.agents import (
    PortfolioManager,
    RiskAgent,
)
from ai_trading_copilot.copilot.domain import (
    ExecutionMode,
    ExecutionStatus,
    MarketRegime,
    PortfolioSnapshot,
    SubscriptionStatus,
    TradeDirection,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.workflow import TradePlanReviewWorkflow


def _workflow() -> TradePlanReviewWorkflow:
    return TradePlanReviewWorkflow(
        persona_config=UserPersonaConfig(),
        subscription_symbols=["AAPL"],
        risk_agent=RiskAgent(),
        portfolio_manager=PortfolioManager(),
    )


def _plan(**overrides):
    data = {
        "symbol": "AAPL",
        "subscription_status": SubscriptionStatus.ACTIONABLE,
        "market_regime": MarketRegime.UPTREND,
        "direction": TradeDirection.BUY,
        "entry_logic": "Pullback to support.",
        "support_level": 180.0,
        "stop_loss": 174.0,
        "targets": [195.0],
        "reward_risk_ratio": 2.2,
        "position_weight": 0.20,
        "holding_period": "1-6 weeks",
        "invalidation_conditions": ["Close below support."],
        "persona_fit_reason": "Fits persona.",
    }
    data.update(overrides)
    return TradePlan(**data)


def test_workflow_approves_valid_subscribed_plan_for_simulation():
    review = _workflow().review(plan=_plan(), portfolio=PortfolioSnapshot())

    assert review.risk_assessment.approved is True
    assert review.risk_assessment.target_weight == 0.20
    assert review.risk_assessment.final_weight == 0.20
    assert review.execution_decision.status == ExecutionStatus.PORTFOLIO_DECIDED


def test_workflow_holds_non_subscription_flat():
    review = _workflow().review(
        plan=_plan(symbol="MSFT"),
        portfolio=PortfolioSnapshot(),
    )

    assert review.risk_assessment.approved is False
    assert review.risk_assessment.warnings
    assert review.risk_assessment.final_weight == 0
    assert review.execution_decision.action == "hold"
    assert review.execution_decision.status == ExecutionStatus.ALERT_ONLY


def test_workflow_live_mode_still_requires_confirmation():
    review = _workflow().review(
        plan=_plan(),
        portfolio=PortfolioSnapshot(),
        mode=ExecutionMode.LIVE,
        user_confirmed=False,
    )

    assert review.risk_assessment.approved is True
    assert review.execution_decision.status == ExecutionStatus.CONFIRMATION_REQUIRED

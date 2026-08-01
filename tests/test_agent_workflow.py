from ai_trading_copilot.copilot.agents import (
    ExecutionAlertManager,
    RiskAgent,
)
from ai_trading_copilot.copilot.domain import (
    ExecutionMode,
    ExecutionStatus,
    MarketRegime,
    PortfolioSnapshot,
    RiskRuleCode,
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
        execution_manager=ExecutionAlertManager(),
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


def _codes(review):
    return {violation.code for violation in review.risk_assessment.violations}


def test_workflow_approves_valid_subscribed_plan_for_simulation():
    review = _workflow().review(plan=_plan(), portfolio=PortfolioSnapshot())

    assert review.risk_assessment.approved is True
    assert review.execution_decision.status == ExecutionStatus.SIMULATION_READY


def test_workflow_blocks_non_subscription_before_execution():
    review = _workflow().review(
        plan=_plan(symbol="MSFT"),
        portfolio=PortfolioSnapshot(),
    )

    assert review.risk_assessment.approved is False
    assert RiskRuleCode.SUBSCRIPTION_REQUIRED in _codes(review)
    assert review.execution_decision.status == ExecutionStatus.BLOCKED_BY_RISK


def test_workflow_live_mode_still_requires_confirmation():
    review = _workflow().review(
        plan=_plan(),
        portfolio=PortfolioSnapshot(),
        mode=ExecutionMode.LIVE,
        user_confirmed=False,
    )

    assert review.risk_assessment.approved is True
    assert review.execution_decision.status == ExecutionStatus.CONFIRMATION_REQUIRED

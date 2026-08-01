from ai_trading_copilot.copilot.domain import (
    ExecutionMode,
    ExecutionStatus,
    MarketRegime,
    RiskAssessment,
    RiskRuleCode,
    RiskSeverity,
    RiskViolation,
    SubscriptionStatus,
    TradeDirection,
    TradePlan,
)
from ai_trading_copilot.copilot.services import prepare_execution_decision


def _plan(direction=TradeDirection.BUY):
    return TradePlan(
        symbol="AAPL",
        subscription_status=SubscriptionStatus.ACTIONABLE,
        market_regime=MarketRegime.UPTREND,
        direction=direction,
        entry_logic="Pullback to support.",
        support_level=180.0,
        stop_loss=174.0,
        targets=[195.0],
        reward_risk_ratio=2.2,
        position_weight=0.20,
        holding_period="1-6 weeks",
        invalidation_conditions=["Close below support."],
        persona_fit_reason="Fits persona.",
    )


def test_execution_defaults_to_simulation_ready_after_risk_approval():
    decision = prepare_execution_decision(
        plan=_plan(),
        risk_assessment=RiskAssessment(approved=True),
    )

    assert decision.mode == ExecutionMode.SIMULATION
    assert decision.status == ExecutionStatus.SIMULATION_READY
    assert decision.requires_user_confirmation is False


def test_execution_is_blocked_when_risk_fails():
    decision = prepare_execution_decision(
        plan=_plan(),
        risk_assessment=RiskAssessment(
            approved=False,
            violations=[
                RiskViolation(
                    code=RiskRuleCode.STOP_LOSS_REQUIRED,
                    severity=RiskSeverity.BLOCK,
                    message="Missing stop.",
                )
            ],
        ),
    )

    assert decision.status == ExecutionStatus.BLOCKED_BY_RISK
    assert decision.approved_by_risk is False


def test_live_execution_requires_explicit_confirmation():
    decision = prepare_execution_decision(
        plan=_plan(),
        risk_assessment=RiskAssessment(approved=True),
        mode=ExecutionMode.LIVE,
        user_confirmed=False,
    )

    assert decision.status == ExecutionStatus.CONFIRMATION_REQUIRED
    assert decision.requires_user_confirmation is True


def test_live_execution_can_only_be_ready_after_confirmation():
    decision = prepare_execution_decision(
        plan=_plan(),
        risk_assessment=RiskAssessment(approved=True),
        mode=ExecutionMode.LIVE,
        user_confirmed=True,
    )

    assert decision.status == ExecutionStatus.LIVE_READY
    assert decision.requires_user_confirmation is False


def test_watch_plan_produces_alert_only():
    decision = prepare_execution_decision(
        plan=_plan(direction=TradeDirection.WATCH),
        risk_assessment=RiskAssessment(approved=True),
    )

    assert decision.status == ExecutionStatus.ALERT_ONLY


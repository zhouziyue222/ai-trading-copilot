"""Execution and alert decision manager."""

from __future__ import annotations

from ai_trading_copilot.copilot.domain.enums import (
    ExecutionMode,
    ExecutionStatus,
    TradeDirection,
)
from ai_trading_copilot.copilot.domain.models import (
    ExecutionDecision,
    RiskAssessment,
    TradePlan,
)


_ORDER_DIRECTIONS = {
    TradeDirection.BUY,
    TradeDirection.REDUCE,
    TradeDirection.SELL,
}


def prepare_execution_decision(
    *,
    plan: TradePlan,
    risk_assessment: RiskAssessment,
    mode: ExecutionMode = ExecutionMode.SIMULATION,
    user_confirmed: bool = False,
    broker_execution_enabled: bool = False,
) -> ExecutionDecision:
    """Convert a risk-checked plan into an execution or alert state.

    This function intentionally does not call any broker API. It only decides
    whether the next step is blocked, a reminder, a simulated order candidate,
    or a live order candidate that still requires explicit confirmation.
    """
    if not risk_assessment.approved:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.BLOCKED_BY_RISK,
            direction=plan.direction,
            approved_by_risk=False,
            requires_user_confirmation=False,
            message="Hard risk checks failed; execution is blocked.",
        )

    if plan.direction not in _ORDER_DIRECTIONS:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.ALERT_ONLY,
            direction=plan.direction,
            approved_by_risk=True,
            requires_user_confirmation=False,
            message="This plan is an alert/watch decision, not an order candidate.",
        )

    if mode == ExecutionMode.SIMULATION:
        suffix = (
            " Simulated broker execution is enabled."
            if broker_execution_enabled
            else " No broker order has been submitted."
        )
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.SIMULATION_READY,
            direction=plan.direction,
            approved_by_risk=True,
            requires_user_confirmation=False,
            message="Simulation candidate is ready." + suffix,
        )

    if not user_confirmed:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.CONFIRMATION_REQUIRED,
            direction=plan.direction,
            approved_by_risk=True,
            requires_user_confirmation=True,
            message="Live execution requires explicit user confirmation.",
        )

    return ExecutionDecision(
        symbol=plan.symbol,
        mode=mode,
        status=ExecutionStatus.LIVE_READY,
        direction=plan.direction,
        approved_by_risk=True,
        requires_user_confirmation=False,
        message="Live order candidate is ready for a broker adapter.",
    )

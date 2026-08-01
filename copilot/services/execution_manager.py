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
) -> ExecutionDecision:
    """Convert a risk-checked plan into an execution or alert state.

    This function intentionally does not call any broker API. It only decides
    whether the next step is blocked, a reminder, a simulated order, or a live
    order candidate that still requires explicit confirmation.
    """
    if not risk_assessment.approved:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.BLOCKED_BY_RISK,
            direction=plan.direction,
            approved_by_risk=False,
            requires_user_confirmation=False,
            message="硬性风险检查未通过，执行已被阻止。",
        )

    if plan.direction not in _ORDER_DIRECTIONS:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.ALERT_ONLY,
            direction=plan.direction,
            approved_by_risk=True,
            requires_user_confirmation=False,
            message="该计划仅用于提醒或观察，不是订单候选。",
        )

    if mode == ExecutionMode.SIMULATION:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.SIMULATION_READY,
            direction=plan.direction,
            approved_by_risk=True,
            requires_user_confirmation=False,
            message="模拟执行已就绪，不会向实盘券商提交订单。",
        )

    if not user_confirmed:
        return ExecutionDecision(
            symbol=plan.symbol,
            mode=mode,
            status=ExecutionStatus.CONFIRMATION_REQUIRED,
            direction=plan.direction,
            approved_by_risk=True,
            requires_user_confirmation=True,
            message="实盘执行需要用户明确确认。",
        )

    return ExecutionDecision(
        symbol=plan.symbol,
        mode=mode,
        status=ExecutionStatus.LIVE_READY,
        direction=plan.direction,
        approved_by_risk=True,
        requires_user_confirmation=False,
        message="实盘订单候选已就绪，可交给券商适配器处理。",
    )

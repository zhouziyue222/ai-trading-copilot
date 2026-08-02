"""Execution and alert manager agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ai_trading_copilot.copilot.domain.enums import ExecutionMode, ExecutionStatus, TradeDirection
from ai_trading_copilot.copilot.domain.models import (
    BrokerExecutionRequest,
    BrokerExecutionResult,
    ExecutionDecision,
    RiskAssessment,
    TradePlan,
)
from ai_trading_copilot.copilot.domain.localization import zh_label
from ai_trading_copilot.copilot.services.execution_manager import prepare_execution_decision


@dataclass
class ExecutionAlertResult:
    decision: ExecutionDecision
    report: str


class ExecutionAlertManager:
    """Prepares alerts or one-shot simulated broker execution.

    Broker execution is opt-in per run. When enabled, this manager only submits
    to a Futu simulated account and only after the deterministic risk decision
    returns ``SIMULATION_READY``.
    """

    def __init__(self, llm=None, execution_adapter=None):
        self.llm = llm
        self.execution_adapter = execution_adapter
        self._broker_results: dict[str, BrokerExecutionResult] = {}

    def prepare(
        self,
        *,
        plan: TradePlan,
        risk_assessment: RiskAssessment,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
        broker_execution_enabled: bool = False,
        run_id: str | None = None,
    ) -> ExecutionDecision:
        decision = prepare_execution_decision(
            plan=plan,
            risk_assessment=risk_assessment,
            mode=mode,
            user_confirmed=user_confirmed,
            broker_execution_enabled=broker_execution_enabled,
        )
        if not broker_execution_enabled:
            return decision
        if decision.status != ExecutionStatus.SIMULATION_READY:
            return decision

        try:
            request = _broker_request_from_plan(plan=plan, run_id=run_id)
        except ValueError as exc:
            return _with_failed_broker_result(
                decision,
                str(exc),
                idempotency_key=_idempotency_key(plan=plan, run_id=run_id, quantity=0, price=None),
            )

        result = self._submit_simulated_order(request)
        return _with_broker_result(decision, result)

    def prepare_with_report(
        self,
        *,
        plan: TradePlan,
        risk_assessment: RiskAssessment,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
        broker_execution_enabled: bool = False,
        run_id: str | None = None,
    ) -> ExecutionAlertResult:
        decision = self.prepare(
            plan=plan,
            risk_assessment=risk_assessment,
            mode=mode,
            user_confirmed=user_confirmed,
            broker_execution_enabled=broker_execution_enabled,
            run_id=run_id,
        )
        fallback = _report_from_decision(decision, "deterministic execution guardrail")
        if self.llm is None:
            return ExecutionAlertResult(decision, fallback)

        prompt = (
            "You are the Execution Alert Manager for an AI trading copilot. "
            "Your job starts after Trader and Risk Check. Convert the deterministic "
            "execution decision into a user-ready alert or simulated broker execution "
            "summary. Never claim live execution happened. Do not weaken confirmation "
            "or risk requirements. Return Markdown only in Simplified Chinese.\n\n"
            f"Trade plan: {plan.model_dump_json()}\n"
            f"Risk assessment: {risk_assessment.model_dump_json()}\n"
            f"Broker execution enabled: {broker_execution_enabled}\n"
            f"Execution decision: {decision.model_dump_json()}"
        )
        try:
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response) or "").strip()
            return ExecutionAlertResult(decision, content or fallback)
        except Exception:
            return ExecutionAlertResult(decision, fallback)

    def _submit_simulated_order(self, request: BrokerExecutionRequest) -> BrokerExecutionResult:
        if request.idempotency_key in self._broker_results:
            return self._broker_results[request.idempotency_key]

        adapter = self.execution_adapter
        if adapter is None:
            from ai_trading_copilot.copilot.adapters.futu_execution import (
                FutuSimulatedExecutionAdapter,
            )

            adapter = FutuSimulatedExecutionAdapter()
            self.execution_adapter = adapter

        try:
            result = adapter.place_order(request)
        except Exception as exc:
            result = BrokerExecutionResult(
                idempotency_key=request.idempotency_key,
                submitted=False,
                status="failed",
                message=f"Simulated broker execution failed: {exc}",
            )
        self._broker_results[request.idempotency_key] = result
        return result


def _broker_request_from_plan(
    *,
    plan: TradePlan,
    run_id: str | None,
) -> BrokerExecutionRequest:
    side = _broker_side(plan.direction)
    quantity = _default_order_quantity()
    price = _order_price(plan)
    if price is None:
        raise ValueError("Cannot submit a simulated order without a positive limit price.")
    return BrokerExecutionRequest(
        idempotency_key=_idempotency_key(plan=plan, run_id=run_id, quantity=quantity, price=price),
        symbol=plan.symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type="NORMAL",
        trd_env="SIMULATE",
    )


def _broker_side(direction: TradeDirection) -> str:
    if direction == TradeDirection.BUY:
        return "BUY"
    if direction in {TradeDirection.SELL, TradeDirection.REDUCE}:
        return "SELL"
    raise ValueError(f"Direction is not broker-executable: {direction.value}")


def _order_price(plan: TradePlan) -> float | None:
    if plan.direction == TradeDirection.BUY and plan.support_level:
        return float(plan.support_level)
    if plan.direction in {TradeDirection.SELL, TradeDirection.REDUCE}:
        if plan.targets:
            return float(plan.targets[0])
        if plan.support_level:
            return float(plan.support_level)
    if plan.stop_loss:
        return float(plan.stop_loss)
    return None


def _default_order_quantity() -> float:
    raw = os.getenv("COPILOT_SIMULATED_ORDER_QUANTITY", "1").strip()
    try:
        quantity = float(raw)
    except ValueError:
        quantity = 1.0
    return max(quantity, 1.0)


def _idempotency_key(
    *,
    plan: TradePlan,
    run_id: str | None,
    quantity: float,
    price: float | None,
) -> str:
    key_run_id = (run_id or "manual").strip() or "manual"
    side = "BUY" if plan.direction == TradeDirection.BUY else "SELL"
    return (
        f"{key_run_id}:{plan.symbol}:{side}:"
        f"qty={quantity:g}:price={price if price is not None else 'none'}"
    )


def _with_broker_result(
    decision: ExecutionDecision,
    result: BrokerExecutionResult,
) -> ExecutionDecision:
    status = (
        ExecutionStatus.SIMULATED_ORDER_SUBMITTED
        if result.submitted
        else ExecutionStatus.SIMULATED_ORDER_FAILED
    )
    message = (
        "Simulated broker order submitted."
        if result.submitted
        else "Simulated broker order failed."
    )
    if result.message:
        message = f"{message} {result.message}"
    return decision.model_copy(
        update={
            "status": status,
            "submitted_to_broker": result.submitted,
            "broker_order_id": result.order_id,
            "broker_message": result.message,
            "broker_idempotency_key": result.idempotency_key,
            "message": message,
        }
    )


def _with_failed_broker_result(
    decision: ExecutionDecision,
    message: str,
    *,
    idempotency_key: str,
) -> ExecutionDecision:
    return decision.model_copy(
        update={
            "status": ExecutionStatus.SIMULATED_ORDER_FAILED,
            "submitted_to_broker": False,
            "broker_message": message,
            "broker_idempotency_key": idempotency_key,
            "message": f"Simulated broker order failed. {message}",
        }
    )


def _report_from_decision(decision: ExecutionDecision, note: str) -> str:
    return (
        f"# Execution Alert: {decision.symbol}\n\n"
        f"- Note: {note}\n"
        f"- Mode: {zh_label(decision.mode)}\n"
        f"- Status: {zh_label(decision.status)}\n"
        f"- Direction: {zh_label(decision.direction)}\n"
        f"- Risk approved: {'yes' if decision.approved_by_risk else 'no'}\n"
        f"- Requires confirmation: {'yes' if decision.requires_user_confirmation else 'no'}\n"
        f"- Submitted to broker: {'yes' if decision.submitted_to_broker else 'no'}\n"
        f"- Broker order id: {decision.broker_order_id or '-'}\n"
        f"- Message: {decision.message}\n"
    )

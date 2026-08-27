"""v2-style deterministic Portfolio Manager."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_copilot.copilot.agents.llm_tools import (
    extract_json_object,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.agents.react_runner import ReActAgentRunner
from ai_trading_copilot.copilot.config.prompts import render_prompt
from ai_trading_copilot.copilot.domain.enums import (
    ExecutionMode,
    ExecutionStatus,
    TradeDirection,
)
from ai_trading_copilot.copilot.domain.models import (
    ExecutionDecision,
    PortfolioSnapshot,
    RiskAssessment,
    TradePlan,
    normalize_symbol,
)
from ai_trading_copilot.copilot.services.cancellation import RunCancelled
from ai_trading_copilot.copilot.services.memory_retrieval import MemoryRetrievalSession


MIN_WEIGHT_DELTA = 1e-6
BROKER_ACTIONS = {"buy", "sell", "reduce", "short", "cover"}


@dataclass
class PortfolioDecisionResult:
    decision: ExecutionDecision
    report: str
    tool_calls: list[str] | None = None


class PortfolioManager:
    """Convert risk-adjusted target weights into final portfolio actions."""

    def __init__(self, llm=None):
        self.llm = llm

    def build_target_weights(
        self,
        *,
        trade_plans: dict[str, TradePlan],
        portfolio: PortfolioSnapshot,
    ) -> dict[str, float]:
        targets: dict[str, float] = {}
        for raw_symbol, plan in trade_plans.items():
            symbol = normalize_symbol(raw_symbol or plan.symbol)
            current_weight = portfolio.position_weights.get(symbol, 0.0)
            planned_weight = float(plan.position_weight or 0.0)
            if plan.direction == TradeDirection.BUY:
                targets[symbol] = planned_weight if planned_weight > 0 else current_weight
            elif plan.direction in {TradeDirection.HOLD, TradeDirection.WATCH}:
                targets[symbol] = current_weight
            elif plan.direction == TradeDirection.REDUCE:
                targets[symbol] = current_weight * 0.5
            elif plan.direction in {TradeDirection.SELL, TradeDirection.COVER}:
                targets[symbol] = 0.0
            elif plan.direction == TradeDirection.SHORT:
                targets[symbol] = -planned_weight
            else:
                targets[symbol] = current_weight
        return targets

    def decide(
        self,
        *,
        plan: TradePlan,
        risk_assessment: RiskAssessment,
        portfolio: PortfolioSnapshot,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
        analyst_context: str = "",
        run_id: str | None = None,
        memory_session: MemoryRetrievalSession | None = None,
    ) -> ExecutionDecision:
        return self.decide_with_report(
            plan=plan,
            risk_assessment=risk_assessment,
            portfolio=portfolio,
            mode=mode,
            user_confirmed=user_confirmed,
            analyst_context=analyst_context,
            run_id=run_id,
            memory_session=memory_session,
        ).decision

    def decide_with_report(
        self,
        *,
        plan: TradePlan,
        risk_assessment: RiskAssessment,
        portfolio: PortfolioSnapshot,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
        analyst_context: str = "",
        run_id: str | None = None,
        memory_session: MemoryRetrievalSession | None = None,
    ) -> PortfolioDecisionResult:
        baseline = _decision_from_risk(
            plan=plan,
            risk=risk_assessment,
            mode=mode,
            user_confirmed=user_confirmed,
        )
        if (
            self.llm is None
            or memory_session is None
            or not _increases_gross_exposure(risk_assessment)
        ):
            return PortfolioDecisionResult(
                decision=baseline,
                report=_report_from_decision(baseline, risk_assessment),
                tool_calls=[],
            )

        try:
            memory_evidence = memory_session.prefetch(
                _portfolio_memory_query(plan, risk_assessment, portfolio),
                tags=["portfolio", "position_sizing", plan.direction.value],
            )
            tools = memory_session.make_tools()
            prompt = render_prompt(
                "portfolio_memory_advisor.v1",
                trade_plan=plan.model_dump_json(),
                risk_assessment=risk_assessment.model_dump_json(),
                portfolio=portfolio.model_dump_json(),
                analyst_context=analyst_context or "-",
                memory_evidence=memory_evidence,
                available_memory_tools=", ".join(tool.name for tool in tools),
            )
            react_result = ReActAgentRunner(
                llm=self.llm,
                tools=tools,
                max_rounds=2,
            ).run(prompt=prompt)
            payload = extract_json_object(react_result.content)
            policy = str(payload.get("decision", "proceed")).strip().lower()
            scale = _memory_scale(policy, payload.get("scale", 1.0))
            raw_citations = payload.get("memory_citations", [])
            citations = memory_session.mark_cited(
                [str(item) for item in raw_citations]
                if isinstance(raw_citations, list)
                else []
            )
            if scale < 1 and not citations:
                scale = 1.0
            influence = (
                str(payload.get("memory_influence", "")).strip()
                if citations
                else ""
            )
            adjusted_risk = _scaled_risk(risk_assessment, scale)
            decision = _decision_from_risk(
                plan=plan,
                risk=adjusted_risk,
                mode=mode,
                user_confirmed=user_confirmed,
            ).model_copy(
                update={
                    "memory_citations": citations,
                    "memory_influence": influence,
                    "memory_scale": scale,
                    "pre_memory_final_weight": risk_assessment.final_weight,
                }
            )
            narrative = strip_trailing_json_object(react_result.content)
            report = _report_from_decision(decision, risk_assessment)
            if narrative:
                report += f"\n## Memory Advisor\n\n{narrative}\n"
            return PortfolioDecisionResult(
                decision=decision,
                report=report,
                tool_calls=["search_trading_memories", *react_result.tool_calls],
            )
        except RunCancelled:
            raise
        except Exception:
            decision = baseline.model_copy(
                update={"pre_memory_final_weight": risk_assessment.final_weight}
            )
            return PortfolioDecisionResult(
                decision=decision,
                report=_report_from_decision(decision, risk_assessment),
                tool_calls=[],
            )


def _decision_from_risk(
    *,
    plan: TradePlan,
    risk: RiskAssessment,
    mode: ExecutionMode,
    user_confirmed: bool,
) -> ExecutionDecision:
    action = _action_from_risk(risk)
    quantity = _estimated_quantity(action, risk)
    if quantity <= 0:
        action = "hold"
    status = _status_for_action(action, quantity, mode, user_confirmed)
    direction = _direction_from_action(action)
    requires_confirmation = (
        mode == ExecutionMode.LIVE
        and action != "hold"
        and quantity > 0
        and not user_confirmed
    )
    pending_broker_order = (
        mode == ExecutionMode.SIMULATION
        and action in BROKER_ACTIONS
        and quantity > 0
        and risk.approved
    )
    return ExecutionDecision(
        symbol=plan.symbol,
        mode=mode,
        status=status,
        direction=direction,
        approved_by_risk=risk.approved,
        requires_user_confirmation=requires_confirmation,
        message=_message(action, quantity, status, risk),
        action=action,
        quantity=quantity,
        confidence=1.0 if risk.approved else 0.0,
        reasoning=_reasoning(action, risk),
        current_weight=risk.current_position_weight,
        target_weight=risk.target_weight,
        final_weight=risk.final_weight,
        delta_weight=risk.delta_weight,
        current_price=risk.current_price,
        estimated_trade_value=risk.estimated_trade_value,
        pending_broker_order=pending_broker_order,
        broker_confirmation_required=pending_broker_order,
        submitted_to_broker=False,
    )


def _increases_gross_exposure(risk: RiskAssessment) -> bool:
    return abs(risk.final_weight) > abs(risk.current_position_weight) + MIN_WEIGHT_DELTA


def _memory_scale(policy: str, raw_scale) -> float:
    if policy == "hold":
        return 0.0
    if policy != "scale":
        return 1.0
    try:
        value = float(raw_scale)
    except (TypeError, ValueError):
        return 1.0
    return min(max(value, 0.0), 1.0)


def _scaled_risk(risk: RiskAssessment, scale: float) -> RiskAssessment:
    if scale >= 1 or not _increases_gross_exposure(risk):
        return risk
    delta = risk.delta_weight * scale
    final_weight = risk.current_position_weight + delta
    estimated_trade_value = risk.portfolio_value * delta
    return risk.model_copy(
        update={
            "final_weight": final_weight,
            "delta_weight": delta,
            "estimated_trade_value": estimated_trade_value,
            "clamped": True,
            "reasoning": {
                **risk.reasoning,
                "memory_advisor": f"risk-increasing delta scaled to {scale:.2f}",
            },
        }
    )


def _portfolio_memory_query(
    plan: TradePlan,
    risk: RiskAssessment,
    portfolio: PortfolioSnapshot,
) -> str:
    weights = ",".join(
        f"{symbol}:{weight:.4f}"
        for symbol, weight in sorted(portfolio.position_weights.items())
    )
    return " ".join(
        [
            plan.symbol,
            f"direction={plan.direction.value}",
            f"market_regime={plan.market_regime.value}",
            f"current_weight={risk.current_position_weight:.4f}",
            f"risk_final_weight={risk.final_weight:.4f}",
            f"delta_weight={risk.delta_weight:.4f}",
            f"risk_clamped={risk.clamped}",
            f"portfolio_weights={weights}",
            "cited_trader_memories=" + ",".join(plan.memory_citations),
            "portfolio concentration position sizing exposure drawdown",
        ]
    )


def _action_from_risk(risk: RiskAssessment) -> str:
    if not risk.approved:
        return "hold"
    if abs(risk.delta_weight) <= MIN_WEIGHT_DELTA:
        return "hold"
    if risk.current_price is None or risk.current_price <= 0:
        return "hold"
    if risk.delta_weight > 0:
        if risk.current_position_weight < 0:
            return "cover"
        return "buy"
    if risk.final_weight < 0:
        return "short"
    if risk.final_weight <= MIN_WEIGHT_DELTA and risk.current_position_weight > 0:
        return "sell"
    return "reduce"


def _estimated_quantity(action: str, risk: RiskAssessment) -> int:
    if action == "hold" or risk.current_price is None or risk.current_price <= 0:
        return 0
    return max(int(abs(risk.estimated_trade_value) // risk.current_price), 0)


def _status_for_action(
    action: str,
    quantity: int,
    mode: ExecutionMode,
    user_confirmed: bool,
) -> ExecutionStatus:
    if action == "hold" or quantity <= 0:
        return ExecutionStatus.ALERT_ONLY
    if mode == ExecutionMode.LIVE and not user_confirmed:
        return ExecutionStatus.CONFIRMATION_REQUIRED
    return ExecutionStatus.PORTFOLIO_DECIDED


def _direction_from_action(action: str) -> TradeDirection:
    try:
        return TradeDirection(action)
    except ValueError:
        return TradeDirection.HOLD


def _reasoning(action: str, risk: RiskAssessment) -> str:
    if action == "hold":
        if not risk.approved:
            return "Hold because the eligibility gate kept the target flat."
        if risk.current_price is None:
            return "Hold because no price is available for share estimation."
        return "Hold because the risk-adjusted target is unchanged or below one share."
    if risk.clamped:
        return "Use the risk-adjusted final weight after v2 clamp."
    return "Use the requested target weight because it is within v2 risk limits."


def _message(
    action: str,
    quantity: int,
    status: ExecutionStatus,
    risk: RiskAssessment,
) -> str:
    if action == "hold" or quantity <= 0:
        return (
            "Portfolio Manager selected hold. "
            f"Current {risk.current_position_weight:.2%}, final {risk.final_weight:.2%}. "
            "No broker order was submitted."
        )
    base = (
        f"Portfolio Manager selected {action} about {quantity} shares, moving "
        f"from {risk.current_position_weight:.2%} to {risk.final_weight:.2%}. "
    )
    if status == ExecutionStatus.CONFIRMATION_REQUIRED:
        return f"{base} Live action requires explicit user confirmation."
    return f"{base} Simulated broker order is pending result-page confirmation."


def _report_from_decision(
    decision: ExecutionDecision,
    risk: RiskAssessment,
) -> str:
    return (
        f"# Portfolio Manager Report: {decision.symbol}\n\n"
        f"- Action: {decision.action}\n"
        f"- Quantity: {decision.quantity}\n"
        f"- Status: {decision.status.value}\n"
        f"- Current weight: {decision.current_weight:.2%}\n"
        f"- Requested target weight: {decision.target_weight:.2%}\n"
        f"- Risk-adjusted final weight: {decision.final_weight:.2%}\n"
        f"- Delta weight: {decision.delta_weight:.2%}\n"
        f"- Estimated trade value: {decision.estimated_trade_value:.2f}\n"
        f"- Pending broker order: {str(decision.pending_broker_order).lower()}\n"
        f"- Broker confirmation required: {str(decision.broker_confirmation_required).lower()}\n"
        f"- Clamped by risk: {str(risk.clamped).lower()}\n"
        f"- Memory scale: {decision.memory_scale:.2f}\n"
        f"- Pre-memory final weight: {decision.pre_memory_final_weight}\n"
        f"- Memory citations: {', '.join(decision.memory_citations) or '-'}\n"
        f"- Memory influence: {decision.memory_influence or '-'}\n"
        f"- Reasoning: {decision.reasoning}\n"
        f"- Message: {decision.message}\n"
    )


__all__ = ["PortfolioDecisionResult", "PortfolioManager"]

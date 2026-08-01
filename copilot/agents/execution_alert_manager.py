"""Execution and alert manager agent."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_copilot.copilot.domain.enums import ExecutionMode
from ai_trading_copilot.copilot.domain.models import (
    ExecutionDecision,
    RiskAssessment,
    TradePlan,
)
from ai_trading_copilot.copilot.domain.localization import zh_label
from ai_trading_copilot.copilot.services.execution_manager import (
    prepare_execution_decision,
)


@dataclass
class ExecutionAlertResult:
    decision: ExecutionDecision
    report: str


class ExecutionAlertManager:
    """Prepares alerts, simulation orders, or live candidates with LLM messaging."""

    def __init__(self, llm=None):
        self.llm = llm

    def prepare(
        self,
        *,
        plan: TradePlan,
        risk_assessment: RiskAssessment,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
    ) -> ExecutionDecision:
        return prepare_execution_decision(
            plan=plan,
            risk_assessment=risk_assessment,
            mode=mode,
            user_confirmed=user_confirmed,
        )

    def prepare_with_report(
        self,
        *,
        plan: TradePlan,
        risk_assessment: RiskAssessment,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
    ) -> ExecutionAlertResult:
        decision = self.prepare(
            plan=plan,
            risk_assessment=risk_assessment,
            mode=mode,
            user_confirmed=user_confirmed,
        )
        fallback = _report_from_decision(decision, "确定性执行护栏。")
        if self.llm is None:
            return ExecutionAlertResult(decision, fallback)

        prompt = (
            "You are the Execution Alert Manager for an AI trading copilot. "
            "Your job starts after Trader and Risk Check. Convert the deterministic "
            "execution decision into a user-ready alert, simulation instruction, or live "
            "candidate checklist. Never place a real order and never weaken confirmation "
            "requirements.\n\n"
            "高效报告格式：\n"
            "1. 执行状态：阻止、仅提醒、模拟就绪、需要确认或实盘就绪。\n"
            "2. 用户动作：如有下一步，请写出具体动作。\n"
            "3. 护栏：止损、失效、确认要求，以及为什么允许或不允许执行。\n"
            "4. 券商说明：明确说明本报告不会提交任何实盘订单。\n\n"
            "Live execution must require explicit user confirmation unless the deterministic decision "
            "already says LIVE_READY. Return Markdown only in Simplified Chinese.\n\n"
            f"Trade plan: {plan.model_dump_json()}\n"
            f"Risk assessment: {risk_assessment.model_dump_json()}\n"
            f"Execution decision: {decision.model_dump_json()}"
        )
        try:
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response) or "").strip()
            return ExecutionAlertResult(decision, content or fallback)
        except Exception:
            return ExecutionAlertResult(decision, fallback)


def _report_from_decision(decision: ExecutionDecision, note: str) -> str:
    return (
        f"# 执行提醒报告：{decision.symbol}\n\n"
        f"- 说明：{note}\n"
        f"- 模式：{zh_label(decision.mode)}\n"
        f"- 状态：{zh_label(decision.status)}\n"
        f"- 方向：{zh_label(decision.direction)}\n"
        f"- 风控通过：{'是' if decision.approved_by_risk else '否'}\n"
        f"- 需要用户确认：{'是' if decision.requires_user_confirmation else '否'}\n"
        f"- 消息：{decision.message}\n"
    )

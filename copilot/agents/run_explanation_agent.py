"""Run explanation agent."""

from __future__ import annotations

from typing import Dict, Iterable, List

from ai_trading_copilot.copilot.domain.enums import SubscriptionStatus
from ai_trading_copilot.copilot.domain.models import (
    CopilotRunReport,
    ExecutionDecision,
    MarketRegimeReport,
    OpportunityRadarItem,
    RiskAssessment,
    SymbolExplanation,
    TechnicalPosition,
    TradePlan,
    SubscriptionBook,
)
from ai_trading_copilot.copilot.domain.localization import zh_join, zh_label
from ai_trading_copilot.copilot.agents.llm_tools import record_response_usage
from ai_trading_copilot.copilot.services.cancellation import RunCancelled, check_cancelled
from ai_trading_copilot.copilot.services.reporting import REPORT_WRITING_RULES, prepare_report, strip_report_json


class RunExplanationAgent:
    """Build user-facing Chinese explanations from structured state."""

    def __init__(self, llm=None):
        self.llm = llm
        self.fallback_reason = ""

    def explain(
        self,
        *,
        market_regime: MarketRegimeReport | None,
        radar_items: Iterable[OpportunityRadarItem],
        technical_positions: Dict[str, TechnicalPosition],
        trade_plans: Dict[str, TradePlan],
        risk_assessments: Dict[str, RiskAssessment],
        execution_decisions: Dict[str, ExecutionDecision],
        risk_challenges: Dict[str, str],
        subscriptions: SubscriptionBook | None = None,
    ) -> CopilotRunReport:
        self.fallback_reason = ""
        symbol_explanations: List[SymbolExplanation] = []
        subscription_by_symbol = (
            {item.symbol: item for item in subscriptions.items}
            if subscriptions is not None
            else {}
        )
        for item in radar_items:
            technical = technical_positions.get(item.symbol)
            plan = trade_plans.get(item.symbol)
            risk = risk_assessments.get(item.symbol)
            execution = execution_decisions.get(item.symbol)

            evidence = [item.reason]
            if technical is not None:
                evidence.extend([
                    f"当前价格 {technical.current_price:.2f}",
                    f"支撑位 {technical.support_level:.2f}",
                    f"距离支撑 {technical.distance_to_support_pct:.1%}",
                ])
                if technical.reward_risk_ratio is not None:
                    evidence.append(f"收益风险比 {technical.reward_risk_ratio:.2f}")

            risk_notes = []
            if risk is not None:
                risk_notes.append(
                    f"申请仓位 {risk.target_weight:.2%}，风控后仓位 {risk.final_weight:.2%}，调整幅度 {risk.delta_weight:.2%}。"
                )
                risk_notes.extend(risk.warnings)
            if item.symbol in risk_challenges:
                risk_notes.append(risk_challenges[item.symbol])
            intent_note = _user_intent_note(
                subscription=subscription_by_symbol.get(item.symbol),
                item=item,
                execution=execution,
            )

            summary = f"{item.symbol}：{item.status_label or zh_label(item.status)}"
            if plan is not None:
                summary += f"，方向 {zh_label(plan.direction)}"

            symbol_explanations.append(
                SymbolExplanation(
                    symbol=item.symbol,
                    status=item.status,
                    summary=summary,
                    key_evidence=evidence,
                    risk_notes=risk_notes,
                    user_intent_note=intent_note,
                    execution_message=execution.message if execution else None,
                )
            )

        fallback_summary = self._fallback_summary(market_regime, symbol_explanations)
        summary = self._llm_summary_or_fallback(fallback_summary, symbol_explanations)
        return CopilotRunReport(
            summary=summary,
            market_regime=market_regime,
            symbols=symbol_explanations,
            risk_challenges=risk_challenges,
        )
    def render_report(self, report: CopilotRunReport) -> str:
        lines = ["# 运行解释报告", "", report.summary, ""]
        for item in report.symbols:
            lines.extend(
                [
                    f"## {item.symbol}",
                    "",
                    f"- 状态：{zh_label(item.status)}",
                    f"- 用户意图：{item.user_intent_note or '-'}",
                    f"- 摘要：{item.summary}",
                    f"- 证据：{zh_join(item.key_evidence, empty='-')}",
                    f"- 风险提示：{zh_join(item.risk_notes, empty='-')}",
                    f"- 执行：{item.execution_message or '-'}",
                    "",
                ]
            )
        if report.risk_challenges:
            lines.extend(["## 风险挑战", ""])
            for symbol, challenge in sorted(report.risk_challenges.items()):
                lines.append(f"- {symbol}: {challenge}")
        return prepare_report("\n".join(lines))

    def _fallback_summary(
        self,
        market_regime: MarketRegimeReport | None,
        symbols: List[SymbolExplanation],
    ) -> str:
        actionable_count = sum(
            1 for item in symbols if item.status == SubscriptionStatus.ACTIONABLE
        )
        regime_text = zh_label(market_regime.regime) if market_regime else "【待补充】未提供整体市场评估"
        return (
            f"本次扫描覆盖 {len(symbols)} 个订阅标的；"
            f"市场环境为{regime_text}；"
            f"发现 {actionable_count} 个可执行候选机会，实际操作仍以风控及执行状态为准。"
        )

    def _llm_summary_or_fallback(
        self,
        fallback: str,
        symbols: List[SymbolExplanation],
    ) -> str:
        if self.llm is None:
            return fallback
        prompt = (
            REPORT_WRITING_RULES + "\n仅输出中文 Markdown 摘要，不输出任何结构化对象。\n" +
            "请用简体中文总结这次 AI 交易助手扫描。保持简洁、面向用户，"
            "不要改变结构化决策、方向、风险结论或执行状态。优先说明可执行机会数量、"
            "主要风险和用户下一步。\n\n"
            f"默认摘要：{fallback}\n"
            f"标的数量：{len(symbols)}\n"
            f"结构化标的摘要：{[item.model_dump() for item in symbols]}"
        )
        try:
            check_cancelled()
            response = self.llm.invoke(prompt)
            check_cancelled()
            record_response_usage(response, kind="llm", model=_llm_model_name(self.llm))
            content = getattr(response, "content", response)
            return strip_report_json(str(content)) or fallback
        except RunCancelled:
            raise
        except Exception as exc:
            self.fallback_reason = f"run_explanation_llm_failed: {exc}"
            return fallback


def _user_intent_note(*, subscription, item, execution) -> str:
    """Explain-only user intent paragraph. It never affects decisions."""
    if subscription is None:
        return ""


    user_status = subscription.status
    user_label = zh_label(user_status)
    system_buy = bool(execution is not None and execution.action == "buy")
    positive_user = user_status in {
        SubscriptionStatus.ACTIONABLE,
        SubscriptionStatus.NEAR_OPPORTUNITY,
    }
    negative_user = user_status in {
        SubscriptionStatus.RISK_ELEVATED,
        SubscriptionStatus.NOT_COMPATIBLE,
    }
    parts = [f"你的判断：{user_label}"]
    reason = _user_text(subscription.reason)
    target_action = _user_text(subscription.target_action)
    if reason:
        parts.append(f"关注理由：{reason}")
    if target_action:
        parts.append(f"目标动作：{target_action}")
    system_summary = (
        getattr(execution, "message", "")
        if execution is not None and getattr(execution, "message", "")
        else f"系统状态：{zh_label(item.status)}"
    )
    parts.append(f"系统结论：{system_summary}")
    if user_status == SubscriptionStatus.OBSERVING:
        parts.append("你尚未表态，系统按自身证据输出结论。")
    elif positive_user and system_buy:
        parts.append("你的判断与系统结论一致。")
    elif positive_user:
        parts.append(
            f"系统当前不产生买入建议，与你的判断冲突。反对理由："
            f"{item.reason or '当前证据未通过买入条件'}。"
            "系统结论不因你的标记改变，若风险解除且条件满足会重新复核。"
        )
    elif negative_user and not system_buy:
        parts.append("系统同样不产生买入建议，与你的判断一致。")
    elif negative_user:
        parts.append(
            "系统当前证据支持买入，与你的标记不一致，请确认该标记是否仍然有效。"
        )
    return " ".join(parts)


def _user_text(value) -> str:
    return " ".join(str(value or "").split())[:120]


def _llm_model_name(llm) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", "") or "")

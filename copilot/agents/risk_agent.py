"""Risk agent wrapper around deterministic hard rules."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_copilot.copilot.domain.models import (
    PortfolioSnapshot,
    RiskAssessment,
    SubscriptionBook,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.localization import zh_label
from ai_trading_copilot.copilot.domain.risk_rules import evaluate_trade_plan


@dataclass
class RiskReviewResult:
    assessment: RiskAssessment
    report: str
    risk_challenge: str


class RiskAgent:
    """Runs hard gates and an optional single LLM risk analyst review."""

    def __init__(self, llm=None):
        self.llm = llm

    def review(
        self,
        *,
        persona: UserPersonaConfig,
        subscriptions: SubscriptionBook,
        portfolio: PortfolioSnapshot,
        plan: TradePlan,
    ) -> RiskAssessment:
        return evaluate_trade_plan(
            persona=persona,
            subscriptions=subscriptions,
            portfolio=portfolio,
            plan=plan,
        )

    def review_with_report(
        self,
        *,
        persona: UserPersonaConfig,
        subscriptions: SubscriptionBook,
        portfolio: PortfolioSnapshot,
        plan: TradePlan,
        analyst_context: str = "",
    ) -> RiskReviewResult:
        hard_assessment = self.review(
            persona=persona,
            subscriptions=subscriptions,
            portfolio=portfolio,
            plan=plan,
        )
        risk_challenge = _risk_challenge_for_assessment(
            plan=plan,
            assessment=hard_assessment,
        )
        fallback = _report_from_assessment(
            plan=plan,
            assessment=hard_assessment,
            note="确定性硬性风控规则。未生成 LLM 风控报告。",
            risk_challenge=risk_challenge,
        )
        if getattr(self, "llm", None) is None:
            return RiskReviewResult(hard_assessment, fallback, risk_challenge)

        prompt = (
            "You are the single Risk Check analyst in an AI trading copilot. "
            "This consolidates TradingAgents' aggressive, neutral, and conservative risk "
            "analysts into one decisive review. Evaluate upside participation, balanced "
            "tradeoff, and capital protection in a single pass against persona constraints, "
            "portfolio exposure, drawdown, analyst evidence, stop-loss discipline, position "
            "sizing, and market risk.\n\n"
            "Hard risk rule blocks are final and cannot be overridden. If hard rules block "
            "the plan, explain the block and the minimum change required before reconsidering. "
            "If hard rules pass, still identify the top residual risk and whether sizing should "
            "remain, shrink, or wait.\n\n"
            "高效报告格式：风险结论、硬性规则结果、残余风险、仓位/执行提示、风险挑战、必要跟进。"
            "Return Markdown only in Simplified Chinese; do not claim approval when hard rules block the plan. "
            "Include one concise 风险挑战 sentence naming the most important human check before action.\n\n"
            f"Persona: {persona.model_dump_json()}\n"
            f"Subscriptions: {subscriptions.model_dump_json()}\n"
            f"Portfolio: {portfolio.model_dump_json()}\n"
            f"Trader plan: {plan.model_dump_json()}\n"
            f"Hard risk assessment: {hard_assessment.model_dump_json()}\n"
            f"Baseline risk challenge: {risk_challenge}\n"
            f"Analyst context:\n{analyst_context or '-'}"
        )
        try:
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response) or "").strip()
            report = (
                _ensure_risk_challenge_section(content, risk_challenge)
                if content
                else fallback
            )
            return RiskReviewResult(hard_assessment, report, risk_challenge)
        except Exception:
            return RiskReviewResult(hard_assessment, fallback, risk_challenge)


def _report_from_assessment(
    *,
    plan: TradePlan,
    assessment: RiskAssessment,
    note: str,
    risk_challenge: str,
) -> str:
    lines = [
        f"# 风控检查报告：{plan.symbol}",
        "",
        f"- 说明：{note}",
        f"- 方向：{zh_label(plan.direction)}",
        f"- 硬性规则通过：{'是' if assessment.approved else '否'}",
        f"- 违规数量：{len(assessment.violations)}",
    ]
    for violation in assessment.violations:
        lines.append(
            f"- {zh_label(violation.severity)}：{zh_label(violation.code)} - {violation.message}"
        )
    lines.extend(["", "## 风险挑战", "", risk_challenge])
    return "\n".join(lines)


def _risk_challenge_for_assessment(
    *,
    plan: TradePlan,
    assessment: RiskAssessment,
) -> str:
    if not assessment.approved:
        codes = ", ".join(zh_label(v.code) for v in assessment.blocking_violations)
        return (
            f"行动前，必须先解决阻断性的硬性风控规则：{codes or '未知'}。"
        )
    if assessment.warnings:
        codes = ", ".join(zh_label(v.code) for v in assessment.warnings)
        return f"行动前，请确认这些警告风险可以接受：{codes}。"
    return (
        f"行动前，请确认 {plan.symbol} 的价格仍满足计划入场、止损和失效条件。"
    )


def _ensure_risk_challenge_section(report: str, risk_challenge: str) -> str:
    if "risk challenge" in report.lower() or "风险挑战" in report:
        return report
    return f"{report}\n\n## 风险挑战\n\n{risk_challenge}"

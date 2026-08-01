"""Opportunity review manager."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_copilot.copilot.agents.llm_tools import (
    extract_json_object,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.domain.enums import SubscriptionStatus
from ai_trading_copilot.copilot.domain.models import (
    FundamentalNewsReport,
    OpportunityRadarItem,
    TechnicalPosition,
)
from ai_trading_copilot.copilot.domain.localization import zh_join, zh_label


@dataclass
class OpportunityReviewResult:
    item: OpportunityRadarItem
    report: str


class OpportunityReviewManager:
    """Merges enabled analyst outputs into one LLM-backed opportunity judgment."""

    def __init__(self, llm=None):
        self.llm = llm

    def review(
        self,
        *,
        symbol: str,
        opportunity: OpportunityRadarItem | None = None,
        technical_position: TechnicalPosition | None = None,
        fundamental_news: FundamentalNewsReport | None = None,
    ) -> OpportunityRadarItem:
        normalized = symbol.strip().upper()
        reviewed = opportunity or OpportunityRadarItem(
            symbol=normalized,
            status=SubscriptionStatus.OBSERVING,
            reason="未选择机会雷达分析。",
        )

        if reviewed.symbol != normalized:
            raise ValueError("opportunity symbol does not match review symbol")

        if technical_position is not None and technical_position.symbol != normalized:
            raise ValueError("technical position symbol does not match review symbol")

        if fundamental_news is not None and fundamental_news.symbol != normalized:
            raise ValueError("fundamental/news report symbol does not match opportunity")

        review_reasons = [reviewed.reason]
        risk_points = []

        if (
            technical_position is not None
            and technical_position.moving_average_50 is not None
            and technical_position.current_price < technical_position.moving_average_50
        ):
            risk_points.append("价格低于 50 日均线。")
            reviewed = reviewed.model_copy(
                update={
                    "status": SubscriptionStatus.RISK_ELEVATED,
                    "status_label": _final_conclusion(
                        SubscriptionStatus.RISK_ELEVATED
                    ),
                    "reason": (
                        reviewed.reason
                        + " 技术复核发现价格低于 50 日均线。"
                    ),
                }
            )

        if fundamental_news is not None and (
            fundamental_news.material_risk or not fundamental_news.thesis_intact
        ):
            reasons = [
                reviewed.reason,
                "基本面/新闻复核发现尚未解除的重大风险。",
            ]
            if fundamental_news.risk_flags:
                reasons.append("风险标记：" + ", ".join(fundamental_news.risk_flags))
                risk_points.extend(fundamental_news.risk_flags)
            reviewed = reviewed.model_copy(
                update={
                    "status": SubscriptionStatus.RISK_ELEVATED,
                    "status_label": _final_conclusion(
                        SubscriptionStatus.RISK_ELEVATED
                    ),
                    "reason": " ".join(reasons),
                }
            )

        if technical_position is not None:
            review_reasons.append(
                "技术复核：支撑位="
                f"{technical_position.support_level:.2f}, "
                f"收益风险比={technical_position.reward_risk_ratio}。"
            )
        if fundamental_news is not None:
            review_reasons.append(
                "基本面/新闻复核："
                + (fundamental_news.summary or "未提供摘要。")
            )

        if reviewed.status == SubscriptionStatus.RISK_ELEVATED and not risk_points:
            risk_points.append("机会状态为风险升高。")

        return reviewed.model_copy(
            update={
                "final_conclusion": _final_conclusion(reviewed.status),
                "review_reasons": review_reasons,
                "risk_points": risk_points,
                "suggested_action": _suggested_action(reviewed.status),
            }
        )

    def review_with_report(
        self,
        *,
        symbol: str,
        opportunity: OpportunityRadarItem | None = None,
        technical_position: TechnicalPosition | None = None,
        fundamental_news: FundamentalNewsReport | None = None,
        analyst_context: str = "",
    ) -> OpportunityReviewResult:
        fallback = self.review(
            symbol=symbol,
            opportunity=opportunity,
            technical_position=technical_position,
            fundamental_news=fundamental_news,
        )
        fallback_report = _report_from_item(
            fallback,
            "Deterministic fallback opportunity review.",
        )
        if self.llm is None:
            return OpportunityReviewResult(fallback, fallback_report)

        prompt = (
            "You are the Opportunity Review Manager for an AI trading copilot. "
            "This mirrors TradingAgents' research manager: synthesize analyst evidence into "
            "one clear action gate for the Trader. Do not redo analysis or quote long report "
            "sections; weigh conflicts and decide whether the opportunity should advance.\n\n"
            "Decision scale:\n"
            "- actionable: evidence supports preparing a trade plan now.\n"
            "- near_opportunity: setup is close but needs confirmation or better entry.\n"
            "- observing: no immediate setup.\n"
            "- risk_elevated: technical or fundamental/news risk blocks new entry.\n"
            "- not_compatible: symbol/setup conflicts with persona or scope.\n\n"
            "高效报告格式：结论、证据权衡、阻断风险、下一步动作。"
            "Write the entire Markdown review and all JSON string values in Simplified Chinese. "
            "Enum value status must still use the allowed English value. "
            "Use only supplied analyst reports and tool-derived evidence. Return a Markdown review followed by "
            "one JSON object with keys: status, reason, final_conclusion, "
            "review_reasons, risk_points, suggested_action. Allowed status values: "
            "observing, near_opportunity, actionable, risk_elevated, not_compatible. "
            "JSON must be the final object.\n\n"
            f"Symbol: {symbol}\n"
            f"Radar item: {opportunity.model_dump_json() if opportunity else '-'}\n"
            f"Technical position: {technical_position.model_dump_json() if technical_position else '-'}\n"
            f"Fundamental/news report: {fundamental_news.model_dump_json() if fundamental_news else '-'}\n"
            f"Fallback review: {fallback.model_dump_json()}\n"
            f"Analyst context:\n{analyst_context or '-'}"
        )
        try:
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response) or "").strip()
            payload = extract_json_object(content)
            item = _item_from_payload(payload, fallback)
            return OpportunityReviewResult(
                item,
                strip_trailing_json_object(content) or _report_from_item(item, "LLM 机会复核。"),
            )
        except Exception:
            return OpportunityReviewResult(fallback, fallback_report)


def _final_conclusion(status: SubscriptionStatus) -> str:
    return zh_label(status)


def _suggested_action(status: SubscriptionStatus) -> str:
    if status == SubscriptionStatus.ACTIONABLE:
        return "通过风险检查后准备买入计划。"
    if status == SubscriptionStatus.NEAR_OPPORTUNITY:
        return "等待支撑附近的回调或确认信号。"
    if status == SubscriptionStatus.RISK_ELEVATED:
        return "风险下降前避免新增入场。"
    return "继续观察。"


def _item_from_payload(
    payload: dict,
    fallback: OpportunityRadarItem,
) -> OpportunityRadarItem:
    updates = {}
    if payload.get("status"):
        updates["status"] = SubscriptionStatus(str(payload["status"]).lower())
    for key in {
        "reason",
        "final_conclusion",
        "review_reasons",
        "risk_points",
        "suggested_action",
    }:
        if key in payload:
            updates[key] = payload[key]
    status = updates.get("status", fallback.status)
    updates.setdefault("final_conclusion", _final_conclusion(status))
    updates.setdefault("suggested_action", _suggested_action(status))
    return fallback.model_copy(update=updates)


def _report_from_item(item: OpportunityRadarItem, note: str) -> str:
    return (
        f"# 机会复核报告：{item.symbol}\n\n"
        f"- 说明：{note}\n"
        f"- 状态：{zh_label(item.status)}\n"
        f"- 最终结论：{item.final_conclusion or _final_conclusion(item.status)}\n"
        f"- 建议动作：{item.suggested_action or _suggested_action(item.status)}\n"
        f"- 原因：{item.reason}\n"
        f"- 风险点：{zh_join(item.risk_points)}\n"
    )

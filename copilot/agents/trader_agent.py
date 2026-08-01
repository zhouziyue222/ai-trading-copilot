"""Trader agent for turning reviewed opportunities into trade plans."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_copilot.copilot.agents.llm_tools import (
    extract_json_object,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.domain.enums import (
    MarketRegime,
    SymbolTrendState,
    SubscriptionStatus,
    TradeDirection,
)
from ai_trading_copilot.copilot.domain.models import (
    OpportunityRadarItem,
    TechnicalPosition,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.localization import zh_label


@dataclass
class TraderAnalysisResult:
    plan: TradePlan
    report: str
    tool_calls: list[str]


class TraderAgent:
    """Generates a structured LLM-backed plan from analyst evidence."""

    def __init__(self, llm=None):
        self.llm = llm

    def create_plan(
        self,
        *,
        opportunity: OpportunityRadarItem,
        technical_position: TechnicalPosition,
        market_regime: MarketRegime | None = None,
        persona: UserPersonaConfig | None = None,
    ) -> TradePlan:
        persona = persona or UserPersonaConfig()
        resolved_regime = market_regime or _market_regime_from_trend(
            opportunity.trend_state
        )
        if opportunity.status == SubscriptionStatus.ACTIONABLE:
            return self._buy_plan(
                opportunity=opportunity,
                technical_position=technical_position,
                market_regime=resolved_regime,
                persona=persona,
            )

        direction = (
            TradeDirection.WATCH
            if opportunity.status
            in {SubscriptionStatus.OBSERVING, SubscriptionStatus.NEAR_OPPORTUNITY}
            else TradeDirection.HOLD
        )
        return TradePlan(
            symbol=opportunity.symbol,
            subscription_status=opportunity.status,
            market_regime=resolved_regime,
            direction=direction,
            entry_logic=opportunity.reason,
            support_level=opportunity.support_level,
            stop_loss=None,
            targets=[],
            reward_risk_ratio=opportunity.reward_risk_ratio,
            position_weight=0.0,
            holding_period="机会变为可执行前不设置主动持有周期。",
            invalidation_conditions=[],
            persona_fit_reason="当前机会尚不符合回调型用户画像的可执行要求。",
        )

    def _buy_plan(
        self,
        *,
        opportunity: OpportunityRadarItem,
        technical_position: TechnicalPosition,
        market_regime: MarketRegime,
        persona: UserPersonaConfig,
    ) -> TradePlan:
        support = technical_position.support_level
        stop_loss = round(support * 0.97, 2)
        first_target = round(technical_position.recent_high, 2)
        second_target = round(
            max(
                technical_position.recent_high,
                technical_position.current_price
                + 2 * (technical_position.current_price - stop_loss),
            ),
            2,
        )

        return TradePlan(
            symbol=opportunity.symbol,
            subscription_status=opportunity.status,
            market_regime=market_regime,
            direction=TradeDirection.BUY,
            entry_logic=(
                "仅在价格保持在已确认支撑附近、且市场环境仍具建设性时买入。"
            ),
            support_level=support,
            stop_loss=stop_loss,
            targets=[first_target, second_target],
            reward_risk_ratio=technical_position.reward_risk_ratio,
            position_weight=persona.default_position_weight_max,
            holding_period="1-6 周",
            invalidation_conditions=[
                "日线收盘跌破支撑位或止损位。",
                "市场环境降级为熊市风险。",
                "入场前收益风险比跌破要求阈值。",
            ],
            persona_fit_reason=(
                "支撑附近出现可执行回调，回撤风险可控，仓位处于默认 10%-25% 区间内。"
            ),
            uses_leverage=False,
            uses_options=False,
        )

    def create_plan_with_report(
        self,
        *,
        opportunity: OpportunityRadarItem,
        technical_position: TechnicalPosition,
        market_regime: MarketRegime | None = None,
        persona: UserPersonaConfig | None = None,
        analyst_context: str = "",
    ) -> TraderAnalysisResult:
        fallback = self.create_plan(
            opportunity=opportunity,
            technical_position=technical_position,
            market_regime=market_regime,
            persona=persona,
        )
        fallback_report = _report_from_plan(fallback, "确定性备用交易计划。")
        if self.llm is None:
            return TraderAnalysisResult(fallback, fallback_report, [])

        prompt = _trader_prompt(
            opportunity=opportunity,
            technical_position=technical_position,
            persona=persona or UserPersonaConfig(),
            fallback=fallback,
            analyst_context=analyst_context,
        )
        try:
            response = self.llm.invoke(prompt)
            content = str(getattr(response, "content", response) or "").strip()
            payload = extract_json_object(content)
            plan = _plan_from_payload(payload, fallback)
            report = strip_trailing_json_object(content) or _report_from_plan(plan, "LLM 交易计划。")
            return TraderAnalysisResult(plan, report, [])
        except Exception:
            return TraderAnalysisResult(fallback, fallback_report, [])


def _market_regime_from_trend(trend_state: SymbolTrendState | None) -> MarketRegime:
    if trend_state in {SymbolTrendState.UPTREND, SymbolTrendState.UPTREND_PULLBACK}:
        return MarketRegime.UPTREND
    if trend_state == SymbolTrendState.DOWNTREND:
        return MarketRegime.WEAKENING
    return MarketRegime.UNCLEAR


def _trader_prompt(
    *,
    opportunity: OpportunityRadarItem,
    technical_position: TechnicalPosition,
    persona: UserPersonaConfig,
    fallback: TradePlan,
    analyst_context: str,
) -> str:
    return (
        "You are the Trader in an AI trading copilot, mirroring the TradingAgents "
        "Trader role. Convert the reviewed opportunity and analyst evidence into a "
        "specific transaction proposal: buy, sell/reduce, hold, or watch. Anchor every "
        "decision in the provided analyst/tool reports and the user's persona. Do not "
        "discover new market facts and do not override a non-actionable review with a buy.\n\n"
        "高效报告格式：\n"
        "1. 交易建议：动作、仓位和时间周期。\n"
        "2. 入场计划：入场条件、止损、目标位和失效条件。\n"
        "3. 证据：来自分析师报告的 2-4 条要点，不要原文大段复制。\n"
        "4. 暂缓原因：如需回避或延后，用一句话说明最主要原因。\n\n"
        "Use only the provided analyst/tool reports as evidence. Write the entire "
        "Markdown Trader report and all JSON string values in Simplified Chinese. "
        "Enum values such as direction and market_regime must still use the allowed English values. "
        "Return a Markdown Trader report followed by one JSON object with keys: direction, entry_logic, "
        "market_regime, support_level, stop_loss, targets, reward_risk_ratio, "
        "position_weight, holding_period, invalidation_conditions, persona_fit_reason, "
        "uses_leverage, uses_options, is_chasing, breakout_confirmed, pullback_confirmed. "
        "Allowed direction values: buy, hold, reduce, sell, watch. JSON must be the final object. "
        "Do not invent market data.\n\n"
        f"Persona: {persona.model_dump_json()}\n"
        f"Reviewed opportunity: {opportunity.model_dump_json()}\n"
        f"Technical position: {technical_position.model_dump_json()}\n"
        f"Fallback safe plan: {fallback.model_dump_json()}\n"
        f"Analyst context:\n{analyst_context or '-'}"
    )


def _plan_from_payload(payload: dict, fallback: TradePlan) -> TradePlan:
    data = fallback.model_dump()
    for key in {
        "entry_logic",
        "support_level",
        "stop_loss",
        "targets",
        "reward_risk_ratio",
        "position_weight",
        "holding_period",
        "invalidation_conditions",
        "persona_fit_reason",
        "uses_leverage",
        "uses_options",
        "is_chasing",
        "breakout_confirmed",
        "pullback_confirmed",
    }:
        if key in payload:
            data[key] = payload[key]
    if payload.get("direction"):
        data["direction"] = TradeDirection(str(payload["direction"]).lower())
    if payload.get("market_regime"):
        data["market_regime"] = MarketRegime(str(payload["market_regime"]).lower())
    return TradePlan(**data)


def _report_from_plan(plan: TradePlan, note: str) -> str:
    return (
        f"# 交易员报告：{plan.symbol}\n\n"
        f"- 说明：{note}\n"
        f"- 方向：{zh_label(plan.direction)}\n"
        f"- 入场逻辑：{plan.entry_logic}\n"
        f"- 止损位：{plan.stop_loss}\n"
        f"- 目标位：{', '.join(str(target) for target in plan.targets) or '-'}\n"
        f"- 仓位权重：{plan.position_weight:.1%}\n"
        f"- 持有周期：{plan.holding_period}\n"
        f"- 用户画像匹配：{plan.persona_fit_reason}\n"
    )

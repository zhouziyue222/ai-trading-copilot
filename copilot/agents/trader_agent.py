"""Trader agent for turning analyst evidence into trade plans."""

from __future__ import annotations

from dataclasses import dataclass

from ai_trading_copilot.copilot.agents.llm_tools import (
    extract_json_object,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.agents.react_runner import ReActAgentRunner
from ai_trading_copilot.copilot.config.prompts import render_prompt
from ai_trading_copilot.copilot.services.cancellation import RunCancelled, check_cancelled
from ai_trading_copilot.copilot.services.tracing import (
    get_current_trace_recorder,
    summarize_text,
)
from ai_trading_copilot.copilot.domain.enums import (
    MarketRegime,
    SubscriptionStatus,
    SymbolTrendState,
    TradeDirection,
)
from ai_trading_copilot.copilot.domain.models import (
    FundamentalAnalysisReport,
    NewsSentimentReport,
    OpportunityRadarItem,
    TechnicalContext,
    TechnicalPosition,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.localization import zh_label
from ai_trading_copilot.copilot.services.memory_retrieval import MemoryRetrievalSession


@dataclass
class TraderAnalysisResult:
    plan: TradePlan
    report: str
    tool_calls: list[str]
    opportunity: OpportunityRadarItem | None = None


class TraderAgent:
    """Generates a structured plan after internal opportunity gating."""

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
        reviewed = _review_opportunity(
            opportunity=opportunity,
            technical_position=technical_position,
            fundamental_analysis=None,
            news_sentiment=None,
        )
        resolved_regime = market_regime or _market_regime_from_trend(reviewed.trend_state)
        if reviewed.status == SubscriptionStatus.ACTIONABLE:
            return self._buy_plan(
                opportunity=reviewed,
                technical_position=technical_position,
                market_regime=resolved_regime,
                persona=persona,
            )

        direction = (
            TradeDirection.WATCH
            if reviewed.status
            in {SubscriptionStatus.OBSERVING, SubscriptionStatus.NEAR_OPPORTUNITY}
            else TradeDirection.HOLD
        )
        return TradePlan(
            symbol=reviewed.symbol,
            subscription_status=reviewed.status,
            market_regime=resolved_regime,
            direction=direction,
            entry_logic=reviewed.reason,
            support_level=reviewed.support_level,
            stop_loss=None,
            targets=[],
            reward_risk_ratio=reviewed.reward_risk_ratio,
            position_weight=0.0,
            holding_period="No active holding period until the setup becomes actionable.",
            invalidation_conditions=[],
            persona_fit_reason="The current evidence does not meet actionable entry requirements.",
        )

    def create_plan_from_evidence(
        self,
        *,
        symbol: str,
        technical_position: TechnicalPosition | None = None,
        technical_context: TechnicalContext | None = None,
        news_sentiment: NewsSentimentReport | None = None,
        fundamental_analysis: FundamentalAnalysisReport | None = None,
        persona: UserPersonaConfig | None = None,
        analyst_context: str = "",
        memory_session: MemoryRetrievalSession | None = None,
    ) -> TraderAnalysisResult:
        position = technical_position or _technical_position_from_context(symbol, technical_context)
        opportunity = _opportunity_from_evidence(
            symbol=symbol,
            technical_position=position,
            technical_context=technical_context,
            news_sentiment=news_sentiment,
            fundamental_analysis=fundamental_analysis,
        )
        return self.create_plan_with_report(
            opportunity=opportunity,
            technical_position=position,
            market_regime=_market_regime_from_context(technical_context, opportunity),
            persona=persona,
            analyst_context=analyst_context,
            technical_context=technical_context,
            news_sentiment=news_sentiment,
            fundamental_analysis=fundamental_analysis,
            memory_session=memory_session,
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
                "Buy only after the stock, sector, and broad market context "
                "support an actionable pullback or confirmed continuation setup."
            ),
            support_level=support,
            stop_loss=stop_loss,
            targets=[first_target, second_target],
            reward_risk_ratio=technical_position.reward_risk_ratio,
            position_weight=persona.default_position_weight_max,
            holding_period="1-6 weeks",
            invalidation_conditions=[
                "Daily close below support or stop loss.",
                "Broad market context deteriorates into a downtrend.",
                "News/fundamental evidence introduces unresolved material risk.",
            ],
            persona_fit_reason=(
                "The plan fits a controlled pullback style with explicit stop loss, "
                "bounded position weight, and no leverage or options."
            ),
            uses_leverage=False,
            uses_options=False,
            pullback_confirmed=True,
        )

    def create_plan_with_report(
        self,
        *,
        opportunity: OpportunityRadarItem,
        technical_position: TechnicalPosition,
        market_regime: MarketRegime | None = None,
        persona: UserPersonaConfig | None = None,
        analyst_context: str = "",
        technical_context: TechnicalContext | None = None,
        news_sentiment: NewsSentimentReport | None = None,
        fundamental_analysis: FundamentalAnalysisReport | None = None,
        memory_session: MemoryRetrievalSession | None = None,
    ) -> TraderAnalysisResult:
        reviewed = _review_opportunity(
            opportunity=opportunity,
            technical_position=technical_position,
            fundamental_analysis=fundamental_analysis,
            news_sentiment=news_sentiment,
        )
        resolved_regime = market_regime or _market_regime_from_context(
            technical_context,
            reviewed,
        )
        fallback = self.create_plan(
            opportunity=reviewed,
            technical_position=technical_position,
            market_regime=resolved_regime,
            persona=persona,
        )
        fallback_report = _report_from_plan(fallback, reviewed, "deterministic trader gate")
        if self.llm is None:
            return TraderAnalysisResult(fallback, fallback_report, [], reviewed)

        memory_evidence = "-"
        memory_tools = []
        if memory_session is not None:
            try:
                memory_evidence = memory_session.prefetch(
                    _trader_memory_query(
                        opportunity=reviewed,
                        technical_position=technical_position,
                        technical_context=technical_context,
                        news_sentiment=news_sentiment,
                        fundamental_analysis=fundamental_analysis,
                        persona=persona or UserPersonaConfig(),
                        market_regime=resolved_regime,
                    ),
                    tags=_trader_memory_tags(reviewed, persona or UserPersonaConfig()),
                )
                memory_tools = memory_session.make_tools()
            except RunCancelled:
                raise
            except Exception:
                memory_session = None
                memory_evidence = "-"
                memory_tools = []

        prompt = _trader_prompt(
            opportunity=reviewed,
            technical_position=technical_position,
            technical_context=technical_context,
            news_sentiment=news_sentiment,
            fundamental_analysis=fundamental_analysis,
            persona=persona or UserPersonaConfig(),
            fallback=fallback,
            analyst_context=analyst_context,
            memory_evidence=memory_evidence,
            available_memory_tools=", ".join(tool.name for tool in memory_tools) or "-",
        )
        try:
            calls: list[str] = []
            if memory_tools:
                react_result = ReActAgentRunner(
                    llm=self.llm,
                    tools=memory_tools,
                    max_rounds=2,
                ).run(prompt=prompt)
                content = react_result.content.strip()
                calls = ["search_trading_memories", *react_result.tool_calls]
            else:
                recorder = get_current_trace_recorder()
                llm_attrs = {
                    "llm.model": _llm_model_name(self.llm),
                    "llm.round": 1,
                    "llm.max_rounds": 1,
                    "llm.message_count": 1,
                    **summarize_text(prompt, "prompt"),
                }
                if recorder is not None:
                    with recorder.start_span(
                        "llm.invoke", kind="client", attributes=llm_attrs
                    ) as span:
                        check_cancelled()
                        response = self.llm.invoke(prompt)
                        check_cancelled()
                        content = str(getattr(response, "content", response) or "").strip()
                        for key, value in summarize_text(content, "llm.response").items():
                            span.set_attribute(key, value)
                else:
                    check_cancelled()
                    response = self.llm.invoke(prompt)
                    check_cancelled()
                    content = str(getattr(response, "content", response) or "").strip()
            payload = extract_json_object(content)
            plan = _plan_from_payload(payload, fallback)
            raw_citations = payload.get("memory_citations", [])
            citations = (
                memory_session.mark_cited(_citation_values(raw_citations))
                if memory_session is not None
                else []
            )
            influence = str(payload.get("memory_influence", "")).strip() if citations else ""
            plan = plan.model_copy(
                update={
                    "memory_citations": citations,
                    "memory_influence": influence,
                }
            )
            plan = _enforce_trader_safety(
                plan,
                fallback=fallback,
                opportunity=reviewed,
                technical_position=technical_position,
                news_sentiment=news_sentiment,
                fundamental_analysis=fundamental_analysis,
                persona=persona or UserPersonaConfig(),
            )
            report = strip_trailing_json_object(content) or _report_from_plan(
                plan,
                reviewed,
                "LLM trader plan",
            )
            return TraderAnalysisResult(plan, report, calls, reviewed)
        except RunCancelled:
            raise
        except Exception:
            return TraderAnalysisResult(fallback, fallback_report, [], reviewed)


def _review_opportunity(
    *,
    opportunity: OpportunityRadarItem,
    technical_position: TechnicalPosition,
    fundamental_analysis: FundamentalAnalysisReport | None,
    news_sentiment: NewsSentimentReport | None,
) -> OpportunityRadarItem:
    if fundamental_analysis is not None and fundamental_analysis.symbol != opportunity.symbol:
        raise ValueError(
            f"Fundamental report symbol {fundamental_analysis.symbol} does not match {opportunity.symbol}"
        )
    if news_sentiment is not None and news_sentiment.symbol != opportunity.symbol:
        raise ValueError(
            f"News sentiment symbol {news_sentiment.symbol} does not match {opportunity.symbol}"
        )

    updates: dict[str, object] = {}
    risk_points = list(opportunity.risk_points)
    review_reasons = [*opportunity.review_reasons, opportunity.reason]

    if (
        technical_position.moving_average_50 is not None
        and technical_position.current_price < technical_position.moving_average_50
    ):
        risk_points.append("price_below_50dma")
        updates["status"] = SubscriptionStatus.RISK_ELEVATED
        review_reasons.append("Technical evidence shows price below the 50-day average.")

    if news_sentiment is not None:
        review_reasons.append(f"News sentiment score={news_sentiment.sentiment_score:.2f}.")
        if news_sentiment.material_risk or news_sentiment.sentiment_score <= -0.4:
            updates["status"] = SubscriptionStatus.RISK_ELEVATED
            risk_points.extend(news_sentiment.risk_flags or ["negative_news_sentiment"])

    if fundamental_analysis is not None:
        score = fundamental_analysis.fundamental_score
        if score is not None:
            review_reasons.append(f"Fundamental score={score:.2f}.")
        if fundamental_analysis.material_risk or not fundamental_analysis.thesis_intact:
            updates["status"] = SubscriptionStatus.RISK_ELEVATED
            risk_points.extend(fundamental_analysis.risk_flags or ["fundamental_thesis_risk"])

    status = updates.get("status", opportunity.status)
    return opportunity.model_copy(
        update={
            **updates,
            "final_conclusion": zh_label(status),
            "review_reasons": _unique_strings(review_reasons),
            "risk_points": _unique_strings(risk_points),
            "suggested_action": _suggested_action(status),
        }
    )


def _opportunity_from_evidence(
    *,
    symbol: str,
    technical_position: TechnicalPosition,
    technical_context: TechnicalContext | None,
    news_sentiment: NewsSentimentReport | None,
    fundamental_analysis: FundamentalAnalysisReport | None,
) -> OpportunityRadarItem:
    trend_state = _stock_trend_state(technical_context, technical_position)
    status = SubscriptionStatus.OBSERVING
    reason = "Technical evidence is insufficient for an active trade."
    if trend_state in {SymbolTrendState.UPTREND, SymbolTrendState.UPTREND_PULLBACK}:
        if (
            technical_position.reward_risk_ratio is not None
            and technical_position.reward_risk_ratio >= 2.0
        ):
            status = SubscriptionStatus.ACTIONABLE
            reason = "Technical setup is actionable with acceptable reward/risk."
        else:
            status = SubscriptionStatus.NEAR_OPPORTUNITY
            reason = "Trend is constructive but reward/risk or entry quality needs confirmation."
    if trend_state == SymbolTrendState.DOWNTREND:
        status = SubscriptionStatus.RISK_ELEVATED
        reason = "Stock technical state is a downtrend."

    item = OpportunityRadarItem(
        symbol=symbol.strip().upper(),
        status=status,
        trend_state=trend_state,
        current_price=technical_position.current_price,
        support_level=technical_position.support_level,
        reward_risk_ratio=technical_position.reward_risk_ratio,
        reason=reason,
    )
    return _review_opportunity(
        opportunity=item,
        technical_position=technical_position,
        fundamental_analysis=fundamental_analysis,
        news_sentiment=news_sentiment,
    )


def _market_regime_from_context(
    technical_context: TechnicalContext | None,
    opportunity: OpportunityRadarItem,
) -> MarketRegime:
    if technical_context is None or technical_context.market is None:
        return _market_regime_from_trend(opportunity.trend_state)
    state = technical_context.market.trend_state
    if state == SymbolTrendState.UPTREND:
        return MarketRegime.UPTREND
    if state == SymbolTrendState.DOWNTREND:
        return MarketRegime.DOWNTREND
    if state == SymbolTrendState.UPTREND_PULLBACK:
        return MarketRegime.RANGE_BOUND
    return _market_regime_from_trend(opportunity.trend_state)


def _market_regime_from_trend(trend_state: SymbolTrendState | None) -> MarketRegime:
    if trend_state in {SymbolTrendState.UPTREND, SymbolTrendState.UPTREND_PULLBACK}:
        return MarketRegime.UPTREND
    if trend_state == SymbolTrendState.DOWNTREND:
        return MarketRegime.DOWNTREND
    return MarketRegime.UNCLEAR


def _stock_trend_state(
    technical_context: TechnicalContext | None,
    technical_position: TechnicalPosition,
) -> SymbolTrendState:
    if technical_context is not None and technical_context.stock.trend_state is not None:
        return technical_context.stock.trend_state
    if technical_position.uptrend:
        return SymbolTrendState.UPTREND_PULLBACK
    if (
        technical_position.moving_average_50 is not None
        and technical_position.current_price < technical_position.moving_average_50
    ):
        return SymbolTrendState.DOWNTREND
    return SymbolTrendState.UNKNOWN


def _technical_position_from_context(
    symbol: str,
    technical_context: TechnicalContext | None,
) -> TechnicalPosition:
    stock = technical_context.stock if technical_context is not None else None
    current = stock.current_price if stock and stock.current_price else 1.0
    ma20 = stock.moving_average_20 if stock else None
    ma50 = stock.moving_average_50 if stock else None
    return TechnicalPosition(
        symbol=symbol.strip().upper(),
        current_price=current,
        support_level=current,
        recent_high=current,
        moving_average_20=ma20,
        moving_average_50=ma50,
        distance_to_support_pct=0.0,
        pullback_from_high_pct=0.0,
        reward_risk_ratio=None,
        uptrend=stock.trend_state == SymbolTrendState.UPTREND if stock else False,
    )


def _trader_prompt(
    *,
    opportunity: OpportunityRadarItem,
    technical_position: TechnicalPosition,
    technical_context: TechnicalContext | None,
    news_sentiment: NewsSentimentReport | None,
    fundamental_analysis: FundamentalAnalysisReport | None,
    persona: UserPersonaConfig,
    fallback: TradePlan,
    analyst_context: str,
    memory_evidence: str,
    available_memory_tools: str,
) -> str:
    return render_prompt(
        "trader.v1",
        persona=persona.model_dump_json(),
        opportunity=opportunity.model_dump_json(),
        technical_position=technical_position.model_dump_json(),
        technical_context=technical_context.model_dump_json() if technical_context else "-",
        news_sentiment=news_sentiment.model_dump_json() if news_sentiment else "-",
        fundamental_analysis=fundamental_analysis.model_dump_json() if fundamental_analysis else "-",
        fallback=fallback.model_dump_json(),
        analyst_context=analyst_context or "-",
        memory_evidence=memory_evidence or "-",
        available_memory_tools=available_memory_tools,
    )


def _llm_model_name(llm) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", "") or "")


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
        "memory_citations",
        "memory_influence",
    }:
        if key in payload:
            data[key] = payload[key]
    if payload.get("direction"):
        data["direction"] = TradeDirection(str(payload["direction"]).lower())
    if payload.get("market_regime"):
        data["market_regime"] = MarketRegime(str(payload["market_regime"]).lower())
    return TradePlan(**data)


def _trader_memory_query(
    *,
    opportunity: OpportunityRadarItem,
    technical_position: TechnicalPosition,
    technical_context: TechnicalContext | None,
    news_sentiment: NewsSentimentReport | None,
    fundamental_analysis: FundamentalAnalysisReport | None,
    persona: UserPersonaConfig,
    market_regime: MarketRegime,
) -> str:
    stock_trend = (
        technical_context.stock.trend_state.value
        if technical_context and technical_context.stock.trend_state
        else opportunity.trend_state.value if opportunity.trend_state else "unknown"
    )
    return " ".join(
        [
            opportunity.symbol,
            f"market_regime={market_regime.value}",
            f"trend={stock_trend}",
            f"opportunity={opportunity.status.value}",
            f"distance_to_support={technical_position.distance_to_support_pct:.4f}",
            f"reward_risk={technical_position.reward_risk_ratio}",
            f"news_material_risk={bool(news_sentiment and news_sentiment.material_risk)}",
            "news_flags=" + ",".join(news_sentiment.risk_flags if news_sentiment else []),
            "fundamental_material_risk="
            + str(bool(fundamental_analysis and fundamental_analysis.material_risk)),
            "fundamental_flags="
            + ",".join(fundamental_analysis.risk_flags if fundamental_analysis else []),
            f"trading_style={persona.trading_style}",
            f"risk_profile={persona.risk_profile}",
            "trade planning entry stop loss position sizing invalidation",
        ]
    )


def _trader_memory_tags(
    opportunity: OpportunityRadarItem,
    persona: UserPersonaConfig,
) -> list[str]:
    return [
        "trading",
        "trade_plan",
        opportunity.status.value,
        persona.trading_style,
        persona.risk_profile,
    ]


def _citation_values(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _enforce_trader_safety(
    plan: TradePlan,
    *,
    fallback: TradePlan,
    opportunity: OpportunityRadarItem,
    technical_position: TechnicalPosition,
    news_sentiment: NewsSentimentReport | None,
    fundamental_analysis: FundamentalAnalysisReport | None,
    persona: UserPersonaConfig,
) -> TradePlan:
    buy_is_safe = (
        opportunity.status == SubscriptionStatus.ACTIONABLE
        and technical_position.reward_risk_ratio is not None
        and technical_position.reward_risk_ratio >= 2.0
        and not bool(news_sentiment and news_sentiment.material_risk)
        and not bool(fundamental_analysis and fundamental_analysis.material_risk)
        and technical_position.distance_to_support_pct <= 0.03
        and not plan.is_chasing
        and plan.stop_loss is not None
        and bool(plan.targets)
        and bool(plan.invalidation_conditions)
    )
    if plan.direction == TradeDirection.BUY and not buy_is_safe:
        if fallback.direction != TradeDirection.BUY:
            return fallback
        return fallback.model_copy(
            update={
                "direction": TradeDirection.HOLD,
                "entry_logic": "Current structured evidence failed the deterministic buy gate.",
                "position_weight": 0.0,
                "targets": [],
                "memory_citations": [],
                "memory_influence": "",
            }
        )
    return plan.model_copy(
        update={
            "position_weight": min(plan.position_weight, persona.default_position_weight_max),
            "uses_leverage": False,
            "uses_options": False,
        }
    )


def _report_from_plan(plan: TradePlan, opportunity: OpportunityRadarItem, note: str) -> str:
    return (
        f"# Trader Report: {plan.symbol}\n\n"
        f"- Note: {note}\n"
        f"- Market state: {zh_label(plan.market_regime)}\n"
        f"- Opportunity status: {zh_label(opportunity.status)}\n"
        f"- Direction: {zh_label(plan.direction)}\n"
        f"- Entry logic: {plan.entry_logic}\n"
        f"- Stop loss: {plan.stop_loss}\n"
        f"- Targets: {', '.join(str(target) for target in plan.targets) or '-'}\n"
        f"- Position weight: {plan.position_weight:.1%}\n"
        f"- Holding period: {plan.holding_period}\n"
        f"- Risk points: {', '.join(opportunity.risk_points) or '-'}\n"
    )


def _suggested_action(status: SubscriptionStatus) -> str:
    if status == SubscriptionStatus.ACTIONABLE:
        return "Prepare a risk-checked trade plan."
    if status == SubscriptionStatus.NEAR_OPPORTUNITY:
        return "Wait for better entry confirmation."
    if status == SubscriptionStatus.RISK_ELEVATED:
        return "Avoid new entry until risk clears."
    return "Continue monitoring."


def _unique_strings(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            output.append(cleaned)
            seen.add(cleaned)
    return output



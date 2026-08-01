"""Subscription opportunity radar."""

from __future__ import annotations

from typing import Dict, List

from ai_trading_copilot.copilot.analysis.technical_position import (
    evaluate_technical_position,
)
from ai_trading_copilot.copilot.domain.enums import (
    SubscriptionStatus,
    SymbolTrendState,
)
from ai_trading_copilot.copilot.domain.models import (
    OpportunityRadarItem,
    PriceBar,
)


def analyze_symbol_opportunity(symbol: str, bars: List[PriceBar] | None) -> OpportunityRadarItem:
    """Classify one symbol's trend state and opportunity status."""
    normalized = symbol.strip().upper()
    if not bars:
        return OpportunityRadarItem(
            symbol=normalized,
            status=SubscriptionStatus.OBSERVING,
            trend_state=SymbolTrendState.UNKNOWN,
            trend_reason="没有可用的价格历史数据。",
            reason="该订阅标的没有可用的价格历史数据。",
        )

    position = evaluate_technical_position(normalized, bars)
    trend_state = _classify_symbol_trend(position)
    status = SubscriptionStatus.OBSERVING
    reason = "未发现明确的短线入场优势。"

    if trend_state == SymbolTrendState.DOWNTREND:
        status = SubscriptionStatus.RISK_ELEVATED
        reason = "标的处于下降趋势，或价格低于关键趋势参考位。"
    elif (
        trend_state == SymbolTrendState.UPTREND_PULLBACK
        and position.distance_to_support_pct <= 0.02
        and position.reward_risk_ratio is not None
        and position.reward_risk_ratio >= 2.0
    ):
        status = SubscriptionStatus.ACTIONABLE
        reason = "上升趋势回调已接近支撑位，收益风险比可接受。"
    elif (
        trend_state in {SymbolTrendState.UPTREND, SymbolTrendState.UPTREND_PULLBACK}
        and position.distance_to_support_pct <= 0.03
    ):
        status = SubscriptionStatus.NEAR_OPPORTUNITY
        reason = "上升趋势标的正在接近支撑区域。"

    return OpportunityRadarItem(
        symbol=normalized,
        status=status,
        trend_state=trend_state,
        trend_reason=_trend_reason(trend_state),
        current_price=position.current_price,
        support_level=position.support_level,
        reward_risk_ratio=position.reward_risk_ratio,
        reason=reason,
    )


def scan_subscription_opportunities(
    *,
    subscription_symbols: List[str],
    price_history_by_symbol: Dict[str, List[PriceBar]],
) -> List[OpportunityRadarItem]:
    """Scan only subscribed symbols and classify per-symbol opportunity state."""
    items: List[OpportunityRadarItem] = []

    for symbol in subscription_symbols:
        normalized = symbol.strip().upper()
        items.append(
            analyze_symbol_opportunity(
                normalized,
                price_history_by_symbol.get(normalized),
            )
        )

    return items


def _classify_symbol_trend(position) -> SymbolTrendState:
    if position.moving_average_20 is None or position.moving_average_50 is None:
        return SymbolTrendState.UNKNOWN
    if position.current_price < position.moving_average_50:
        return SymbolTrendState.DOWNTREND
    if position.uptrend and position.pullback_from_high_pct >= 0.03:
        return SymbolTrendState.UPTREND_PULLBACK
    if position.current_price >= position.moving_average_20 >= position.moving_average_50:
        return SymbolTrendState.UPTREND
    return SymbolTrendState.UNKNOWN


def _trend_reason(trend_state: SymbolTrendState) -> str:
    if trend_state == SymbolTrendState.UPTREND:
        return "价格位于短期和中期趋势参考位之上。"
    if trend_state == SymbolTrendState.UPTREND_PULLBACK:
        return "价格仍处于上升趋势，但已从近期高点回调。"
    if trend_state == SymbolTrendState.DOWNTREND:
        return "价格低于中期趋势参考位。"
    return "证据不足或信号混杂，无法明确判断趋势。"

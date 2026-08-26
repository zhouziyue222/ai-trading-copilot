"""Chinese display helpers for generated reports."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .enums import (
    ExecutionMode,
    ExecutionStatus,
    MarketRegime,
    SubscriptionStatus,
    SymbolTrendState,
    TradeDirection,
)


_LABELS: dict[Enum, str] = {
    SubscriptionStatus.OBSERVING: "观察中",
    SubscriptionStatus.NEAR_OPPORTUNITY: "接近机会",
    SubscriptionStatus.ACTIONABLE: "可执行",
    SubscriptionStatus.RISK_ELEVATED: "风险升高",
    SubscriptionStatus.NOT_COMPATIBLE: "不匹配",
    SymbolTrendState.UPTREND: "上升趋势",
    SymbolTrendState.DOWNTREND: "下降趋势",
    SymbolTrendState.UPTREND_PULLBACK: "上升趋势回调",
    SymbolTrendState.UNKNOWN: "趋势不明确",
    MarketRegime.BULL_MARKET: "牛市",
    MarketRegime.UPTREND: "上升趋势",
    MarketRegime.DOWNTREND: "下降趋势",
    MarketRegime.RANGE_BOUND: "震荡区间",
    MarketRegime.REVERSAL_POINT: "趋势反转点",
    MarketRegime.TRADABLE_RANGE: "可交易震荡区间",
    MarketRegime.UNCLEAR: "不明确",
    MarketRegime.WEAKENING: "走弱",
    MarketRegime.BEAR_RISK: "熊市风险",
    TradeDirection.BUY: "买入",
    TradeDirection.HOLD: "持有",
    TradeDirection.REDUCE: "减仓",
    TradeDirection.SELL: "卖出",
    TradeDirection.SHORT: "做空",
    TradeDirection.COVER: "回补",
    TradeDirection.WATCH: "观察",
    ExecutionMode.SIMULATION: "模拟",
    ExecutionMode.LIVE: "实盘",
    ExecutionStatus.ALERT_ONLY: "仅提醒",
    ExecutionStatus.PORTFOLIO_DECIDED: "组合经理已决策",
    ExecutionStatus.CONFIRMATION_REQUIRED: "需要用户确认",
}


def zh_label(value) -> str:
    """Return a Chinese label for enum values used in human reports."""
    if isinstance(value, Enum):
        return _LABELS.get(value, value.value)
    return str(value)


def zh_bool(value: bool) -> str:
    return "是" if value else "否"


def zh_join(values: Iterable[object], *, empty: str = "-") -> str:
    items = [str(value) for value in values if str(value).strip()]
    return ", ".join(items) if items else empty

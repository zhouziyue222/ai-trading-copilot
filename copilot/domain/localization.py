"""Chinese display helpers for generated reports."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .enums import (
    ExecutionMode,
    ExecutionStatus,
    MarketRegime,
    RiskRuleCode,
    RiskSeverity,
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
    TradeDirection.WATCH: "观察",
    ExecutionMode.SIMULATION: "模拟",
    ExecutionMode.LIVE: "实盘",
    ExecutionStatus.BLOCKED_BY_RISK: "被风险规则阻止",
    ExecutionStatus.ALERT_ONLY: "仅提醒",
    ExecutionStatus.SIMULATION_READY: "模拟执行就绪",
    ExecutionStatus.SIMULATED_ORDER_SUBMITTED: "模拟订单已提交",
    ExecutionStatus.SIMULATED_ORDER_FAILED: "模拟订单失败",
    ExecutionStatus.CONFIRMATION_REQUIRED: "需要用户确认",
    ExecutionStatus.LIVE_READY: "实盘候选就绪",
    RiskSeverity.INFO: "信息",
    RiskSeverity.WARN: "警告",
    RiskSeverity.BLOCK: "阻止",
    RiskRuleCode.SUBSCRIPTION_REQUIRED: "需要订阅标的",
    RiskRuleCode.MARKET_NOT_ALLOWED: "市场类型不允许",
    RiskRuleCode.FORBIDDEN_INSTRUMENT: "禁用工具",
    RiskRuleCode.STOP_LOSS_REQUIRED: "需要止损",
    RiskRuleCode.INVALIDATION_REQUIRED: "需要失效条件",
    RiskRuleCode.POSITION_LIMIT_EXCEEDED: "超过单标的仓位上限",
    RiskRuleCode.HIGH_POSITION_SIZE: "仓位高于默认区间",
    RiskRuleCode.DRAWDOWN_CAUTION: "回撤警戒",
    RiskRuleCode.DRAWDOWN_DEFENSIVE: "回撤防守",
    RiskRuleCode.DRAWDOWN_LIMIT_REACHED: "达到最大回撤限制",
    RiskRuleCode.MARKET_REGIME_WEAK: "市场环境偏弱",
    RiskRuleCode.CHASE_CONFIRMATION_REQUIRED: "追高需要确认",
    RiskRuleCode.REWARD_RISK_TOO_LOW: "收益风险比不足",
    RiskRuleCode.ACTIONABLE_STATUS_REQUIRED: "需要可执行状态",
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

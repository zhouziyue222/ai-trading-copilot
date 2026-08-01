"""Subscription opportunity radar agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    make_market_tools,
    tool_names,
)
from ai_trading_copilot.copilot.analysis.opportunity_radar import (
    analyze_symbol_opportunity,
    scan_subscription_opportunities,
)
from ai_trading_copilot.copilot.agents.llm_tools import (
    collect_tool_evidence,
    extract_json_object,
    run_tool_calling_llm,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.domain.enums import SubscriptionStatus, SymbolTrendState
from ai_trading_copilot.copilot.domain.models import (
    OpportunityRadarItem,
    PriceBar,
)
from ai_trading_copilot.copilot.domain.localization import zh_label


@dataclass
class OpportunityRadarAnalysisResult:
    item: OpportunityRadarItem
    report: str
    tool_calls: List[str]


class OpportunityRadarAgent:
    """Classifies per-symbol trend and opportunity state for subscribed symbols."""

    def __init__(self, *, llm=None, tools=None, stock_info_getter=None):
        self.llm = llm
        self.tools = tools
        self.stock_info_getter = stock_info_getter

    def analyze_symbol(
        self,
        *,
        symbol: str,
        bars: List[PriceBar] | None,
    ) -> OpportunityRadarItem:
        return analyze_symbol_opportunity(symbol, bars)

    def analyze_symbol_with_report(
        self,
        *,
        symbol: str,
        bars: List[PriceBar] | None,
        trade_date: str | None = None,
        look_back_days: int = 90,
        rag_context: str = "",
    ) -> OpportunityRadarAnalysisResult:
        fallback = self.analyze_symbol(symbol=symbol, bars=bars)
        fallback_report = _markdown_from_item(fallback)
        if self.llm is None:
            return OpportunityRadarAnalysisResult(
                item=fallback,
                report=fallback_report,
                tool_calls=[],
            )

        kwargs = {}
        if self.stock_info_getter is not None:
            kwargs["stock_info_getter"] = self.stock_info_getter
        tools = self.tools or make_market_tools(**kwargs)
        start_date, end_date = date_window(trade_date, look_back_days)
        tool_evidence, pre_calls = collect_tool_evidence(
            tools=tools,
            requests=[
                ("get_stock_info", {"symbol": symbol}),
                ("get_stock_data", {"symbol": symbol, "start_date": start_date, "end_date": end_date}),
                (
                    "get_indicators",
                    {
                        "symbol": symbol,
                        "indicator": "close_50_sma",
                        "curr_date": end_date,
                        "look_back_days": look_back_days,
                    },
                ),
            ],
        )
        prompt = (
            "You are the Opportunity Radar analyst for an AI trading copilot. "
            "Your role mirrors TradingAgents' market analyst, but narrowed to one job: "
            "decide whether this subscribed symbol is worth escalating as an entry setup. "
            "Use Futu stock information, price data, and indicators only. Prefer a clear "
            "stance over a broad market essay.\n\n"
            "高效报告格式：\n"
            "1. 结论：用一句话说明机会状态和原因。\n"
            "2. 证据：2-4 条来自工具的具体价格/趋势事实。\n"
            "3. 观察/否决原因：说明改善或失效该设置的单一关键条件。\n"
            "4. 缺失数据：列出不可用的工具证据，不要用猜测补全。\n\n"
            "Do not repeat raw tool output. Do not discuss non-subscribed symbols. "
            "Write the entire Markdown report and all JSON string values in Simplified Chinese. "
            "Enum values such as status and trend_state must still use the allowed English values. "
            "Return the Markdown report followed by one JSON object with keys: status, trend_state, reason, "
            "current_price, support_level, reward_risk_ratio, trend_reason. "
            "Allowed status values: observing, near_opportunity, actionable, "
            "risk_elevated, not_compatible. Allowed trend_state values: uptrend, "
            "downtrend, uptrend_pullback, unknown. JSON must be the final object in the response.\n\n"
            f"Symbol: {symbol}\n"
            f"Date window: {start_date} to {end_date}\n"
            f"Fallback classification: status={fallback.status.value}, "
            f"trend={fallback.trend_state.value if fallback.trend_state else 'unknown'}, "
            f"reason={fallback.reason}\n"
            f"Available tools: {tool_names(tools)}\n\n"
            f"Prefetched tool evidence:\n{tool_evidence or '-'}"
        )
        content, calls = run_tool_calling_llm(llm=self.llm, prompt=prompt, tools=tools)
        try:
            payload = extract_json_object(content)
            item = fallback.model_copy(
                update={
                    "status": SubscriptionStatus(str(payload.get("status", fallback.status.value))),
                    "trend_state": SymbolTrendState(
                        str(payload.get(
                            "trend_state",
                            fallback.trend_state.value if fallback.trend_state else "unknown",
                        ))
                    ),
                    "reason": str(payload.get("reason") or fallback.reason),
                    "trend_reason": str(payload.get("trend_reason") or fallback.trend_reason),
                    "current_price": _optional_float(
                        payload.get("current_price"),
                        fallback.current_price,
                    ),
                    "support_level": _optional_float(
                        payload.get("support_level"),
                        fallback.support_level,
                    ),
                    "reward_risk_ratio": _optional_float(
                        payload.get("reward_risk_ratio"),
                        fallback.reward_risk_ratio,
                    ),
                }
            )
        except Exception:
            item = fallback
        return OpportunityRadarAnalysisResult(
            item=item,
            report=strip_trailing_json_object(content) or fallback_report,
            tool_calls=[*pre_calls, *calls],
        )

    def scan(
        self,
        *,
        subscription_symbols: List[str],
        price_history_by_symbol: Dict[str, List[PriceBar]],
    ) -> List[OpportunityRadarItem]:
        return scan_subscription_opportunities(
            subscription_symbols=subscription_symbols,
            price_history_by_symbol=price_history_by_symbol,
        )


def _optional_float(value, fallback):
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _markdown_from_item(item: OpportunityRadarItem) -> str:
    return (
        f"# 机会雷达：{item.symbol}\n\n"
        f"- 状态：{zh_label(item.status)}\n"
        f"- 趋势：{zh_label(item.trend_state) if item.trend_state else '未知'}\n"
        f"- 当前价格：{item.current_price}\n"
        f"- 支撑位：{item.support_level}\n"
        f"- 收益风险比：{item.reward_risk_ratio}\n"
        f"- 原因：{item.reason}\n"
    )

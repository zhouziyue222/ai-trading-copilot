"""Technical position agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ai_trading_copilot.copilot.analysis.technical_position import (
    evaluate_technical_position,
)
from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    make_market_tools,
    tool_names,
)
from ai_trading_copilot.copilot.agents.llm_tools import collect_tool_evidence
from ai_trading_copilot.copilot.agents.llm_tools import run_tool_calling_llm
from ai_trading_copilot.copilot.agents.llm_tools import extract_json_object
from ai_trading_copilot.copilot.agents.llm_tools import strip_trailing_json_object
from ai_trading_copilot.copilot.domain.models import PriceBar, TechnicalPosition


@dataclass
class TechnicalAnalysisResult:
    position: TechnicalPosition
    report: str
    tool_calls: List[str]


class TechnicalPositionAgent:
    """Evaluates support, pullback, trend, reward/risk, and optional LLM report."""

    def __init__(self, *, llm=None, tools=None, stock_info_getter=None):
        self.llm = llm
        self.tools = tools
        self.stock_info_getter = stock_info_getter

    def analyze(self, symbol: str, bars: List[PriceBar]) -> TechnicalPosition:
        return evaluate_technical_position(symbol, bars)

    def analyze_with_report(
        self,
        *,
        symbol: str,
        bars: List[PriceBar] | None,
        trade_date: str | None = None,
        look_back_days: int = 90,
        rag_context: str = "",
    ) -> TechnicalAnalysisResult:
        position = self.analyze(symbol, bars) if bars else _empty_position(symbol)
        fallback = _fallback_report(position)
        if self.llm is None:
            return TechnicalAnalysisResult(position=position, report=fallback, tool_calls=[])

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
            "You are the Technical Position analyst for an AI trading copilot. "
            "This mirrors TradingAgents' market analyst indicator discipline, but focuses "
            "only on actionable technical levels for the subscribed symbol. Use available "
            "tools to inspect Futu stock info, price history, and relevant indicators.\n\n"
            "高效报告格式：\n"
            "1. 技术立场：趋势、回调质量，以及价格是否接近支撑。\n"
            "2. 关键位置：当前价格、支撑、止损参考、近期高点/目标区。\n"
            "3. 收益风险质量：用一句话说明是否可用。\n"
            "4. 数据质量：列出失败或缺失的工具证据。\n\n"
            "Avoid generic indicator education and avoid repeating raw tool output. "
            "Write the entire Markdown report and all JSON string values in Simplified Chinese. "
            "Write a concise Markdown report for the exact subscribed symbol followed by one JSON "
            "object with keys: current_price, support_level, recent_high, "
            "moving_average_20, moving_average_50, distance_to_support_pct, "
            "pullback_from_high_pct, reward_risk_ratio, uptrend. JSON must be the final object. "
            "Do not invent data.\n\n"
            f"Symbol: {symbol}\n"
            f"Date window: {start_date} to {end_date}\n"
            f"Deterministic position: current={position.current_price:.2f}, "
            f"support={position.support_level:.2f}, recent_high={position.recent_high:.2f}, "
            f"ma20={position.moving_average_20}, ma50={position.moving_average_50}, "
            f"reward_risk={position.reward_risk_ratio}\n"
            f"Available tools: {tool_names(tools)}\n\n"
            f"Prefetched tool evidence:\n{tool_evidence or '-'}"
        )
        report, calls = run_tool_calling_llm(llm=self.llm, prompt=prompt, tools=tools)
        try:
            payload = extract_json_object(report)
            position = TechnicalPosition(
                symbol=symbol,
                current_price=float(payload.get("current_price", position.current_price)),
                support_level=float(payload.get("support_level", position.support_level)),
                recent_high=float(payload.get("recent_high", position.recent_high)),
                moving_average_20=_optional_float(payload.get("moving_average_20")),
                moving_average_50=_optional_float(payload.get("moving_average_50")),
                distance_to_support_pct=float(
                    payload.get("distance_to_support_pct", position.distance_to_support_pct)
                ),
                pullback_from_high_pct=float(
                    payload.get("pullback_from_high_pct", position.pullback_from_high_pct)
                ),
                reward_risk_ratio=_optional_float(
                    payload.get("reward_risk_ratio"), position.reward_risk_ratio
                ),
                uptrend=bool(payload.get("uptrend", position.uptrend)),
            )
        except Exception:
            pass
        return TechnicalAnalysisResult(
            position=position,
            report=strip_trailing_json_object(report) or fallback,
            tool_calls=[*pre_calls, *calls],
        )


def _fallback_report(position: TechnicalPosition) -> str:
    rr = "N/A" if position.reward_risk_ratio is None else f"{position.reward_risk_ratio:.2f}"
    return (
        f"# 技术位置：{position.symbol}\n\n"
        f"- 当前价格：{position.current_price:.2f}\n"
        f"- 支撑位：{position.support_level:.2f}\n"
        f"- 近期高点：{position.recent_high:.2f}\n"
        f"- 20 日均线：{position.moving_average_20}\n"
        f"- 50 日均线：{position.moving_average_50}\n"
        f"- 距离支撑：{position.distance_to_support_pct:.1%}\n"
        f"- 从高点回调：{position.pullback_from_high_pct:.1%}\n"
        f"- 收益风险比：{rr}\n"
    )


def _optional_float(value, fallback=None):
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _empty_position(symbol: str) -> TechnicalPosition:
    return TechnicalPosition(
        symbol=symbol,
        current_price=1.0,
        support_level=1.0,
        recent_high=1.0,
        moving_average_20=None,
        moving_average_50=None,
        distance_to_support_pct=0.0,
        pullback_from_high_pct=0.0,
        reward_risk_ratio=None,
        uptrend=False,
    )

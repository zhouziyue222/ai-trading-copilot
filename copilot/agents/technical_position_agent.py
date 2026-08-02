"""Technical position agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ai_trading_copilot.copilot.analysis.technical_position import (
    evaluate_technical_context,
    evaluate_technical_position,
)
from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    make_market_tools,
    tool_names,
)
from ai_trading_copilot.copilot.agents.llm_tools import extract_json_object
from ai_trading_copilot.copilot.agents.llm_tools import strip_trailing_json_object
from ai_trading_copilot.copilot.agents.react_runner import ReActAgentRunner
from ai_trading_copilot.copilot.domain.enums import SymbolTrendState
from ai_trading_copilot.copilot.domain.models import (
    PriceBar,
    TechnicalContext,
    TechnicalDimension,
    TechnicalPosition,
)


@dataclass
class TechnicalAnalysisResult:
    position: TechnicalPosition
    report: str
    tool_calls: List[str]
    context: TechnicalContext | None = None


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
        sector_symbol = _sector_benchmark_for_symbol(symbol)
        context = (
            evaluate_technical_context(symbol, bars, sector_symbol=sector_symbol)
            if bars
            else _empty_context(symbol, sector_symbol)
        )
        fallback = _fallback_report(position)
        if self.llm is None:
            return TechnicalAnalysisResult(
                position=position,
                report=fallback,
                tool_calls=[],
                context=context,
            )

        kwargs = {}
        if self.stock_info_getter is not None:
            kwargs["stock_info_getter"] = self.stock_info_getter
        tools = self.tools or make_market_tools(**kwargs)
        start_date, end_date = date_window(trade_date, look_back_days)
        prefetch = [
            ("get_stock_info", {"symbol": symbol}),
            ("get_stock_data", {"symbol": symbol, "start_date": start_date, "end_date": end_date}),
            (
                "get_indicators",
                {
                    "symbol": symbol,
                    "indicator": "close_20_sma,close_50_sma,close_200_sma,rsi,macd,macds,macdh",
                    "curr_date": end_date,
                    "look_back_days": look_back_days,
                },
            ),
            (
                "get_indicators",
                {
                    "symbol": sector_symbol,
                    "indicator": "close_20_sma,close_50_sma,close_200_sma,rsi,macd,macds,macdh",
                    "curr_date": end_date,
                    "look_back_days": look_back_days,
                },
            ),
            (
                "get_indicators",
                {
                    "symbol": "SPY",
                    "indicator": "close_20_sma,close_50_sma,close_200_sma,rsi,macd,macds,macdh",
                    "curr_date": end_date,
                    "look_back_days": look_back_days,
                },
            ),
            (
                "get_indicators",
                {
                    "symbol": "QQQ",
                    "indicator": "close_20_sma,close_50_sma,close_200_sma,rsi,macd,macds,macdh",
                    "curr_date": end_date,
                    "look_back_days": look_back_days,
                },
            ),
        ]
        tool_result_cache = {}
        runner = ReActAgentRunner(llm=self.llm, tools=tools)
        evidence_result = runner.run(
            prompt="",
            prefetch=prefetch,
            tool_result_cache=tool_result_cache,
        )
        tool_evidence = evidence_result.prefetched_evidence
        pre_calls = evidence_result.prefetched_calls
        prompt = (
            "You are the Technical Position analyst for an AI trading copilot. "
            "This mirrors TradingAgents' market analyst indicator discipline, but focuses "
            "on actionable technical levels across three dimensions: the subscribed stock, "
            "its sector/index proxy, and the broad US market. Use available tools to inspect "
            "Futu stock info, price history, and MA/MACD/RSI indicators. The final report "
            "must explicitly state stock, sector, and broad market technical states.\n\n"
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
            "pullback_from_high_pct, reward_risk_ratio, uptrend, stock_trend_state, "
            "sector_symbol, sector_trend_state, market_symbol, market_trend_state, "
            "technical_summary, technical_warnings. Trend state values must be one of "
            "uptrend, downtrend, uptrend_pullback, unknown. JSON must be the final object. "
            "Do not invent data.\n\n"
            f"Symbol: {symbol}\n"
            f"Date window: {start_date} to {end_date}\n"
            f"Deterministic position: current={position.current_price:.2f}, "
            f"support={position.support_level:.2f}, recent_high={position.recent_high:.2f}, "
            f"ma20={position.moving_average_20}, ma50={position.moving_average_50}, "
            f"reward_risk={position.reward_risk_ratio}\n"
            f"Sector benchmark proxy: {sector_symbol}\n"
            f"Broad market proxies: SPY, QQQ\n"
            f"Available tools: {tool_names(tools)}\n\n"
            f"Prefetched tool evidence:\n{tool_evidence or '-'}"
        )
        react_result = runner.run(
            prompt=prompt,
            tool_result_cache=tool_result_cache,
        )
        report = react_result.content
        calls = react_result.tool_calls
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
            context = _context_from_payload(
                symbol=symbol,
                sector_symbol=sector_symbol,
                payload=payload,
                fallback=context,
            )
        except Exception:
            pass
        return TechnicalAnalysisResult(
            position=position,
            report=strip_trailing_json_object(report) or fallback,
            tool_calls=[*pre_calls, *calls],
            context=context,
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


def _empty_context(symbol: str, sector_symbol: str) -> TechnicalContext:
    return TechnicalContext(
        symbol=symbol.strip().upper(),
        stock=TechnicalDimension(
            scope="stock",
            symbol=symbol.strip().upper(),
            label=symbol.strip().upper(),
            trend_state=SymbolTrendState.UNKNOWN,
            reason="Missing price history.",
            data_available=False,
        ),
        sector=TechnicalDimension(
            scope="sector",
            symbol=sector_symbol,
            label=f"{sector_symbol} sector proxy",
            trend_state=SymbolTrendState.UNKNOWN,
            reason="Missing sector benchmark evidence.",
            data_available=False,
        ),
        market=TechnicalDimension(
            scope="market",
            symbol="SPY",
            label="SPY broad market proxy",
            trend_state=SymbolTrendState.UNKNOWN,
            reason="Missing broad market benchmark evidence.",
            data_available=False,
        ),
        summary="Insufficient technical evidence.",
        warnings=["missing_price_history"],
    )


def _context_from_payload(
    *,
    symbol: str,
    sector_symbol: str,
    payload: dict,
    fallback: TechnicalContext,
) -> TechnicalContext:
    stock_symbol = symbol.strip().upper()
    sector = str(payload.get("sector_symbol") or sector_symbol).strip().upper()
    market = str(payload.get("market_symbol") or "SPY").strip().upper()
    summary = str(payload.get("technical_summary") or fallback.summary or "").strip()
    warnings = payload.get("technical_warnings") or fallback.warnings
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    return TechnicalContext(
        symbol=stock_symbol,
        stock=TechnicalDimension(
            scope="stock",
            symbol=stock_symbol,
            label=stock_symbol,
            current_price=_optional_float(payload.get("current_price")),
            moving_average_20=_optional_float(payload.get("moving_average_20")),
            moving_average_50=_optional_float(payload.get("moving_average_50")),
            moving_average_200=_optional_float(payload.get("moving_average_200")),
            rsi=_optional_float(payload.get("rsi")),
            macd=_optional_float(payload.get("macd")),
            macd_signal=_optional_float(payload.get("macd_signal")),
            macd_histogram=_optional_float(payload.get("macd_histogram")),
            trend_state=_trend_state(payload.get("stock_trend_state")),
            reason=summary or fallback.stock.reason,
            data_available=True,
        ),
        sector=TechnicalDimension(
            scope="sector",
            symbol=sector,
            label=f"{sector} sector proxy",
            trend_state=_trend_state(payload.get("sector_trend_state")),
            reason=str(payload.get("sector_reason") or ""),
            data_available=True,
        ),
        market=TechnicalDimension(
            scope="market",
            symbol=market,
            label=f"{market} broad market proxy",
            trend_state=_trend_state(payload.get("market_trend_state")),
            reason=str(payload.get("market_reason") or ""),
            data_available=True,
        ),
        summary=summary,
        warnings=[str(item) for item in warnings if str(item).strip()],
    )


def _trend_state(value) -> SymbolTrendState:
    if value is None:
        return SymbolTrendState.UNKNOWN
    try:
        return SymbolTrendState(str(value).strip().lower())
    except ValueError:
        return SymbolTrendState.UNKNOWN


def _sector_benchmark_for_symbol(symbol: str) -> str:
    semiconductors = {
        "AMD",
        "AVGO",
        "INTC",
        "MU",
        "NVDA",
        "QCOM",
        "SMCI",
        "SOXX",
        "TSM",
    }
    normalized = symbol.strip().upper().split(".")[-1]
    if normalized in semiconductors:
        return "SOXX"
    if normalized in {"AAPL", "MSFT", "GOOG", "GOOGL", "META", "AMZN", "NFLX"}:
        return "QQQ"
    return "SPY"

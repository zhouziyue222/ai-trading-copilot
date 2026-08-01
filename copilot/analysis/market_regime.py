"""Deterministic market regime classification."""

from __future__ import annotations

from typing import List, Optional

from ai_trading_copilot.copilot.domain.enums import MarketRegime
from ai_trading_copilot.copilot.domain.models import MarketRegimeReport, PriceBar


def _sma(values: List[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def classify_market_regime(bars: List[PriceBar]) -> MarketRegimeReport:
    if not bars:
        raise ValueError("bars are required")

    closes = [bar.close for bar in bars]
    current = closes[-1]
    ma50 = _sma(closes, 50)
    ma200 = _sma(closes, 200)

    if ma50 is None or ma200 is None:
        return MarketRegimeReport(
            regime=MarketRegime.UNCLEAR,
            current_price=current,
            moving_average_50=ma50,
            moving_average_200=ma200,
            reason="历史数据不足，无法判断 50/200 日趋势。",
        )

    if current > ma50 > ma200:
        return MarketRegimeReport(
            regime=MarketRegime.BULL_MARKET,
            current_price=current,
            moving_average_50=ma50,
            moving_average_200=ma200,
            reason="价格位于上行的中长期趋势参考位之上。",
        )
    if current > ma50 and current > ma200:
        return MarketRegimeReport(
            regime=MarketRegime.UPTREND,
            current_price=current,
            moving_average_50=ma50,
            moving_average_200=ma200,
            reason="价格位于关键均线之上，但趋势排列不够理想。",
        )
    if current > ma200:
        return MarketRegimeReport(
            regime=MarketRegime.TRADABLE_RANGE,
            current_price=current,
            moving_average_50=ma50,
            moving_average_200=ma200,
            reason="长期趋势仍然完好，但中期趋势信号混杂。",
        )
    if current < ma50 and ma50 < ma200:
        return MarketRegimeReport(
            regime=MarketRegime.BEAR_RISK,
            current_price=current,
            moving_average_50=ma50,
            moving_average_200=ma200,
            reason="价格和中期趋势均低于长期趋势参考位。",
        )
    return MarketRegimeReport(
        regime=MarketRegime.WEAKENING,
        current_price=current,
        moving_average_50=ma50,
        moving_average_200=ma200,
        reason="市场已跌破重要趋势参考位。",
    )

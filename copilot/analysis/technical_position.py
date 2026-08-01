"""Technical position calculations for subscribed symbols."""

from __future__ import annotations

from typing import List, Optional

from ai_trading_copilot.copilot.domain.models import PriceBar, TechnicalPosition


def _sma(values: List[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def evaluate_technical_position(
    symbol: str,
    bars: List[PriceBar],
    *,
    support_window: int = 20,
) -> TechnicalPosition:
    if not bars:
        raise ValueError("bars are required")

    closes = [bar.close for bar in bars]
    lookback = bars[-support_window:] if len(bars) >= support_window else bars
    current = bars[-1].close
    support = min(bar.low for bar in lookback)
    recent_high = max(bar.high for bar in lookback)
    ma20 = _sma(closes, 20)
    ma50 = _sma(closes, 50)

    downside = max(current - support, 0)
    upside = max(recent_high - current, 0)
    reward_risk = (upside / downside) if downside > 0 else None

    uptrend = (
        ma20 is not None
        and ma50 is not None
        and current >= ma20 >= ma50
    )

    return TechnicalPosition(
        symbol=symbol.strip().upper(),
        current_price=current,
        support_level=support,
        recent_high=recent_high,
        moving_average_20=ma20,
        moving_average_50=ma50,
        distance_to_support_pct=(current - support) / support,
        pullback_from_high_pct=(recent_high - current) / recent_high,
        reward_risk_ratio=reward_risk,
        uptrend=uptrend,
    )


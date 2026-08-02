"""Technical position calculations for subscribed symbols."""

from __future__ import annotations

from typing import List, Optional

from ai_trading_copilot.copilot.domain.enums import SymbolTrendState
from ai_trading_copilot.copilot.domain.models import (
    PriceBar,
    TechnicalContext,
    TechnicalDimension,
    TechnicalPosition,
)


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


def evaluate_technical_context(
    symbol: str,
    bars: List[PriceBar],
    *,
    sector_symbol: str | None = None,
    market_symbol: str = "SPY",
) -> TechnicalContext:
    position = evaluate_technical_position(symbol, bars)
    stock_dimension = evaluate_technical_dimension(
        scope="stock",
        symbol=symbol,
        label=symbol.strip().upper(),
        bars=bars,
    )
    warnings = []
    sector_dimension = None
    if sector_symbol:
        sector_dimension = TechnicalDimension(
            scope="sector",
            symbol=sector_symbol.strip().upper(),
            label=f"{sector_symbol.strip().upper()} sector proxy",
            trend_state=SymbolTrendState.UNKNOWN,
            reason="Sector benchmark requires live tool evidence.",
            data_available=False,
        )
        warnings.append("sector benchmark not available in deterministic price history")
    market_dimension = TechnicalDimension(
        scope="market",
        symbol=market_symbol.strip().upper(),
        label=f"{market_symbol.strip().upper()} broad market proxy",
        trend_state=SymbolTrendState.UNKNOWN,
        reason="Broad market benchmark requires live tool evidence.",
        data_available=False,
    )
    warnings.append("broad market benchmark not available in deterministic price history")
    return TechnicalContext(
        symbol=position.symbol,
        stock=stock_dimension,
        sector=sector_dimension,
        market=market_dimension,
        summary=(
            "Stock trend is up"
            if position.uptrend
            else "Stock trend is not confirmed by deterministic moving averages"
        ),
        warnings=warnings,
    )


def evaluate_technical_dimension(
    *,
    scope: str,
    symbol: str,
    label: str,
    bars: List[PriceBar],
) -> TechnicalDimension:
    closes = [bar.close for bar in bars]
    current = closes[-1] if closes else None
    ma20 = _sma(closes, 20)
    ma50 = _sma(closes, 50)
    ma200 = _sma(closes, 200)
    macd, macd_signal, macd_histogram = _macd(closes)
    trend_state = _trend_state(current, ma20, ma50, ma200)
    return TechnicalDimension(
        scope=scope,
        symbol=symbol.strip().upper(),
        label=label,
        current_price=current,
        moving_average_20=ma20,
        moving_average_50=ma50,
        moving_average_200=ma200,
        rsi=_rsi(closes),
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        trend_state=trend_state,
        reason=_trend_reason(trend_state),
        data_available=bool(closes),
    )


def _rsi(values: List[float], window: int = 14) -> Optional[float]:
    if len(values) <= window:
        return None
    gains = []
    losses = []
    for previous, current in zip(values[-window - 1 : -1], values[-window:]):
        delta = current - previous
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(values: List[float], span: int) -> List[float]:
    if not values:
        return []
    multiplier = 2 / (span + 1)
    output = [values[0]]
    for value in values[1:]:
        output.append((value - output[-1]) * multiplier + output[-1])
    return output


def _macd(values: List[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(values) < 35:
        return None, None, None
    ema12 = _ema(values, 12)
    ema26 = _ema(values, 26)
    macd_values = [left - right for left, right in zip(ema12, ema26)]
    signal_values = _ema(macd_values, 9)
    macd = macd_values[-1]
    signal = signal_values[-1]
    return macd, signal, macd - signal


def _trend_state(
    current: Optional[float],
    ma20: Optional[float],
    ma50: Optional[float],
    ma200: Optional[float],
) -> SymbolTrendState:
    if current is None or ma20 is None or ma50 is None:
        return SymbolTrendState.UNKNOWN
    if current >= ma20 >= ma50 and (ma200 is None or ma50 >= ma200):
        return SymbolTrendState.UPTREND
    if current < ma50:
        return SymbolTrendState.DOWNTREND
    return SymbolTrendState.UPTREND_PULLBACK


def _trend_reason(state: SymbolTrendState) -> str:
    if state == SymbolTrendState.UPTREND:
        return "Price and moving averages support an uptrend."
    if state == SymbolTrendState.DOWNTREND:
        return "Price is below the intermediate trend reference."
    if state == SymbolTrendState.UPTREND_PULLBACK:
        return "Trend is mixed or pulling back above longer references."
    return "Insufficient technical evidence."

"""Technical position calculations for subscribed symbols."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

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


def build_compact_technical_summary(
    symbol: str,
    bars: List[PriceBar],
    *,
    curr_date: str | None = None,
    look_back_days: int = 90,
) -> dict[str, Any]:
    """Build the compact technical evidence passed to the LLM."""
    normalized_symbol = symbol.strip().upper()
    filtered = _bars_until_date(bars, curr_date)
    if not filtered:
        return {
            "symbol": normalized_symbol,
            "window": f"{int(look_back_days)}d",
            "current_price": None,
            "latest_ohlcv_5d": [],
            "returns": {"5d": None, "20d": None, "50d": None, "90d": None},
            "moving_averages": {
                "ma20": None,
                "ma50": None,
                "ma200": None,
                "price_vs_ma": {},
            },
            "trend": SymbolTrendState.UNKNOWN.value,
            "support": None,
            "resistance": None,
            "rsi": {"current": None, "state": "unknown", "percentile_90d": None},
            "macd": {"state": "unknown", "value": None, "signal": None, "histogram": None},
            "atr": {"value": None, "pct": None},
            "bollinger": {"middle": None, "upper": None, "lower": None, "position": "unknown"},
            "volume": {"state": "unknown", "latest": None, "avg20": None},
            "windows": {},
            "key_events": [],
            "data_quality": "unavailable",
            "warnings": ["empty_price_history"],
        }

    frame = _bars_to_frame(filtered)
    closes = frame["close"]
    current = float(closes.iloc[-1])
    ma20 = _series_last(closes.rolling(window=20).mean())
    ma50 = _series_last(closes.rolling(window=50).mean())
    ma200 = _series_last(closes.rolling(window=200).mean())
    ma_values = {"ma20": ma20, "ma50": ma50, "ma200": ma200}
    support_lookback = frame.tail(min(20, len(frame)))
    support = float(support_lookback["low"].min())
    resistance = float(support_lookback["high"].max())
    rsi_series = _rsi_series(closes)
    rsi_current = _series_last(rsi_series)
    macd_line, macd_signal, macd_hist = _macd_series(closes)
    atr_series = _atr_series(frame)
    bollinger_middle = _series_last(closes.rolling(window=20).mean())
    bollinger_std = _series_last(closes.rolling(window=20).std())
    bollinger_upper = (
        None if bollinger_middle is None or bollinger_std is None else bollinger_middle + 2 * bollinger_std
    )
    bollinger_lower = (
        None if bollinger_middle is None or bollinger_std is None else bollinger_middle - 2 * bollinger_std
    )
    warnings = _summary_warnings(len(frame), int(look_back_days), ma200)
    return {
        "symbol": normalized_symbol,
        "window": f"{int(look_back_days)}d",
        "current_price": _rounded(current),
        "latest_ohlcv_5d": [_bar_snapshot(row) for _, row in frame.tail(5).iterrows()],
        "returns": {
            "5d": _period_return(closes, 5),
            "20d": _period_return(closes, 20),
            "50d": _period_return(closes, 50),
            "90d": _period_return(closes, 90),
        },
        "moving_averages": {
            **{key: _rounded(value) for key, value in ma_values.items()},
            "price_vs_ma": _price_vs_ma(current, ma_values),
        },
        "trend": _trend_state(current, ma20, ma50, ma200).value,
        "support": _rounded(support),
        "resistance": _rounded(resistance),
        "rsi": {
            "current": _rounded(rsi_current),
            "state": _rsi_state(rsi_current),
            "percentile_90d": _percentile_latest(rsi_series.tail(90)),
        },
        "macd": _compact_macd(macd_line, macd_signal, macd_hist),
        "atr": _compact_atr(atr_series, current),
        "bollinger": {
            "middle": _rounded(bollinger_middle),
            "upper": _rounded(bollinger_upper),
            "lower": _rounded(bollinger_lower),
            "position": _bollinger_position(current, bollinger_lower, bollinger_upper),
        },
        "volume": _compact_volume(frame),
        "windows": {
            "5d": _window_summary(frame, 5),
            "20d": _window_summary(frame, 20),
            "60d": _window_summary(frame, 60),
            "90d": _window_summary(frame, 90),
        },
        "key_events": _key_events(
            frame=frame,
            current=current,
            support=support,
            resistance=resistance,
            ma20=ma20,
            ma50=ma50,
            rsi_current=rsi_current,
        ),
        "data_quality": "available" if len(frame) >= int(look_back_days) else "partial",
        "warnings": warnings,
    }


def _bars_until_date(bars: List[PriceBar], curr_date: str | None) -> List[PriceBar]:
    if not curr_date:
        return list(bars)
    cutoff = pd.Timestamp(curr_date)
    return [bar for bar in bars if pd.Timestamp(bar.date) <= cutoff]


def _bars_to_frame(bars: List[PriceBar]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def _bar_snapshot(row) -> dict[str, Any]:
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "open": _rounded(row["open"]),
        "high": _rounded(row["high"]),
        "low": _rounded(row["low"]),
        "close": _rounded(row["close"]),
        "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
    }


def _series_last(series: pd.Series) -> Optional[float]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[-1])


def _rounded(value: Any, digits: int = 4) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _period_return(closes: pd.Series, days: int) -> Optional[float]:
    if len(closes) <= days:
        return None
    base = float(closes.iloc[-days - 1])
    if base == 0:
        return None
    return _rounded(float(closes.iloc[-1]) / base - 1)


def _price_vs_ma(current: float, ma_values: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    output = {}
    for key, value in ma_values.items():
        output[key] = None if value in (None, 0) else _rounded(current / value - 1)
    return output


def _rsi_series(closes: pd.Series, window: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    return rsi


def _rsi_state(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value >= 70:
        return "overbought"
    if value <= 30:
        return "oversold"
    return "neutral"


def _percentile_latest(series: pd.Series) -> Optional[float]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    latest = float(cleaned.iloc[-1])
    return _rounded((cleaned <= latest).mean())


def _macd_series(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _compact_macd(
    macd_line: pd.Series,
    signal: pd.Series,
    histogram: pd.Series,
) -> dict[str, Any]:
    current_hist = _series_last(histogram)
    previous_hist = _series_previous(histogram)
    if current_hist is None:
        state = "unknown"
    elif previous_hist is not None and previous_hist <= 0 < current_hist:
        state = "bullish_cross"
    elif previous_hist is not None and previous_hist >= 0 > current_hist:
        state = "bearish_cross"
    elif current_hist > 0 and previous_hist is not None and current_hist < previous_hist:
        state = "bullish_but_weakening"
    elif current_hist > 0:
        state = "bullish"
    elif current_hist < 0 and previous_hist is not None and current_hist > previous_hist:
        state = "bearish_but_improving"
    elif current_hist < 0:
        state = "bearish"
    else:
        state = "neutral"
    return {
        "state": state,
        "value": _rounded(_series_last(macd_line)),
        "signal": _rounded(_series_last(signal)),
        "histogram": _rounded(current_hist),
    }


def _series_previous(series: pd.Series) -> Optional[float]:
    cleaned = series.dropna()
    if len(cleaned) < 2:
        return None
    return float(cleaned.iloc[-2])


def _atr_series(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window).mean()


def _compact_atr(series: pd.Series, current: float) -> dict[str, Optional[float]]:
    value = _series_last(series)
    return {
        "value": _rounded(value),
        "pct": None if value is None or current == 0 else _rounded(value / current),
    }


def _bollinger_position(
    current: float,
    lower: Optional[float],
    upper: Optional[float],
) -> str:
    if lower is None or upper is None:
        return "unknown"
    if current > upper:
        return "above_upper"
    if current < lower:
        return "below_lower"
    midpoint = (upper + lower) / 2
    return "upper_half" if current >= midpoint else "lower_half"


def _compact_volume(frame: pd.DataFrame) -> dict[str, Any]:
    latest = float(frame["volume"].iloc[-1])
    avg20 = _series_last(frame["volume"].tail(20).rolling(window=min(20, len(frame))).mean())
    if avg20 is None or avg20 == 0:
        state = "unknown"
    elif latest >= avg20 * 1.5:
        state = "above_20d_average"
    elif latest <= avg20 * 0.7:
        state = "below_20d_average"
    else:
        state = "near_20d_average"
    return {"state": state, "latest": int(latest), "avg20": _rounded(avg20)}


def _window_summary(frame: pd.DataFrame, days: int) -> dict[str, Any]:
    window = frame.tail(min(days, len(frame)))
    first_close = float(window["close"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    avg_volume = float(window["volume"].mean())
    volatility = window["close"].pct_change().std()
    return {
        "days": int(len(window)),
        "open": _rounded(window["open"].iloc[0]),
        "high": _rounded(window["high"].max()),
        "low": _rounded(window["low"].min()),
        "close": _rounded(last_close),
        "return": None if first_close == 0 else _rounded(last_close / first_close - 1),
        "volume_trend": _window_volume_trend(frame, days),
        "volatility": _rounded(volatility),
        "key_levels": {
            "low": _rounded(window["low"].min()),
            "high": _rounded(window["high"].max()),
        },
    }


def _window_volume_trend(frame: pd.DataFrame, days: int) -> str:
    current = frame["volume"].tail(min(days, len(frame))).mean()
    previous = frame["volume"].iloc[max(0, len(frame) - days * 2) : max(0, len(frame) - days)]
    if previous.empty or previous.mean() == 0:
        return "unknown"
    ratio = current / previous.mean()
    if ratio >= 1.2:
        return "rising"
    if ratio <= 0.8:
        return "falling"
    return "stable"


def _key_events(
    *,
    frame: pd.DataFrame,
    current: float,
    support: float,
    resistance: float,
    ma20: Optional[float],
    ma50: Optional[float],
    rsi_current: Optional[float],
) -> List[str]:
    events: List[str] = []
    previous = frame.iloc[-2] if len(frame) >= 2 else None
    prior = frame.iloc[:-1]
    if support and (current - support) / support <= 0.03:
        events.append("near_support")
    if not prior.empty and current >= float(prior["high"].tail(20).max()):
        events.append("breakout_20d_high")
    if not prior.empty and current <= float(prior["low"].tail(20).min()):
        events.append("new_20d_low")
    if not prior.empty and current >= float(prior["high"].tail(90).max()):
        events.append("new_90d_high")
    if not prior.empty and current <= float(prior["low"].tail(90).min()):
        events.append("new_90d_low")
    if ma20 is not None and previous is not None:
        if float(previous["close"]) >= ma20 > current:
            events.append("broke_below_ma20")
        elif float(previous["close"]) <= ma20 < current:
            events.append("reclaimed_ma20")
    if ma50 is not None and current < ma50:
        events.append("below_ma50")
    if previous is not None and float(previous["close"]) != 0:
        gap = float(frame.iloc[-1]["open"]) / float(previous["close"]) - 1
        if gap >= 0.02:
            events.append("gap_up")
        elif gap <= -0.02:
            events.append("gap_down")
    volume = _compact_volume(frame)
    if volume["state"] == "above_20d_average":
        events.append("volume_expansion")
    if rsi_current is not None and rsi_current >= 70:
        events.append("rsi_overbought")
    if rsi_current is not None and rsi_current <= 30:
        events.append("rsi_oversold")
    if resistance and current >= resistance:
        events.append("at_resistance")
    return sorted(set(events))


def _summary_warnings(
    bar_count: int,
    look_back_days: int,
    ma200: Optional[float],
) -> List[str]:
    warnings = []
    if bar_count < look_back_days:
        warnings.append(f"partial_window:{bar_count}_bars")
    if ma200 is None:
        warnings.append("insufficient_bars_for_ma200")
    return warnings


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

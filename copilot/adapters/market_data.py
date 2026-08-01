"""Copilot-local Futu OpenD market data adapter."""

from __future__ import annotations

import csv
from io import StringIO
from typing import List

import pandas as pd

from ai_trading_copilot.copilot.domain.models import PriceBar
from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_symbol


class FutuMarketDataError(RuntimeError):
    """Raised when Futu OpenD cannot return historical market data."""


def parse_price_bars_from_csv(raw: str) -> List[PriceBar]:
    """Parse OHLCV CSV text into PriceBar models."""
    data_lines = [
        line for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        return []

    reader = csv.DictReader(StringIO("\n".join(data_lines)))
    bars: List[PriceBar] = []
    for row in reader:
        date = row.get("Date") or row.get("") or row.get("date")
        if not date:
            continue
        bars.append(
            PriceBar(
                date=date,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume") or 0),
            )
        )
    return bars


def get_stock_data_text(symbol: str, start_date: str, end_date: str) -> str:
    """Return Futu OpenD daily OHLCV data as a yfinance-like CSV string."""
    data = fetch_history_dataframe(symbol=symbol, start_date=start_date, end_date=end_date)
    formatted = format_ohlcv_for_csv(data)
    futu_code = normalize_futu_symbol(symbol)
    header = (
        f"# {symbol.strip().upper()} 通过 Futu OpenD ({futu_code}) 获取的价格数据 "
        f"({start_date} 至 {end_date})\n"
        f"# 总记录数：{len(formatted)}\n\n"
    )
    return header + formatted.to_csv(index=False)


def fetch_history_dataframe(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars directly from Futu OpenD."""
    try:
        from futu import AuType, KLType, OpenQuoteContext, RET_OK
    except Exception as exc:  # pragma: no cover - depends on local SDK install
        raise FutuMarketDataError(f"futu-api is not available: {exc}") from exc

    code = normalize_futu_symbol(symbol)
    ctx = None
    try:
        ctx = _create_quote_context(OpenQuoteContext)
        all_data = []
        page_req_key = None
        while True:
            ret, data, page_req_key = ctx.request_history_kline(
                code,
                start=start_date,
                end=end_date,
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
                max_count=1000,
                page_req_key=page_req_key,
            )
            if ret != RET_OK:
                raise FutuMarketDataError(str(data))
            if data is not None and not data.empty:
                all_data.append(data)
            if page_req_key is None:
                break
    except Exception as exc:
        if isinstance(exc, FutuMarketDataError):
            raise
        raise FutuMarketDataError(
            f"Futu OpenD history request failed for {code}: {exc}"
        ) from exc
    finally:
        if ctx is not None:
            ctx.close()

    if not all_data:
        return pd.DataFrame(columns=["time_key", "open", "high", "low", "close", "volume"])
    return pd.concat(all_data, ignore_index=True)


def format_ohlcv_for_csv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize Futu history rows to Date/Open/High/Low/Close/Volume CSV columns."""
    if data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    frame = data.copy()
    date_source = frame["time_key"] if "time_key" in frame else frame.iloc[:, 0]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(date_source).dt.strftime("%Y-%m-%d"),
            "Open": pd.to_numeric(frame["open"], errors="coerce"),
            "High": pd.to_numeric(frame["high"], errors="coerce"),
            "Low": pd.to_numeric(frame["low"], errors="coerce"),
            "Close": pd.to_numeric(frame["close"], errors="coerce"),
            "Volume": pd.to_numeric(frame.get("volume", 0), errors="coerce").fillna(0),
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])


def _create_quote_context(context_cls):
    import os

    host = os.getenv("FUTU_OPEND_HOST", "127.0.0.1")
    port = int(os.getenv("FUTU_OPEND_PORT", "11111"))
    try:
        return context_cls(host=host, port=port, ai_type=1)
    except TypeError:
        return context_cls(host=host, port=port)


class FutuMarketDataAdapter:
    """Fetch OHLCV data directly through Futu OpenD."""

    def get_price_history(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> List[PriceBar]:
        raw = get_stock_data_text(symbol, start_date, end_date)
        return parse_price_bars_from_csv(raw)

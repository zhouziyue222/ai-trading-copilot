from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ai_trading_copilot.copilot.adapters.runtime import ensure_yfinance_proxy


def get_fundamentals_yfinance(ticker: str, curr_date: str | None = None) -> str:
    """Return company fundamentals from yfinance as formatted text."""
    yf = _import_yfinance()
    symbol = ticker.strip().upper()
    info = _call_yfinance(lambda: yf.Ticker(symbol).info)
    if not info:
        return f"未找到标的 {symbol} 的基本面数据。"

    fields = [
        ("名称", info.get("longName")),
        ("板块", info.get("sector")),
        ("行业", info.get("industry")),
        ("市值", info.get("marketCap")),
        ("市盈率 (TTM)", info.get("trailingPE")),
        ("远期市盈率", info.get("forwardPE")),
        ("PEG 比率", info.get("pegRatio")),
        ("市净率", info.get("priceToBook")),
        ("每股收益 (TTM)", info.get("trailingEps")),
        ("远期每股收益", info.get("forwardEps")),
        ("股息率", info.get("dividendYield")),
        ("Beta", info.get("beta")),
        ("52 周高点", info.get("fiftyTwoWeekHigh")),
        ("52 周低点", info.get("fiftyTwoWeekLow")),
        ("50 日均价", info.get("fiftyDayAverage")),
        ("200 日均价", info.get("twoHundredDayAverage")),
        ("收入 (TTM)", info.get("totalRevenue")),
        ("毛利润", info.get("grossProfits")),
        ("EBITDA", info.get("ebitda")),
        ("净利润", info.get("netIncomeToCommon")),
        ("利润率", info.get("profitMargins")),
        ("营业利润率", info.get("operatingMargins")),
        ("净资产收益率", info.get("returnOnEquity")),
        ("资产收益率", info.get("returnOnAssets")),
        ("债务权益比", info.get("debtToEquity")),
        ("流动比率", info.get("currentRatio")),
        ("账面价值", info.get("bookValue")),
        ("自由现金流", info.get("freeCashflow")),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value is not None]
    if not lines:
        return f"未找到标的 {symbol} 的基本面数据。"
    return _header(f"{symbol} 公司基本面") + "\n".join(lines)


def get_balance_sheet_yfinance(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    symbol = ticker.strip().upper()
    yf = _import_yfinance()
    ticker_obj = yf.Ticker(symbol)
    data = _call_yfinance(
        lambda: ticker_obj.quarterly_balance_sheet
        if freq.lower() == "quarterly"
        else ticker_obj.balance_sheet
    )
    return _format_statement(data, symbol, freq, curr_date, "资产负债表")


def get_cashflow_yfinance(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    symbol = ticker.strip().upper()
    yf = _import_yfinance()
    ticker_obj = yf.Ticker(symbol)
    data = _call_yfinance(
        lambda: ticker_obj.quarterly_cashflow
        if freq.lower() == "quarterly"
        else ticker_obj.cashflow
    )
    return _format_statement(data, symbol, freq, curr_date, "现金流量表")


def get_income_statement_yfinance(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    symbol = ticker.strip().upper()
    yf = _import_yfinance()
    ticker_obj = yf.Ticker(symbol)
    data = _call_yfinance(
        lambda: ticker_obj.quarterly_income_stmt
        if freq.lower() == "quarterly"
        else ticker_obj.income_stmt
    )
    return _format_statement(data, symbol, freq, curr_date, "利润表")


def is_unusable_yfinance_text(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith("no fundamentals data found")
        or lowered.startswith("no balance sheet data found")
        or lowered.startswith("no cash flow data found")
        or lowered.startswith("no income statement data found")
        or lowered.startswith("未找到标的")
        or lowered.startswith("error retrieving")
        or "too many requests" in lowered
        or "rate limit" in lowered
        or "429" in lowered
    )


def _import_yfinance():
    ensure_yfinance_proxy()
    import yfinance as yf

    return yf


def _call_yfinance(fetcher):
    try:
        return fetcher()
    except Exception as exc:
        raise RuntimeError(f"yfinance 请求失败：{exc}") from exc


def _format_statement(
    data: Any,
    symbol: str,
    freq: str,
    curr_date: str | None,
    title: str,
) -> str:
    if data is None:
        return f"未找到标的 {symbol} 的 {title} 数据。"
    frame = pd.DataFrame(data)
    frame = _filter_financials_by_date(frame, curr_date)
    if frame.empty:
        return f"未找到标的 {symbol} 的 {title} 数据。"
    return _header(f"{symbol} {title} 数据（{freq}）") + frame.to_csv()


def _filter_financials_by_date(data: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


def _header(title: str) -> str:
    return f"# {title}\n# 数据获取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

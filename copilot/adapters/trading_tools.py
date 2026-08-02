"""Copilot-local wrappers around market, news, and fundamentals tools."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Iterable, List

import pandas as pd

from ai_trading_copilot.copilot.adapters.market_data import (
    FutuMarketDataError,
    fetch_history_dataframe,
    format_ohlcv_for_csv,
    get_stock_data_text as get_futu_stock_data_text,
)
from ai_trading_copilot.copilot.adapters.stock_info import (
    format_stock_info,
    get_futu_stock_info,
)


def get_stock_data_text(symbol: str, start_date: str, end_date: str) -> str:
    try:
        return get_futu_stock_data_text(symbol, start_date, end_date)
    except Exception as exc:
        return f"工具 get_stock_data 失败：{exc}"


def get_indicator_text(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    outputs = []
    for item in [part.strip().lower() for part in indicator.split(",") if part.strip()]:
        try:
            outputs.append(_get_single_indicator_text(symbol, item, curr_date, look_back_days))
        except Exception as exc:
            outputs.append(f"工具 get_indicators 处理 {item} 失败：{exc}")
    return "\n\n".join(outputs)


def _get_single_indicator_text(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    supported = {
        "close_50_sma": "收盘价 50 日简单移动平均线。",
        "close_200_sma": "收盘价 200 日简单移动平均线。",
        "close_20_sma": "收盘价 20 日简单移动平均线。",
        "rsi": "相对强弱指数。",
        "macd": "MACD 快慢线差值。",
        "macds": "MACD 信号线。",
        "macdh": "MACD 柱状图。",
    }
    if indicator not in supported:
        raise ValueError(
            f"不支持指标 {indicator}。可选指标：{sorted(supported)}"
        )

    curr_dt = pd.Timestamp(curr_date)
    start_dt = curr_dt - pd.Timedelta(days=max(int(look_back_days), 260))
    data = fetch_history_dataframe(
        symbol=symbol,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=curr_dt.strftime("%Y-%m-%d"),
    )
    frame = format_ohlcv_for_csv(data)
    if frame.empty:
        raise FutuMarketDataError("Futu OpenD 未返回价格历史。")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame[frame["Date"] <= curr_dt].copy()
    frame["close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if indicator.endswith("_sma"):
        window = int(indicator.split("_")[1])
        frame[indicator] = frame["close"].rolling(window=window).mean()
    elif indicator == "rsi":
        delta = frame["close"].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, pd.NA)
        frame[indicator] = 100 - (100 / (1 + rs))
    else:
        ema_12 = frame["close"].ewm(span=12, adjust=False).mean()
        ema_26 = frame["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        values = {
            "macd": macd,
            "macds": signal,
            "macdh": macd - signal,
        }
        frame[indicator] = values[indicator]

    start_output = curr_dt - pd.Timedelta(days=int(look_back_days))
    output = frame[frame["Date"] >= start_output][["Date", indicator]]
    lines = [
        f"## {indicator} 指标值（{start_output.strftime('%Y-%m-%d')} "
        f"至 {curr_dt.strftime('%Y-%m-%d')}，来源：Futu OpenD）",
        "",
    ]
    if output.empty:
        lines.append("指定日期范围内没有可用数据。")
    else:
        for _, row in output.iterrows():
            value = row[indicator]
            formatted = "N/A" if pd.isna(value) else f"{float(value):.4f}"
            lines.append(f"{row['Date'].strftime('%Y-%m-%d')}: {formatted}")
    lines.extend(["", supported[indicator]])
    return "\n".join(lines)


def get_news_text(ticker: str, start_date: str, end_date: str) -> str:
    from ai_trading_copilot.copilot.adapters.yfinance_news import get_news_yfinance

    return get_news_yfinance(ticker, start_date, end_date)


def get_global_news_text(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
    from ai_trading_copilot.copilot.adapters.yfinance_news import get_global_news_yfinance

    return get_global_news_yfinance(curr_date, look_back_days, limit)


def get_fundamentals_text(ticker: str, curr_date: str) -> str:
    from ai_trading_copilot.copilot.adapters import finnhub, yfinance_fundamentals

    return _with_finnhub_fallback(
        lambda: yfinance_fundamentals.get_fundamentals_yfinance(ticker, curr_date),
        lambda: finnhub.get_basic_fundamentals_text(ticker),
    )


def get_balance_sheet_text(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    from ai_trading_copilot.copilot.adapters import finnhub, yfinance_fundamentals

    return _with_finnhub_fallback(
        lambda: yfinance_fundamentals.get_balance_sheet_yfinance(ticker, freq, curr_date),
        lambda: finnhub.get_balance_sheet_text(ticker, freq, curr_date),
    )


def get_cashflow_text(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    from ai_trading_copilot.copilot.adapters import finnhub, yfinance_fundamentals

    return _with_finnhub_fallback(
        lambda: yfinance_fundamentals.get_cashflow_yfinance(ticker, freq, curr_date),
        lambda: finnhub.get_cashflow_text(ticker, freq, curr_date),
    )


def get_income_statement_text(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    from ai_trading_copilot.copilot.adapters import finnhub, yfinance_fundamentals

    return _with_finnhub_fallback(
        lambda: yfinance_fundamentals.get_income_statement_yfinance(ticker, freq, curr_date),
        lambda: finnhub.get_income_statement_text(ticker, freq, curr_date),
    )


def get_stock_info_text(symbol: str) -> str:
    return format_stock_info(get_futu_stock_info(symbol))


def make_market_tools(
    *,
    stock_info_getter: Callable[[str], str] = get_stock_info_text,
) -> List:
    from langchain_core.tools import tool

    @tool
    def get_stock_info(symbol: str) -> str:
        """Retrieve current Futu quote and snapshot information for a ticker."""
        return stock_info_getter(symbol)

    @tool
    def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
        """Retrieve OHLCV stock price data for a ticker and date range."""
        return get_stock_data_text(symbol, start_date, end_date)

    @tool
    def get_indicators(
        symbol: str,
        indicator: str,
        curr_date: str,
        look_back_days: int = 30,
    ) -> str:
        """Retrieve a technical indicator for a ticker."""
        return get_indicator_text(symbol, indicator, curr_date, look_back_days)

    return [get_stock_info, get_stock_data, get_indicators]


def make_fundamental_news_tools() -> List:
    from langchain_core.tools import tool

    @tool
    def get_news(ticker: str, start_date: str, end_date: str) -> str:
        """Retrieve company-specific news for a ticker."""
        return get_news_text(ticker, start_date, end_date)

    @tool
    def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
        """Retrieve broad macro and market news."""
        return get_global_news_text(curr_date, look_back_days, limit)

    @tool
    def get_fundamentals(ticker: str, curr_date: str) -> str:
        """Retrieve comprehensive company fundamentals."""
        return get_fundamentals_text(ticker, curr_date)

    @tool
    def get_balance_sheet(
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        """Retrieve balance sheet data."""
        return get_balance_sheet_text(ticker, freq, curr_date)

    @tool
    def get_cashflow(
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        """Retrieve cash flow statement data."""
        return get_cashflow_text(ticker, freq, curr_date)

    @tool
    def get_income_statement(
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        """Retrieve income statement data."""
        return get_income_statement_text(ticker, freq, curr_date)

    return [
        get_news,
        get_global_news,
        get_fundamentals,
        get_balance_sheet,
        get_cashflow,
        get_income_statement,
    ]


def _safe_tool(fetcher: Callable[[], str]) -> str:
    try:
        return str(fetcher())
    except Exception as exc:
        return f"Tool data unavailable: {exc}"


def make_news_sentiment_tools() -> List:
    from langchain_core.tools import tool

    @tool
    def get_company_news(ticker: str, start_date: str, end_date: str) -> str:
        """Retrieve Finnhub company news for a ticker."""
        from ai_trading_copilot.copilot.adapters import finnhub

        return _safe_tool(lambda: finnhub.get_company_news_text(ticker, start_date, end_date))

    @tool
    def get_market_news(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
        """Retrieve Finnhub broad market news."""
        from ai_trading_copilot.copilot.adapters import finnhub

        return _safe_tool(lambda: finnhub.get_global_news_text(curr_date, look_back_days, limit))

    @tool
    def get_news_sentiment(ticker: str) -> str:
        """Retrieve Finnhub news sentiment for a ticker."""
        from ai_trading_copilot.copilot.adapters import finnhub

        return _safe_tool(lambda: finnhub.get_news_sentiment_text(ticker))

    @tool
    def get_social_sentiment(ticker: str) -> str:
        """Retrieve Finnhub social sentiment for a ticker."""
        from ai_trading_copilot.copilot.adapters import finnhub

        return _safe_tool(lambda: finnhub.get_social_sentiment_text(ticker))

    @tool
    def get_earnings_calendar(ticker: str, start_date: str, end_date: str) -> str:
        """Retrieve Finnhub earnings calendar rows for a ticker."""
        from ai_trading_copilot.copilot.adapters import finnhub

        return _safe_tool(lambda: finnhub.get_earnings_calendar_text(ticker, start_date, end_date))

    return [
        get_company_news,
        get_market_news,
        get_news_sentiment,
        get_social_sentiment,
        get_earnings_calendar,
    ]


def make_fundamental_tools(*, rag_retriever=None) -> List:
    from langchain_core.tools import tool

    @tool
    def get_fundamentals(ticker: str, curr_date: str) -> str:
        """Retrieve comprehensive company fundamentals."""
        return get_fundamentals_text(ticker, curr_date)

    @tool
    def get_balance_sheet(
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        """Retrieve balance sheet data."""
        return get_balance_sheet_text(ticker, freq, curr_date)

    @tool
    def get_cashflow(
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        """Retrieve cash flow statement data."""
        return get_cashflow_text(ticker, freq, curr_date)

    @tool
    def get_income_statement(
        ticker: str,
        freq: str = "quarterly",
        curr_date: str | None = None,
    ) -> str:
        """Retrieve income statement data."""
        return get_income_statement_text(ticker, freq, curr_date)

    tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

    if rag_retriever is not None:

        @tool
        def retrieve_fundamental_rag(ticker: str, query: str, limit: int = 5) -> str:
            """Retrieve company fundamental RAG documents."""
            docs = rag_retriever.retrieve_fundamental_rag_context(
                symbol=ticker,
                tags=["fundamentals"],
                query=query,
                limit=limit,
            )
            if not docs:
                return "No fundamental RAG documents found."
            lines = [f"# Fundamental RAG for {ticker}", ""]
            for item in docs:
                title = getattr(item, "title", "") or getattr(item, "id", "")
                text = str(getattr(item, "text", "")).replace("\n", " ")
                lines.append(f"- {title}: {text[:800]}")
            return "\n".join(lines)

        tools.append(retrieve_fundamental_rag)

    return tools


def date_window(curr_date: str | None, look_back_days: int) -> tuple[str, str]:
    end = date.fromisoformat(curr_date) if curr_date else date.today()
    start = end - timedelta(days=look_back_days)
    return start.isoformat(), end.isoformat()


def tool_names(tools: Iterable) -> str:
    return ", ".join(getattr(tool_item, "name", str(tool_item)) for tool_item in tools)


def _with_finnhub_fallback(yfinance_fetcher: Callable[[], str], finnhub_fetcher: Callable[[], str]) -> str:
    from ai_trading_copilot.copilot.adapters.yfinance_fundamentals import is_unusable_yfinance_text

    try:
        result = str(yfinance_fetcher())
    except Exception as exc:
        yfinance_error = f"yfinance 不可用：{exc}"
    else:
        if not is_unusable_yfinance_text(result):
            return result
        yfinance_error = result

    try:
        return str(finnhub_fetcher())
    except Exception as exc:
        return f"{yfinance_error}\n\nFinnhub 备用数据不可用：{exc}"

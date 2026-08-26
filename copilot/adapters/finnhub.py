"""Optional Finnhub fallback for news and basic fundamentals."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests

from ai_trading_copilot.copilot.adapters.runtime import load_project_env_files


_BASE_URL = "https://finnhub.io/api/v1"
_TIMEOUT_SECONDS = 15


class FinnhubUnavailableError(RuntimeError):
    """Raised when Finnhub is not configured or cannot serve data."""


def get_company_news_text(ticker: str, start_date: str, end_date: str) -> str:
    data = _get_json(
        "/company-news",
        {"symbol": ticker.strip().upper(), "from": start_date, "to": end_date},
    )
    if not isinstance(data, list) or not data:
        return f"未找到 {ticker} 在 {start_date} 至 {end_date} 期间的 Finnhub 新闻。"
    return (
        f"## {ticker.strip().upper()} 新闻（来源：Finnhub，{start_date} 至 {end_date}）\n\n"
        + "\n".join(_format_news_item_with_metadata(item) for item in data[:10])
    )


def get_global_news_text(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
    data = _get_json("/news", {"category": "general"})
    if not isinstance(data, list) or not data:
        return f"未找到 {curr_date} 的 Finnhub 全球市场新闻。"

    cutoff = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    filtered = [
        item for item in data
        if _item_datetime(item) is None or _item_datetime(item) >= cutoff
    ]
    selected = filtered[:limit]
    if not selected:
        return f"未找到 {curr_date} 的 Finnhub 全球市场新闻。"
    start_date = cutoff.strftime("%Y-%m-%d")
    return (
        f"## 全球市场新闻（来源：Finnhub，{start_date} 至 {curr_date}）\n\n"
        + "\n".join(_format_news_item_with_metadata(item) for item in selected)
    )


def get_news_sentiment_text(ticker: str) -> str:
    symbol = ticker.strip().upper()
    data = _get_json("/news-sentiment", {"symbol": symbol})
    if not isinstance(data, dict) or not data:
        return f"No Finnhub news sentiment data found for {symbol}."
    lines = [f"# {symbol} Finnhub news sentiment", ""]
    for key in [
        "buzz",
        "companyNewsScore",
        "sectorAverageBullishPercent",
        "sectorAverageNewsScore",
        "sentiment",
    ]:
        value = data.get(key)
        if value not in (None, "", {}, []):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def get_social_sentiment_text(ticker: str) -> str:
    symbol = ticker.strip().upper()
    data = _get_json("/stock/social-sentiment", {"symbol": symbol})
    if not isinstance(data, dict) or not data:
        return f"No Finnhub social sentiment data found for {symbol}."
    lines = [f"# {symbol} Finnhub social sentiment", ""]
    for source in ["reddit", "twitter"]:
        rows = data.get(source, [])
        lines.append(f"## {source}")
        if not rows:
            lines.append("No rows.")
            continue
        for row in rows[:10]:
            lines.append(str(_clean_json_dict(row)))
    return "\n".join(lines)


def get_earnings_calendar_text(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    symbol = ticker.strip().upper()
    data = _get_json(
        "/calendar/earnings",
        {"symbol": symbol, "from": start_date, "to": end_date},
    )
    rows = data.get("earningsCalendar", []) if isinstance(data, dict) else []
    if not rows:
        return f"No Finnhub earnings calendar rows found for {symbol} from {start_date} to {end_date}."
    lines = [f"# {symbol} Finnhub earnings calendar ({start_date} to {end_date})", ""]
    for row in rows[:10]:
        lines.append(str(_clean_json_dict(row)))
    return "\n".join(lines)


def get_basic_fundamentals_text(ticker: str) -> str:
    symbol = ticker.strip().upper()
    profile = _get_json("/stock/profile2", {"symbol": symbol})
    metrics = _get_json("/stock/metric", {"symbol": symbol, "metric": "all"})
    metric_values = metrics.get("metric", {}) if isinstance(metrics, dict) else {}
    if not profile and not metric_values:
        return f"未找到标的 {symbol} 的 Finnhub 基本面数据。"

    lines = [f"# {symbol} 公司基本面（来源：Finnhub）", ""]
    if isinstance(profile, dict):
        for label, key in [
            ("名称", "name"),
            ("交易所", "exchange"),
            ("行业", "finnhubIndustry"),
            ("IPO 日期", "ipo"),
            ("市值", "marketCapitalization"),
            ("流通股数", "shareOutstanding"),
            ("货币", "currency"),
        ]:
            value = profile.get(key)
            if value not in (None, ""):
                lines.append(f"{label}: {value}")
    for label, key in [
        ("市盈率 (TTM)", "peNormalizedAnnual"),
        ("市销率 (TTM)", "psTTM"),
        ("市净率", "pbAnnual"),
        ("EPS 三年增长", "epsGrowth3Y"),
        ("收入三年增长", "revenueGrowth3Y"),
        ("毛利率 TTM", "grossMarginTTM"),
        ("营业利润率 TTM", "operatingMarginTTM"),
        ("净利率 TTM", "netProfitMarginTTM"),
        ("年度流动比率", "currentRatioAnnual"),
        ("年度债务/权益", "totalDebt/totalEquityAnnual"),
    ]:
        value = metric_values.get(key)
        if value not in (None, ""):
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def get_balance_sheet_text(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _get_financial_statement_text(ticker, "bs", freq, curr_date, "资产负债表")


def get_cashflow_text(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _get_financial_statement_text(ticker, "cf", freq, curr_date, "现金流量表")


def get_income_statement_text(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _get_financial_statement_text(ticker, "ic", freq, curr_date, "利润表")


def _get_json(path: str, params: Dict[str, Any]) -> Any:
    load_project_env_files()
    token = os.getenv("FINNHUB_API_KEY", "").strip()
    if not token:
        raise FinnhubUnavailableError("FINNHUB_API_KEY is not set")
    response = requests.get(
        f"{_BASE_URL}{path}",
        params={**params, "token": token},
        timeout=_TIMEOUT_SECONDS,
    )
    if response.status_code == 429:
        raise FinnhubUnavailableError("Finnhub rate limit exceeded")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise FinnhubUnavailableError(f"Finnhub request failed: {exc}") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise FinnhubUnavailableError(f"Finnhub returned a non-JSON response: {exc}") from exc


def _get_financial_statement_text(
    ticker: str,
    statement_key: str,
    freq: str,
    curr_date: str | None,
    title: str,
) -> str:
    symbol = ticker.strip().upper()
    finnhub_freq = "quarterly" if freq.lower() == "quarterly" else "annual"
    data = _get_json(
        "/stock/financials-reported",
        {"symbol": symbol, "freq": finnhub_freq},
    )
    reports = data.get("data", []) if isinstance(data, dict) else []
    if curr_date:
        reports = [report for report in reports if str(report.get("endDate", "")) <= curr_date]
    if not reports:
        return f"未找到标的 {symbol} 的 Finnhub {title} 数据。"

    rows: List[Dict[str, Any]] = []
    for report in reports[:4]:
        period = report.get("endDate") or report.get("year")
        for item in report.get("report", {}).get(statement_key, []):
            rows.append(
                {
                    "period": period,
                    "concept": item.get("concept"),
                    "label": item.get("label"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                }
            )
    if not rows:
        return f"未找到标的 {symbol} 的 Finnhub {title} 数据。"

    header = f"# {symbol} {title} 数据（{finnhub_freq}，来源：Finnhub）\n"
    header += f"# 数据获取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + pd.DataFrame(rows).to_csv(index=False)


def _format_news_item(item: Dict[str, Any]) -> str:
    title = item.get("headline") or item.get("title") or "无标题"
    source = item.get("source") or "Finnhub"
    summary = item.get("summary") or ""
    url = item.get("url") or ""
    lines = [f"### {title}（来源：{source}）"]
    if summary:
        lines.append(str(summary))
    if url:
        lines.append(f"链接：{url}")
    return "\n".join(lines) + "\n"


def _format_news_item_with_metadata(item: Dict[str, Any]) -> str:
    title = item.get("headline") or item.get("title") or "unknown"
    source = item.get("source") or "Finnhub"
    summary = item.get("summary") or "unknown"
    url = item.get("url") or "unknown"
    published_at = _item_datetime(item)
    lines = [
        f"### {title}",
        f"published_at: {published_at.isoformat() if published_at else 'unknown'}",
        f"source: {source or 'unknown'}",
        f"title: {title or 'unknown'}",
        f"url: {url or 'unknown'}",
        f"summary: {summary}",
    ]
    return "\n".join(lines) + "\n"


def _item_datetime(item: Dict[str, Any]):
    value = item.get("datetime")
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError):
        return None


def _clean_json_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in row.items() if value not in (None, "")}

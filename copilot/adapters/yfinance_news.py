"""Copilot-local yfinance news access with request coalescing and cooldown."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Tuple

from dateutil.relativedelta import relativedelta

from ai_trading_copilot.copilot.adapters.runtime import ensure_yfinance_proxy


_SUCCESS_TTL_SECONDS = 30 * 60
_RATE_LIMIT_TTL_SECONDS = 5 * 60
_RETRY_DELAYS_SECONDS = (10.0, 30.0, 90.0)
_COMPANY_NEWS_COUNT = 10


@dataclass
class _CacheEntry:
    value: str
    expires_at: float


_CACHE: Dict[Tuple[Any, ...], _CacheEntry] = {}
_COOLDOWN_UNTIL = 0.0


def get_news_yfinance(ticker: str, start_date: str, end_date: str) -> str:
    """Return company news from Yahoo Finance via yfinance."""
    normalized = ticker.strip().upper()
    key = ("company", normalized, start_date, end_date)
    return _cached_fetch(
        key,
        lambda: _format_company_news(
            ticker=normalized,
            start_date=start_date,
            end_date=end_date,
            articles=_fetch_company_news_from_yahoo(normalized),
        ),
        fallback_fetcher=lambda: _fetch_company_news_from_finnhub(
            normalized,
            start_date,
            end_date,
        ),
        error_prefix=f"获取 {normalized} 新闻失败",
    )


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 5,
) -> str:
    """Return broad market news from Yahoo Finance via one yfinance search."""
    key = ("global", curr_date, int(look_back_days), int(limit))
    return _cached_fetch(
        key,
        lambda: _format_global_news(
            curr_date=curr_date,
            look_back_days=look_back_days,
            articles=_fetch_global_news_from_yahoo(limit),
            limit=limit,
        ),
        fallback_fetcher=lambda: _fetch_global_news_from_finnhub(
            curr_date,
            look_back_days,
            limit,
        ),
        error_prefix="获取全球市场新闻失败",
    )


def clear_yfinance_news_cache() -> None:
    """Clear process-local yfinance news cache and cooldown; intended for tests."""
    global _COOLDOWN_UNTIL
    _CACHE.clear()
    _COOLDOWN_UNTIL = 0.0


def _cached_fetch(
    key: Tuple[Any, ...],
    fetcher: Callable[[], str],
    *,
    error_prefix: str,
    fallback_fetcher: Callable[[], str] | None = None,
) -> str:
    global _COOLDOWN_UNTIL

    cached = _get_cache(key)
    if cached is not None:
        return cached

    now = time.monotonic()
    if now < _COOLDOWN_UNTIL:
        fallback = _fetch_fallback(fallback_fetcher)
        if fallback is not None:
            _set_cache(key, fallback, time.monotonic() + _SUCCESS_TTL_SECONDS)
            return fallback
        message = (
            f"{error_prefix}：Yahoo Finance 触发限流后正在冷却。"
            f"请在 {int(_COOLDOWN_UNTIL - now)} 秒后重试。"
        )
        _set_cache(key, message, _COOLDOWN_UNTIL)
        return message

    try:
        value = _retry_yfinance(fetcher)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            _COOLDOWN_UNTIL = time.monotonic() + _RATE_LIMIT_TTL_SECONDS
            fallback = _fetch_fallback(fallback_fetcher)
            if fallback is not None:
                _set_cache(key, fallback, time.monotonic() + _SUCCESS_TTL_SECONDS)
                return fallback
            value = f"{error_prefix}: {exc}"
            _set_cache(key, value, _COOLDOWN_UNTIL)
            return value
        value = f"{error_prefix}: {exc}"
    _set_cache(key, value, time.monotonic() + _SUCCESS_TTL_SECONDS)
    return value


def _retry_yfinance(fetcher: Callable[[], str]) -> str:
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS_SECONDS)):
        if delay:
            time.sleep(delay)
        try:
            return fetcher()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == len(_RETRY_DELAYS_SECONDS):
                raise
    raise RuntimeError("unreachable yfinance retry state")


def _fetch_company_news_from_yahoo(ticker: str) -> List[dict]:
    ensure_yfinance_proxy()
    import yfinance as yf

    search = yf.Search(
        query=f"{ticker} stock news",
        max_results=0,
        news_count=_COMPANY_NEWS_COUNT,
        lists_count=0,
        include_cb=False,
        enable_fuzzy_query=False,
        recommended=0,
    )
    return list(search.news or [])


def _fetch_global_news_from_yahoo(limit: int) -> List[dict]:
    ensure_yfinance_proxy()
    import yfinance as yf

    search = yf.Search(
        query="stock market economy Federal Reserve inflation",
        max_results=0,
        news_count=max(1, int(limit)),
        lists_count=0,
        include_cb=False,
        enable_fuzzy_query=False,
        recommended=0,
    )
    return list(search.news or [])


def _fetch_company_news_from_finnhub(ticker: str, start_date: str, end_date: str) -> str:
    from ai_trading_copilot.copilot.adapters.finnhub import get_company_news_text

    return get_company_news_text(ticker, start_date, end_date)


def _fetch_global_news_from_finnhub(curr_date: str, look_back_days: int, limit: int) -> str:
    from ai_trading_copilot.copilot.adapters.finnhub import get_global_news_text

    return get_global_news_text(curr_date, look_back_days, limit)


def _fetch_fallback(fallback_fetcher: Callable[[], str] | None) -> str | None:
    if fallback_fetcher is None:
        return None
    try:
        return fallback_fetcher()
    except Exception:
        return None


def _format_company_news(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    articles: Iterable[dict],
) -> str:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    lines: List[str] = []
    for article in articles:
        data = _extract_article_data(article)
        pub_date = data.get("pub_date")
        if pub_date is not None:
            pub_date_naive = pub_date.replace(tzinfo=None)
            if not (start_dt <= pub_date_naive <= end_dt + relativedelta(days=1)):
                continue
        lines.append(_format_article(data))

    if not lines:
        return f"未找到 {ticker} 在 {start_date} 至 {end_date} 期间的新闻。"
    return f"## {ticker} 新闻（{start_date} 至 {end_date}）\n\n" + "\n".join(lines)


def _format_global_news(
    *,
    curr_date: str,
    look_back_days: int,
    articles: Iterable[dict],
    limit: int,
) -> str:
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    lines: List[str] = []
    for article in list(articles)[:limit]:
        data = _extract_article_data(article)
        pub_date = data.get("pub_date")
        if pub_date is not None:
            pub_date_naive = pub_date.replace(tzinfo=None)
            if pub_date_naive > curr_dt + relativedelta(days=1):
                continue
        lines.append(_format_article(data))

    start_date = start_dt.strftime("%Y-%m-%d")
    if not lines:
        return f"未找到 {curr_date} 的全球市场新闻。"
    return f"## 全球市场新闻（{start_date} 至 {curr_date}）\n\n" + "\n".join(lines)


def _extract_article_data(article: dict) -> dict:
    if "content" in article:
        content = article["content"]
        provider = content.get("provider", {})
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        return {
            "title": content.get("title", "无标题"),
            "summary": content.get("summary", ""),
            "publisher": provider.get("displayName", "未知"),
            "link": url_obj.get("url", ""),
            "pub_date": _parse_pub_date(content.get("pubDate")),
        }

    provider = article.get("provider", "未知")
    if isinstance(provider, dict):
        provider = provider.get("displayName", "未知")
    return {
        "title": article.get("title", "无标题"),
        "summary": article.get("summary", ""),
        "publisher": article.get("publisher") or provider,
        "link": article.get("link", ""),
        "pub_date": _parse_pub_date(article.get("providerPublishTime")),
    }


def _format_article(data: dict) -> str:
    lines = [f"### {data['title']}（来源：{data['publisher']}）"]
    if data.get("summary"):
        lines.append(str(data["summary"]))
    if data.get("link"):
        lines.append(f"链接：{data['link']}")
    return "\n".join(lines) + "\n"


def _parse_pub_date(raw: Any):
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "too many requests" in text
        or "rate limited" in text
        or "rate limit" in text
        or "429" in text
        or exc.__class__.__name__ == "YFRateLimitError"
    )


def _get_cache(key: Tuple[Any, ...]) -> str | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry.expires_at < time.monotonic():
        _CACHE.pop(key, None)
        return None
    return entry.value


def _set_cache(key: Tuple[Any, ...], value: str, expires_at: float) -> None:
    _CACHE[key] = _CacheEntry(value=value, expires_at=expires_at)

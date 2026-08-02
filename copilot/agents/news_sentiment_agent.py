"""News and sentiment analyst backed by Finnhub tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    make_news_sentiment_tools,
    tool_names,
)
from ai_trading_copilot.copilot.agents.llm_tools import (
    extract_json_object,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.agents.react_runner import ReActAgentRunner
from ai_trading_copilot.copilot.domain.models import NewsSentimentReport


@dataclass
class NewsSentimentAnalysisResult:
    report: NewsSentimentReport
    markdown: str
    tool_calls: List[str]


class NewsSentimentAgent:
    """Scores market news, social sentiment, and event warnings."""

    def __init__(self, *, llm=None, tools=None):
        self.llm = llm
        self.tools = tools

    def analyze_symbol(
        self,
        *,
        symbol: str,
        trade_date: str | None = None,
        look_back_days: int = 7,
    ) -> NewsSentimentAnalysisResult:
        fallback = NewsSentimentReport(
            symbol=symbol,
            sentiment_score=0.0,
            summary="LLM news sentiment analysis was not generated.",
            data_availability={"llm": "unavailable"},
        )
        if self.llm is None:
            return NewsSentimentAnalysisResult(
                report=fallback,
                markdown=_markdown_from_report(fallback),
                tool_calls=[],
            )

        tools = self.tools or make_news_sentiment_tools()
        start_date, end_date = date_window(trade_date, look_back_days)
        runner = ReActAgentRunner(llm=self.llm, tools=tools)
        prefetch = [
            ("get_company_news", {"ticker": symbol, "start_date": start_date, "end_date": end_date}),
            ("get_market_news", {"curr_date": end_date, "look_back_days": look_back_days, "limit": 5}),
            ("get_news_sentiment", {"ticker": symbol}),
            ("get_social_sentiment", {"ticker": symbol}),
            ("get_earnings_calendar", {"ticker": symbol, "start_date": start_date, "end_date": end_date}),
        ]
        evidence = runner.run(prompt="", prefetch=prefetch)
        prompt = (
            "You are the News Sentiment Analyst for an AI trading copilot. "
            "Use Finnhub-sourced company news, broad market news, social sentiment, "
            "news sentiment, and earnings calendar evidence. Produce a concise event "
            "risk warning and sentiment score for the exact symbol. Do not invent data. "
            "If a tool is unavailable or rate-limited, mark it in data_availability.\n\n"
            "Return a Markdown report followed by one final JSON object with keys: "
            "sentiment_score, company_news_score, social_sentiment_score, "
            "earnings_event_score, material_risk, risk_flags, key_events, alerts, "
            "summary, data_availability. Scores must be between -1 and 1. JSON must "
            "be the final object.\n\n"
            f"Symbol: {symbol}\n"
            f"News window: {start_date} to {end_date}\n"
            f"Available tools: {tool_names(tools)}\n\n"
            f"Prefetched Finnhub evidence:\n{evidence.prefetched_evidence or '-'}"
        )
        result = runner.run(prompt=prompt)
        content = result.content
        try:
            payload = extract_json_object(content)
            report = NewsSentimentReport(
                symbol=symbol,
                sentiment_score=_score(payload.get("sentiment_score"), 0.0),
                company_news_score=_optional_score(payload.get("company_news_score")),
                social_sentiment_score=_optional_score(payload.get("social_sentiment_score")),
                earnings_event_score=_optional_score(payload.get("earnings_event_score")),
                material_risk=bool(payload.get("material_risk", False)),
                risk_flags=[str(item) for item in payload.get("risk_flags", []) if str(item).strip()],
                key_events=[str(item) for item in payload.get("key_events", []) if str(item).strip()],
                alerts=[str(item) for item in payload.get("alerts", []) if str(item).strip()],
                summary=str(payload.get("summary") or "").strip(),
                data_availability={
                    str(key): str(value)
                    for key, value in (payload.get("data_availability") or {}).items()
                },
            )
        except Exception:
            report = fallback.model_copy(update={"summary": content.strip() or fallback.summary})
        return NewsSentimentAnalysisResult(
            report=report,
            markdown=strip_trailing_json_object(content) or _markdown_from_report(report),
            tool_calls=[*evidence.prefetched_calls, *result.tool_calls],
        )


def _score(value, fallback: float) -> float:
    parsed = _optional_score(value)
    return fallback if parsed is None else parsed


def _optional_score(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(min(float(value), 1.0), -1.0)
    except (TypeError, ValueError):
        return None


def _markdown_from_report(report: NewsSentimentReport) -> str:
    return (
        f"# News Sentiment: {report.symbol}\n\n"
        f"- Sentiment score: {report.sentiment_score:.2f}\n"
        f"- Material risk: {'yes' if report.material_risk else 'no'}\n"
        f"- Risk flags: {', '.join(report.risk_flags) or '-'}\n"
        f"- Key events: {', '.join(report.key_events) or '-'}\n"
        f"- Alerts: {', '.join(report.alerts) or '-'}\n"
        f"- Summary: {report.summary or '-'}\n"
    )

"""News and sentiment analyst backed by Finnhub tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
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
from ai_trading_copilot.copilot.config.prompts import render_prompt
from ai_trading_copilot.copilot.domain.models import NewsSentimentReport


@dataclass
class NewsSentimentAnalysisResult:
    report: NewsSentimentReport
    markdown: str
    tool_calls: List[str]


class NewsSentimentAgent:
    """Scores market news, social sentiment, and event warnings."""

    UPCOMING_EARNINGS_DAYS = 45

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
        earnings_start_date = end_date
        earnings_end_date = (
            date.fromisoformat(end_date) + timedelta(days=self.UPCOMING_EARNINGS_DAYS)
        ).isoformat()
        runner = ReActAgentRunner(llm=self.llm, tools=tools)
        prefetch = [
            ("get_company_news", {"ticker": symbol, "start_date": start_date, "end_date": end_date}),
            ("get_market_news", {"curr_date": end_date, "look_back_days": look_back_days, "limit": 5}),
            ("get_news_sentiment", {"ticker": symbol}),
            ("get_social_sentiment", {"ticker": symbol}),
            (
                "get_earnings_calendar",
                {
                    "ticker": symbol,
                    "start_date": earnings_start_date,
                    "end_date": earnings_end_date,
                },
            ),
        ]
        evidence = runner.run(prompt="", prefetch=prefetch)
        prompt = render_prompt(
            "news_sentiment.v1",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            earnings_start_date=earnings_start_date,
            earnings_end_date=earnings_end_date,
            available_tools=tool_names(tools),
            news_evidence=evidence.prefetched_evidence or "-",
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
                news_references=_news_references(payload.get("news_references")),
                summary=str(payload.get("summary") or "").strip(),
                data_availability={
                    str(key): str(value)
                    for key, value in (payload.get("data_availability") or {}).items()
                },
                decision_basis=_string_list(payload.get("decision_basis")),
                uncertainties=_string_list(payload.get("uncertainties")),
                downstream_summary=str(
                    payload.get("downstream_summary") or payload.get("summary") or ""
                ).strip(),
            )
        except Exception:
            report = fallback.model_copy(update={"summary": content.strip() or fallback.summary})
        markdown = strip_trailing_json_object(content) or _markdown_from_report(report)
        return NewsSentimentAnalysisResult(
            report=report,
            markdown=_markdown_with_references(markdown, report),
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


def _news_references(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    allowed_keys = ["title", "published_at", "url", "source", "event_type", "relevance"]
    references: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        reference = {
            key: str(item.get(key) or "unknown").strip() or "unknown"
            for key in allowed_keys
        }
        references.append(reference)
    return references


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _markdown_from_report(report: NewsSentimentReport) -> str:
    content = (
        f"# News Sentiment: {report.symbol}\n\n"
        f"- Sentiment score: {report.sentiment_score:.2f}\n"
        f"- Material risk: {'yes' if report.material_risk else 'no'}\n"
        f"- Risk flags: {', '.join(report.risk_flags) or '-'}\n"
        f"- Key events: {', '.join(report.key_events) or '-'}\n"
        f"- Alerts: {', '.join(report.alerts) or '-'}\n"
        f"- Decision basis: {', '.join(report.decision_basis) or '-'}\n"
        f"- Uncertainties: {', '.join(report.uncertainties) or '-'}\n"
        f"- Downstream summary: {report.downstream_summary or '-'}\n"
        f"- Summary: {report.summary or '-'}\n"
    )
    if report.news_references:
        lines = [content, "", "## News References", ""]
        for item in report.news_references:
            lines.append(
                "- "
                f"{item.get('published_at', 'unknown')} | "
                f"{item.get('source', 'unknown')} | "
                f"{item.get('title', 'unknown')} | "
                f"{item.get('url', 'unknown')}"
            )
        return "\n".join(lines) + "\n"
    return content


def _markdown_with_references(markdown: str, report: NewsSentimentReport) -> str:
    if not report.news_references:
        return markdown
    missing = [
        item
        for item in report.news_references
        if item.get("url", "unknown") not in markdown
        or item.get("published_at", "unknown") not in markdown
    ]
    if not missing:
        return markdown
    lines = [markdown.rstrip(), "", "## News References", ""]
    for item in missing:
        lines.append(
            "- "
            f"{item.get('published_at', 'unknown')} | "
            f"{item.get('source', 'unknown')} | "
            f"{item.get('title', 'unknown')} | "
            f"{item.get('url', 'unknown')}"
        )
    return "\n".join(lines) + "\n"

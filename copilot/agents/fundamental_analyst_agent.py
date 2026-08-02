"""Fundamental analyst with owned RAG retrieval tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    make_fundamental_tools,
    tool_names,
)
from ai_trading_copilot.copilot.agents.llm_tools import (
    extract_json_object,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.agents.react_runner import ReActAgentRunner
from ai_trading_copilot.copilot.domain.models import FundamentalNewsReport


@dataclass
class FundamentalAnalysisResult:
    report: FundamentalNewsReport
    markdown: str
    tool_calls: List[str]


class FundamentalAnalystAgent:
    """Scores company fundamentals and exposes fundamental RAG as a tool."""

    def __init__(self, *, llm=None, tools=None, rag_retriever=None):
        self.llm = llm
        self.tools = tools
        self.rag_retriever = rag_retriever

    def set_rag_query_llm(self, llm) -> None:
        if self.rag_retriever is not None and hasattr(self.rag_retriever, "set_rag_query_llm"):
            self.rag_retriever.set_rag_query_llm(llm)

    def retrieve_fundamental_rag_context(self, **kwargs):
        if self.rag_retriever is None:
            return []
        return self.rag_retriever.retrieve_fundamental_rag_context(**kwargs)

    def analyze_symbol(
        self,
        *,
        symbol: str,
        trade_date: str | None = None,
        look_back_days: int = 90,
    ) -> FundamentalAnalysisResult:
        fallback = FundamentalNewsReport(
            symbol=symbol,
            thesis_intact=True,
            material_risk=False,
            risk_flags=[],
            summary="LLM fundamental analysis was not generated.",
            fundamental_score=0.0,
            data_availability={"llm": "unavailable"},
        )
        if self.llm is None:
            return FundamentalAnalysisResult(
                report=fallback,
                markdown=_markdown_from_report(fallback),
                tool_calls=[],
            )

        tools = self.tools or make_fundamental_tools(rag_retriever=self.rag_retriever)
        _, end_date = date_window(trade_date, look_back_days)
        runner = ReActAgentRunner(llm=self.llm, tools=tools)
        rag_query = (
            f"{symbol} revenue margin earnings guidance cash flow balance sheet "
            "valuation risk factors management discussion"
        )
        prefetch = [
            ("get_fundamentals", {"ticker": symbol, "curr_date": end_date}),
            ("get_balance_sheet", {"ticker": symbol, "freq": "quarterly", "curr_date": end_date}),
            ("get_cashflow", {"ticker": symbol, "freq": "quarterly", "curr_date": end_date}),
            ("get_income_statement", {"ticker": symbol, "freq": "quarterly", "curr_date": end_date}),
            ("retrieve_fundamental_rag", {"ticker": symbol, "query": rag_query, "limit": 5}),
        ]
        evidence = runner.run(prompt="", prefetch=prefetch)
        prompt = (
            "You are the Fundamental Analyst for an AI trading copilot. Use fresh "
            "fundamental tools first and use retrieved fundamental RAG documents as "
            "background evidence. Score fundamental quality from -1 to 1, identify "
            "material risks, and decide whether the trading thesis is intact. Do not "
            "invent data.\n\n"
            "Return a Markdown report followed by one final JSON object with keys: "
            "thesis_intact, material_risk, risk_flags, summary, fundamental_score, "
            "key_events, data_availability. JSON must be the final object.\n\n"
            f"Symbol: {symbol}\n"
            f"Current date: {end_date}\n"
            f"Available tools: {tool_names(tools)}\n\n"
            f"Prefetched fundamental evidence:\n{evidence.prefetched_evidence or '-'}"
        )
        result = runner.run(prompt=prompt)
        content = result.content
        try:
            payload = extract_json_object(content)
            report = FundamentalNewsReport(
                symbol=symbol,
                thesis_intact=bool(payload.get("thesis_intact", True)),
                material_risk=bool(payload.get("material_risk", False)),
                risk_flags=[str(flag) for flag in payload.get("risk_flags", []) if str(flag).strip()],
                summary=str(payload.get("summary") or "").strip(),
                fundamental_score=_score(payload.get("fundamental_score"), 0.0),
                key_events=[str(item) for item in payload.get("key_events", []) if str(item).strip()],
                data_availability={
                    str(key): str(value)
                    for key, value in (payload.get("data_availability") or {}).items()
                },
            )
        except Exception:
            report = fallback.model_copy(update={"summary": content.strip() or fallback.summary})
        return FundamentalAnalysisResult(
            report=report,
            markdown=strip_trailing_json_object(content) or _markdown_from_report(report),
            tool_calls=[*evidence.prefetched_calls, *result.tool_calls],
        )


def _score(value, fallback: float) -> float:
    if value in (None, ""):
        return fallback
    try:
        return max(min(float(value), 1.0), -1.0)
    except (TypeError, ValueError):
        return fallback


def _markdown_from_report(report: FundamentalNewsReport) -> str:
    return (
        f"# Fundamental Analysis: {report.symbol}\n\n"
        f"- Fundamental score: {report.fundamental_score if report.fundamental_score is not None else '-'}\n"
        f"- Thesis intact: {'yes' if report.thesis_intact else 'no'}\n"
        f"- Material risk: {'yes' if report.material_risk else 'no'}\n"
        f"- Risk flags: {', '.join(report.risk_flags) or '-'}\n"
        f"- Key events: {', '.join(report.key_events) or '-'}\n"
        f"- Summary: {report.summary or '-'}\n"
    )

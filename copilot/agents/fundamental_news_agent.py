"""Fundamental/news risk agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ai_trading_copilot.copilot.adapters.trading_tools import (
    date_window,
    make_fundamental_news_tools,
    tool_names,
)
from ai_trading_copilot.copilot.agents.llm_tools import (
    collect_tool_evidence,
    extract_json_object,
    run_tool_calling_llm,
    strip_trailing_json_object,
)
from ai_trading_copilot.copilot.domain.models import FundamentalNewsReport


@dataclass
class FundamentalNewsAnalysisResult:
    report: FundamentalNewsReport
    markdown: str
    tool_calls: List[str]


class FundamentalNewsAgent:
    """Normalizes or generates structured fundamental/news risk input."""

    def __init__(self, *, llm=None, tools=None):
        self.llm = llm
        self.tools = tools

    def from_report(self, report: FundamentalNewsReport) -> FundamentalNewsReport:
        return report

    def analyze_symbol(
        self,
        *,
        symbol: str,
        trade_date: str | None = None,
        look_back_days: int = 7,
        rag_context: str = "",
    ) -> FundamentalNewsAnalysisResult:
        fallback = FundamentalNewsReport(
            symbol=symbol,
            thesis_intact=True,
            material_risk=False,
            risk_flags=[],
            summary="未生成 LLM 基本面/新闻复核。",
        )
        if self.llm is None:
            return FundamentalNewsAnalysisResult(
                report=fallback,
                markdown=_markdown_from_report(fallback),
                tool_calls=[],
            )

        tools = self.tools or make_fundamental_news_tools()
        start_date, end_date = date_window(trade_date, look_back_days)
        tool_result_cache = {}
        tool_evidence, pre_calls = collect_tool_evidence(
            tools=tools,
            requests=[
                ("get_news", {"ticker": symbol, "start_date": start_date, "end_date": end_date}),
                (
                    "get_global_news",
                    {"curr_date": end_date, "look_back_days": look_back_days, "limit": 5},
                ),
                ("get_fundamentals", {"ticker": symbol, "curr_date": end_date}),
            ],
            tool_result_cache=tool_result_cache,
        )
        prompt = (
            "You are the Fundamental News Review analyst for an AI trading copilot. "
            "This combines TradingAgents' news and fundamentals analyst roles into a compact "
            "risk screen. Use company news, global news, and fundamentals tools to decide "
            "whether the trade thesis is intact or blocked by material risk.\n\n"
            "高效报告格式：\n"
            "1. 交易逻辑状态：完好或受损，并说明决定性原因。\n"
            "2. 重大风险：只列出会改变交易时机、仓位或回避决策的风险。\n"
            "3. 支撑证据：2-4 条来自工具的新闻/基本面数据。\n"
            "4. 缺失数据：失败或不可用的工具，不要绕开缺口推断。\n\n"
            "Avoid broad company summaries unless directly relevant to this trade decision. "
            "Write the entire Markdown report and all JSON string values in Simplified Chinese. "
            "Return the Markdown report followed by one JSON object with keys: thesis_intact (bool), "
            "material_risk (bool), risk_flags (array of strings), summary (string). "
            "JSON must be the final object. Do not invent facts not present in tool outputs.\n\n"
            f"Symbol: {symbol}\n"
            f"News window: {start_date} to {end_date}\n"
            f"Current date: {end_date}\n"
            f"Available tools: {tool_names(tools)}\n\n"
            f"Prefetched tool evidence:\n{tool_evidence or '-'}\n\n"
            "Fundamental RAG context. Treat stored company fundamental documents "
            "as background only; fresh news/fundamental tool evidence remains authoritative:\n"
            f"{rag_context or '-'}"
        )
        content, calls = run_tool_calling_llm(
            llm=self.llm,
            prompt=prompt,
            tools=tools,
            tool_result_cache=tool_result_cache,
        )
        try:
            payload = extract_json_object(content)
            report = FundamentalNewsReport(
                symbol=symbol,
                thesis_intact=bool(payload.get("thesis_intact", True)),
                material_risk=bool(payload.get("material_risk", False)),
                risk_flags=[
                    str(flag)
                    for flag in payload.get("risk_flags", [])
                    if str(flag).strip()
                ],
                summary=str(payload.get("summary") or "").strip(),
            )
        except Exception:
            report = fallback.model_copy(update={"summary": content.strip() or fallback.summary})
        return FundamentalNewsAnalysisResult(
            report=report,
            markdown=strip_trailing_json_object(content) or _markdown_from_report(report),
            tool_calls=[*pre_calls, *calls],
        )


def _markdown_from_report(report: FundamentalNewsReport) -> str:
    return (
        f"# 基本面/新闻复核：{report.symbol}\n\n"
        f"- 交易逻辑完好：{'是' if report.thesis_intact else '否'}\n"
        f"- 重大风险：{'是' if report.material_risk else '否'}\n"
        f"- 风险标记：{', '.join(report.risk_flags) or '-'}\n"
        f"- 摘要：{report.summary or '-'}\n"
    )

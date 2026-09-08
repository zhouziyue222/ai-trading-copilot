"""Fundamental analyst with owned RAG retrieval tool."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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
from ai_trading_copilot.copilot.config.prompts import render_prompt
from ai_trading_copilot.copilot.domain.models import FundamentalAnalysisReport
from ai_trading_copilot.copilot.services.eval_samples import record_eval_sample


@dataclass
class FundamentalAnalysisResult:
    report: FundamentalAnalysisReport
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
        started = perf_counter()
        fallback = FundamentalAnalysisReport(
            symbol=symbol,
            thesis_intact=True,
            material_risk=False,
            risk_flags=[],
            summary="【待补充】未生成基本面分析，默认评分不代表中性判断，无法确认投资逻辑或排除重大风险。",
            fundamental_score=0.0,
            data_availability={"llm": "unavailable"},
        )
        if self.llm is None:
            result = FundamentalAnalysisResult(
                report=fallback,
                markdown=_markdown_from_report(fallback),
                tool_calls=[],
            )
            _record_fundamental_agent_eval_sample(
                symbol=symbol,
                trade_date=trade_date,
                look_back_days=look_back_days,
                result=result,
                latency_ms=(perf_counter() - started) * 1000.0,
                evidence="",
                error="",
            )
            return result

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
        prompt = render_prompt(
            "fundamental_analyst.v1",
            symbol=symbol,
            current_date=end_date,
            available_tools=tool_names(tools),
            fundamental_evidence=evidence.prefetched_evidence or "-",
        )
        result = runner.run(prompt=prompt)
        content = result.content
        try:
            payload = extract_json_object(content)
            report = FundamentalAnalysisReport(
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
                decision_basis=_string_list(payload.get("decision_basis")),
                uncertainties=_string_list(payload.get("uncertainties")),
                downstream_summary=str(
                    payload.get("downstream_summary") or payload.get("summary") or ""
                ).strip(),
            )
        except Exception:
            report = fallback.model_copy(update={"summary": content.strip() or fallback.summary})
        final_result = FundamentalAnalysisResult(
            report=report,
            markdown=strip_trailing_json_object(content) or _markdown_from_report(report),
            tool_calls=[*evidence.prefetched_calls, *result.tool_calls],
        )
        _record_fundamental_agent_eval_sample(
            symbol=symbol,
            trade_date=end_date,
            look_back_days=look_back_days,
            result=final_result,
            latency_ms=(perf_counter() - started) * 1000.0,
            evidence=evidence.prefetched_evidence,
            error="",
        )
        return final_result


def _score(value, fallback: float) -> float:
    if value in (None, ""):
        return fallback
    try:
        return max(min(float(value), 1.0), -1.0)
    except (TypeError, ValueError):
        return fallback


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _markdown_from_report(report: FundamentalAnalysisReport) -> str:
    return (
        f"# 基本面分析： {report.symbol}\n\n"
        f"- 基本面评分： {report.fundamental_score if report.fundamental_score is not None else '-'}\n"
        f"- 投资逻辑是否完整： {'是' if report.thesis_intact else '否'}\n"
        f"- 是否存在重大风险： {'是' if report.material_risk else '否'}\n"
        f"- 风险事项： {', '.join(report.risk_flags) or '-'}\n"
        f"- 关键事件： {', '.join(report.key_events) or '-'}\n"
        f"- 关键依据： {', '.join(report.decision_basis) or '-'}\n"
        f"- 【待补充】不确定性： {', '.join(report.uncertainties) or '-'}\n"
        f"- 综合结论： {report.downstream_summary or '-'}\n"
        f"- 结论： {report.summary or '-'}\n"
    )


def _record_fundamental_agent_eval_sample(
    *,
    symbol: str,
    trade_date: str | None,
    look_back_days: int,
    result: FundamentalAnalysisResult,
    latency_ms: float,
    evidence: str,
    error: str,
) -> None:
    report = result.report
    response = result.markdown or _markdown_from_report(report)
    record_eval_sample(
        stage="fundamental_agent",
        payload={
            "user_input": (
                f"Evaluate {symbol} fundamentals as of {trade_date or 'latest'} "
                f"using a {look_back_days} day evidence window."
            ),
            "symbol": report.symbol,
            "trade_date": trade_date,
            "look_back_days": look_back_days,
            "response": response,
            "answer": response,
            "reference": _default_fundamental_reference(report),
            "retrieved_contexts": [evidence] if evidence.strip() else [],
            "tool_calls": list(result.tool_calls),
            "latency_ms": latency_ms,
            "error": error,
            "structured_output": {
                "thesis_intact": report.thesis_intact,
                "material_risk": report.material_risk,
                "risk_flags": list(report.risk_flags),
                "summary": report.summary,
                "fundamental_score": report.fundamental_score,
                "key_events": list(report.key_events),
                "decision_basis": list(report.decision_basis),
                "uncertainties": list(report.uncertainties),
                "data_availability": dict(report.data_availability),
            },
        },
    )


def _default_fundamental_reference(report: FundamentalAnalysisReport) -> str:
    risk_clause = "material risk is present" if report.material_risk else "no material risk is present"
    thesis_clause = "thesis is intact" if report.thesis_intact else "thesis is impaired"
    return f"The fundamental {thesis_clause}, and {risk_clause}."


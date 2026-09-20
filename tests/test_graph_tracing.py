import json

import pytest

from ai_trading_copilot.copilot.domain import AnalystType, PortfolioSnapshot
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.run_tracker import RunTracker


def _tracker(tmp_path, run_id="run_trace"):
    return RunTracker(
        output_dir=tmp_path,
        run_id=run_id,
        symbols=["AAPL"],
        params={"subscription_symbols": ["AAPL"]},
        defaults={},
        nodes=CopilotLangGraph.NODE_ORDER,
    )


def test_langgraph_writes_trace_reports_for_successful_run(tmp_path):
    tracker = _tracker(tmp_path, "run_trace_success")

    state = CopilotLangGraph(enable_default_llm=False, run_tracker=tracker).run(
        subscription_symbols=["AAPL"],
        portfolio=PortfolioSnapshot(),
        report_output_dir=tmp_path,
        run_id="run_trace_success",
    )

    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert status["reports"]["trace_json"]["exists"] is True
    assert status["reports"]["trace_markdown"]["exists"] is True
    assert (tmp_path / "trace.md").exists()
    assert trace["run_id"] == "run_trace_success"
    assert "copilot.run" in {span["name"] for span in trace["spans"]}
    assert "graph.node" in {span["name"] for span in trace["spans"]}
    assert state["trace_persisted"] is True

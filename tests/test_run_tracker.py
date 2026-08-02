import json
from pathlib import Path

from ai_trading_copilot.copilot.domain import AnalystType, PortfolioSnapshot
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.run_tracker import RunTracker


class FailingRadarAgent:
    def analyze_symbol_with_report(self, **kwargs):
        raise RuntimeError("radar failed")


def _tracker(tmp_path):
    return RunTracker(
        output_dir=tmp_path,
        run_id="run_test",
        symbols=["AAPL"],
        params={"subscription_symbols": ["AAPL"]},
        defaults={"look_back_days": 90},
        nodes=CopilotLangGraph.NODE_ORDER,
    )


def test_run_tracker_records_node_and_report_status(tmp_path):
    tracker = _tracker(tmp_path)
    report_path = tmp_path / "report.md"
    report_path.write_text("# Report\n", encoding="utf-8")

    tracker.start_node("Node A")
    tracker.succeed_node("Node A")
    tracker.start_node("Node B")
    tracker.fail_node("Node B", "boom")
    tracker.record_report("report", report_path)
    tracker.finish(failed=True)

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert payload["nodes"]["Node A"]["status"] == "succeeded"
    assert payload["nodes"]["Node B"]["status"] == "failed"
    assert payload["reports"]["report"]["exists"] is True
    assert payload["status"] == "failed"


def test_langgraph_tracker_records_reports_for_successful_run(tmp_path):
    tracker = _tracker(tmp_path)
    state = CopilotLangGraph(
        enable_default_llm=False,
        run_tracker=tracker,
    ).run(
        subscription_symbols=["AAPL"],
        portfolio=PortfolioSnapshot(),
        report_output_dir=tmp_path,
    )
    tracker.finish()
    tracker.write_audit(state=state)

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert payload["nodes"][CopilotLangGraph.NODE_TECHNICAL_POSITION]["status"] == "succeeded"
    assert payload["nodes"][CopilotLangGraph.NODE_NEWS_SENTIMENT]["status"] == "succeeded"
    assert payload["nodes"][CopilotLangGraph.NODE_FUNDAMENTAL_ANALYSIS]["status"] == "succeeded"
    assert payload["nodes"][CopilotLangGraph.NODE_OPPORTUNITY_RADAR]["status"] == "skipped"
    assert payload["reports"]["futu_portfolio"]["exists"] is True
    assert payload["reports"]["run_explanation"]["exists"] is True
    assert (tmp_path / "run_audit.md").exists()


def test_run_tracker_writes_decision_summary_from_state(tmp_path):
    tracker = _tracker(tmp_path)
    state = {
        "subscription_symbols": ["AAPL"],
        "radar_items": [
            {
                "symbol": "AAPL",
                "status": "actionable",
                "status_label": "可行动",
                "current_price": 101.25,
                "support_level": 98.5,
                "reward_risk_ratio": 2.6,
                "suggested_action": "Wait for support confirmation",
            }
        ],
        "technical_positions": {
            "AAPL": {
                "symbol": "AAPL",
                "current_price": 101.25,
                "support_level": 98.5,
                "reward_risk_ratio": 2.6,
            }
        },
        "trade_plans": {
            "AAPL": {
                "symbol": "AAPL",
                "subscription_status": "actionable",
                "market_regime": "uptrend",
                "direction": "buy",
                "entry_logic": "Buy near support",
            }
        },
        "risk_assessments": {"AAPL": {"approved": False, "violations": []}},
        "execution_decisions": {
            "AAPL": {
                "symbol": "AAPL",
                "status": "blocked_by_risk",
                "direction": "buy",
                "approved_by_risk": False,
                "message": "Blocked by hard risk rule",
            }
        },
        "report": {
            "summary": "One actionable setup was blocked by risk.",
            "market_regime": {"regime": "uptrend"},
        },
        "errors": [],
    }

    tracker.finish()
    tracker.write_audit(state=state)

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    summary = payload["decision_summary"]
    assert summary["summary"] == "One actionable setup was blocked by risk."
    assert summary["market_regime"] == "uptrend"
    assert summary["metrics"]["symbol_count"] == 1
    assert summary["metrics"]["actionable_count"] == 1
    assert summary["metrics"]["risk_blocked_count"] == 1
    assert summary["symbols"][0]["symbol"] == "AAPL"
    assert summary["symbols"][0]["direction"] == "buy"
    assert summary["symbols"][0]["execution_status"] == "blocked_by_risk"


def test_langgraph_tracker_records_failed_node(tmp_path):
    tracker = _tracker(tmp_path)

    try:
        CopilotLangGraph(
            llm=None,
            opportunity_radar_agent=FailingRadarAgent(),
            run_tracker=tracker,
        ).run(
            subscription_symbols=["AAPL"],
            portfolio=PortfolioSnapshot(),
            selected_analysts=[AnalystType.OPPORTUNITY_RADAR],
            report_output_dir=tmp_path,
        )
    except RuntimeError:
        tracker.finish(failed=True)
        tracker.write_audit(state=None, error=RuntimeError("radar failed"))

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert payload["nodes"][CopilotLangGraph.NODE_OPPORTUNITY_RADAR]["status"] == "failed"
    assert "radar failed" in payload["nodes"][CopilotLangGraph.NODE_OPPORTUNITY_RADAR]["error"]
    assert (tmp_path / "run_audit.md").exists()


def test_cli_creates_status_and_audit_with_mock_graph(tmp_path, monkeypatch):
    from ai_trading_copilot.copilot import run as run_module

    class MockGraph:
        NODE_ORDER = CopilotLangGraph.NODE_ORDER

        def __init__(self, *, run_tracker):
            self.run_tracker = run_tracker

        def run(self, **kwargs):
            self.run_tracker.start_node(CopilotLangGraph.NODE_LOAD_PERSONA_MARKDOWN)
            self.run_tracker.succeed_node(CopilotLangGraph.NODE_LOAD_PERSONA_MARKDOWN)
            return {
                "subscription_symbols": kwargs["subscription_symbols"],
                "errors": [],
                "agent_reports": {},
                "analyst_reports": {},
                "explanations": {},
            }

    monkeypatch.setattr(run_module, "CopilotLangGraph", MockGraph)

    assert run_module.main(["CRCL", "--output-dir", str(tmp_path), "--portfolio-mode", "live"]) == 0
    assert (tmp_path / "run_status.json").exists()
    assert (tmp_path / "run_audit.md").exists()

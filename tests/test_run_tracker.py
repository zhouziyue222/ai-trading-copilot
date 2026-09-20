import json
from pathlib import Path

from ai_trading_copilot.copilot.domain import AnalystType, PortfolioSnapshot
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.run_tracker import RunTracker


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


def test_tracker_tracks_all_active_nodes_and_cancels_each(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.start_node("Technical Position")
    tracker.start_node("News Sentiment")
    assert tracker.status["current_nodes"] == ["Technical Position", "News Sentiment"]
    assert tracker.status["current_node"] is None
    tracker.succeed_node("News Sentiment")
    assert tracker.status["current_node"] == "Technical Position"
    tracker.start_node("Fundamental Analysis")
    tracker.cancel()
    assert tracker.status["current_nodes"] == []
    assert tracker.status["current_node"] is None
    assert all(node["status"] != "running" for node in tracker.status["nodes"].values())


def test_run_tracker_records_sanitized_node_activity(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.start_node("Technical Position")

    tracker.record_node_activity(
        "Technical Position",
        {
            "phase": "tool.call",
            "status": "running",
            "title": "正在调用 get_stock_info",
            "tool_name": "get_stock_info",
            "args": {"symbol": "CRCL", "api_key": "secret", "nested": {"token": "hidden"}},
            "summary": {"tool.output": {"chars": 10, "sha256": "abc"}},
        },
    )
    for index in range(25):
        tracker.record_node_activity(
            "Technical Position",
            {
                "phase": "tool.call",
                "status": "ok",
                "title": f"完成调用 tool_{index}",
                "tool_name": f"tool_{index}",
                "args": {"api_key": "secret", "token": "hidden", "password": "masked"},
            },
            completed=True,
        )

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    node = payload["nodes"]["Technical Position"]
    assert node["activity"]["tool_name"] == "tool_24"
    assert node["activity"]["args"]["password"] == "<redacted>"
    assert len(node["activity_history"]) == 20
    first = node["activity_history"][0]
    assert first["tool_name"] == "tool_5"
    assert "secret" not in json.dumps(node, ensure_ascii=False)
    assert "hidden" not in json.dumps(node, ensure_ascii=False)
    assert "masked" not in json.dumps(node, ensure_ascii=False)


def test_run_tracker_records_cancel_request_and_cancelled_terminal_state(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.start_node("Technical Position")

    assert tracker.request_cancel() is True
    assert tracker.is_cancel_requested() is True
    tracker.cancel()
    tracker.finish()

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["cancel_requested"] is True
    assert payload["cancel_requested_at"]
    assert payload["cancelled_at"]
    assert payload["finished_at"] == payload["cancelled_at"]
    assert payload["current_node"] is None
    assert payload["nodes"]["Technical Position"]["status"] == "skipped"
    assert payload["nodes"]["Technical Position"]["error"] == "cancelled"


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
    assert payload["reports"]["futu_portfolio"]["exists"] is True
    assert payload["reports"]["run_explanation"]["exists"] is True
    assert (tmp_path / "run_audit.md").exists()
    audit = (tmp_path / "run_audit.md").read_text(encoding="utf-8")
    assert "# Trading Copilot Run Audit" in audit
    for marker in ["锛", "涓", "鏅", "浜", "鍩", "椋"]:
        assert marker not in audit


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
        "risk_assessments": {
            "AAPL": {
                "symbol": "AAPL",
                "approved": True,
                "current_position_weight": 0.0,
                "target_weight": 0.2,
                "final_weight": 0.2,
                "delta_weight": 0.2,
                "estimated_trade_value": 20_000,
                "clamped": False,
            }
        },
        "execution_decisions": {
            "AAPL": {
                "symbol": "AAPL",
                "status": "portfolio_decided",
                "direction": "buy",
                "approved_by_risk": True,
                "message": "Portfolio Manager selected buy 10 shares within risk limits.",
                "action": "buy",
                "quantity": 10,
                "current_weight": 0.0,
                "target_weight": 0.2,
                "final_weight": 0.2,
                "delta_weight": 0.2,
                "estimated_trade_value": 20_000,
            }
        },
        "report": {
            "summary": "One actionable setup received a portfolio decision.",
            "market_regime": {"regime": "uptrend"},
        },
        "errors": [],
    }

    tracker.finish()
    tracker.write_audit(state=state)

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    summary = payload["decision_summary"]
    assert summary["summary"] == "One actionable setup received a portfolio decision."
    assert summary["market_regime"] == "uptrend"
    assert summary["metrics"]["symbol_count"] == 1
    assert summary["metrics"]["actionable_count"] == 1
    assert summary["metrics"]["risk_adjusted_count"] == 0
    assert summary["metrics"]["risk_held_count"] == 0
    assert summary["metrics"]["portfolio_decided_count"] == 1
    assert summary["symbols"][0]["symbol"] == "AAPL"
    assert summary["symbols"][0]["direction"] == "buy"
    assert summary["symbols"][0]["execution_status"] == "portfolio_decided"
    assert summary["symbols"][0]["target_weight"] == 0.2
    assert summary["symbols"][0]["final_weight"] == 0.2


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

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.ui.app import UISettings, create_app


class FakeGraph:
    NODE_ORDER = ["Load", "Finish"]
    calls = []
    memory_agents = []

    def __init__(self, *, run_tracker, memory_agent):
        self.run_tracker = run_tracker
        self.__class__.memory_agents.append(memory_agent)

    def run(self, **kwargs):
        self.__class__.calls.append(kwargs)
        report_dir = Path(kwargs["report_output_dir"]) / "6_explanation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "run_explanation.md"
        report_path.write_text("# Summary\n\nUI run complete.\n", encoding="utf-8")
        self.run_tracker.record_report("run_explanation", report_path)
        return {
            "subscription_symbols": kwargs["subscription_symbols"],
            "agent_reports": {"run_explanation": str(report_path)},
            "analyst_reports": {},
            "radar_items": [],
            "technical_positions": {},
            "fundamental_news_by_symbol": {},
            "trade_plans": {},
            "risk_assessments": {},
            "execution_decisions": {},
            "explanations": {},
            "errors": [],
        }


@pytest.fixture()
def client(tmp_path):
    FakeGraph.calls = []
    FakeGraph.memory_agents = []
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_file=tmp_path / "config" / "memory.jsonl",
            reports_dir=tmp_path / "reports",
            graph_cls=FakeGraph,
            run_in_background=False,
        )
    )
    return TestClient(app)


def test_subscription_list_loads_when_file_is_missing(client):
    response = client.get("/api/subscriptions")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_subscription_crud_persists_valid_json(client, tmp_path):
    create_response = client.post(
        "/api/subscriptions",
        json={
            "symbol": "aapl",
            "market_type": "US_STOCK",
            "reason": "pullback",
            "target_action": "watch support",
        },
    )
    assert create_response.status_code == 200
    assert create_response.json()["items"][0]["symbol"] == "AAPL"

    update_response = client.patch(
        "/api/subscriptions/AAPL",
        json={"status": "near_opportunity", "reason": "near support"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["items"][0]["status"] == "near_opportunity"

    delete_response = client.delete("/api/subscriptions/AAPL")
    assert delete_response.status_code == 200
    assert delete_response.json()["items"] == []

    payload = json.loads((tmp_path / "config" / "subscriptions.json").read_text(encoding="utf-8"))
    assert payload == {"items": []}


def test_create_run_constructs_graph_params_and_status(client):
    response = client.post(
        "/api/runs",
        json={
            "symbols": ["aapl"],
            "manual_symbols": "msft, CRCL",
            "selected_analysts": ["opportunity_radar", "fundamental_news"],
            "trade_date": "2026-05-13",
            "look_back_days": 45,
            "portfolio_mode": "live",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"].startswith("run_UI_")
    assert payload["status"]["status"] == "succeeded"
    assert payload["reports"]["run_explanation"]["exists"] is True
    assert payload["status"]["decision_summary"]["metrics"]["symbol_count"] == 3
    assert payload["status"]["decision_summary"]["symbols"][0]["symbol"] == "AAPL"
    assert FakeGraph.calls[0]["subscription_symbols"] == ["AAPL", "MSFT", "CRCL"]
    assert [item.value for item in FakeGraph.calls[0]["selected_analysts"]] == [
        "opportunity_radar",
        "fundamental_news",
    ]
    assert FakeGraph.calls[0]["mode"].value == "simulation"
    assert FakeGraph.calls[0]["portfolio_mode"].value == "live"
    assert FakeGraph.memory_agents[0].retrieve_context(symbol="AAPL") == []


def test_get_run_returns_status(client):
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = client.get(f"/api/runs/{created['run_id']}")

    assert response.status_code == 200
    assert response.json()["status"]["symbols"] == ["AAPL"]


def test_rag_status_endpoint_degrades_when_chroma_is_unavailable(client):
    response = client.get("/api/rag/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "fundamental_chroma"
    assert payload["scope"] == "fundamental_only"
    assert "available" in payload
    assert "document_count" in payload


def test_rag_ingest_text_endpoint_returns_ingest_result(client):
    response = client.post(
        "/api/rag/ingest-text",
        json={
            "title": "AAPL note",
            "text": "AAPL earnings guidance improved, but event risk remains.",
            "symbols": ["AAPL"],
            "tags": ["earnings"],
        },
    )

    assert response.status_code == 200
    assert set(response.json()) == {"added", "skipped", "errors"}


def test_report_endpoint_returns_markdown_and_rejects_unsafe_paths(client, tmp_path):
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = client.get(f"/api/reports/{created['run_id']}/run_explanation")
    assert response.status_code == 200
    assert "# Summary" in response.text

    run_dir = tmp_path / "reports" / "run_bad"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "reports": {
                    "leak": {
                        "path": str(outside),
                        "exists": True,
                        "bytes": 6,
                    }
                },
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    unsafe = client.get("/api/reports/run_bad/leak")
    assert unsafe.status_code == 400

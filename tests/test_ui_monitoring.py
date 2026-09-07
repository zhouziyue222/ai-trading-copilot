from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.ui.app import UISettings, create_app
from tests.lifecycle_fakes import BoundaryGraph


def test_monitoring_keeps_main_lifecycle_and_correlates_request(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_INTERNAL_UI", "1")
    BoundaryGraph.finish_second.clear()
    app = create_app(UISettings(reports_dir=tmp_path / "reports", subscriptions_file=tmp_path / "subscriptions.json",
                               memory_database=tmp_path / "memory.sqlite3", graph_cls=BoundaryGraph,
                               long_term_memory_enabled=False))
    with TestClient(app) as client:
        runtime = client.get("/api/runtime")
        assert runtime.json()["status"] == "idle"
        assert runtime.headers["x-session-id"] == runtime.json()["session_id"]
        created = client.post("/api/runs", json={"symbols": ["AAPL"]})
        run = created.json()
        request_id = created.headers["x-request-id"]
        assert run["status"]["origin_request_id"] == request_id
        runtime = client.get("/api/runtime").json()
        assert runtime["active_runs"][0]["worker_alive"] is True
        events = client.get("/api/internal/observability", params={"query": request_id}).json()["events"]
        assert any(e.get("run_id") == run["run_id"] for e in events)
        assert app.state.runtime.__class__.__name__ == "RunLifecycle"


def test_internal_monitoring_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("COPILOT_INTERNAL_UI", raising=False)
    app = create_app(UISettings(reports_dir=tmp_path / "reports"))
    with TestClient(app) as client:
        assert client.get("/internal/observability").status_code == 404
        assert client.get("/api/internal/observability").status_code == 404
        response = client.get("/unknown")
        assert response.json()["request_id"] == response.headers["x-request-id"]

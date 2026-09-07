import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.ui.app import UISettings, create_app


@pytest.fixture(autouse=True)
def enable_internal_trace(monkeypatch):
    monkeypatch.setenv("COPILOT_INTERNAL_UI", "1")


def _client(tmp_path):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
        )
    )
    return TestClient(app)


def test_trace_endpoint_returns_local_trace(tmp_path):
    client = _client(tmp_path)
    run_dir = tmp_path / "reports" / "run_trace"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps({"reports": {}, "errors": []}),
        encoding="utf-8",
    )
    (run_dir / "trace.json").write_text(
        json.dumps({"trace_id": "abc", "spans": [{"name": "copilot.run"}]}),
        encoding="utf-8",
    )

    response = client.get("/api/runs/run_trace/trace")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run_trace",
        "trace": {"trace_id": "abc", "spans": [{"name": "copilot.run"}]},
    }


def test_trace_endpoint_returns_404_when_trace_is_missing(tmp_path):
    client = _client(tmp_path)
    run_dir = tmp_path / "reports" / "run_without_trace"
    run_dir.mkdir(parents=True)

    response = client.get("/api/runs/run_without_trace/trace")

    assert response.status_code == 404


def test_trace_endpoint_rejects_unsafe_run_id(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/runs/../trace")

    assert response.status_code in {400, 404}


def test_trace_report_download_requires_internal_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("COPILOT_INTERNAL_UI", raising=False)
    client = _client(tmp_path)
    for key in ("trace_json", "trace_markdown"):
        assert client.get(f"/api/reports/run_trace/{key}").status_code == 404

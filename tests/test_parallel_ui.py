import json
from collections import Counter
import pytest
from itertools import permutations

from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.ui.app import UISettings, create_app
from tests.parallel_fakes import OfflineParallelGraph as Graph
from tests.test_run_lifecycle import wait_status, cancel_saved


def make_app(tmp_path):
    return create_app(UISettings(reports_dir=tmp_path / "reports", graph_cls=Graph,
                                subscriptions_file=tmp_path / "subscriptions.json",
                                memory_database=tmp_path / "memory.sqlite3", long_term_memory_enabled=False))


@pytest.mark.parametrize("resume_serial", [False, True])
def test_parallel_cancel_resume_via_ui(tmp_path, monkeypatch, resume_serial):
    monkeypatch.delenv("COPILOT_FORCE_SEQUENTIAL", raising=False)
    Graph.reset()
    app = make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False})
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, lambda s: set(s["current_nodes"]) == set(Graph.names))
        Graph.releases["News Sentiment"].set()
        wait_status(client, run_id, lambda s: s["nodes"]["News Sentiment"]["status"] == "succeeded")
        checkpoint = cancel_saved(client, run_id)
        payload = json.loads((app.state.runtime.checkpoints_dir / f"{checkpoint}.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert set(payload["snapshot"]["completed_nodes"]) & set(Graph.names) == {"News Sentiment"}
        assert set(payload["snapshot"]["state"]["analyst_reports"]) == {"news_sentiment"}
        for event in Graph.releases.values():
            event.set()
        if resume_serial:
            monkeypatch.setenv("COPILOT_FORCE_SEQUENTIAL", "1")
        resumed = client.post(f"/api/checkpoints/{checkpoint}/resume")
        assert resumed.status_code == 200, resumed.text
        wait_status(client, resumed.json()["run_id"], lambda s: s["status"] == "succeeded")
        calls = Counter(Graph.calls)
        assert calls["News Sentiment"] == 1
        assert calls["Technical Position"] == calls["Fundamental Analysis"] == 2
        assert calls["Trader"] == 1


def test_parallel_failure_never_runs_trader_or_saves_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("COPILOT_FORCE_SEQUENTIAL", raising=False)
    Graph.reset()
    Graph.fail_name = "News Sentiment"
    with TestClient(make_app(tmp_path)) as client:
        run_id = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False}).json()["run_id"]
        wait_status(client, run_id, lambda s: len(s["current_nodes"]) == 3)
        Graph.releases["News Sentiment"].set()
        final = wait_status(client, run_id, lambda s: s["status"] == "failed")
        assert not final["resumable"]
        assert "Trader" not in Graph.calls
        assert client.get("/api/checkpoints").json()["items"] == []


@pytest.mark.parametrize("serial", [False, True])
@pytest.mark.parametrize("analysts", [[], ["news_sentiment"], ["technical_position", "fundamental_analysis"],
                                     ["technical_position", "news_sentiment", "fundamental_analysis"]])
def test_selected_analysts_and_serial_fallback(tmp_path, monkeypatch, analysts, serial):
    monkeypatch.setenv("COPILOT_FORCE_SEQUENTIAL", "1" if serial else "0")
    Graph.reset()
    for event in Graph.releases.values():
        event.set()
    with TestClient(make_app(tmp_path)) as client:
        response = client.post("/api/runs", json={"symbols": ["AAPL"], "selected_analysts": analysts,
                                                 "long_term_memory_enabled": False})
        assert response.status_code == 200, response.text
        final = wait_status(client, response.json()["run_id"], lambda s: s["status"] in {"succeeded", "failed"})
        assert final["status"] == "succeeded", final
        expected = [name for name in Graph.names if name.lower().replace(" ", "_") in analysts]
        actual = [name for name in Graph.calls if name in Graph.names]
        assert (actual == expected) if serial else (set(actual) == set(expected))
        assert Graph.calls.count("Trader") == 1


@pytest.mark.parametrize("completion_order", list(permutations(Graph.names)))
def test_all_completion_orders_commit_in_fixed_report_order(tmp_path, monkeypatch, completion_order):
    monkeypatch.delenv("COPILOT_FORCE_SEQUENTIAL", raising=False)
    Graph.reset()
    with TestClient(make_app(tmp_path)) as client:
        run_id = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False}).json()["run_id"]
        wait_status(client, run_id, lambda s: len(s["current_nodes"]) == 3)
        for name in completion_order:
            Graph.releases[name].set()
            wait_status(client, run_id, lambda s: s["nodes"][name]["status"] == "succeeded")
        final = wait_status(client, run_id, lambda s: s["status"] == "succeeded")
        assert Graph.calls.count("Trader") == 1
        # Per-node files are published only at success, without partial content.
        for name in Graph.names:
            key = name.lower().replace(" ", "_")
            text = client.get(f"/api/reports/{run_id}/{key}").text
            assert "中文分析报告" in text and "尚未完成" not in text
        assert not final["resumable"]


def test_version_one_checkpoint_remains_read_only_until_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_FORCE_SEQUENTIAL", "1")
    Graph.reset()
    Graph.releases["Technical Position"].set()
    app = make_app(tmp_path)
    with TestClient(app) as client:
        run_id = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False}).json()["run_id"]
        wait_status(client, run_id, lambda s: s["nodes"]["News Sentiment"]["status"] == "running")
        checkpoint = cancel_saved(client, run_id)
        path = app.state.runtime.checkpoints_dir / f"{checkpoint}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        payload["snapshot"].pop("completed_nodes")
        payload["snapshot"].pop("schema_version")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        original = path.read_bytes()
        assert client.get("/api/checkpoints").json()["items"][0]["checkpoint_id"] == checkpoint
        assert path.read_bytes() == original
        monkeypatch.delenv("COPILOT_FORCE_SEQUENTIAL")
        for event in Graph.releases.values():
            event.set()
        response = client.post(f"/api/checkpoints/{checkpoint}/resume")
        assert response.status_code == 200, response.text
        wait_status(client, response.json()["run_id"], lambda s: s["status"] == "succeeded")
        assert Graph.calls.count("Technical Position") == 1
        assert not path.exists()


def test_parallel_checkpoint_disk_failure_is_not_resumable(tmp_path, monkeypatch):
    import ai_trading_copilot.copilot.services.run_lifecycle as lifecycle
    monkeypatch.delenv("COPILOT_FORCE_SEQUENTIAL", raising=False)
    Graph.reset()
    original = lifecycle.write_json_atomic
    def fail_checkpoint(path, *args, **kwargs):
        if path.parent.name == ".checkpoints":
            raise OSError("disk full")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(lifecycle, "write_json_atomic", fail_checkpoint)
    with TestClient(make_app(tmp_path)) as client:
        run_id = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False}).json()["run_id"]
        wait_status(client, run_id, lambda s: len(s["current_nodes"]) == 3)
        Graph.releases["News Sentiment"].set()
        wait_status(client, run_id, lambda s: s["nodes"]["News Sentiment"]["status"] == "succeeded")
        client.post(f"/api/runs/{run_id}/cancel")
        final = wait_status(client, run_id, lambda s: s["status"] == "failed")
        assert not final["resumable"]
        assert client.get("/api/checkpoints").json()["items"] == []
        assert "Trader" not in Graph.calls

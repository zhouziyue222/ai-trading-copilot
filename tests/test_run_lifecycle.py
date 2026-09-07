# -*- coding: utf-8 -*-
import json
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.domain.models import UserPersonaConfig
from ai_trading_copilot.copilot.services import run_lifecycle
from ai_trading_copilot.copilot.services.run_checkpoint import restore_snapshot, validate_snapshot
from ai_trading_copilot.copilot.services.run_lifecycle import RunLifecycle, remove_owned
from ai_trading_copilot.copilot.services.run_tracker import write_json_atomic
from ai_trading_copilot.copilot.ui.app import UISettings, create_app
from tests.lifecycle_fakes import BoundaryGraph


@pytest.fixture
def app(tmp_path):
    BoundaryGraph.calls = []
    BoundaryGraph.finish_second.clear()
    return create_app(UISettings(reports_dir=tmp_path / "reports", graph_cls=BoundaryGraph,
                                 subscriptions_file=tmp_path / "subscriptions.json",
                                 memory_database=tmp_path / "memory.sqlite3", long_term_memory_enabled=False))


def wait_status(client, run_id, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        if response.status_code == 200 and predicate(response.json()["status"]):
            return response.json()["status"]
        time.sleep(0.01)
    raise AssertionError(f"Run did not reach expected state: {response.text}")


def start_blocked(client):
    response = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False})
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]
    wait_status(client, run_id, lambda status: status["nodes"]["Second"]["status"] == "running")
    return run_id


def cancel_saved(client, run_id):
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
    status = wait_status(client, run_id, lambda status: status["status"] == "cancelled")
    assert status["resumable"] is True
    return status["checkpoint_id"]


def test_startup_cleans_only_incomplete_ui_work(app):
    root = app.state.runtime.root
    for directory in (".runtime/old/run_UI_work", "run_UI_old", "run_UI_good", "run_CLI_existing"):
        (root / directory).mkdir(parents=True)
        write_json_atomic(root / directory / "run_status.json", {"status": "succeeded" if "good" in directory else "running"})
    (root / ".checkpoints").mkdir()
    (root / ".checkpoints" / "broken.tmp").write_text("half")
    with TestClient(app) as client:
        runtime = client.get("/api/runtime").json()
        assert runtime["status"] == "idle" and runtime["active_runs"] == []
        assert BoundaryGraph.calls == []
        assert not (root / "run_UI_old").exists()
        assert not (root / ".runtime/old").exists()
        assert not (root / ".checkpoints/broken.tmp").exists()
        assert (root / "run_UI_good").exists()
        assert (root / "run_CLI_existing").exists()


def test_cancel_resume_preserves_boundary_and_consumes_checkpoint(app):
    with TestClient(app) as client:
        run_id = start_blocked(client)
        active = client.get("/api/runtime").json()
        assert active["active_runs"][0]["run_id"] == run_id
        checkpoint_id = cancel_saved(client, run_id)
        assert client.get("/api/runtime").json()["status"] == "idle"
        path = app.state.runtime.checkpoints_dir / f"{checkpoint_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot = payload["snapshot"]
        assert snapshot["next_node_index"] == 1
        assert snapshot["state"]["explanations"]["items"] == ["完整节点"]
        assert not any("partial" in name for name in snapshot["reports"])
        restored = restore_snapshot(snapshot, app.state.runtime.work_root / "type_check", "test")
        assert isinstance(restored["state"]["persona_config"], UserPersonaConfig)
        assert client.get(f"/api/reports/{run_id}/first").text == "中文报告：已完成\n"
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
        assert len(client.get("/api/checkpoints").json()["items"]) == 1
        BoundaryGraph.finish_second.set()
        resumed = client.post(f"/api/checkpoints/{checkpoint_id}/resume")
        assert resumed.status_code == 200, resumed.text
        new_id = resumed.json()["run_id"]
        assert new_id != run_id
        wait_status(client, new_id, lambda status: status["status"] == "succeeded")
        assert BoundaryGraph.calls == ["First", "Second", "Second"]
        assert not path.exists()
        assert client.post(f"/api/checkpoints/{checkpoint_id}/resume").status_code == 409


def test_saved_cancel_survives_service_restart_and_never_autoruns(app):
    with TestClient(app) as client:
        checkpoint_id = cancel_saved(client, start_blocked(client))
        session = client.get("/api/runtime").json()["session_id"]
    new_app = create_app(app.state.ui_settings)
    calls = list(BoundaryGraph.calls)
    with TestClient(new_app) as client:
        status = client.get("/api/runtime").json()
        assert status["session_id"] != session and status["status"] == "idle"
        assert client.get("/api/checkpoints").json()["items"][0]["checkpoint_id"] == checkpoint_id
        assert BoundaryGraph.calls == calls


def test_shutdown_discards_active_and_consumed_resume(app):
    with TestClient(app) as client:
        checkpoint_id = cancel_saved(client, start_blocked(client))
        resumed = client.post(f"/api/checkpoints/{checkpoint_id}/resume").json()["run_id"]
        wait_status(client, resumed, lambda status: status["nodes"]["Second"]["status"] == "running")
    assert not (app.state.runtime.root / resumed).exists()
    assert list(app.state.runtime.checkpoints_dir.glob("*.json")) == []
    assert not app.state.runtime.work_root.exists()


def test_shutdown_wins_over_uncommitted_cancel(app):
    with TestClient(app) as client:
        run_id = start_blocked(client)
        runtime = app.state.runtime
        with runtime.lock:
            runtime.request_user_cancel(run_id)
            runtime.request_discard()
    assert list(runtime.checkpoints_dir.glob("*.json")) == []
    assert not (runtime.root / run_id).exists()


def test_checkpoint_disk_failure_is_not_reported_as_saved(app, monkeypatch):
    original = run_lifecycle.write_json_atomic
    def fail_checkpoint(path, payload, **kwargs):
        if path.parent.name == ".checkpoints":
            raise OSError("disk full")
        return original(path, payload, **kwargs)
    with TestClient(app) as client:
        run_id = start_blocked(client)
        monkeypatch.setattr(run_lifecycle, "write_json_atomic", fail_checkpoint)
        client.post(f"/api/runs/{run_id}/cancel")
        status = wait_status(client, run_id, lambda status: status["status"] == "failed")
        assert status["resumable"] is False
        assert "断点保存失败" in str(status["errors"])
        assert client.get("/api/checkpoints").json()["items"] == []


@pytest.mark.parametrize("corruption", ["schema", "cursor", "missing_report", "path", "state", "order", "nodes", "empty_state"])
def test_incompatible_checkpoint_is_rejected_before_consuming(app, corruption):
    with TestClient(app) as client:
        checkpoint_id = cancel_saved(client, start_blocked(client))
        path = app.state.runtime.checkpoints_dir / f"{checkpoint_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "schema": payload["schema_version"] += 1
        if corruption == "cursor": payload["snapshot"]["next_node_index"] = 99
        if corruption == "missing_report": payload["snapshot"]["reports"] = {}
        if corruption == "path": payload["snapshot"]["reports"]["../escape.md"] = "bad"
        if corruption == "state": payload["snapshot"]["state"]["persona_config"] = "invalid"
        if corruption == "order": payload["snapshot"]["node_order"].reverse()
        if corruption == "nodes": del payload["snapshot"]["tracker_status"]["nodes"]
        if corruption == "empty_state": payload["snapshot"]["state"] = {}
        write_json_atomic(path, payload)
        response = client.post(f"/api/checkpoints/{checkpoint_id}/resume")
        assert response.status_code == 409, response.text
        assert path.exists()
        assert client.get("/api/runtime").json()["status"] == "idle"


def test_instance_lock_does_not_clean_another_instances_work(tmp_path):
    first, second = RunLifecycle(tmp_path / "reports"), RunLifecycle(tmp_path / "reports")
    first.start()
    marker = first.work_root / "marker"
    marker.write_text("active")
    try:
        with pytest.raises(RuntimeError, match="Another UI"):
            second.start()
        assert marker.exists()
    finally:
        first.shutdown()
    second.start()
    second.shutdown()


def test_cleanup_rejects_paths_outside_root_and_links(tmp_path):
    root = tmp_path / "reports"
    root.mkdir()
    other = tmp_path / "business"
    other.mkdir()
    (other / "keep").write_text("keep")
    with pytest.raises(ValueError): remove_owned(root, other)
    try:
        (root / "linked").symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink creation is unavailable.")
    with pytest.raises(ValueError): remove_owned(root, root / "linked")
    assert (other / "keep").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
def test_cleanup_rejects_windows_junction(tmp_path):
    root, other = tmp_path / "reports", tmp_path / "business"
    root.mkdir()
    other.mkdir()
    (other / "keep").write_text("keep")
    junction = root / "junction"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(other)], capture_output=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError):
        remove_owned(root, junction)
    assert (other / "keep").read_text() == "keep"


def test_atomic_checkpoint_failure_does_not_publish_partial_json(tmp_path):
    path = tmp_path / "checkpoint.json"
    def stopped(): raise RuntimeError("shutdown")
    with pytest.raises(RuntimeError):
        write_json_atomic(path, {"中文": "正常"}, before_replace=stopped)
    assert not path.exists() and list(tmp_path.iterdir()) == []


def test_optional_future_node_can_be_lazily_registered():
    snapshot = {"state": {"selected_analysts": ["opportunity_radar"], "subscription_symbols": [],
                          "report_output_dir": "__run__", "run_id": "source", "persona_config": {}},
                "node_order": ["Load", "Opportunity Radar"], "next_node_index": 1,
                "reports": {}, "tracker_status": {"reports": {}, "nodes": {"Load": {"status": "succeeded"}}}}
    validate_snapshot(snapshot, ["Load", "Opportunity Radar"])
    snapshot["next_node_index"] = 2
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, ["Load", "Opportunity Radar"])


def test_utf8_json_errors_files_and_python_subprocess(app, tmp_path):
    with TestClient(app, raise_server_exceptions=False) as client:
        for response in (client.get("/missing"), client.post("/api/runs", json={"look_back_days": -1}),
                         client.get("/api/runtime")):
            assert response.headers["content-type"] == "application/json; charset=utf-8"
        payload = {"symbol": "AAPL", "reason": "中文验证：空闲、运行中、已取消"}
        response = client.post("/api/subscriptions", json=payload)
        assert response.json()["items"][0]["reason"] == payload["reason"]
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", "print('中文验证')"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    assert result.stdout.strip() == "中文验证"
    write_json_atomic(tmp_path / "unicode.json", payload)
    assert json.loads((tmp_path / "unicode.json").read_text(encoding="utf-8")) == payload
    with (tmp_path / "unicode.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerow(["状态", payload["reason"]])
    with (tmp_path / "unicode.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == ["状态", payload["reason"]]


def test_cancel_before_initialization_saves_only_inputs(app):
    from threading import Event
    entered, proceed = Event(), Event()
    class InitializingGraph(BoundaryGraph):
        def __init__(self, **kwargs):
            entered.set()
            assert proceed.wait(5)
            super().__init__(**kwargs)
    settings = app.state.ui_settings
    from dataclasses import replace
    app = create_app(replace(settings, graph_cls=InitializingGraph))
    with TestClient(app) as client:
        run_id = client.post("/api/runs", json={"symbols": ["AAPL"]}).json()["run_id"]
        assert entered.wait(3)
        client.post(f"/api/runs/{run_id}/cancel")
        proceed.set()
        status = wait_status(client, run_id, lambda status: status["status"] == "cancelled")
        path = app.state.runtime.checkpoints_dir / f"{status['checkpoint_id']}.json"
        assert json.loads(path.read_text(encoding="utf-8"))["snapshot"] == {"state": None}
        assert BoundaryGraph.calls == []
        BoundaryGraph.finish_second.set()
        resumed = client.post(f"/api/checkpoints/{status['checkpoint_id']}/resume")
        assert resumed.status_code == 200
        wait_status(client, resumed.json()["run_id"], lambda status: status["status"] == "succeeded")
        assert BoundaryGraph.calls == ["First", "Second"]


def test_cancel_resumed_run_during_initialization_keeps_boundary(app):
    from threading import Event
    from dataclasses import replace
    entered, proceed = Event(), Event()
    class SlowResumeGraph(BoundaryGraph):
        def __init__(self, **kwargs):
            entered.set()
            assert proceed.wait(5)
            super().__init__(**kwargs)
    with TestClient(app) as client:
        checkpoint_id = cancel_saved(client, start_blocked(client))
    app = create_app(replace(app.state.ui_settings, graph_cls=SlowResumeGraph))
    with TestClient(app) as client:
        response = client.post(f"/api/checkpoints/{checkpoint_id}/resume")
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        assert entered.wait(3)
        client.post(f"/api/runs/{run_id}/cancel")
        proceed.set()
        status = wait_status(client, run_id, lambda status: status["status"] == "cancelled")
        new_path = app.state.runtime.checkpoints_dir / f"{status['checkpoint_id']}.json"
        snapshot = json.loads(new_path.read_text(encoding="utf-8"))["snapshot"]
        assert snapshot["next_node_index"] == 1
        assert snapshot["state"]["explanations"]["items"] == ["完整节点"]


@pytest.mark.parametrize("cancel", [False, True])
def test_shutdown_during_final_flush_discards_publication(app, monkeypatch, cancel):
    with TestClient(app) as client:
        run_id = start_blocked(client)
        runtime = app.state.runtime
        control = runtime.controls[run_id]
        original = control.tracker.flush
        def flush():
            # The last flush contains paths rebased to the final directory.
            reports = control.tracker.status.get("reports", {})
            if "run_audit" in reports and ".runtime" not in reports["run_audit"]["path"]:
                runtime.request_discard()
            return original()
        monkeypatch.setattr(control.tracker, "flush", flush)
        if cancel:
            client.post(f"/api/runs/{run_id}/cancel")
        else:
            BoundaryGraph.finish_second.set()
        control.thread.join(5)
        assert not control.thread.is_alive()
        assert not (runtime.root / run_id).exists()
        assert list(runtime.checkpoints_dir.glob("*.json")) == []

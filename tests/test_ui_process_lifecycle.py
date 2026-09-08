# -*- coding: utf-8 -*-
"""Exercise signals and OS process death against an isolated offline server."""

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.test_run_lifecycle import start_blocked, cancel_saved, wait_status


def process_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel.CloseHandle(handle)


def force_kill(pid):
    if os.name != "nt":
        os.kill(pid, signal.SIGKILL)
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(1, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.TerminateProcess(handle, 99):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def offline_server(root, mode=None):
    root.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (root / f"server-{time.time_ns()}.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "tests.lifecycle_server", str(root), str(port), *([mode] if mode else [])],
                               cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log,
                               env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5)
    service_pid = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.flush()
                raise AssertionError(Path(log.name).read_text(encoding="utf-8"))
            try:
                response = client.get("/api/runtime")
                if response.status_code == 200:
                    service_pid = response.json()["pid"]
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        assert service_pid is not None, "Test server did not start."
        yield client, process, service_pid
    finally:
        if process.poll() is None:
            try:
                client.post("/test/sigint")
                process.wait(10)
            except (httpx.TransportError, subprocess.TimeoutExpired):
                if service_pid and process_alive(service_pid):
                    force_kill(service_pid)
                if process.poll() is None:
                    process.kill()
                process.wait(5)
        client.close()
        log.close()


@pytest.mark.parametrize("termination", ["sigint", "kill"])
def test_real_service_exit_discards_work_and_starts_idle(tmp_path, termination):
    with offline_server(tmp_path) as (client, process, pid):
        session = client.get("/api/runtime").json()["session_id"]
        run_id = start_blocked(client)
        if termination == "sigint":
            client.post("/test/sigint")
        else:
            force_kill(pid)
        process.wait(15)
    with offline_server(tmp_path) as (client, _, __):
        status = client.get("/api/runtime").json()
        assert status["status"] == "idle" and status["session_id"] != session
        assert client.get("/api/checkpoints").json()["items"] == []
        assert client.get(f"/api/runs/{run_id}").status_code == 404
        assert client.get("/test/calls").json()["calls"] == []
        assert not (tmp_path / "reports/.runtime" / session).exists()


def test_cancel_restart_resume_then_kill_consumes_progress(tmp_path):
    with offline_server(tmp_path) as (client, process, pid):
        checkpoint = cancel_saved(client, start_blocked(client))
        force_kill(pid)
        process.wait(10)
    with offline_server(tmp_path) as (client, process, pid):
        assert client.get("/api/runtime").json()["status"] == "idle"
        assert len(client.get("/api/checkpoints").json()["items"]) == 1
        response = client.post(f"/api/checkpoints/{checkpoint}/resume")
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        wait_status(client, run_id, lambda status: status["nodes"]["Second"]["status"] == "running")
        assert client.get("/test/calls").json()["calls"] == ["Second"]
        force_kill(pid)
        process.wait(10)
    with offline_server(tmp_path) as (client, _, __):
        assert client.get("/api/checkpoints").json()["items"] == []
        assert client.get("/api/runtime").json()["status"] == "idle"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree contract")
def test_windows_job_kills_children_when_service_is_killed(tmp_path):
    with offline_server(tmp_path) as (client, process, pid):
        child_pid = client.post("/test/child").json()["pid"]
        assert process_alive(child_pid)
        force_kill(pid)
        process.wait(10)
        deadline = time.monotonic() + 5
        while process_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_alive(child_pid)


def test_kill_during_checkpoint_write_never_exposes_partial_checkpoint(tmp_path):
    directory = tmp_path / "reports/.checkpoints"
    directory.mkdir(parents=True)
    destination = directory / "checkpoint_partial.json"
    ready = tmp_path / "writer-ready"
    code = (
        "import os,sys,time\nfrom pathlib import Path\n"
        "from ai_trading_copilot.copilot.services.run_tracker import write_json_atomic\n"
        "def pause():\n Path(sys.argv[2]).write_text(str(os.getpid()))\n time.sleep(300)\n"
        "write_json_atomic(Path(sys.argv[1]), {'origin':'user_cancel'}, before_replace=pause)\n"
    )
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-c", code, str(destination), str(ready)])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        force_kill(int(ready.read_text()))
        process.wait(10)
        assert not destination.exists()
        assert list(directory.glob("*.tmp"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
    with offline_server(tmp_path) as (client, _, __):
        assert client.get("/api/checkpoints").json()["items"] == []
        assert list(directory.glob("*.tmp")) == []


def test_occupied_port_does_not_trigger_legacy_cleanup(tmp_path):
    import json
    legacy = tmp_path / "reports/run_UI_old"
    legacy.mkdir(parents=True)
    status = legacy / "run_status.json"
    status.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        result = subprocess.run([sys.executable, "-X", "utf8", "-m", "ai_trading_copilot.copilot.ui.server",
                                 "--port", str(port), "--reports-dir", str(tmp_path / "reports")],
                                capture_output=True, timeout=20)
    assert result.returncode != 0
    assert status.exists()


@pytest.mark.parametrize("termination", ["sigint", "kill"])
def test_parallel_workers_discard_on_process_exit(tmp_path, termination):
    with offline_server(tmp_path, "parallel") as (client, process, pid):
        run_id = client.post("/api/runs", json={"symbols": ["AAPL"], "long_term_memory_enabled": False}).json()["run_id"]
        wait_status(client, run_id, lambda s: len(s["current_nodes"]) == 3)
        if termination == "kill":
            force_kill(pid)
        else:
            client.post("/test/sigint")
        process.wait(10)
    with offline_server(tmp_path, "parallel") as (client, _, __):
        assert client.get("/api/runtime").json()["status"] == "idle"
        assert client.get("/api/checkpoints").json()["items"] == []
        assert not list((tmp_path / "reports/.runtime").glob("*/run_*"))

# -*- coding: utf-8 -*-
"""Own UI work, ephemeral outputs and explicitly committed checkpoints."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
import uuid
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock, Thread

from ai_trading_copilot.copilot.services.cancellation import GLOBAL_CANCELLATION_MANAGER, StopReason
from ai_trading_copilot.copilot.services.run_checkpoint import (
    SCHEMA_VERSION, capture_snapshot, json_value, move_paths, restore_snapshot, validate_snapshot,
)
from ai_trading_copilot.copilot.services.run_tracker import RunTracker, write_json_atomic

ACTIVE_RUNTIMES: weakref.WeakSet = weakref.WeakSet()
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (ValueError, UnicodeError):
        return {}


def safe_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value) and ".." not in value)


def checked_path(root: Path, path: Path) -> Path:
    """Reject links/junctions before resolving, including linked ancestors."""
    root, path = root.absolute(), path.absolute()
    if not path.is_relative_to(root):
        raise ValueError("Path is outside the report directory.")
    for item in [path, *path.parents]:
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0)
                                & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"Linked paths are not allowed: {item}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path is outside the report directory.")
    return path


def remove_owned(root: Path, path: Path) -> None:
    path = checked_path(root, path)
    if path == root.absolute():
        raise ValueError("Cannot remove the report root.")
    if not path.exists():
        return
    if path.is_dir():
        for directory, directories, files in os.walk(path, followlinks=False):
            for name in [*directories, *files]:
                checked_path(root, Path(directory) / name)
        shutil.rmtree(path)
    else:
        path.unlink()


@dataclass
class RunControl:
    run_id: str
    output_dir: Path
    tracker: RunTracker
    params: dict
    cancel_event: Event = field(default_factory=Event)
    thread: Thread | None = None
    reason: StopReason | None = None
    snapshot: dict = field(default_factory=lambda: {"state": None})
    resume_snapshot: dict | None = None
    checkpoint_id: str | None = None


class RunLifecycle:
    def __init__(self, reports_dir: Path):
        self.root = reports_dir.absolute()
        self.session_id = uuid.uuid4().hex
        self.controls: dict[str, RunControl] = {}
        # ponytail: one lock serializes local UI commits; split per run only if
        # simultaneous report publication becomes a measured bottleneck.
        self.lock = RLock()
        self.shutting_down = Event()
        self._lease = None
        self.started = False

    @property
    def work_root(self) -> Path:
        return self.root / ".runtime" / self.session_id

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / ".checkpoints"

    def start(self) -> None:
        with self.lock:
            if self.started:
                return
            checked_path(self.root, self.root).mkdir(parents=True, exist_ok=True)
            lease_path = checked_path(self.root, self.root / ".ui.lock")
            lease = lease_path.open("a+b")
            try:
                if lease.tell() == 0:
                    lease.write(b"0")
                    lease.flush()
                lease.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                lease.close()
                raise RuntimeError("Another UI service owns this reports directory.") from exc
            self._lease = lease
            try:
                remove_owned(self.root, self.root / ".runtime")
                checked_path(self.root, self.checkpoints_dir).mkdir(exist_ok=True)
                for path in self.checkpoints_dir.iterdir():
                    checked_path(self.root, path)
                    if path.name.endswith(".tmp"):
                        remove_owned(self.root, path)
                for path in self.root.glob("run_UI_*"):
                    checked_path(self.root, path)
                    if path.is_dir() and read_json(path / "run_status.json").get("status") not in TERMINAL:
                        remove_owned(self.root, path)
                self.work_root.mkdir(parents=True)
                self.started = True
                ACTIVE_RUNTIMES.add(self)
            except BaseException:
                self._release()
                raise

    def _release(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        ACTIVE_RUNTIMES.discard(self)

    def new_run_dir(self) -> tuple[str, Path]:
        self.start()
        if self.shutting_down.is_set():
            raise RuntimeError("UI service is shutting down.")
        run_id = "run_UI_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12]
        output = checked_path(self.root, self.work_root / run_id)
        output.mkdir()
        return run_id, output

    def locate(self, run_id: str) -> Path:
        if not safe_id(run_id):
            raise ValueError("Invalid run id.")
        with self.lock:
            control = self.controls.get(run_id)
            path = control.output_dir if control else self.root / run_id
            return checked_path(self.root, path)

    def runtime_status(self) -> dict:
        self.start()
        with self.lock:
            active = [
                {"run_id": c.run_id, "status": "running", "cancel_requested": c.reason == StopReason.USER_CANCEL,
                 "started_at": c.tracker.status["started_at"]}
                for c in self.controls.values()
                if c.thread is not None and c.thread.is_alive()
            ]
            active.sort(key=lambda item: item["started_at"], reverse=True)
            return {"session_id": self.session_id, "pid": os.getpid(),
                    "project_root": str(Path(__file__).resolve().parents[2]),
                    "status": "running" if active else "idle", "active_runs": active}

    def capture(self, control: RunControl, state: dict, order: list[str], index: int) -> None:
        snapshot = capture_snapshot(state, order, index, control.tracker)
        with self.lock:
            if not self.shutting_down.is_set():
                control.snapshot = snapshot

    def request_user_cancel(self, run_id: str) -> bool:
        with self.lock:
            control = self.controls.get(run_id)
            if control is None or self.shutting_down.is_set():
                return False
            control.reason = StopReason.USER_CANCEL
            control.tracker.request_cancel()
            control.cancel_event.set()
        GLOBAL_CANCELLATION_MANAGER.request_cancel(run_id)
        return True

    def request_discard(self) -> None:
        # Set the gate before waiting for a worker's lock / current IO.
        self.shutting_down.set()
        with self.lock:
            controls = list(self.controls.values())
            for control in controls:
                control.reason = StopReason.SERVER_SHUTDOWN
                control.cancel_event.set()
        # Third party cancellation callbacks may block. Do not run them in the
        # signal handler; daemon workers and child processes die with the server.

    def shutdown(self, timeout: float = 5.0) -> None:
        self.request_discard()
        deadline = time.monotonic() + timeout
        for control in list(self.controls.values()):
            if control.thread is not None:
                control.thread.join(max(0, deadline - time.monotonic()))
        with self.lock:
            if not self.controls:
                remove_owned(self.root, self.work_root)
                self._release()
            # A stuck daemon still owns its output directory. Keep the lease
            # until it exits or the OS closes it on process termination.

    def unregister(self, control: RunControl) -> None:
        with self.lock:
            self.controls.pop(control.run_id, None)
            control.snapshot = {"state": None}
            control.resume_snapshot = None
            if self.shutting_down.is_set() and not self.controls:
                remove_owned(self.root, self.work_root)
                self._release()

    def list_checkpoints(self) -> list[dict]:
        items = []
        for path in self.checkpoints_dir.glob("checkpoint_*.json"):
            payload = read_json(checked_path(self.root, path))
            if payload.get("origin") == "user_cancel" and payload.get("schema_version") == SCHEMA_VERSION:
                items.append({"checkpoint_id": path.stem, "run_id": payload.get("run_id"),
                              "symbols": payload.get("params", {}).get("subscription_symbols", []),
                              "created_at": payload.get("created_at")})
        return sorted(items, key=lambda item: item["created_at"] or "", reverse=True)

    def read_checkpoint(self, checkpoint_id: str, allowed_nodes: list[str]) -> dict:
        if not safe_id(checkpoint_id) or not checkpoint_id.startswith("checkpoint_"):
            raise ValueError("Invalid checkpoint id.")
        path = checked_path(self.root, self.checkpoints_dir / f"{checkpoint_id}.json")
        if not path.exists():
            if (self.checkpoints_dir / f"{checkpoint_id}.consumed").exists():
                raise ValueError("Checkpoint has already been consumed.")
            raise FileNotFoundError("Checkpoint not found.")
        payload = read_json(path)
        if (payload.get("schema_version") != SCHEMA_VERSION or payload.get("origin") != "user_cancel"
                or payload.get("checkpoint_id") != checkpoint_id or not isinstance(payload.get("params"), dict)):
            raise ValueError("Invalid or incompatible checkpoint.")
        validate_snapshot(payload.get("snapshot"), allowed_nodes)
        return payload

    def consume(self, checkpoint_id: str, control: RunControl, payload: dict) -> None:
        restored = restore_snapshot(payload["snapshot"], control.output_dir, control.run_id)
        source = checked_path(self.root, self.checkpoints_dir / f"{checkpoint_id}.json")
        # A receipt has no progress; it only distinguishes consumed from absent.
        write_json_atomic(self.checkpoints_dir / f"{checkpoint_id}.consumed", {"run_id": control.run_id})
        os.replace(source, control.output_dir / "resume-input.json")
        control.resume_snapshot = restored
        # A second cancel during agent initialization must retain the consumed
        # boundary even before graph.resume() has a chance to publish a snapshot.
        control.snapshot = deepcopy(payload["snapshot"])
        if restored is not None:
            for key in ("nodes", "reports"):
                control.tracker.status[key] = restored["tracker_status"][key]
            control.tracker.flush()

    def finalize(self, control: RunControl, *, state: dict | None, error: Exception | None) -> None:
        with self.lock:
            if self.shutting_down.is_set() or control.reason == StopReason.SERVER_SHUTDOWN:
                remove_owned(self.root, control.output_dir)
                return
            tracker = control.tracker
            cancelled = control.reason == StopReason.USER_CANCEL
            if cancelled:
                checkpoint_id = "checkpoint_" + uuid.uuid4().hex
                payload = {"schema_version": SCHEMA_VERSION, "checkpoint_id": checkpoint_id,
                           "origin": "user_cancel", "run_id": control.run_id,
                           "created_at": datetime.now(timezone.utc).isoformat(),
                           "params": json_value(control.params), "defaults": tracker.status["defaults"],
                           "snapshot": control.snapshot}
                # Paths in parameters are execution-local and regenerated on resume.
                payload["params"].pop("report_output_dir", None)
                payload["params"].pop("run_id", None)
                try:
                    if self.shutting_down.is_set():
                        remove_owned(self.root, control.output_dir)
                        return
                    def check_commit():
                        if self.shutting_down.is_set():
                            raise RuntimeError("Shutdown interrupted checkpoint commit.")
                    write_json_atomic(self.checkpoints_dir / f"{checkpoint_id}.json", payload,
                                      before_replace=check_commit)
                    control.checkpoint_id = checkpoint_id
                except Exception as exc:
                    error = RuntimeError(f"已停止，断点保存失败：{exc}")
                    cancelled = False
                if self.shutting_down.is_set():
                    remove_owned(self.root, self.checkpoints_dir / f"{checkpoint_id}.json")
                    remove_owned(self.root, control.output_dir)
                    return
                if cancelled:
                    # Retain only stable reports; never publish partial node files.
                    remove_owned(self.root, control.output_dir)
                    control.output_dir.mkdir()
                    restored = restore_snapshot(control.snapshot, control.output_dir, control.run_id)
                    if restored is not None:
                        tracker.status["nodes"] = restored["tracker_status"]["nodes"]
                        tracker.status["reports"] = restored["tracker_status"]["reports"]
                        state = restored["state"]
                    else:
                        tracker.status["reports"] = {}
                        state = None
                    tracker.cancel()
                    tracker.status.update(resumable=True, checkpoint_id=checkpoint_id)
            if not cancelled:
                if error is not None:
                    remove_owned(self.root, control.output_dir)
                    control.output_dir.mkdir()
                    tracker.status["reports"] = {}
                    tracker.status["params"] = {}
                    tracker.status["nodes"] = {}
                    state = None
                    tracker.add_error(str(error))
                tracker.status["resumable"] = False
                tracker.finish(failed=error is not None)
            tracker.write_audit(state=state, error=error if not cancelled else None)
            if self.shutting_down.is_set():
                if control.checkpoint_id:
                    remove_owned(self.root, self.checkpoints_dir / f"{control.checkpoint_id}.json")
                remove_owned(self.root, control.output_dir)
                return
            # Publish under one directory rename. Readers hold this same lock.
            final = checked_path(self.root, self.root / control.run_id)
            tracker.status = move_paths(tracker.status, str(control.output_dir), str(final))
            tracker.flush()
            if self.shutting_down.is_set():
                if control.checkpoint_id:
                    remove_owned(self.root, self.checkpoints_dir / f"{control.checkpoint_id}.json")
                remove_owned(self.root, control.output_dir)
                return
            (control.output_dir / "resume-input.json").unlink(missing_ok=True)
            os.replace(control.output_dir, final)
            if self.shutting_down.is_set():
                if control.checkpoint_id:
                    remove_owned(self.root, self.checkpoints_dir / f"{control.checkpoint_id}.json")
                remove_owned(self.root, final)
                return
            tracker.output_dir = final
            tracker.status_path = final / "run_status.json"
            control.output_dir = final


def discard_active_runs() -> None:
    for runtime in list(ACTIVE_RUNTIMES):
        runtime.request_discard()

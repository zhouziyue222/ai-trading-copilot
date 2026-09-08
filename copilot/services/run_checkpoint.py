# -*- coding: utf-8 -*-
"""Lossless, versioned node-boundary snapshots.

Version 2 records completed nodes explicitly so analyst branches can finish in
different orders.  Version 1 snapshots remain readable and are normalized in
memory until the checkpoint is consumed.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from pydantic import TypeAdapter

from ai_trading_copilot.copilot.graph.state import CopilotGraphState

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
STATE_ADAPTER = TypeAdapter(CopilotGraphState)
JSON_ADAPTER = TypeAdapter(dict[str, Any])
PATH_MARKER = "__run__"


def json_value(value: dict) -> dict:
    return json.loads(JSON_ADAPTER.dump_json(value))


def relative_file(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or not path.parts or path.is_absolute() or ".." in path.parts
            or "\\" in value or ":" in value or path.as_posix() != value):
        raise ValueError("Invalid checkpoint report path.")
    return value


def move_paths(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, dict):
        return {key: move_paths(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [move_paths(item, source, destination) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        base = source.replace("\\", "/").rstrip("/")
        if normalized == base:
            return destination
        if normalized.startswith(base + "/"):
            suffix = relative_file(normalized[len(base) + 1:])
            return destination.rstrip("/\\") + "/" + suffix
    return value


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value.replace("\\", "/")


def normalize_snapshot(snapshot: dict, schema_version: int | None = None) -> dict:
    """Return a v2 in-memory view without rewriting the source checkpoint."""
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid checkpoint snapshot.")
    version = schema_version or (2 if "completed_nodes" in snapshot else 1)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported checkpoint schema.")
    normalized = deepcopy(snapshot)
    if normalized.get("state") is None:
        normalized["completed_nodes"] = []
        normalized["schema_version"] = 2
        return normalized
    order = normalized.get("node_order") or []
    if version == 1:
        index = normalized.get("next_node_index")
        normalized["completed_nodes"] = list(order[:index]) if type(index) is int else []
    elif not isinstance(normalized.get("completed_nodes"), list):
        raise ValueError("Checkpoint completed node set is missing.")
    normalized["schema_version"] = 2
    return normalized


def capture_snapshot(
    state: dict,
    node_order: list[str],
    next_index: int,
    tracker,
    *,
    completed_nodes: list[str] | None = None,
) -> dict:
    """Capture only a stable boundary; callers provide already-merged state."""
    root = tracker.output_dir.resolve()
    completed = list(completed_nodes) if completed_nodes is not None else list(node_order[:next_index])
    if len(set(completed)) != len(completed) or any(node not in node_order for node in completed):
        raise ValueError("Invalid completed node set.")
    reports = {}
    with tracker._lock:
        status = deepcopy(tracker.status)
    original_nodes = status.get("nodes", {})
    status["nodes"] = {}
    for node, value in original_nodes.items():
        if node in completed or value.get("status") == "skipped":
            status["nodes"][node] = value
        else:
            pending = deepcopy(value)
            pending.update(status="pending", started_at=None, finished_at=None,
                           error=None, activity=None, activity_history=[])
            status["nodes"][node] = pending
    status["current_node"] = None
    status["current_nodes"] = []
    state_payload = json.loads(STATE_ADAPTER.dump_json(state))
    # Windows may spell the same directory with an 8.3 alias, so normalize the
    # canonical execution root explicitly before rebasing nested path values.
    if isinstance(state_payload, dict) and "report_output_dir" in state_payload:
        state_payload["report_output_dir"] = str(root)
    serialized_state = move_paths(state_payload, str(root), PATH_MARKER)
    referenced = {
        value[len(PATH_MARKER) + 1:]
        for value in _strings(serialized_state)
        if value.startswith(PATH_MARKER + "/")
    }
    retained_report_keys = set()
    for key, report in status.get("reports", {}).items():
        path = Path(report["path"])
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if relative in referenced and path.is_file():
            reports[relative_file(relative)] = path.read_text(encoding="utf-8")
            retained_report_keys.add(key)
    status["reports"] = {
        key: value for key, value in status.get("reports", {}).items()
        if key in retained_report_keys
    }
    return {
        "schema_version": 2,
        "state": serialized_state,
        "node_order": list(node_order),
        "next_node_index": next_index,
        "completed_nodes": completed,
        "reports": reports,
        "tracker_status": move_paths(status, str(root), PATH_MARKER),
    }


def validate_snapshot(
    snapshot: dict,
    allowed_nodes: list[str],
    *,
    schema_version: int | None = None,
) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid checkpoint snapshot.")
    if snapshot.get("state") is None:
        if ("state" not in snapshot or set(snapshot) - {"state", "schema_version", "completed_nodes"}
                or snapshot.get("completed_nodes", []) != []):
            raise ValueError("Invalid initialization checkpoint.")
        return
    snapshot = normalize_snapshot(snapshot, schema_version)
    order = snapshot.get("node_order")
    index = snapshot.get("next_node_index")
    completed = snapshot.get("completed_nodes")
    if (not isinstance(order, list) or len(set(order)) != len(order)
            or any(node not in allowed_nodes for node in order)
            or type(index) is not int or not 0 <= index <= len(order)
            or not isinstance(completed, list) or len(set(completed)) != len(completed)
            or any(node not in order for node in completed)):
        raise ValueError("Incompatible checkpoint node order or cursor.")
    expected_index = next((position for position, node in enumerate(order) if node not in completed), len(order))
    if index != expected_index:
        raise ValueError("Checkpoint cursor disagrees with completed nodes.")
    analysts = {"Technical Position", "News Sentiment", "Fundamental Analysis"}
    for position, node in enumerate(order):
        if node in completed:
            prerequisites = set(order[:position])
            if node in analysts:
                prerequisites -= analysts
            if not prerequisites.issubset(completed):
                raise ValueError("Checkpoint skips a required predecessor.")
    STATE_ADAPTER.validate_python(snapshot["state"])
    required = {"subscription_symbols", "selected_analysts", "report_output_dir", "run_id", "persona_config"}
    if not required.issubset(snapshot["state"]) or snapshot["state"]["report_output_dir"] != PATH_MARKER:
        raise ValueError("Checkpoint is missing required graph inputs.")
    reports = snapshot.get("reports")
    if not isinstance(reports, dict):
        raise ValueError("Checkpoint reports are missing.")
    for name, content in reports.items():
        relative_file(name)
        if not isinstance(content, str):
            raise ValueError("Invalid checkpoint report content.")
    status = snapshot.get("tracker_status")
    if (not isinstance(status, dict) or not isinstance(status.get("nodes"), dict)
            or not isinstance(status.get("reports"), dict)):
        raise ValueError("Checkpoint tracker status is missing.")
    for name in completed:
        node = status["nodes"].get(name)
        if not isinstance(node, dict) or node.get("status") != "succeeded":
            raise ValueError("Checkpoint completed node is not stable.")
    for name, node in status["nodes"].items():
        if not isinstance(node, dict) or (name not in completed and node.get("status") not in {"pending", "skipped"}):
            raise ValueError("Checkpoint contains an uncommitted node.")
    for report in status.get("reports", {}).values():
        if not isinstance(report, dict):
            raise ValueError("Invalid checkpoint report metadata.")
        path = report.get("path", "")
        if not path.startswith(PATH_MARKER + "/") or path[len(PATH_MARKER) + 1:] not in reports:
            raise ValueError("Checkpoint report is missing or outside the run directory.")
    for value in _strings(snapshot["state"]):
        if value.startswith(PATH_MARKER + "/") and value[len(PATH_MARKER) + 1:] not in reports:
            raise ValueError("Checkpoint state references a missing report.")


def restore_snapshot(snapshot: dict, output_dir: Path, run_id: str) -> dict | None:
    if snapshot["state"] is None:
        return None
    restored = move_paths(deepcopy(snapshot), PATH_MARKER, str(output_dir.resolve()))
    for relative, content in restored["reports"].items():
        path = output_dir / relative_file(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    restored["state"] = STATE_ADAPTER.validate_python(restored["state"])
    restored["state"]["run_id"] = run_id
    restored["state"]["report_output_dir"] = str(output_dir)
    return restored

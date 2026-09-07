# -*- coding: utf-8 -*-
"""Lossless, versioned node-boundary snapshots. No disk writes during capture."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import TypeAdapter

from ai_trading_copilot.copilot.graph.state import CopilotGraphState

SCHEMA_VERSION = 1
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


def capture_snapshot(state: dict, node_order: list[str], next_index: int, tracker) -> dict:
    root = tracker.output_dir.resolve()
    reports = {}
    with tracker._lock:
        status = deepcopy(tracker.status)
    for report in status.get("reports", {}).values():
        path = Path(report["path"])
        relative = path.resolve().relative_to(root).as_posix()
        if path.is_file():
            reports[relative_file(relative)] = path.read_text(encoding="utf-8")
    return {
        "state": move_paths(json.loads(STATE_ADAPTER.dump_json(state)), str(root), PATH_MARKER),
        "node_order": list(node_order),
        "next_node_index": next_index,
        "reports": reports,
        "tracker_status": move_paths(status, str(root), PATH_MARKER),
    }


def validate_snapshot(snapshot: dict, allowed_nodes: list[str]) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid checkpoint snapshot.")
    if snapshot.get("state") is None:
        if snapshot != {"state": None}:
            raise ValueError("Invalid initialization checkpoint.")
        return
    order = snapshot.get("node_order")
    index = snapshot.get("next_node_index")
    if (not isinstance(order, list) or len(set(order)) != len(order)
            or any(node not in allowed_nodes for node in order)
            or type(index) is not int or not 0 <= index <= len(order)):
        raise ValueError("Incompatible checkpoint node order or cursor.")
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
    for position, name in enumerate(order):
        node = status["nodes"].get(name)
        # Optional nodes (e.g. Opportunity Radar) are registered lazily when
        # execution reaches them. A missing future node is still pending.
        if node is None and position >= index:
            continue
        if not isinstance(node, dict) or node.get("status") not in {"pending", "succeeded", "skipped"}:
            raise ValueError("Invalid checkpoint node status.")
        if position < index and node["status"] != "succeeded":
            raise ValueError("Checkpoint cursor is ahead of completed nodes.")
    references = list(status.get("reports", {}).values())
    references += [{"path": path} for key in ("agent_reports", "analyst_reports")
                   for path in snapshot["state"].get(key, {}).values()]
    for report in references:
        if not isinstance(report, dict):
            raise ValueError("Invalid checkpoint report metadata.")
        path = report.get("path", "")
        if not path.startswith(PATH_MARKER + "/") or path[len(PATH_MARKER) + 1:] not in reports:
            raise ValueError("Checkpoint report is missing or outside the run directory.")


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

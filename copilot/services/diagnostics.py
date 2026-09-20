# -*- coding: utf-8 -*-
"""Bounded, local diagnostic events; never a source of resumable run state."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock

CONTEXT = ContextVar("copilot_diagnostics", default={})
ROOT = Path(__file__).resolve().parents[2]
SECRET = re.compile(
    r'''(?i)(bearer\s+)[\w.\-]+|((?:api[_-]?key|token|password|secret|authorization|cookie)['"]?\s*[=:]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)'''
)


def redact(value):
    if isinstance(value, dict):
        return {
            str(k): (
                "<redacted>"
                if (
                    not str(k).lower().startswith("usage.")
                    and re.search(
                        r"key|token|secret|password|authorization|cookie|account|header",
                        str(k),
                        re.I,
                    )
                )
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value[:100]]
    if isinstance(value, str):
        value = SECRET.sub(lambda m: (m.group(1) or m.group(2) or "") + "<redacted>", value)
        for key, secret in os.environ.items():
            if len(secret) >= 8 and re.search(r"key|token|secret|password", key, re.I):
                value = value.replace(secret, "<redacted>")
        return value[:12000]
    return value if value is None or isinstance(value, (int, float, bool)) else str(value)[:500]


def exception_data(exc: BaseException) -> dict:
    frames = []
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": redact(str(current))})
        for frame in traceback.extract_tb(current.__traceback__):
            path = Path(frame.filename)
            try:
                relative = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                relative = path.name
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            except OSError:
                digest = None
            frames.append({"file": relative, "function": frame.name, "line": frame.lineno, "source_sha256": digest})
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    own = next((f for f in reversed(frames) if f["file"].startswith("copilot/")), frames[-1] if frames else {})
    fingerprint = hashlib.sha256(f"{type(exc).__name__}:{own.get('file')}:{own.get('function')}".encode()).hexdigest()[:16]
    return {"error.type": type(exc).__name__, "error.message": redact(str(exc)),
            "exception.stacktrace": frames, "exception.chain": chain, "error.group": fingerprint,
            "code.file.path": own.get("file"), "code.function.name": own.get("function"),
            "code.line.number": own.get("line"), "code.source_sha256": own.get("source_sha256")}


class DiagnosticStore:
    def __init__(self, root: Path):
        self.root = root
        self.lock = RLock()
        self.dropped = 0
        self.last_error = None
        self.last_event_at = None
        self.logger = None

    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / "events.sqlite3", timeout=1)
        connection.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, timestamp TEXT, request_id TEXT, run_id TEXT, trace_id TEXT, payload TEXT)")
        for column in ("request_id", "run_id", "trace_id"):
            connection.execute(f"CREATE INDEX IF NOT EXISTS events_{column} ON events ({column})")
        return connection

    def record(self, event: str, **values):
        context = {k: v for k, v in CONTEXT.get().items() if k != "store"}
        payload = redact({"timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
                          "service.name": "ai-trading-copilot", "service.version": os.getenv("COPILOT_BUILD_VERSION", "working-tree"),
                          **context, **values})
        try:
            with self.lock:
                encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
                with self._connect() as db:
                    db.execute("INSERT INTO events(timestamp,request_id,run_id,trace_id,payload) VALUES(?,?,?,?,?)",
                               (payload["timestamp"], payload.get("origin_request_id") or payload.get("request_id"),
                                payload.get("run_id"), payload.get("trace_id"), encoded))
                    db.execute("DELETE FROM events WHERE id <= (SELECT COALESCE(MAX(id),0)-10000 FROM events)")
                if self.logger is None:
                    self.logger = logging.Logger(f"copilot.diagnostics.{id(self)}")
                    self.logger.addHandler(RotatingFileHandler(self.root / "events.jsonl", maxBytes=5_000_000, backupCount=2, encoding="utf-8"))
                self.logger.info(encoded)
                self.last_event_at = payload["timestamp"]
        except Exception as exc:
            self.dropped += 1
            self.last_error = type(exc).__name__

    def query(self, query: str = "") -> dict:
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE ?='' OR request_id=? OR run_id=? OR trace_id=? ORDER BY id DESC LIMIT 1000",
                              (query, query, query, query)).fetchall()
        events = [json.loads(row[0]) for row in rows]
        groups, services = {}, {}
        usage: dict[str, dict[str, Any]] = {}
        for item in events:
            if item.get("error.group"):
                group = groups.setdefault(item["error.group"], {"count": 0, "latest": item})
                group["count"] += 1
            if item.get("event") == "span.end" and item.get("name") in {"tool.call", "llm.invoke", "http.client"}:
                name = item.get("tool") or item.get("name")
                stats = services.setdefault(name, {"samples": [], "errors": 0, "cache_hits": 0})
                stats["samples"].append(item.get("duration_ms", 0))
                stats["errors"] += item.get("outcome") == "error"
                stats["cache_hits"] += bool(item.get("cache_hit"))
            if item.get("event") == "usage.record":
                kind = str(item.get("usage_kind") or "llm")
                model = str(item.get("usage_model") or "unknown")
                key = f"{kind}:{model}"
                row = usage.setdefault(
                    key,
                    {
                        "name": key,
                        "count": 0,
                        "missing_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                row["count"] += 1
                prompt = item.get("usage.prompt_tokens")
                completion = item.get("usage.completion_tokens")
                total = item.get("usage.total_tokens")
                if prompt is None and completion is None:
                    row["missing_count"] += 1
                row["prompt_tokens"] += int(prompt or 0)
                row["completion_tokens"] += int(completion or 0)
                row["total_tokens"] += int(total or 0)
        metrics = []
        for name, stats in services.items():
            times = sorted(stats["samples"])
            metrics.append({"name": name, "count": len(times), "p50_ms": times[(len(times)-1)//2],
                            "p95_ms": times[min(len(times)-1, int(len(times)*.95))],
                            "errors": stats["errors"], "cache_hits": stats["cache_hits"]})
        usage_rows = sorted(
            (
                {
                    **row,
                    "avg_total_tokens": round(
                        row["total_tokens"] / row["count"], 1
                    )
                    if row["count"]
                    else 0,
                }
                for row in usage.values()
            ),
            key=lambda row: row["total_tokens"],
            reverse=True,
        )
        return {"events": events, "errors": list(groups.values()), "services": metrics, "usage": usage_rows,
                "health": {"last_event_at": self.last_event_at, "dropped": self.dropped, "error": self.last_error},
                "window": "最近 1000 条匹配事件；本地最多保留 10000 条"}

    def close(self):
        if self.logger:
            for handler in self.logger.handlers:
                handler.close()


def record(event: str, **values):
    store = CONTEXT.get().get("store")
    if store:
        store.record(event, **values)


def record_usage(
    *,
    kind: str,
    model: str | None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> None:
    """Record one model/embedding token usage event for diagnostics."""
    record(
        "usage.record",
        usage_kind=kind,
        usage_model=model or "",
        **{
            "usage.prompt_tokens": prompt_tokens,
            "usage.completion_tokens": completion_tokens,
            "usage.total_tokens": total_tokens,
            "usage.reasoning_tokens": reasoning_tokens,
        },
    )


@contextmanager
def external_request(service: str, timeout_ms=None):
    from ai_trading_copilot.copilot.services.tracing import get_current_trace_recorder
    recorder = get_current_trace_recorder()
    if recorder is None:
        yield None
        return
    with recorder.start_span("http.client", kind="client", attributes={"tool.name": service, "timeout_ms": timeout_ms}) as span:
        yield span

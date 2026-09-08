"""OTEL-compatible local tracing for copilot runs."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from threading import RLock
from ai_trading_copilot.copilot.services.diagnostics import CONTEXT, record, redact, exception_data


TRACE_JSON_REPORT_KEY = "trace_json"
_PROVIDERS = {}
_PROVIDER_LOCK = RLock()
TRACE_MARKDOWN_REPORT_KEY = "trace_markdown"
DEFAULT_SERVICE_NAME = "ai-trading-copilot"
DEFAULT_JAEGER_UI_URL = "http://127.0.0.1:16686"
_EXPORT_HEALTH = {"last_success_at": None, "last_error": None, "failed_batches": 0}


class ObservedExporter:
    def __init__(self, exporter):
        self.exporter = exporter

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult
        try:
            result = self.exporter.export(spans)
            if result == SpanExportResult.SUCCESS:
                _EXPORT_HEALTH.update(last_success_at=_utc_now(), last_error=None)
            else:
                _EXPORT_HEALTH["last_error"] = "OTLP export failed"
                _EXPORT_HEALTH["failed_batches"] += 1
            return result
        except Exception as exc:
            _EXPORT_HEALTH["last_error"] = type(exc).__name__
            _EXPORT_HEALTH["failed_batches"] += 1
            return SpanExportResult.FAILURE

    def shutdown(self):
        self.exporter.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self.exporter.force_flush(timeout_millis)


def configure_otel(endpoint=None, service_name=None):
    endpoint = (endpoint if endpoint is not None else os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip()
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    if not endpoint:
        return None, None
    with _PROVIDER_LOCK:
        key = (endpoint, service_name)
        if key not in _PROVIDERS:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                exporter = ObservedExporter(OTLPSpanExporter(endpoint=endpoint, headers=_parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))))
                provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
                provider.add_span_processor(BatchSpanProcessor(exporter))
                _PROVIDERS[key] = (provider, provider.get_tracer(__name__))
            except Exception as exc:
                _EXPORT_HEALTH["last_error"] = type(exc).__name__
                return None, None
        return _PROVIDERS[key]


def start_http_span(incoming):
    _, tracer = configure_otel()
    if tracer is None:
        return None
    from opentelemetry import propagate
    from opentelemetry.trace import SpanKind
    return tracer.start_span("http.server", context=propagate.extract({"traceparent": incoming}), kind=SpanKind.SERVER)


def trace_headers():
    recorder = get_current_trace_recorder()
    context = CONTEXT.get()
    trace_id = recorder.trace_id if recorder else context.get("trace_id")
    span_id = _current_span_id.get() or context.get("span_id")
    return {"traceparent": f"00-{trace_id}-{span_id}-{context.get('trace_flags', '01')}"} if trace_id and span_id else {}


SENSITIVE_KEY_PARTS = (
    "key",
    "token",
    "secret",
    "password",
    "header",
    "authorization",
)

_current_recorder: ContextVar["TraceRecorder | None"] = ContextVar(
    "copilot_trace_recorder",
    default=None,
)
_current_span_id: ContextVar[str | None] = ContextVar(
    "copilot_trace_span_id",
    default=None,
)

TraceActivityCallback = Callable[[str, "SpanRecord", "TraceRecorder"], None]


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str = "internal"
    status: str = "running"
    started_at: str = ""
    ended_at: str | None = None
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    _start_perf: float = field(default=0.0, repr=False)
    _otel_span: Any = field(default=None, repr=False)


class TraceSpanScope:
    """Context manager returned by TraceRecorder.start_span."""

    def __init__(self, recorder: "TraceRecorder", span: SpanRecord):
        self.recorder = recorder
        self.span = span
        self._span_token = None

    def __enter__(self) -> "TraceSpanScope":
        self._span_token = _current_span_id.set(self.span.span_id)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None and type(exc).__name__ == "RunCancelled":
            self.end(status="cancelled")
        elif exc is not None:
            self.record_exception(exc)
            self.end(status="error", error=str(exc))
        else:
            self.end(status="ok")
        if self._span_token is not None:
            _current_span_id.reset(self._span_token)
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        sanitized = sanitize_value(value, text_limit=_attribute_text_limit(key))
        with self.recorder._lock:
            self.span.attributes[key] = sanitized
            if self.span._otel_span is not None:
                self.span._otel_span.set_attribute(key, _otel_value(key, value))
        self.recorder.notify_activity("attribute", self.span)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        event = {
            "name": name,
            "timestamp": _utc_now(),
            "attributes": sanitize_value(attributes or {}),
        }
        with self.recorder._lock:
            self.span.events.append(event)
            if self.span._otel_span is not None:
                self.span._otel_span.add_event(name, _otel_attributes(attributes or {}))
        self.recorder.notify_activity("event", self.span)

    def record_exception(self, exc: BaseException) -> None:
        self.add_event(
            "exception",
            {
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
                **exception_data(exc),
            },
        )
        record("span.exception", trace_id=self.span.trace_id, span_id=self.span.span_id,
               level="ERROR", **exception_data(exc))

    def end(self, *, status: str = "ok", error: str | None = None) -> None:
        if self.span.ended_at is not None:
            return
        self.recorder.end_span(self.span.span_id, status=status, error=error)


class TraceRecorder:
    """Records local spans and optionally exports them through OTLP gRPC."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_id: str,
        symbols: list[str] | None = None,
        service_name: str | None = None,
        otlp_endpoint: str | None = None,
        activity_callback: TraceActivityCallback | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.symbols = list(symbols or [])
        self.service_name = service_name or os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
        self.trace_id = CONTEXT.get().get("trace_id") or secrets.token_hex(16)
        self.spans: list[SpanRecord] = []
        self._span_by_id: dict[str, SpanRecord] = {}
        # The graph can finish spans from several analyst worker threads.  Keep
        # registry mutations and snapshots serialized, but never invoke user
        # callbacks while holding this lock (callbacks may touch RunTracker).
        self._lock = RLock()
        self._otel_provider = None
        self._otel_tracer = None
        self._otel_enabled = False
        self._otlp_endpoint = ""
        self._activity_callback = activity_callback
        self._configure_otel(otlp_endpoint)

    @property
    def trace_json_path(self) -> Path:
        return self.output_dir / "trace.json"

    @property
    def trace_markdown_path(self) -> Path:
        return self.output_dir / "trace.md"

    @contextmanager
    def activate(self) -> Iterator["TraceRecorder"]:
        recorder_token = _current_recorder.set(self)
        try:
            yield self
        finally:
            _current_recorder.reset(recorder_token)

    def start_span(
        self,
        name: str,
        *,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> TraceSpanScope:
        parent_id = parent_span_id if parent_span_id is not None else (_current_span_id.get() or CONTEXT.get().get("span_id"))
        resolved_attributes = {
            **self._common_span_attributes(),
            **(attributes or {}),
        }
        span = SpanRecord(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            started_at=_utc_now(),
            attributes=sanitize_value(resolved_attributes),
            _start_perf=time.perf_counter(),
        )
        with self._lock:
            span._otel_span = self._start_otel_span(span)
            if span._otel_span is not None:
                sdk_context = span._otel_span.get_span_context()
                span.trace_id = self.trace_id = format(sdk_context.trace_id, "032x")
                span.span_id = format(sdk_context.span_id, "016x")
            self.spans.append(span)
            self._span_by_id[span.span_id] = span
        self.notify_activity("start", span)
        record("span.start", trace_id=span.trace_id, span_id=span.span_id, parent_span_id=parent_id,
               name=name, node=resolved_attributes.get("graph.node.name"), tool=resolved_attributes.get("tool.name"),
               input_summary={key: value for key, value in resolved_attributes.get("tool.args", {}).items()
                              if key in {"symbol", "ticker", "start_date", "end_date", "look_back_days", "limit"}})
        return TraceSpanScope(self, span)

    def instant_span(
        self,
        name: str,
        *,
        kind: str = "internal",
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> SpanRecord:
        with self.start_span(
            name,
            kind=kind,
            attributes=attributes,
            parent_span_id=parent_span_id,
        ) as scope:
            scope.end(status=status)
            return scope.span

    def end_span(self, span_id: str, *, status: str = "ok", error: str | None = None) -> None:
        error = redact(error)
        with self._lock:
            span = self._span_by_id[span_id]
            if span.ended_at is not None:
                return
            span.status = status
            span.error = redact(error)
            span.ended_at = _utc_now()
            span.duration_ms = round((time.perf_counter() - span._start_perf) * 1000, 3)
            span.attributes["status"] = status
            span.attributes["duration_ms"] = span.duration_ms
            if error:
                span.attributes["error"] = _truncate(error)
            if span._otel_span is not None:
                span._otel_span.set_attribute("status", status)
                span._otel_span.set_attribute("duration_ms", span.duration_ms)
                if error:
                    span._otel_span.set_attribute("error", _truncate(error))
                _set_otel_status(span._otel_span, status, error)
                span._otel_span.end()
            completed = _snapshot_span(span)
        self.notify_activity("end", completed)
        record("span.end", trace_id=completed.trace_id, span_id=completed.span_id,
               parent_span_id=completed.parent_span_id, name=completed.name,
               tool=completed.attributes.get("tool.name"),
               node=completed.attributes.get("graph.node.name"),
               cache_hit=completed.attributes.get("tool.cache_hit", False),
               duration_ms=completed.duration_ms, outcome=status,
               output_summary={key: value for key, value in completed.attributes.items()
                   if key.startswith(("tool.output.", "llm.response.")) and key.endswith((".chars", ".bytes", ".sha256"))},
               **{"error.message": redact(error)})

    def notify_activity(self, event: str, span: SpanRecord) -> None:
        if self._activity_callback is None:
            return
        with self._lock:
            snapshot = _snapshot_span(span)
        try:
            # The callback is deliberately outside ``self._lock``.  A tracker
            # callback may synchronously request another trace snapshot.
            self._activity_callback(event, snapshot, self)
        except Exception:
            return

    def flush(self) -> tuple[Path, Path]:
        payload = self.to_dict()
        self.trace_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.trace_markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        # Batch export must not block a run's completion.
        return self.trace_json_path, self.trace_markdown_path

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            spans = [_snapshot_span(span) for span in self.spans]
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "service_name": self.service_name,
            "symbols": self.symbols,
            "otel_export_enabled": self._otel_enabled,
            "otel": self.otel_status(),
            "generated_at": _utc_now(),
            "spans": [_span_to_dict(span) for span in spans],
        }

    def otel_status(self) -> dict[str, Any]:
        return {
            "local_trace_enabled": True,
            "otel_configured": bool(self._otlp_endpoint),
            "otel_export_enabled": self._otel_enabled,
            "otlp_endpoint": self._otlp_endpoint or None,
            "service_name": self.service_name,
            "jaeger_ui_url": os.getenv("JAEGER_UI_URL", DEFAULT_JAEGER_UI_URL),
        "export_health": dict(_EXPORT_HEALTH),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Copilot Trace",
            "",
            f"- Run ID: `{self.run_id or '-'}`",
            f"- Trace ID: `{self.trace_id}`",
            f"- Service: `{self.service_name}`",
            f"- OTEL export enabled: `{self._otel_enabled}`",
            "",
            "## Timeline",
            "",
            "| Span | Parent | Name | Kind | Status | Duration ms | Attributes |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
        with self._lock:
            spans = [_snapshot_span(span) for span in self.spans]
        for span in spans:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(span.span_id),
                        _md(span.parent_span_id or "-"),
                        _md(span.name),
                        _md(span.kind),
                        _md(span.status),
                        _md(span.duration_ms if span.duration_ms is not None else "-"),
                        _md(_attribute_summary(span.attributes)),
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def _configure_otel(self, endpoint: str | None) -> None:
        self._otlp_endpoint = (endpoint if endpoint is not None else os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip()
        self._otel_provider, self._otel_tracer = configure_otel(self._otlp_endpoint, self.service_name)
        self._otel_enabled = self._otel_provider is not None

    def _start_otel_span(self, span: SpanRecord):
        if self._otel_tracer is None:
            return None
        try:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
            from opentelemetry.trace import SpanContext, NonRecordingSpan, TraceFlags

            parent_span = self.get_span(span.parent_span_id)
            context = (
                trace.set_span_in_context(parent_span._otel_span)
                if parent_span is not None and parent_span._otel_span is not None
                else trace.set_span_in_context(NonRecordingSpan(SpanContext(
                    trace_id=int(self.trace_id, 16),
                    span_id=int(span.parent_span_id or secrets.token_hex(8), 16),
                    is_remote=True, trace_flags=TraceFlags(int(CONTEXT.get().get("trace_flags", "01"), 16)),
                )))
            )
            return self._otel_tracer.start_span(
                span.name,
                context=context,
                kind=_otel_kind(span.kind, SpanKind),
                attributes=_otel_attributes(span.attributes),
            )
        except Exception:
            return None

    def get_span(self, span_id: str | None) -> SpanRecord | None:
        """Return a span from the registry under the recorder lock.

        The returned record remains the live internal object for backwards
        compatibility; callers that need a stable value should use
        ``snapshot_span`` or ``to_dict``.
        """
        if not span_id:
            return None
        with self._lock:
            return self._span_by_id.get(span_id)

    def snapshot_span(self, span_id: str) -> SpanRecord:
        with self._lock:
            return _snapshot_span(self._span_by_id[span_id])

    def _common_span_attributes(self) -> dict[str, Any]:
        return {
            "run.id": self.run_id,
            "run.symbol_count": len(self.symbols),
            "run.symbols": self.symbols,
            "service.name": self.service_name,
        }


def get_current_trace_recorder() -> TraceRecorder | None:
    return _current_recorder.get()


def get_observability_status() -> dict[str, Any]:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    service_name = os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    return {
        "local_trace_enabled": True,
        "otel_configured": bool(endpoint),
        "otel_export_enabled": bool(endpoint),
        "otlp_endpoint": endpoint or None,
        "service_name": service_name,
        "jaeger_ui_url": os.getenv("JAEGER_UI_URL", DEFAULT_JAEGER_UI_URL),
        "export_health": dict(_EXPORT_HEALTH),
    }


def summarize_text(value: Any, prefix: str) -> dict[str, Any]:
    text = "" if value is None else str(value)
    raw = text.encode("utf-8", errors="replace")
    return {
        f"{prefix}.chars": len(text),
        f"{prefix}.bytes": len(raw),
        f"{prefix}.sha256": hashlib.sha256(raw).hexdigest(),
    }


def sanitize_value(value: Any, *, text_limit: int = 500) -> Any:
    value = redact(value)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_sensitive_key(text_key):
                sanitized[text_key] = "<redacted>"
            else:
                sanitized[text_key] = sanitize_value(item, text_limit=text_limit)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_value(item, text_limit=text_limit) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else _truncate(value, text_limit)
    return _truncate(str(value), text_limit)


def _attribute_text_limit(key: str) -> int:
    if key in {
        "llm.request.messages",
        "llm.response.content",
        "llm.response.reasoning_content",
    }:
        return 20000
    return 500


def _span_to_dict(span: SpanRecord) -> dict[str, Any]:
    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "kind": span.kind,
        "status": span.status,
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "duration_ms": span.duration_ms,
        "attributes": span.attributes,
        "events": span.events,
        "error": span.error,
    }


def _snapshot_span(span: SpanRecord) -> SpanRecord:
    """Make a stable callback/serialization view without copying OTEL handles."""
    events = []
    for event in span.events:
        copied = dict(event)
        if isinstance(copied.get("attributes"), dict):
            copied["attributes"] = dict(copied["attributes"])
        events.append(copied)
    return SpanRecord(
        trace_id=span.trace_id,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        name=span.name,
        kind=span.kind,
        status=span.status,
        started_at=span.started_at,
        ended_at=span.ended_at,
        duration_ms=span.duration_ms,
        attributes=dict(span.attributes),
        events=events,
        error=span.error,
        _start_perf=span._start_perf,
        _otel_span=span._otel_span,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _truncate(value: str, limit: int = 500) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _parse_headers(raw: str) -> dict[str, str] | None:
    if not raw.strip():
        return None
    headers = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip():
            headers[key.strip()] = value.strip()
    return headers or None


def _otel_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: _otel_value(key, value) for key, value in attributes.items()}


def _otel_value(key: str, value: Any) -> Any:
    value = sanitize_value(value)
    if key in {
        "llm.request.messages",
        "llm.response.content",
        "llm.response.reasoning_content",
    }:
        text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        raw = text.encode("utf-8", errors="replace")
        return json.dumps(
            {
                "chars": len(text),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _otel_kind(kind: str, span_kind):
    normalized = kind.lower()
    if normalized == "client":
        return span_kind.CLIENT
    if normalized == "server":
        return span_kind.SERVER
    return span_kind.INTERNAL


def _set_otel_status(otel_span, status: str, error: str | None) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        if status == "error":
            otel_span.set_status(Status(StatusCode.ERROR, error or "error"))
        else:
            otel_span.set_status(Status(StatusCode.OK))
    except Exception:
        return


def _attribute_summary(attributes: dict[str, Any]) -> str:
    if not attributes:
        return "-"
    parts = []
    for key, value in sorted(attributes.items())[:6]:
        parts.append(f"{key}={_markdown_attribute_value(key, value)}")
    return "; ".join(parts)


def _md(value: Any) -> str:
    text = "-" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_attribute_value(key: str, value: Any) -> Any:
    if key in {
        "llm.request.messages",
        "llm.response.content",
        "llm.response.reasoning_content",
    }:
        text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        return f"<detail chars={len(text)}>"
    return value

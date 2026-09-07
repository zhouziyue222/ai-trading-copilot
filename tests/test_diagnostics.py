# -*- coding: utf-8 -*-
import json
from pathlib import Path

from ai_trading_copilot.copilot.services.diagnostics import CONTEXT, DiagnosticStore, exception_data, redact
from ai_trading_copilot.copilot.services.tracing import TraceRecorder
import ai_trading_copilot.copilot.services.tracing as tracing


def test_local_ids_match_exported_sdk_spans(tmp_path, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "in-memory")
    monkeypatch.setitem(tracing._PROVIDERS, ("in-memory", "ai-trading-copilot"), (provider, provider.get_tracer("test")))
    token = CONTEXT.set({"trace_id": "1" * 32, "span_id": "2" * 16})
    try:
        recorder = TraceRecorder(output_dir=tmp_path, run_id="run_ids")
        with recorder.activate(), recorder.start_span("copilot.run"):
            with recorder.start_span("tool.call"):
                pass
        exported = {format(span.context.span_id, "016x"): span for span in exporter.get_finished_spans()}
        assert len(exported) == 2
        for span in recorder.spans:
            assert span.span_id in exported
            assert span.trace_id == format(exported[span.span_id].context.trace_id, "032x") == "1" * 32
    finally:
        CONTEXT.reset(token)
        provider.shutdown()


def test_failures_are_searchable_without_leaking_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "live-sensitive-secret")
    store = DiagnosticStore(tmp_path / "diagnostics")
    token = CONTEXT.set({"store": store, "request_id": "poll", "origin_request_id": "start-request",
                         "trace_id": "a" * 32, "run_id": "run_failure"})
    try:
        recorder = TraceRecorder(output_dir=tmp_path / "trace", run_id="run_failure", otlp_endpoint="")
        try:
            with recorder.activate(), recorder.start_span("tool.call", attributes={"tool.name": "service"}):
                raise ValueError("api_key=live-sensitive-secret rejected")
        except ValueError:
            pass
        path, _ = recorder.flush()
        results = store.query("start-request")
        assert results["errors"][0]["latest"]["error.type"] == "ValueError"
        assert results["errors"][0]["latest"]["code.line.number"]
        assert results["services"][0]["errors"] == 1
        raw = path.read_text(encoding="utf-8") + json.dumps(results)
        assert "live-sensitive-secret" not in raw
        assert "<redacted>" in raw
    finally:
        CONTEXT.reset(token)
        store.close()


def test_diagnostic_failure_does_not_fail_work(tmp_path, monkeypatch):
    store = DiagnosticStore(tmp_path)
    def unavailable():
        raise OSError("disk unavailable")
    monkeypatch.setattr(store, "_connect", unavailable)
    store.record("test")
    assert store.dropped == 1


def test_quoted_response_secrets_are_redacted():
    message = '''response: {"api_key": "unknown-live-key", "password": "private with spaces", 'token': 'another-secret'}'''
    clean = redact(message)
    assert all(secret not in clean for secret in ("unknown-live-key", "private with spaces", "another-secret"))
    assert clean.count("<redacted>") == 3


def test_export_failure_is_observable_and_does_not_raise(monkeypatch):
    class Broken:
        def export(self, spans):
            raise OSError("offline")
    monkeypatch.setattr(tracing, "_EXPORT_HEALTH", {"last_success_at": None, "last_error": None, "failed_batches": 0})
    exporter = tracing.ObservedExporter(Broken())
    exporter.export([])
    assert tracing.get_observability_status()["export_health"]["failed_batches"] == 1

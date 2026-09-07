import json

from ai_trading_copilot.copilot.services import tracing as tracing_module
from ai_trading_copilot.copilot.services.tracing import TraceRecorder


def test_trace_recorder_writes_parented_spans_and_files(tmp_path):
    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_trace", symbols=["AAPL"])

    with recorder.activate():
        with recorder.start_span("copilot.run") as root:
            with recorder.start_span("graph.node", attributes={"graph.node.name": "Technical"}):
                pass
            recorder.instant_span("tool.call", kind="client", attributes={"tool.name": "get_stock_info"})

    json_path, markdown_path = recorder.flush()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_trace"
    assert payload["service_name"] == "ai-trading-copilot"
    assert markdown_path.exists()
    root_span = payload["spans"][0]
    child_spans = payload["spans"][1:]
    assert root_span["name"] == "copilot.run"
    assert root_span["attributes"]["run.id"] == "run_trace"
    assert root_span["attributes"]["duration_ms"] is not None
    assert root_span["attributes"]["status"] == "ok"
    assert all(span["attributes"]["run.id"] == "run_trace" for span in child_spans)
    assert all(span["parent_span_id"] == root.span.span_id for span in child_spans)
    assert {span["name"] for span in payload["spans"]} == {
        "copilot.run",
        "graph.node",
        "tool.call",
    }


def test_trace_recorder_redacts_sensitive_values_and_summarizes_text(tmp_path):
    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_secret")

    with recorder.activate():
        with recorder.start_span(
            "tool.call",
            attributes={
                "api_key": "secret-value",
                "headers": {"Authorization": "Bearer abc"},
                "regular": "visible",
            },
        ) as span:
            for key, value in tracing_module.summarize_text("raw prompt text", "prompt").items():
                span.set_attribute(key, value)

    json_path, _ = recorder.flush()
    raw = json_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    attrs = payload["spans"][0]["attributes"]

    assert "secret-value" not in raw
    assert "Bearer abc" not in raw
    assert attrs["api_key"] == "<redacted>"
    assert attrs["headers"] == "<redacted>"
    assert attrs["regular"] == "visible"
    assert attrs["prompt.chars"] == 15
    assert "prompt.sha256" in attrs
    assert "raw prompt text" not in raw


def test_trace_recorder_without_endpoint_does_not_export(tmp_path, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_local")

    assert recorder.to_dict()["otel_export_enabled"] is False
    recorder.flush()


def test_trace_recorder_configures_otlp_exporter_when_endpoint_is_set(tmp_path, monkeypatch):
    calls = {}

    class FakeExporter:
        def __init__(self, *, endpoint, headers=None):
            calls["endpoint"] = endpoint
            calls["headers"] = headers

    class FakeProcessor:
        def __init__(self, exporter):
            calls["processor_exporter"] = exporter

    class FakeProvider:
        def __init__(self, *, resource):
            calls["resource"] = resource
            self.processors = []
            self.flushed = False

        def add_span_processor(self, processor):
            self.processors.append(processor)

        def get_tracer(self, name):
            return None

        def force_flush(self):
            calls["flushed"] = True

    class FakeResource:
        @classmethod
        def create(cls, attrs):
            calls["resource_attrs"] = attrs
            return cls()

    monkeypatch.setattr(tracing_module, "OTLPSpanExporter", FakeExporter, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=test")
    monkeypatch.setattr("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr("opentelemetry.sdk.trace.export.BatchSpanProcessor", FakeProcessor)
    monkeypatch.setattr("opentelemetry.sdk.trace.TracerProvider", FakeProvider)
    monkeypatch.setattr("opentelemetry.sdk.resources.Resource", FakeResource)

    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_otel")
    recorder.flush()

    assert calls["endpoint"] == "http://collector:4317"
    assert calls["headers"] == {"x-api-key": "test"}
    assert calls["resource_attrs"] == {"service.name": "ai-trading-copilot"}
    assert "flushed" not in calls  # Remote export must not block report completion.


def test_otel_attributes_use_summaries_for_large_llm_payloads():
    attrs = tracing_module._otel_attributes(
        {
            "llm.response.reasoning_content": "private reasoning body",
            "llm.request.messages": [{"role": "user", "content": "prompt body"}],
            "tool.args": {"api_key": "secret", "symbol": "AAPL"},
        }
    )

    assert "private reasoning body" not in attrs["llm.response.reasoning_content"]
    assert "prompt body" not in attrs["llm.request.messages"]
    assert "sha256" in attrs["llm.response.reasoning_content"]
    assert "secret" not in attrs["tool.args"]
    assert "<redacted>" in attrs["tool.args"]


def test_observability_status_reports_local_only_by_default(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

    payload = tracing_module.get_observability_status()

    assert payload["local_trace_enabled"] is True
    assert payload["otel_configured"] is False
    assert payload["otel_export_enabled"] is False
    assert payload["otlp_endpoint"] is None
    assert payload["service_name"] == "ai-trading-copilot"

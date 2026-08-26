import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.agents import llm_tools
from ai_trading_copilot.copilot.agents import TechnicalPositionAgent
from ai_trading_copilot.copilot.agents.llm_tools import run_tool_calling_llm
from ai_trading_copilot.copilot.domain import PriceBar
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.cancellation import (
    CancellationManager,
    CancellationToken,
    RunCancelled,
    activate_cancellation,
)
from ai_trading_copilot.copilot.services.run_tracker import RunTracker
from ai_trading_copilot.copilot.services.tracing import TraceRecorder


class RecordingTool:
    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.output


class ToolCallingFakeLLM:
    def __init__(self):
        self.invocations = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations += 1
        if self.invocations == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_info",
                        "args": {"symbol": "CRCL"},
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content="# Technical report\n\nCRCL is holding support.")


class CancellableTool(RecordingTool):
    def __init__(self, name, output, cancel_state):
        super().__init__(name, output)
        self.cancel_state = cancel_state
        self.cancelled_execution_ids = []

    def invoke_with_cancellation(self, args, *, cancellation_token, execution_id):
        self.calls.append(args)
        self.cancel_state["cancelled"] = True
        cancellation_token.manager.request_cancel(cancellation_token.run_id)
        return self.output

    def cancel_execution(self, execution_id):
        self.cancelled_execution_ids.append(execution_id)


class DeepSeekLikeLLM:
    model_name = "deepseek-v4-pro"
    openai_api_base = "https://api.deepseek.com"
    openai_api_key = "test-key"
    reasoning_effort = "high"
    extra_body = {"thinking": {"type": "enabled"}}


class FakeOpenAIClient:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            reasoning_content="Need quote before final answer.",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="get_stock_info",
                                        arguments='{"symbol":"CRCL","api_key":"secret"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="# Technical report\n\nCRCL is holding support.",
                        reasoning_content="Tool output supports final answer.",
                        tool_calls=None,
                    )
                )
            ]
        )


def _bars_from_closes(closes):
    return [
        PriceBar(
            date=f"2026-05-{idx + 1:02d}",
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1000,
        )
        for idx, close in enumerate(closes)
    ]


def test_react_runner_records_llm_and_tool_spans(tmp_path):
    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_react_trace")
    tool = RecordingTool("get_stock_info", "CRCL quote")
    summary_tool = RecordingTool("get_technical_summary", '{"symbol": "CRCL", "window": "90d"}')
    agent = TechnicalPositionAgent(llm=ToolCallingFakeLLM(), tools=[tool, summary_tool])

    with recorder.activate():
        with recorder.start_span("copilot.run"):
            agent.analyze_with_report(
                symbol="CRCL",
                bars=_bars_from_closes([100 + i for i in range(60)]),
                trade_date="2026-05-08",
            )

    trace_path, _ = recorder.flush()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    spans = payload["spans"]
    tool_spans = [span for span in spans if span["name"] == "tool.call"]
    llm_spans = [span for span in spans if span["name"] == "llm.invoke"]

    assert llm_spans
    assert all("llm.model" in span["attributes"] for span in llm_spans)
    assert {span["attributes"]["tool.phase"] for span in tool_spans} == {"prefetch", "react"}
    assert any(span["attributes"]["tool.name"] == "get_technical_summary" for span in tool_spans)
    assert all("CRCL quote" not in json.dumps(span, ensure_ascii=False) for span in tool_spans)
    assert all("tool.output.sha256" in span["attributes"] for span in tool_spans)


def test_deepseek_sdk_tool_loop_preserves_reasoning_content_and_traces_details(
    tmp_path,
    monkeypatch,
):
    fake_client = FakeOpenAIClient()
    monkeypatch.setattr(
        llm_tools,
        "_create_openai_client",
        lambda *, api_key, base_url: fake_client,
    )
    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_deepseek_trace")
    tool = RecordingTool("get_stock_info", "CRCL quote")

    with recorder.activate():
        with recorder.start_span("copilot.run"):
            content, calls = run_tool_calling_llm(
                llm=DeepSeekLikeLLM(),
                prompt="Analyze CRCL.",
                tools=[tool],
                max_rounds=4,
            )

    assert content.startswith("# Technical report")
    assert calls == ["get_stock_info"]
    assert len(fake_client.requests) == 2
    assistant_message = fake_client.requests[1]["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["reasoning_content"] == "Need quote before final answer."
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_stock_info"

    trace_path, _ = recorder.flush()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    llm_spans = [span for span in payload["spans"] if span["name"] == "llm.invoke"]
    assert all(span["attributes"]["llm.model"] == "deepseek-v4-pro" for span in llm_spans)
    assert llm_spans[0]["attributes"]["llm.response.reasoning_content"] == (
        "Need quote before final answer."
    )
    assert llm_spans[1]["attributes"]["llm.response.content"].startswith("# Technical report")
    assert (
        llm_spans[0]["attributes"]["llm.response.tool_calls"][0]["args"]["api_key"]
        == "<redacted>"
    )


def test_tool_loop_discards_tool_result_when_cancelled_during_tool_call(tmp_path):
    recorder = TraceRecorder(output_dir=tmp_path, run_id="run_cancel_trace")
    manager = CancellationManager()
    cancel_state = {"cancelled": False}
    token = CancellationToken(
        run_id="run_cancel_trace",
        checker=lambda: cancel_state["cancelled"],
        manager=manager,
    )
    llm = ToolCallingFakeLLM()
    tool = CancellableTool("get_stock_info", "CRCL quote", cancel_state)

    try:
        with activate_cancellation(token), recorder.activate():
            with recorder.start_span("copilot.run"):
                run_tool_calling_llm(
                    llm=llm,
                    prompt="Analyze CRCL.",
                    tools=[tool],
                    max_rounds=4,
                )
    except RunCancelled:
        pass
    else:
        raise AssertionError("Expected RunCancelled")

    assert llm.invocations == 1
    assert tool.calls == [{"symbol": "CRCL"}]
    assert len(tool.cancelled_execution_ids) == 1

    trace_path, _ = recorder.flush()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    tool_span = next(span for span in payload["spans"] if span["name"] == "tool.call")
    assert tool_span["attributes"]["tool.execution_id"] == tool.cancelled_execution_ids[0]
    assert tool_span["attributes"]["tool.cancellable"] is True
    assert tool_span["attributes"]["tool.cancel_requested"] is True


def test_technical_position_activity_is_written_as_safe_run_status(tmp_path):
    tracker = RunTracker(
        output_dir=tmp_path,
        run_id="run_react_activity",
        symbols=["CRCL"],
        params={},
        defaults={},
        nodes=CopilotLangGraph.NODE_ORDER,
    )
    graph = CopilotLangGraph(enable_default_llm=False, run_tracker=tracker)
    recorder = TraceRecorder(
        output_dir=tmp_path,
        run_id="run_react_activity",
        symbols=["CRCL"],
        activity_callback=graph._record_trace_activity,
    )
    tool = RecordingTool("get_stock_info", "CRCL quote should stay out of run status")
    summary_tool = RecordingTool(
        "get_technical_summary",
        '{"symbol": "CRCL", "secret_raw_summary": "should stay out of run status"}',
    )
    agent = TechnicalPositionAgent(llm=ToolCallingFakeLLM(), tools=[tool, summary_tool])

    tracker.start_node(CopilotLangGraph.NODE_TECHNICAL_POSITION)
    with recorder.activate():
        with recorder.start_span(
            "graph.node",
            attributes={"graph.node.name": CopilotLangGraph.NODE_TECHNICAL_POSITION},
        ):
            agent.analyze_with_report(
                symbol="CRCL",
                bars=_bars_from_closes([100 + i for i in range(60)]),
                trade_date="2026-05-08",
            )

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    node = payload["nodes"][CopilotLangGraph.NODE_TECHNICAL_POSITION]
    phases = {item["phase"] for item in node["activity_history"]}

    assert {"llm.invoke", "tool.call"} <= phases
    assert any(item.get("tool_name") == "get_stock_info" for item in node["activity_history"])
    assert any(item.get("tool_name") == "get_technical_summary" for item in node["activity_history"])
    assert any("tool.output" in item.get("summary", {}) for item in node["activity_history"])
    assert "CRCL quote should stay out of run status" not in json.dumps(node, ensure_ascii=False)
    assert "should stay out of run status" not in json.dumps(node, ensure_ascii=False)

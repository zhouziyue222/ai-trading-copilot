"""Small LLM tool-calling helpers for copilot analysts."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from ai_trading_copilot.copilot.services.cancellation import (
    RunCancelled,
    check_cancelled,
    get_current_cancellation_token,
    tool_cancel_callback,
)
from ai_trading_copilot.copilot.services.diagnostics import record_usage
from ai_trading_copilot.copilot.services.tracing import (
    get_current_trace_recorder,
    llm_usage_attributes,
    sanitize_value,
    summarize_text,
)


ToolResultCache = Dict[Tuple[str, str], str]


def run_tool_calling_llm(
    *,
    llm: Any,
    prompt: str,
    tools: Iterable[Any],
    max_rounds: int = 4,
    tool_result_cache: ToolResultCache | None = None,
) -> Tuple[str, List[str]]:
    """Run a bounded LangChain-style tool-calling loop.

    The helper is intentionally local and small; it lets the copilot agents use
    the same tool-calling style as TradingAgents without embedding that graph.
    """
    if llm is None:
        return "", []

    deepseek_config = _deepseek_sdk_config(llm)
    if deepseek_config is not None:
        return _run_deepseek_tool_calling_llm(
            config=deepseek_config,
            prompt=prompt,
            tools=tools,
            max_rounds=max_rounds,
            tool_result_cache=tool_result_cache,
        )

    from langchain_core.messages import HumanMessage, ToolMessage

    tool_list = list(tools)
    tool_by_name = {tool.name: tool for tool in tool_list}
    bound = llm.bind_tools(tool_list) if hasattr(llm, "bind_tools") else llm
    messages: List[Any] = [HumanMessage(content=prompt)]
    calls: List[str] = []
    last_content = ""

    recorder = get_current_trace_recorder()
    for round_index in range(max_rounds):
        check_cancelled()
        llm_attrs = {
            "llm.model": _llm_model_name(llm),
            "llm.round": round_index + 1,
            "llm.max_rounds": max_rounds,
            "llm.message_count": len(messages),
            "llm.tool_count": len(tool_list),
            **summarize_text(prompt, "prompt"),
        }
        if recorder is not None:
            with recorder.start_span("llm.invoke", kind="client", attributes=llm_attrs) as span:
                result = bound.invoke(messages)
                last_content = str(getattr(result, "content", result) or "")
                tool_calls = list(getattr(result, "tool_calls", []) or [])
                _set_usage_attributes(span, result)
                span.set_attribute("llm.tool_call_count", len(tool_calls))
                for key, value in summarize_text(last_content, "llm.response").items():
                    span.set_attribute(key, value)
        else:
            result = bound.invoke(messages)
            last_content = str(getattr(result, "content", result) or "")
            tool_calls = list(getattr(result, "tool_calls", []) or [])
        record_response_usage(result, kind="llm", model=_llm_model_name(llm))
        check_cancelled()
        if not tool_calls:
            return last_content, calls

        messages.append(result)
        for call in tool_calls:
            name = _tool_call_name(call)
            args = _tool_call_args(call)
            tool_item = tool_by_name.get(name)
            cache_key = _tool_cache_key(name, args)
            cache_hit = tool_result_cache is not None and cache_key in tool_result_cache
            output = ""
            if recorder is not None:
                with recorder.start_span(
                    "tool.call",
                    kind="client",
                    attributes={
                        "tool.name": name,
                        "tool.phase": "react",
                        "tool.cache_hit": cache_hit,
                        "tool.args": sanitize_value(args),
                    },
                ) as span:
                    output = _invoke_registered_tool(
                        tool_item=tool_item,
                        name=name,
                        args=args,
                        cache_key=cache_key,
                        tool_result_cache=tool_result_cache,
                        span=span,
                    )
                    for key, value in summarize_text(output, "tool.output").items():
                        span.set_attribute(key, value)
                    if tool_item is None:
                        span.end(status="error", error=f"Unknown tool: {name}")
                    elif output.startswith(f"Tool {name} failed:"):
                        span.end(status="error", error=output)
            else:
                output = _invoke_registered_tool(
                    tool_item=tool_item,
                    name=name,
                    args=args,
                    cache_key=cache_key,
                    tool_result_cache=tool_result_cache,
                )
            check_cancelled()
            calls.append(name)
            messages.append(
                ToolMessage(
                    content=output,
                    tool_call_id=_tool_call_id(call),
                    name=name,
                )
            )

    return last_content, calls


def _run_deepseek_tool_calling_llm(
    *,
    config: dict[str, Any],
    prompt: str,
    tools: Iterable[Any],
    max_rounds: int,
    tool_result_cache: ToolResultCache | None,
) -> Tuple[str, List[str]]:
    """Run DeepSeek thinking mode through the OpenAI SDK.

    LangChain currently exposes token-level reasoning metadata for DeepSeek but
    drops the reasoning_content field needed for thinking-mode tool rounds.
    This path preserves the assistant message exactly enough for follow-up tool
    calls and for UI trace inspection.
    """
    tool_list = list(tools)
    tool_by_name = {tool.name: tool for tool in tool_list}
    tool_schemas = [_tool_schema(tool) for tool in tool_list]
    client = _create_openai_client(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )
    messages: List[dict[str, Any]] = [{"role": "user", "content": prompt}]
    calls: List[str] = []
    last_content = ""
    recorder = get_current_trace_recorder()

    for round_index in range(max_rounds):
        check_cancelled()
        llm_attrs = {
            "llm.provider": "deepseek_openai_sdk",
            "llm.model": config["model"],
            "llm.round": round_index + 1,
            "llm.max_rounds": max_rounds,
            "llm.message_count": len(messages),
            "llm.tool_count": len(tool_list),
            **summarize_text(prompt, "prompt"),
        }
        if recorder is not None:
            with recorder.start_span("llm.invoke", kind="client", attributes=llm_attrs) as span:
                span.set_attribute("llm.request.messages", _messages_for_trace(messages))
                response = _create_deepseek_completion(
                    client=client,
                    config=config,
                    messages=messages,
                    tools=tool_schemas,
                )
                message = response.choices[0].message
                _set_usage_attributes(span, response)
                record_response_usage(response, kind="llm", model=config["model"])
                last_content, reasoning_content, tool_calls = _openai_message_parts(message)
                _record_llm_response(
                    span=span,
                    content=last_content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                )
        else:
            response = _create_deepseek_completion(
                client=client,
                config=config,
                messages=messages,
                tools=tool_schemas,
            )
            message = response.choices[0].message
            record_response_usage(response, kind="llm", model=config["model"])
            last_content, reasoning_content, tool_calls = _openai_message_parts(message)

        check_cancelled()
        if not tool_calls:
            return last_content, calls

        messages.append(
            _assistant_message(
                content=last_content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )
        )
        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            tool_item = tool_by_name.get(name)
            cache_key = _tool_cache_key(name, args)
            cache_hit = tool_result_cache is not None and cache_key in tool_result_cache
            output = ""
            if recorder is not None:
                with recorder.start_span(
                    "tool.call",
                    kind="client",
                    attributes={
                        "tool.name": name,
                        "tool.phase": "react",
                        "tool.cache_hit": cache_hit,
                        "tool.args": sanitize_value(args),
                    },
                ) as span:
                    output = _invoke_registered_tool(
                        tool_item=tool_item,
                        name=name,
                        args=args,
                        cache_key=cache_key,
                        tool_result_cache=tool_result_cache,
                        span=span,
                    )
                    for key, value in summarize_text(output, "tool.output").items():
                        span.set_attribute(key, value)
                    if tool_item is None:
                        span.end(status="error", error=f"Unknown tool: {name}")
                    elif output.startswith(f"Tool {name} failed:"):
                        span.end(status="error", error=output)
            else:
                output = _invoke_registered_tool(
                    tool_item=tool_item,
                    name=name,
                    args=args,
                    cache_key=cache_key,
                    tool_result_cache=tool_result_cache,
                )
            check_cancelled()
            calls.append(name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output,
                }
            )

    return last_content, calls


def _create_deepseek_completion(
    *,
    client: Any,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    request: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "stream": False,
    }
    if tools:
        request["tools"] = tools
    if config.get("reasoning_effort"):
        request["reasoning_effort"] = config["reasoning_effort"]
    if config.get("extra_body"):
        request["extra_body"] = config["extra_body"]
    from ai_trading_copilot.copilot.services.diagnostics import external_request
    from ai_trading_copilot.copilot.services.tracing import trace_headers
    with external_request("DeepSeek"):
        headers = trace_headers()
        if headers:
            request["extra_headers"] = headers
        response = client.chat.completions.create(**request)
    observer = config.get("completion_observer")
    if observer is not None:
        observer(response)
    return response


def _record_llm_response(
    *,
    span,
    content: str,
    reasoning_content: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    span.set_attribute("llm.tool_call_count", len(tool_calls))
    span.set_attribute("llm.response.content", content)
    span.set_attribute("llm.response.reasoning_content", reasoning_content)
    span.set_attribute("llm.response.tool_calls", _tool_calls_for_trace(tool_calls))
    for key, value in summarize_text(content, "llm.response").items():
        span.set_attribute(key, value)
    for key, value in summarize_text(reasoning_content, "llm.response.reasoning_content").items():
        span.set_attribute(key, value)


def _set_usage_attributes(span, response) -> None:
    for key, value in llm_usage_attributes(response).items():
        span.set_attribute(key, value)


def record_response_usage(
    response,
    *,
    kind: str,
    model: str,
) -> None:
    attrs = llm_usage_attributes(response)
    if not attrs:
        return
    record_usage(
        kind=kind,
        model=model,
        prompt_tokens=attrs.get("usage.prompt_tokens"),
        completion_tokens=attrs.get("usage.completion_tokens"),
        total_tokens=attrs.get("usage.total_tokens"),
        reasoning_tokens=attrs.get("usage.reasoning_tokens"),
    )


def _openai_message_parts(message: Any) -> tuple[str, str, list[dict[str, Any]]]:
    content = str(getattr(message, "content", "") or "")
    reasoning_content = str(
        getattr(message, "reasoning_content", None)
        or (getattr(message, "model_extra", None) or {}).get("reasoning_content")
        or ""
    )
    raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
    return content, reasoning_content, [_normalize_openai_tool_call(call) for call in raw_tool_calls]


def _assistant_message(
    *,
    content: str,
    reasoning_content: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [call["raw"] for call in tool_calls],
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return message


def _normalize_openai_tool_call(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    name = str(getattr(function, "name", "") or "")
    raw_arguments = getattr(function, "arguments", "{}") or "{}"
    args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
    call_id = str(getattr(call, "id", "") or name or "tool_call")
    raw = {
        "id": call_id,
        "type": getattr(call, "type", None) or "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False, sort_keys=True),
        },
    }
    return {"id": call_id, "name": name, "args": args, "raw": raw}


def _tool_calls_for_trace(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": call["id"],
            "name": call["name"],
            "args": sanitize_value(call["args"]),
        }
        for call in tool_calls
    ]


def _messages_for_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_messages = []
    for message in messages:
        trace_message = {
            "role": message.get("role"),
            "content": message.get("content", ""),
        }
        if message.get("reasoning_content"):
            trace_message["reasoning_content"] = message["reasoning_content"]
        if message.get("tool_calls"):
            trace_message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": (call.get("function") or {}).get("name"),
                    "args": _json_loads_or_text((call.get("function") or {}).get("arguments", "")),
                }
                for call in message["tool_calls"]
            ]
        if message.get("tool_call_id"):
            trace_message["tool_call_id"] = message["tool_call_id"]
        trace_messages.append(trace_message)
    return trace_messages


def _tool_schema(tool_item: Any) -> dict[str, Any]:
    args_schema = getattr(tool_item, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        parameters = args_schema.model_json_schema()
    else:
        parameters = {
            "type": "object",
            "properties": getattr(tool_item, "args", {}) or {},
            "required": [],
        }
    parameters = dict(parameters)
    parameters.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": str(getattr(tool_item, "name", "")),
            "description": str(getattr(tool_item, "description", "") or ""),
            "parameters": parameters,
        },
    }


def _deepseek_sdk_config(llm: Any) -> dict[str, Any] | None:
    model = _llm_model_name(llm)
    base_url = str(
        getattr(llm, "openai_api_base", None)
        or getattr(llm, "base_url", None)
        or ""
    ).rstrip("/")
    extra_body = dict(getattr(llm, "extra_body", None) or {})
    thinking = extra_body.get("thinking") if isinstance(extra_body.get("thinking"), dict) else {}
    thinking_enabled = str(thinking.get("type", "")).lower() == "enabled"
    if "deepseek" not in f"{model} {base_url}".lower() or not thinking_enabled:
        return None
    api_key = _secret_value(
        getattr(llm, "openai_api_key", None)
        or getattr(llm, "api_key", None)
    )
    if not api_key:
        return None
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "reasoning_effort": getattr(llm, "reasoning_effort", None),
        "extra_body": extra_body,
        "completion_observer": getattr(llm, "_copilot_completion_observer", None),
    }


def _llm_model_name(llm: Any) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", "") or "")


def _secret_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())
    return str(value)


def _create_openai_client(*, api_key: str, base_url: str) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _json_loads_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def collect_tool_evidence(
    *,
    tools: Iterable[Any],
    requests: Iterable[Tuple[str, Dict[str, Any]]],
    tool_result_cache: ToolResultCache | None = None,
) -> Tuple[str, List[str]]:
    """Invoke available tools once to ground the prompt in tool-sourced data."""
    tool_by_name = {tool.name: tool for tool in tools}
    sections: List[str] = []
    calls: List[str] = []
    for name, args in requests:
        check_cancelled()
        tool_item = tool_by_name.get(name)
        if tool_item is None:
            continue
        cache_key = _tool_cache_key(name, args)
        cache_hit = tool_result_cache is not None and cache_key in tool_result_cache
        recorder = get_current_trace_recorder()
        if recorder is not None:
            with recorder.start_span(
                "tool.call",
                kind="client",
                attributes={
                    "tool.name": name,
                    "tool.phase": "prefetch",
                    "tool.cache_hit": cache_hit,
                    "tool.args": sanitize_value(args),
                },
            ) as span:
                output = _invoke_registered_tool(
                    tool_item=tool_item,
                    name=name,
                    args=args,
                    cache_key=cache_key,
                    tool_result_cache=tool_result_cache,
                    span=span,
                )
                for key, value in summarize_text(output, "tool.output").items():
                    span.set_attribute(key, value)
                if output.startswith(f"Tool {name} failed:"):
                    span.end(status="error", error=output)
        else:
            output = _invoke_registered_tool(
                tool_item=tool_item,
                name=name,
                args=args,
                cache_key=cache_key,
                tool_result_cache=tool_result_cache,
            )
        check_cancelled()
        calls.append(name)
        sections.extend([f"## {name}", json.dumps(args, ensure_ascii=False), output, ""])
    return "\n".join(sections).strip(), calls


def _invoke_tool(
    *,
    tool_item: Any,
    name: str,
    args: Dict[str, Any],
    cache_key: Tuple[str, str],
    tool_result_cache: ToolResultCache | None,
    cancellation_token=None,
    execution_id: str | None = None,
) -> str:
    if cancellation_token is not None:
        cancellation_token.check()
    if tool_result_cache is not None and cache_key in tool_result_cache:
        return tool_result_cache[cache_key]
    if tool_item is None:
        return f"Unknown tool: {name}"
    try:
        output = str(_invoke_tool_item(tool_item, args, cancellation_token, execution_id))
    except Exception as exc:
        if isinstance(exc, RunCancelled):
            raise
        output = f"Tool {name} failed: {exc}"
    if cancellation_token is not None:
        cancellation_token.check()
    if tool_result_cache is not None:
        tool_result_cache[cache_key] = output
    return output


def _invoke_registered_tool(
    *,
    tool_item: Any,
    name: str,
    args: Dict[str, Any],
    cache_key: Tuple[str, str],
    tool_result_cache: ToolResultCache | None,
    span=None,
) -> str:
    cancellation_token = get_current_cancellation_token()
    if cancellation_token is None:
        return _invoke_tool(
            tool_item=tool_item,
            name=name,
            args=args,
            cache_key=cache_key,
            tool_result_cache=tool_result_cache,
        )

    cancellation_token.check()
    with cancellation_token.manager.register(run_id=cancellation_token.run_id, tool_name=name) as execution:
        execution.cancel_callback = tool_cancel_callback(tool_item, execution.execution_id)
        if span is not None:
            span.set_attribute("tool.execution_id", execution.execution_id)
            span.set_attribute("tool.cancellable", execution.cancel_callback is not None)
        try:
            output = _invoke_tool(
                tool_item=tool_item,
                name=name,
                args=args,
                cache_key=cache_key,
                tool_result_cache=tool_result_cache,
                cancellation_token=cancellation_token,
                execution_id=execution.execution_id,
            )
        finally:
            if span is not None:
                span.set_attribute("tool.cancel_requested", execution.cancel_requested)
        if execution.cancel_requested:
            if span is not None:
                span.end(status="error", error="Tool result discarded after cancellation.")
            raise RunCancelled("Run cancelled by user.")
        cancellation_token.check()
        return output


def _invoke_tool_item(
    tool_item: Any,
    args: Dict[str, Any],
    cancellation_token,
    execution_id: str | None,
) -> Any:
    invoke_with_cancellation = getattr(tool_item, "invoke_with_cancellation", None)
    if callable(invoke_with_cancellation):
        return invoke_with_cancellation(
            args,
            cancellation_token=cancellation_token,
            execution_id=execution_id,
        )
    return tool_item.invoke(args)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response")
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def strip_trailing_json_object(text: str) -> str:
    """Remove machine payloads from the human report, including fenced JSON."""
    from ai_trading_copilot.copilot.services.reporting import strip_report_json

    return strip_report_json(text)


def _tool_call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "")
    return str(getattr(call, "name", ""))


def _tool_call_args(call: Any) -> Dict[str, Any]:
    if isinstance(call, dict):
        args = call.get("args") or {}
    else:
        args = getattr(call, "args", {}) or {}
    if isinstance(args, str):
        return json.loads(args)
    return dict(args)


def _tool_call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or call.get("name") or "tool_call")
    return str(getattr(call, "id", None) or getattr(call, "name", "tool_call"))


def _tool_cache_key(name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)

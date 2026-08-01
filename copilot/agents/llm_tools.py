"""Small LLM tool-calling helpers for copilot analysts."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


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

    from langchain_core.messages import HumanMessage, ToolMessage

    tool_list = list(tools)
    tool_by_name = {tool.name: tool for tool in tool_list}
    bound = llm.bind_tools(tool_list) if hasattr(llm, "bind_tools") else llm
    messages: List[Any] = [HumanMessage(content=prompt)]
    calls: List[str] = []
    last_content = ""

    for _ in range(max_rounds):
        result = bound.invoke(messages)
        last_content = str(getattr(result, "content", result) or "")
        tool_calls = list(getattr(result, "tool_calls", []) or [])
        if not tool_calls:
            return last_content, calls

        messages.append(result)
        for call in tool_calls:
            name = _tool_call_name(call)
            args = _tool_call_args(call)
            tool_item = tool_by_name.get(name)
            cache_key = _tool_cache_key(name, args)
            if tool_result_cache is not None and cache_key in tool_result_cache:
                output = tool_result_cache[cache_key]
            elif tool_item is None:
                output = f"Unknown tool: {name}"
            else:
                try:
                    output = str(tool_item.invoke(args))
                except Exception as exc:
                    output = f"Tool {name} failed: {exc}"
                if tool_result_cache is not None:
                    tool_result_cache[cache_key] = output
            calls.append(name)
            messages.append(
                ToolMessage(
                    content=output,
                    tool_call_id=_tool_call_id(call),
                    name=name,
                )
            )

    return last_content, calls


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
        tool_item = tool_by_name.get(name)
        if tool_item is None:
            continue
        cache_key = _tool_cache_key(name, args)
        if tool_result_cache is not None and cache_key in tool_result_cache:
            output = tool_result_cache[cache_key]
        else:
            try:
                output = str(tool_item.invoke(args))
            except Exception as exc:
                output = f"Tool {name} failed: {exc}"
            if tool_result_cache is not None:
                tool_result_cache[cache_key] = output
        calls.append(name)
        sections.extend([f"## {name}", json.dumps(args, ensure_ascii=False), output, ""])
    return "\n".join(sections).strip(), calls


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
    """Remove a final JSON object from an LLM response before saving Markdown reports."""
    stripped = text.strip()
    if not stripped.endswith("}"):
        return stripped

    depth = 0
    for index in range(len(stripped) - 1, -1, -1):
        char = stripped[index]
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                return stripped[:index].rstrip()
    return stripped


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

"""Shared bounded ReAct-style tool runner for LLM agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple

from ai_trading_copilot.copilot.agents.llm_tools import (
    ToolResultCache,
    collect_tool_evidence,
    run_tool_calling_llm,
)


@dataclass
class ReActRunResult:
    content: str
    tool_calls: List[str] = field(default_factory=list)
    prefetched_evidence: str = ""
    prefetched_calls: List[str] = field(default_factory=list)

    @property
    def all_tool_calls(self) -> List[str]:
        return [*self.prefetched_calls, *self.tool_calls]


class ReActAgentRunner:
    """Small shared runner for bounded think-act-observe loops.

    It uses LangChain tool calling under the hood. The agent prompts are
    responsible for describing role, action policy, and final JSON format.
    """

    def __init__(self, *, llm: Any, tools: Iterable[Any], max_rounds: int = 4):
        self.llm = llm
        self.tools = list(tools)
        self.max_rounds = max_rounds

    def run(
        self,
        *,
        prompt: str,
        prefetch: Iterable[Tuple[str, dict[str, Any]]] = (),
        tool_result_cache: ToolResultCache | None = None,
    ) -> ReActRunResult:
        cache = tool_result_cache if tool_result_cache is not None else {}
        evidence, pre_calls = collect_tool_evidence(
            tools=self.tools,
            requests=prefetch,
            tool_result_cache=cache,
        )
        if not prompt.strip():
            return ReActRunResult(
                content="",
                prefetched_evidence=evidence,
                prefetched_calls=pre_calls,
            )
        content, calls = run_tool_calling_llm(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools,
            max_rounds=self.max_rounds,
            tool_result_cache=cache,
        )
        return ReActRunResult(
            content=content,
            tool_calls=calls,
            prefetched_evidence=evidence,
            prefetched_calls=pre_calls,
        )

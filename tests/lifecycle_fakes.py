# -*- coding: utf-8 -*-
"""Deterministic, offline graph used by lifecycle and browser verification."""

from threading import Event

from ai_trading_copilot.copilot.graph import CopilotLangGraph


class BoundaryGraph(CopilotLangGraph):
    NODE_ORDER = ["First", "Second"]
    calls = []
    finish_second = Event()

    def __init__(self, *, run_tracker, memory_agent=None, cancellation_checker=None,
                 snapshot_callback=None, **kwargs):
        super().__init__(run_tracker=run_tracker, memory_agent=memory_agent,
                         cancellation_checker=cancellation_checker, snapshot_callback=snapshot_callback,
                         enable_default_llm=False, force_sequential=True)

    def _compile_graph(self):
        return None

    def _sequential_node_order(self, state):
        return list(self.NODE_ORDER)

    @classmethod
    def checkpoint_node_order(cls, selected_analysts):
        return list(cls.NODE_ORDER)

    def _node_map(self):
        return {"First": self._tracked_node("First", self.first),
                "Second": self._tracked_node("Second", self.second)}

    def run(self, **params):
        return self.invoke({"subscription_symbols": params["subscription_symbols"],
                            "selected_analysts": params["selected_analysts"],
                            "report_output_dir": params["report_output_dir"], "run_id": params["run_id"],
                            "explanations": {"items": []}})

    def first(self, state):
        type(self).calls.append("First")
        state["explanations"]["items"].append("完整节点")
        report = self._save_agent_report(state=state, stage="1_complete", agent_name="first", content="中文报告：已完成\n")
        return {"agent_reports": {"first": report}}

    def second(self, state):
        type(self).calls.append("Second")
        assert state["explanations"]["items"] == ["完整节点"]
        state["explanations"]["items"].append("未完成变更")
        self._save_agent_report(state=state, stage="2_partial", agent_name="second", content="部分结果\n")
        while not type(self).finish_second.wait(0.01):
            self._check_cancelled()
        self._check_cancelled()
        return {}

"""Offline three-analyst graph for API, process and browser verification."""
from threading import Event, Lock
import time

from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.cancellation import check_cancelled


class OfflineParallelGraph(CopilotLangGraph):
    names = ["Technical Position", "News Sentiment", "Fundamental Analysis"]
    releases = {name: Event() for name in names}
    calls = []
    calls_lock = Lock()
    fail_name = None
    delay = 0.0

    def __init__(self, **kwargs):
        kwargs.pop("memory_agent", None)
        super().__init__(enable_default_llm=False, **kwargs)

    def _compile_graph(self):
        return None

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.fail_name = None
        cls.delay = 0.0
        for event in cls.releases.values():
            event.clear()

    def run(self, **params):
        return self.invoke({key: params[key] for key in
                            ("subscription_symbols", "selected_analysts", "report_output_dir", "run_id")})

    def _node_map(self):
        def node(name):
            def execute(state):
                with self.calls_lock:
                    self.calls.append(name)
                if name in self.names:
                    key = name.lower().replace(" ", "_")
                    self._save_agent_report(state=state, stage="1_analysts", agent_name=key,
                                            content="尚未完成的报告\n")
                    while not self.releases[name].wait(0.01):
                        check_cancelled()
                    check_cancelled()
                    if self.fail_name == name:
                        raise RuntimeError("offline analyst failed")
                    time.sleep(self.delay)
                    report = self._save_agent_report(state=state, stage="1_analysts", agent_name=key,
                                                     content=f"中文分析报告：{name}\n")
                    return {"agent_reports": {key: report}, "analyst_reports": {key: report}}
                if name == self.NODE_TRADER:
                    expected = set(self._route_analyst_nodes(state)) & set(self.names)
                    assert {item.lower().replace(" ", "_") for item in expected} <= set(state["analyst_reports"])
                    assert list(state["analyst_reports"]) == [item.lower().replace(" ", "_") for item in self.names if item in expected]
                return {}
            return self._tracked_node(name, execute)
        return {name: node(name) for name in [*self.NODE_ORDER, self.NODE_OPPORTUNITY_RADAR]}

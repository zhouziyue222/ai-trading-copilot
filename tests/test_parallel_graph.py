# -*- coding: utf-8 -*-
"""Focused tests for the lifecycle-controlled analyst fan-out."""

from __future__ import annotations

import threading
import time

import pytest

from ai_trading_copilot.copilot.domain.enums import AnalystType
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.cancellation import RunCancelled


class ParallelGraph(CopilotLangGraph):
    def __init__(self, **kwargs):
        super().__init__(
            enable_default_llm=False,
            force_sequential=True,
            parallel_analysts=True,
            default_selected_analysts=[
                AnalystType.TECHNICAL_POSITION,
                AnalystType.NEWS_SENTIMENT,
                AnalystType.FUNDAMENTAL_ANALYSIS,
            ],
            **kwargs,
        )
        self.started = []
        self.started_lock = threading.Lock()
        self.barrier = threading.Barrier(3)
        self.trader_calls = 0

    def _compile_graph(self):
        return None

    def _node_map(self):
        return {
            self.NODE_LOAD_PERSONA_MARKDOWN: self._tracked_node(
                self.NODE_LOAD_PERSONA_MARKDOWN, self._simple
            ),
            self.NODE_LOAD_SUBSCRIPTION_SYMBOLS: self._tracked_node(
                self.NODE_LOAD_SUBSCRIPTION_SYMBOLS, self._simple
            ),
            self.NODE_TECHNICAL_POSITION: self._tracked_node(
                self.NODE_TECHNICAL_POSITION, self._technical_position
            ),
            self.NODE_NEWS_SENTIMENT: self._tracked_node(
                self.NODE_NEWS_SENTIMENT, self._news_sentiment
            ),
            self.NODE_FUNDAMENTAL_ANALYSIS: self._tracked_node(
                self.NODE_FUNDAMENTAL_ANALYSIS, self._fundamental_analysis
            ),
            self.NODE_TRADER: self._tracked_node(self.NODE_TRADER, self._trader_test),
            self.NODE_RISK_CHECK: self._tracked_node(self.NODE_RISK_CHECK, self._simple),
            self.NODE_PORTFOLIO_MANAGER: self._tracked_node(
                self.NODE_PORTFOLIO_MANAGER, self._simple
            ),
            self.NODE_EXPLAIN_RUN: self._tracked_node(
                self.NODE_EXPLAIN_RUN, self._simple
            ),
            self.NODE_PERSIST_TRACE: self._tracked_node(
                self.NODE_PERSIST_TRACE, self._simple
            ),
        }

    @staticmethod
    def _simple(state):
        return {}

    def _branch(self, name, key):
        with self.started_lock:
            self.started.append(name)
        self.barrier.wait(timeout=3)
        return {key: {name: "done"}}

    def _technical_position(self, state):
        return self._branch(self.NODE_TECHNICAL_POSITION, "technical_positions")

    def _news_sentiment(self, state):
        return self._branch(self.NODE_NEWS_SENTIMENT, "news_sentiment_by_symbol")

    def _fundamental_analysis(self, state):
        return self._branch(self.NODE_FUNDAMENTAL_ANALYSIS, "fundamental_analysis_by_symbol")

    def _trader_test(self, state):
        self.trader_calls += 1
        assert set(state) >= {
            "technical_positions",
            "news_sentiment_by_symbol",
            "fundamental_analysis_by_symbol",
        }
        return {"trader_ran": True}


def _state():
    return {
        "subscription_symbols": ["AAPL"],
        "selected_analysts": [
            AnalystType.TECHNICAL_POSITION,
            AnalystType.NEWS_SENTIMENT,
            AnalystType.FUNDAMENTAL_ANALYSIS,
        ],
    }


def test_three_analysts_enter_the_phase_together(tmp_path):
    graph = ParallelGraph(report_output_dir=tmp_path)
    state = graph.invoke(_state())

    assert set(graph.started) == {
        graph.NODE_TECHNICAL_POSITION,
        graph.NODE_NEWS_SENTIMENT,
        graph.NODE_FUNDAMENTAL_ANALYSIS,
    }
    assert state["trader_ran"] is True
    assert graph.trader_calls == 1


def test_cancel_keeps_a_completed_branch_boundary(tmp_path):
    cancel = threading.Event()
    snapshots = []

    class CancelGraph(ParallelGraph):
        def _technical_position(self, state):
            result = {"technical_positions": {"Technical Position": "done"}}
            return result

        def _news_sentiment(self, state):
            while not cancel.wait(0.01):
                self._check_cancelled()
            self._check_cancelled()
            return {}

        def _fundamental_analysis(self, state):
            while not cancel.wait(0.01):
                self._check_cancelled()
            self._check_cancelled()
            return {}

    graph = CancelGraph(
        report_output_dir=tmp_path,
        cancellation_checker=cancel.is_set,
        snapshot_callback=lambda state, order, index, **kw: (
            snapshots.append((index, kw.get("completed_nodes", []))),
            cancel.set() if graph.NODE_TECHNICAL_POSITION in kw.get("completed_nodes", []) else None,
        ),
    )
    with pytest.raises(RunCancelled):
        graph.invoke(_state())
    assert any(graph.NODE_TECHNICAL_POSITION in completed for _, completed in snapshots)
    assert graph.trader_calls == 0


def test_branch_failure_does_not_run_trader(tmp_path):
    class FailingGraph(ParallelGraph):
        def _news_sentiment(self, state):
            raise ValueError("news failed")

    graph = FailingGraph(report_output_dir=tmp_path)
    with pytest.raises(ValueError, match="news failed"):
        graph.invoke(_state())
    assert graph.trader_calls == 0


def test_default_parallel_analysts_have_separate_model_clients(monkeypatch):
    import ai_trading_copilot.copilot.graph.copilot_langgraph as module
    monkeypatch.setattr(module, "create_default_deepseek_llm", lambda: object())
    graph = CopilotLangGraph(parallel_analysts=True)
    models = [graph.technical_position_agent.llm, graph.news_sentiment_agent.llm,
              graph.fundamental_analyst_agent.llm]
    assert len({id(model) for model in models}) == 3

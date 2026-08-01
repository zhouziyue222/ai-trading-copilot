import json
from pathlib import Path

import pytest

from ai_trading_copilot.copilot import run
from ai_trading_copilot.copilot.domain import AnalystType, MarketType
from ai_trading_copilot.copilot.services import SubscriptionStore


def test_parse_single_symbol_keeps_legacy_cli():
    args = run._parse_args(["CRCL"])

    assert run._resolve_symbols(args) == ["CRCL"]


def test_resolve_symbols_merges_positionals_and_comma_list():
    args = run._parse_args(["aapl", "MSFT", "--symbols", "msft, crcl"])

    assert run._resolve_symbols(args) == ["AAPL", "MSFT", "CRCL"]


def test_resolve_analysts_defaults_to_all():
    args = run._parse_args(["CRCL"])

    assert run._resolve_analysts(args) == [
        AnalystType.OPPORTUNITY_RADAR,
        AnalystType.TECHNICAL_POSITION,
        AnalystType.FUNDAMENTAL_NEWS,
    ]


def test_resolve_analysts_accepts_subset():
    args = run._parse_args(
        [
            "CRCL",
            "--analysts",
            "opportunity_radar,fundamental_news",
        ]
    )

    assert run._resolve_analysts(args) == [
        AnalystType.OPPORTUNITY_RADAR,
        AnalystType.FUNDAMENTAL_NEWS,
    ]


def test_parse_args_rejects_unknown_analyst():
    with pytest.raises(SystemExit):
        run._parse_args(["CRCL", "--analysts", "market"])


def test_load_subscription_symbols_from_file(tmp_path):
    path = tmp_path / "subscriptions.json"
    store = SubscriptionStore(path)
    store.add("aapl", MarketType.US_STOCK)
    store.add("spy", MarketType.US_ETF)

    assert run._load_subscription_symbols(path) == ["AAPL", "SPY"]


def test_all_subscribed_uses_subscription_file(tmp_path):
    path = tmp_path / "subscriptions.json"
    store = SubscriptionStore(path)
    store.add("aapl", MarketType.US_STOCK)
    store.add("spy", MarketType.US_ETF)
    args = run._parse_args(
        [
            "--subscriptions-file",
            str(path),
            "--all-subscribed",
        ]
    )

    assert run._resolve_symbols(args) == ["AAPL", "SPY"]


def test_no_symbols_and_missing_subscription_file_exits(tmp_path):
    args = run._parse_args(["--subscriptions-file", str(tmp_path / "missing.json")])

    with pytest.raises(SystemExit, match="No symbols provided"):
        run._resolve_symbols(args)


def test_no_symbols_and_existing_subscription_file_enters_selection(monkeypatch, tmp_path):
    path = tmp_path / "subscriptions.json"
    SubscriptionStore(path).add("aapl", MarketType.US_STOCK)
    args = run._parse_args(["--subscriptions-file", str(path)])
    monkeypatch.setattr(run, "_select_symbols", lambda symbols: ["AAPL"])

    assert run._resolve_symbols(args) == ["AAPL"]


def test_main_passes_multiple_symbols_and_selected_analysts(monkeypatch, tmp_path, capsys):
    calls = {}

    class FakeTracker:
        def __init__(self, **kwargs):
            calls["tracker"] = kwargs

        def add_error(self, error):
            calls["error"] = error

        def finish(self, *, failed):
            calls["failed"] = failed

        def write_audit(self, *, state, error):
            calls["audit_state"] = state
            calls["audit_error"] = error
            return str(Path(calls["tracker"]["output_dir"]) / "run_audit.md")

    class FakeGraph:
        NODE_ORDER = ["node"]

        def __init__(self, *, run_tracker, memory_agent):
            calls["run_tracker"] = run_tracker
            calls["memory_agent"] = memory_agent

        def run(self, **kwargs):
            calls["graph_run"] = kwargs
            return {
                "subscription_symbols": kwargs["subscription_symbols"],
                "agent_reports": {},
            }

    monkeypatch.setattr(run, "RunTracker", FakeTracker)
    monkeypatch.setattr(run, "CopilotLangGraph", FakeGraph)

    result = run.main(
        [
            "AAPL",
            "--symbols",
            "MSFT,CRCL",
            "--analysts",
            "opportunity_radar,fundamental_news",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert calls["tracker"]["symbols"] == ["AAPL", "MSFT", "CRCL"]
    assert calls["tracker"]["params"]["subscription_symbols"] == ["AAPL", "MSFT", "CRCL"]
    assert calls["tracker"]["defaults"]["selected_analysts"] == [
        "opportunity_radar",
        "fundamental_news",
    ]
    assert calls["graph_run"]["selected_analysts"] == [
        AnalystType.OPPORTUNITY_RADAR,
        AnalystType.FUNDAMENTAL_NEWS,
    ]
    assert calls["memory_agent"] is not None
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(tmp_path)


def test_default_memory_agent_uses_requested_path(tmp_path):
    path = tmp_path / "memory.jsonl"
    agent = run.create_default_memory_agent(path)

    assert agent.retrieve_context(symbol="AAPL") == []

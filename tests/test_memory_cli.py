from __future__ import annotations

import json

import pytest

from ai_trading_copilot.copilot import memory


def test_memory_cli_uses_sqlite_database_option(tmp_path, capsys):
    database = tmp_path / "memory.sqlite3"

    result = memory.main(["--memory-database", str(database), "status"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["database"] == str(database)
    assert database.exists()
    assert not (tmp_path / "memory.jsonl").exists()


def test_memory_cli_rejects_removed_jsonl_and_export_interfaces(tmp_path):
    with pytest.raises(SystemExit):
        memory._parse_args(
            ["--memory-file", str(tmp_path / "memory.jsonl"), "status"]
        )
    with pytest.raises(SystemExit):
        memory._parse_args(["export"])


def test_existing_jsonl_is_never_opened_by_memory_entrypoints(tmp_path, monkeypatch, capsys):
    import builtins
    import io
    from pathlib import Path
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from ai_trading_copilot.copilot.domain import DistilledMemory, MemoryStatus, MemoryType
    from ai_trading_copilot.copilot.run import create_default_memory_agent
    from ai_trading_copilot.copilot.services.memory_learning import LangMemCandidate, LangMemProposal
    from ai_trading_copilot.copilot.ui.app import UISettings, create_app

    legacy = tmp_path / "memory.jsonl"
    contents = DistilledMemory(
        memory_id="legacy-only", memory_type=MemoryType.STRATEGY_PERFORMANCE,
        status=MemoryStatus.APPROVED, lesson="Legacy lesson must not load.", symbols=["AAPL"],
    ).model_dump_json() + "\n"
    legacy.write_text(contents, encoding="utf-8")
    before = legacy.stat()
    database = tmp_path / "memory.sqlite3"

    def guard(opener):
        def checked(file, *args, **kwargs):
            if not isinstance(file, int) and Path(file).suffix.lower() == ".jsonl":
                raise AssertionError(f"Long-term memory opened JSONL: {file}")
            return opener(file, *args, **kwargs)
        return checked

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", guard(builtins.open))
        patch.setattr(io, "open", guard(io.open))
        agent = create_default_memory_agent(database)
        assert agent.store.list_all() == []
        agent.learning_service.reflector.extractor = SimpleNamespace(
            available=True, extract=lambda snapshot, existing=(): [LangMemProposal(
                operation="insert", candidate=LangMemCandidate(
                    memory_type=MemoryType.STRATEGY_PERFORMANCE,
                    lesson="Reflected SQLite observation.", symbols=["AAPL"],
                ),
            )],
        )
        learned = agent.learn_from_run({"run_id": "jsonl-isolation", "trade_plans": {"AAPL": {}}})
        assert learned[0].status == MemoryStatus.CANDIDATE
        candidate = agent.store.save_candidate(DistilledMemory(
            memory_type=MemoryType.STRATEGY_PERFORMANCE, lesson="SQLite candidate.", symbols=["AAPL"],
        ))
        assert agent.store.retrieve(symbol="AAPL") == []
        agent.store.repository.upsert(candidate.model_copy(update={"status": MemoryStatus.APPROVED}))
        assert [m.memory_id for m in agent.store.retrieve(symbol="AAPL")] == [candidate.memory_id]
        assert memory.main(["--memory-database", str(database), "list"]) == 0
        assert "legacy-only" not in capsys.readouterr().out
        with TestClient(create_app(UISettings(
            memory_database=database, reports_dir=tmp_path / "reports",
            subscriptions_file=tmp_path / "subscriptions.json", run_in_background=False,
        ))) as client:
            response = client.get("/api/memories")
            assert response.status_code == 200
            assert "legacy-only" not in response.text
    assert legacy.read_text(encoding="utf-8") == contents
    assert legacy.stat().st_mtime_ns == before.st_mtime_ns
    assert sorted(p.name for p in tmp_path.glob("memory.jsonl*")) == ["memory.jsonl"]

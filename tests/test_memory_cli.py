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

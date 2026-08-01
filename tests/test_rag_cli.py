import json

from ai_trading_copilot.copilot import rag


def test_rag_cli_status_outputs_json(tmp_path, capsys):
    result = rag.main(
        [
            "--memory-file",
            str(tmp_path / "memory.jsonl"),
            "--chroma-dir",
            str(tmp_path / "rag_chroma"),
            "status",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["backend"] == "fundamental_chroma"
    assert payload["scope"] == "fundamental_only"
    assert "available" in payload

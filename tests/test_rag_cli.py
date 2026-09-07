import json

from ai_trading_copilot.copilot import rag


def test_print_json_falls_back_to_ascii_when_console_rejects_unicode(monkeypatch):
    calls = []

    def fake_print(value):
        calls.append(value)
        if len(calls) == 1:
            raise UnicodeEncodeError("gbk", "中文\xa0", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr("builtins.print", fake_print)

    rag._print_json({"text": "中文\xa0SEC"})

    assert len(calls) == 2
    assert "\\u4e2d\\u6587" in calls[1]
    assert json.loads(calls[1]) == {"text": "中文\xa0SEC"}


def test_rag_cli_status_outputs_json(tmp_path, capsys):
    result = rag.main(
        [
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
    assert payload["probe_status"] == "not_run"
    assert payload["document_count"] is None

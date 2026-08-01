from pathlib import Path

import ai_trading_copilot
from ai_trading_copilot.copilot.config import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_ENV_PATH,
    DEFAULT_REPORT_OUTPUT_DIR,
    create_default_deepseek_llm,
    load_copilot_env,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph


def test_copilot_env_path_and_report_dir_are_package_local():
    package_root = Path(ai_trading_copilot.__file__).resolve().parent

    assert DEFAULT_ENV_PATH.name == ".env"
    assert DEFAULT_ENV_PATH.parent == package_root
    assert DEFAULT_REPORT_OUTPUT_DIR == package_root / "reports"


def test_load_copilot_env_reads_key_without_overriding_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=file-key\nDEEPSEEK_MODEL=deepseek-test\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    loaded = load_copilot_env(env_file)

    assert loaded["DEEPSEEK_API_KEY"] == "process-key"
    assert loaded["DEEPSEEK_MODEL"] == "deepseek-test"


def test_default_deepseek_llm_uses_deepseek_settings(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    llm = create_default_deepseek_llm()

    assert llm.model_name == DEFAULT_DEEPSEEK_MODEL
    assert str(llm.openai_api_base).rstrip("/") == DEFAULT_DEEPSEEK_BASE_URL


def test_graph_default_report_dir_is_copilot_reports():
    graph = CopilotLangGraph(enable_default_llm=False)

    assert Path(graph.report_output_dir) == DEFAULT_REPORT_OUTPUT_DIR

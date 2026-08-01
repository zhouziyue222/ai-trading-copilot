"""Default LLM configuration for the AI trading copilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PACKAGE_ROOT / ".env"
DEFAULT_REPORT_OUTPUT_DIR = PACKAGE_ROOT / "reports"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def load_copilot_env(path: str | Path = DEFAULT_ENV_PATH) -> Dict[str, str]:
    """Load key-value pairs from the copilot-local .env file.

    Existing process environment values win over file values.
    """
    env_path = Path(path)
    loaded: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
            if key:
                loaded[key] = os.environ.get(key, value)
    return loaded


def create_default_deepseek_llm():
    """Create the default DeepSeek chat model, or return None when unconfigured."""
    load_copilot_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
    )

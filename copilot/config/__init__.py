"""Default configuration for the AI trading copilot."""

from .defaults import DEFAULT_PERSONA_CONFIG, DEFAULT_PERSONA_MARKDOWN, DEFAULT_PRODUCT_CONFIG
from .llm import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_ENV_PATH,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_REPORT_OUTPUT_DIR,
    create_default_deepseek_llm,
    load_copilot_env,
)

__all__ = [
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "DEFAULT_ENV_PATH",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "DEFAULT_PERSONA_CONFIG",
    "DEFAULT_PERSONA_MARKDOWN",
    "DEFAULT_PRODUCT_CONFIG",
    "DEFAULT_REPORT_OUTPUT_DIR",
    "create_default_deepseek_llm",
    "load_copilot_env",
]

"""Prompt template loading for LLM-backed agents."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any


PROMPT_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


def load_prompt_template(name: str) -> str:
    """Load a versioned prompt template from the project config directory."""
    if not name.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise ValueError(f"invalid prompt template name: {name}")
    path = PROMPT_DIR / name
    if path.suffix != ".md":
        path = path.with_name(f"{path.name}.md")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **context: Any) -> str:
    values = {key: str(value) for key, value in context.items()}
    return Template(load_prompt_template(name)).safe_substitute(values)

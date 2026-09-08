"""Prompt template loading for LLM-backed agents."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any


from ai_trading_copilot.copilot.services.reporting import REPORT_WRITING_RULES


PROMPT_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"
DATA_BOUNDARY_BEGIN = "<<<UNTRUSTED_{kind}_DATA_BEGIN>>>"
DATA_BOUNDARY_END = "<<<UNTRUSTED_{kind}_DATA_END>>>"


def untrusted_data_block(kind: str, content: Any) -> str:
    """Wrap external evidence so models can distinguish data from instructions.

    The marker strings are neutralized inside the content so a hostile article
    or document cannot forge an earlier closing boundary.
    """
    label = "".join(
        character
        for character in str(kind).strip().upper()
        if character.isalnum() or character == "_"
    )
    if not label:
        label = "EVIDENCE"
    if content is None or not str(content).strip():
        return "-"
    text = (
        str(content)
        .replace("<<<", "< < <")
        .replace(">>>", "> > >")
    )
    return (
        DATA_BOUNDARY_BEGIN.format(kind=label)
        + "\n"
        + text
        + "\n"
        + DATA_BOUNDARY_END.format(kind=label)
    )


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
    return Template(load_prompt_template(name)).safe_substitute(values) + "\n\n" + REPORT_WRITING_RULES

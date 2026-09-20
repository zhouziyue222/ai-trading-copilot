"""Default product and persona configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_trading_copilot.copilot.domain.models import UserPersonaConfig


PERSONA_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "persona.default.json"
PRODUCT_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "product.default.json"


def _load_default_persona() -> dict[str, Any]:
    payload = json.loads(PERSONA_DEFAULT_PATH.read_text(encoding="utf-8"))
    return UserPersonaConfig(**payload).model_dump(mode="python")


DEFAULT_PERSONA_CONFIG = _load_default_persona()


def _display_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, list):
        return ", ".join(_display_value(item) for item in value)
    return str(value)


DEFAULT_PERSONA_MARKDOWN = "# User Persona\n\n" + "\n".join(
    f"- {key}: {_display_value(value)}"
    for key, value in DEFAULT_PERSONA_CONFIG.items()
)


DEFAULT_PRODUCT_CONFIG = json.loads(PRODUCT_DEFAULT_PATH.read_text(encoding="utf-8"))

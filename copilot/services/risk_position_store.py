"""Persistent store for risk limits and target position weights."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ai_trading_copilot.copilot.config.defaults import DEFAULT_PERSONA_CONFIG
from ai_trading_copilot.copilot.domain.models import (
    UserPersonaConfig,
    normalize_symbol,
)
from ai_trading_copilot.copilot.services.run_tracker import write_json_atomic


class RiskPositionStore:
    """Load, save, and reset user risk/position settings."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        default = self._default_snapshot()
        if not self.path.exists():
            return default
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            persona = UserPersonaConfig(**payload["persona"])
            target_weights = self._normalize_target_weights(
                payload.get("target_weights", {})
            )
            return {
                "source": "user",
                "path": str(self.path),
                "updated_at": payload.get("updated_at"),
                "persona": persona.model_dump(mode="json"),
                "target_weights": target_weights,
                "warning": "",
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return {
                **default,
                "warning": "用户配置不可用，已回退到默认配置。",
            }

    def save(
        self,
        persona: UserPersonaConfig,
        target_weights: dict[str, float],
    ) -> dict:
        normalized_weights = self._normalize_target_weights(target_weights)
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "version": 1,
            "persona": persona.model_dump(mode="json"),
            "target_weights": normalized_weights,
            "updated_at": updated_at,
        }
        write_json_atomic(self.path, payload)
        return {
            "source": "user",
            "path": str(self.path),
            "updated_at": updated_at,
            "persona": persona.model_dump(mode="json"),
            "target_weights": normalized_weights,
            "warning": "",
        }

    def reset(self) -> dict:
        self.path.unlink(missing_ok=True)
        return self._default_snapshot()

    def _default_snapshot(self) -> dict:
        persona = UserPersonaConfig(**DEFAULT_PERSONA_CONFIG)
        return {
            "source": "default",
            "path": str(self.path),
            "updated_at": None,
            "persona": persona.model_dump(mode="json"),
            "target_weights": {},
            "warning": "",
        }

    def _normalize_target_weights(
        self,
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for raw_symbol, raw_weight in target_weights.items():
            symbol = normalize_symbol(raw_symbol)
            if not symbol:
                raise ValueError("target weight symbol is required")
            weight = float(raw_weight)
            if weight != weight or weight < -1 or weight > 1:
                raise ValueError(f"target weight for {symbol} must be in [-1, 1]")
            normalized[symbol] = round(weight, 6)
        return normalized

"""JSON-backed subscription list storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ai_trading_copilot.copilot.domain.enums import MarketType, SubscriptionStatus
from ai_trading_copilot.copilot.domain.models import (
    Subscription,
    SubscriptionBook,
    normalize_symbol,
)
from ai_trading_copilot.copilot.services.state_machine import (
    transition_subscription_status,
)


class SubscriptionStore:
    """Persist and mutate the user subscription list."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> SubscriptionBook:
        if not self.path.exists():
            return SubscriptionBook()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = {"items": payload}
        return SubscriptionBook(**payload)

    def save(self, book: SubscriptionBook) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = book.model_dump(mode="json")
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def add(
        self,
        symbol: str,
        market_type: MarketType,
        *,
        reason: str = "",
        target_action: str = "",
        status: SubscriptionStatus = SubscriptionStatus.OBSERVING,
    ) -> SubscriptionBook:
        book = self.load()
        normalized = normalize_symbol(symbol)
        if book.contains(normalized):
            raise ValueError(f"Symbol is already subscribed: {normalized}")

        next_book = SubscriptionBook(
            items=[
                *book.items,
                Subscription(
                    symbol=normalized,
                    market_type=market_type,
                    reason=reason,
                    target_action=target_action,
                    status=status,
                ),
            ]
        )
        self.save(next_book)
        return next_book

    def remove(self, symbol: str) -> SubscriptionBook:
        normalized = normalize_symbol(symbol)
        book = self.load()
        next_book = SubscriptionBook(
            items=[item for item in book.items if item.symbol != normalized]
        )
        self.save(next_book)
        return next_book

    def update_status(
        self,
        symbol: str,
        status: SubscriptionStatus,
    ) -> SubscriptionBook:
        normalized = normalize_symbol(symbol)
        book = self.load()
        updated = []
        found = False
        for item in book.items:
            if item.symbol == normalized:
                updated.append(transition_subscription_status(item, status))
                found = True
            else:
                updated.append(item)
        if not found:
            raise ValueError(f"Symbol is not subscribed: {normalized}")
        next_book = SubscriptionBook(items=updated)
        self.save(next_book)
        return next_book

    def get(self, symbol: str) -> Optional[Subscription]:
        return self.load().get(symbol)


import pytest

from ai_trading_copilot.copilot.domain import (
    MarketType,
    Subscription,
    SubscriptionStatus,
)
from ai_trading_copilot.copilot.services import (
    SubscriptionStore,
    can_transition,
    transition_subscription_status,
)


def test_state_machine_allows_same_status():
    assert can_transition(
        SubscriptionStatus.OBSERVING,
        SubscriptionStatus.OBSERVING,
    )


def test_state_machine_blocks_direct_risk_to_actionable():
    assert not can_transition(
        SubscriptionStatus.RISK_ELEVATED,
        SubscriptionStatus.ACTIONABLE,
    )


def test_transition_returns_updated_subscription():
    subscription = Subscription(
        symbol="AAPL",
        market_type=MarketType.US_STOCK,
        status=SubscriptionStatus.OBSERVING,
    )

    updated = transition_subscription_status(
        subscription,
        SubscriptionStatus.NEAR_OPPORTUNITY,
    )

    assert subscription.status == SubscriptionStatus.OBSERVING
    assert updated.status == SubscriptionStatus.NEAR_OPPORTUNITY


def test_invalid_transition_raises():
    subscription = Subscription(
        symbol="AAPL",
        market_type=MarketType.US_STOCK,
        status=SubscriptionStatus.RISK_ELEVATED,
    )

    with pytest.raises(ValueError):
        transition_subscription_status(subscription, SubscriptionStatus.ACTIONABLE)


def test_store_loads_empty_book_when_file_missing(tmp_path):
    store = SubscriptionStore(tmp_path / "subscriptions.json")

    assert store.load().items == []


def test_store_adds_and_persists_subscription(tmp_path):
    path = tmp_path / "subscriptions.json"
    store = SubscriptionStore(path)

    store.add("spy", MarketType.US_ETF, reason="Core market benchmark")
    reloaded = SubscriptionStore(path).load()

    assert reloaded.symbols == ["SPY"]
    assert reloaded.get("SPY").market_type == MarketType.US_ETF
    assert reloaded.get("SPY").reason == "Core market benchmark"


def test_store_rejects_duplicate_symbols(tmp_path):
    store = SubscriptionStore(tmp_path / "subscriptions.json")
    store.add("AAPL", MarketType.US_STOCK)

    with pytest.raises(ValueError):
        store.add("aapl", MarketType.US_STOCK)


def test_store_updates_status_with_state_machine(tmp_path):
    store = SubscriptionStore(tmp_path / "subscriptions.json")
    store.add("AAPL", MarketType.US_STOCK)

    updated = store.update_status("aapl", SubscriptionStatus.NEAR_OPPORTUNITY)

    assert updated.get("AAPL").status == SubscriptionStatus.NEAR_OPPORTUNITY
    assert store.load().get("AAPL").status == SubscriptionStatus.NEAR_OPPORTUNITY


def test_store_rejects_invalid_status_transition(tmp_path):
    store = SubscriptionStore(tmp_path / "subscriptions.json")
    store.add(
        "AAPL",
        MarketType.US_STOCK,
        status=SubscriptionStatus.RISK_ELEVATED,
    )

    with pytest.raises(ValueError):
        store.update_status("AAPL", SubscriptionStatus.ACTIONABLE)


def test_store_removes_subscription(tmp_path):
    store = SubscriptionStore(tmp_path / "subscriptions.json")
    store.add("AAPL", MarketType.US_STOCK)
    store.add("SPY", MarketType.US_ETF)

    updated = store.remove("aapl")

    assert updated.symbols == ["SPY"]
    assert store.load().symbols == ["SPY"]


import pytest

from ai_trading_copilot.copilot.services.delayed_review import execution_metrics, price_metrics
from ai_trading_copilot.copilot.services.review_repository import ReviewRepository
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository


def fill(deal, side, quantity, price, date="2026-01-05", environment="REAL"):
    return dict(deal_id=deal, symbol="AAPL", environment=environment, account_id="1", side=side,
                quantity=quantity, price=price, executed_at=date + "T15:00:00+00:00", fees=None)


def bars():
    return [dict(date="2026-01-05", open=100, close=110, low=95, high=115, volume=100),
            dict(date="2026-01-06", open=110, close=105, low=101, high=116, volume=100)]


def test_partial_exit_separates_realized_and_floating_pnl():
    result = execution_metrics([fill("1", "BUY", 10, 100), fill("2", "SELL", 4, 110, "2026-01-06")], bars(), {"direction": "buy"}, .01)[0]
    assert result["realized_pnl_gross"] == 40
    assert result["unrealized_pnl_gross"] == 30
    assert result["remaining_quantity"] == 6
    assert result["realized_return"] == pytest.approx(.07)
    assert result["max_drawdown"] == pytest.approx(1.07 / 1.1 - 1)
    assert result["pnl_net"] is None
    assert result["fee_basis"] == "gross_fees_unavailable"
    assert result["eligible"]


def test_accounts_are_not_combined_and_unknown_inventory_has_no_invented_pnl():
    results = execution_metrics([fill("1", "SELL", 10, 100), fill("2", "BUY", 10, 100, environment="SIMULATE")], bars(), {"direction": "sell"})
    real = next(r for r in results if r["category"] == "REAL")
    assert not real["eligible"]
    assert real["realized_pnl_gross"] is None
    assert real["unrealized_pnl_gross"] is None
    assert len(results) == 2


def test_same_day_stop_and_target_never_invents_sequence():
    result = price_metrics(bars(), {"direction": "buy", "stop_loss": 97, "targets": [112], "invalidation_conditions": ["trend breaks"]})
    assert result["price_events"][0]["sequence"] == "unknown"
    assert result["invalidation"] == "insufficient_evidence"
    assert result["return_basis"] == "hypothetical_asset_observation_not_trade_pnl"


def test_short_cover_and_fees():
    fills = [fill("1", "SELL", 10, 110), fill("2", "BUY", 10, 100, "2026-01-06")]
    for item in fills:
        item["fees"] = 1
    result = execution_metrics(fills, bars(), {"direction": "short"})[0]
    assert result["realized_pnl_gross"] == 100
    assert result["unrealized_pnl_gross"] == 0
    assert result["pnl_net"] == 98
    assert result["remaining_quantity"] == 0
    assert result["eligible"]


def test_order_state_after_cutoff_does_not_leak_into_earlier_review(tmp_path):
    repo = ReviewRepository(SQLiteMemoryRepository(tmp_path / "orders.db"))
    repo.register(dict(run_id="run", symbol="AAPL", generated_at="2026-01-01T00:00:00+00:00"), [5])
    repo.record_order("REAL", "1", "o1", "run", "AAPL")
    order = dict(environment="REAL", account_id=1, order_id="o1", symbol="US.AAPL", side="BUY",
                 quantity=10, price=100, filled_quantity=0, status="SUBMITTED",
                 created_at="2026-01-02T15:00:00+00:00", order_updated_at="2026-01-02T15:00:00+00:00")
    repo.save_orders([order, {**order, "status":"CANCELLED_ALL", "order_updated_at":"2026-01-20T15:00:00+00:00"}])
    before = repo.orders_for_review("REAL", "1", "run", "AAPL", "2026-01-09T21:00:00+00:00")
    assert before[0]["status"] == "SUBMITTED"
    after = repo.orders_for_review("REAL", "1", "run", "AAPL", "2026-01-21T21:00:00+00:00")
    assert after[0]["status"] == "CANCELLED_ALL"
    assert repo.orders_for_review("SIMULATE", "1", "run", "AAPL", "2026-01-21T21:00:00+00:00") == []


def test_order_without_timestamp_does_not_claim_past_cancellation(tmp_path):
    repo = ReviewRepository(SQLiteMemoryRepository(tmp_path / "orders.db"))
    repo.register(dict(run_id="run", symbol="AAPL", generated_at="2026-01-01T00:00:00+00:00"), [5])
    repo.record_order("REAL", "1", "o1", "run", "AAPL")
    repo.save_orders([dict(environment="REAL", account_id=1, order_id="o1", symbol="AAPL", side="BUY",
                          quantity=10, price=100, filled_quantity=0, status="CANCELLED_ALL", created_at="2026-01-02T15:00:00+00:00")])
    observed = repo.orders_for_review("REAL", "1", "run", "AAPL", "2026-01-09T21:00:00+00:00")[0]
    assert observed["status"] == "unknown_at_cutoff"
    assert "filled_quantity" not in observed

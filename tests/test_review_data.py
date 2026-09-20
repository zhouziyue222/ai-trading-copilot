from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai_trading_copilot.copilot.adapters import review_data as module


@pytest.fixture
def provider(monkeypatch):
    quote, trade = Mock(), Mock()
    sdk = SimpleNamespace(
        RET_OK=0, OpenQuoteContext=Mock(return_value=quote),
        OpenSecTradeContext=Mock(return_value=trade),
        TrdMarket=SimpleNamespace(NONE="NONE"), SecurityFirm=SimpleNamespace(NONE="NONE"),
        TrdEnv=SimpleNamespace(REAL="REAL", SIMULATE="SIMULATE"),
        KLType=SimpleNamespace(K_DAY="K_DAY"), AuType=SimpleNamespace(NONE="NONE"),
    )
    monkeypatch.setattr(module, "_load_futu", lambda: sdk)
    return module.FutuReviewDataAdapter(security_firm="NONE"), quote, trade, sdk


def test_calendar_dst_half_day_and_holidays(provider):
    adapter, quote, _, _ = provider
    quote.request_trading_days.return_value = (0, [
        {"time": "2026-03-06", "trade_date_type": "WHOLE"},
        {"time": "2026-03-09", "trade_date_type": "WHOLE"},
        {"time": "2026-11-27", "trade_date_type": "MORNING"},
    ])
    result = adapter.trading_sessions("US.AAPL", "2026-01-01", "2026-12-31")
    assert [r["close_at"] for r in result] == [
        "2026-03-06T16:00:00-05:00", "2026-03-09T16:00:00-04:00", "2026-11-27T13:00:00-05:00",
    ]
    assert len(result) == 3  # No invented weekend or holiday sessions.
    assert result[0]["open_at"] == "2026-03-06T09:30:00-05:00"
    assert result[1]["open_at"] == "2026-03-09T09:30:00-04:00"
    quote.close.assert_called_once()


@pytest.mark.parametrize("symbol,closing", [("HK.00700", "12:10"), ("SH.600000", "11:30"), ("SZ.000001", "11:30")])
def test_asian_half_days(provider, symbol, closing):
    adapter, quote, _, _ = provider
    quote.request_trading_days.return_value = (0, [{"time": "2026-02-16", "trade_date_type": "MORNING"}])
    assert adapter.trading_sessions(symbol, "2026-02-16", "2026-02-16")[0]["close_at"] == f"2026-02-16T{closing}:00+08:00"


def test_calendar_fail_closed(provider):
    adapter, quote, _, sdk = provider
    with pytest.raises(module.ReviewDataError):
        adapter.trading_sessions("SG.TEST", "2026-01-01", "2026-01-02")
    sdk.OpenQuoteContext.assert_not_called()
    quote.request_trading_days.return_value = (0, [{"time": "2026-01-01", "trade_date_type": "UNKNOWN"}])
    with pytest.raises(module.ReviewDataError):
        adapter.trading_sessions("US.AAPL", "2026-01-01", "2026-01-02")
    quote.close.assert_called_once()


def bar(day):
    return dict(time_key=day + " 00:00:00", open=10, high=12, low=9, close=11, volume=100)


@pytest.mark.parametrize("rehab,status", [((0, []), "clear"), ((0, [{"ex_div_date": "2026-01-02"}]), "detected"), ((-1, "permission"), "unknown")])
def test_raw_paginated_history_and_action_safety(provider, rehab, status):
    adapter, quote, _, _ = provider
    quote.get_rehab.return_value = rehab
    quote.request_history_kline.side_effect = [(0, [bar("2026-01-01")], b"next"), (0, [bar("2026-01-02")], None)]
    rows = adapter.history("US.AAPL", "2026-01-01", "2026-01-02")
    assert len(rows) == 2
    assert all(r["adjustment"] == "RAW" and r["corporate_action_status"] == status and r["unsafe"] == (status != "clear") for r in rows)
    assert all(call.kwargs["autype"] == "NONE" for call in quote.request_history_kline.call_args_list)
    assert quote.request_history_kline.call_args_list[1].kwargs["page_req_key"] == b"next"
    quote.close.assert_called_once()


def test_history_cursor_loop_is_not_partial_success(provider):
    adapter, quote, _, _ = provider
    quote.get_rehab.return_value = (0, [])
    quote.request_history_kline.return_value = (0, [bar("2026-01-01")], b"same")
    with pytest.raises(module.ReviewDataError, match="cursor"):
        adapter.history("US.AAPL", "2026-01-01", "2026-01-02")
    quote.close.assert_called_once()


def deal(identifier="d1", **overrides):
    return dict(dict(code="US.AAPL", deal_id=identifier, order_id="o1", qty=2, price=10,
                     trd_side="BUY", create_time="2026-03-09 10:00:00", status="OK"), **overrides)


def test_explicit_accounts_partial_fills_and_no_writes(provider):
    adapter, _, trade, _ = provider
    trade.history_deal_list_query.return_value = (0, [deal(), deal(), deal("d2", qty=3)])
    rows = adapter.fills([{"acc_id": 7, "trd_env": "REAL"}, {"acc_id": 7, "trd_env": "SIMULATE"}], "2026-03-09", "2026-03-09")
    assert len(rows) == 4
    assert {r["environment"] for r in rows} == {"REAL", "SIMULATE"}
    assert all(r["account_id"] == 7 and r["executed_at"].endswith("-04:00") and "fees" not in r for r in rows)
    assert {call.kwargs["trd_env"] for call in trade.history_deal_list_query.call_args_list} == {"REAL", "SIMULATE"}
    assert all(call.kwargs["acc_id"] == 7 for call in trade.history_deal_list_query.call_args_list)
    assert {call[0] for call in trade.method_calls} == {"history_deal_list_query", "close"}


@pytest.mark.parametrize("accounts", [[{}], [{"acc_id": 0, "trd_env": "REAL"}], [{"acc_id": True, "trd_env": "REAL"}], [{"acc_id": 1, "trd_env": "BAD"}]])
def test_invalid_account_never_opens_connection(provider, accounts):
    adapter, _, _, sdk = provider
    with pytest.raises(module.ReviewDataError):
        adapter.fills(accounts, "2026-01-01", "2026-01-02")
    sdk.OpenSecTradeContext.assert_not_called()


def test_empty_accounts_and_sdk_failure(provider):
    adapter, _, trade, sdk = provider
    assert adapter.fills([], "2026-01-01", "2026-01-02") == []
    sdk.OpenSecTradeContext.assert_not_called()
    trade.history_deal_list_query.return_value = (-1, "unavailable")
    with pytest.raises(module.ReviewDataError):
        adapter.fills([{"acc_id": 1, "trd_env": "REAL"}], "2026-01-01", "2026-01-02")
    trade.close.assert_called_once()


@pytest.mark.parametrize("rows", [
    [deal(status="CANCELLED")], [deal(status="CHANGED")],
    [deal(), deal(qty=99)], [deal(code="SG.TEST")], [deal(qty=0)],
])
def test_unsafe_deals_require_reconciliation(provider, rows):
    adapter, _, trade, _ = provider
    trade.history_deal_list_query.return_value = (0, rows)
    with pytest.raises(module.ReviewDataError):
        adapter.fills([{"acc_id": 7, "trd_env": "REAL"}], "2026-03-09", "2026-03-09")
    trade.close.assert_called_once()


@pytest.mark.parametrize("start,end", [("2026-03-09", "2026-03-08"), ("2024-01-01", "2026-01-01")])
def test_bounded_ranges(provider, start, end):
    adapter, _, _, sdk = provider
    with pytest.raises(module.ReviewDataError):
        adapter.history("US.AAPL", start, end)
    sdk.OpenQuoteContext.assert_not_called()


def test_simulated_history_failure_never_claims_today_as_past(provider):
    adapter, _, trade, _ = provider
    trade.history_deal_list_query.return_value = (-1, "unsupported simulated history")
    with pytest.raises(module.ReviewDataError, match="Waiting for historical"):
        adapter.fills([{"acc_id": 7, "trd_env": "SIMULATE"}], "2020-01-01", "2020-01-02")
    trade.deal_list_query.assert_not_called()


def test_simulated_current_day_fallback_is_explicit(provider, monkeypatch):
    from datetime import datetime, timezone

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 9, 12, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(module, "datetime", FrozenDatetime)
    adapter, _, trade, _ = provider
    trade.history_deal_list_query.return_value = (-1, "unsupported")
    trade.deal_list_query.return_value = (0, [deal()])
    result = adapter.fills([{"acc_id": 7, "trd_env": "SIMULATE"}], "2026-03-09", "2026-03-09")
    assert result[0]["coverage"] == "current_day_only"
    trade.deal_list_query.assert_called_once_with(acc_id=7, trd_env="SIMULATE", refresh_cache=True)


def test_orders_preserve_partial_cancellations_and_account_scope(provider):
    adapter, _, trade, _ = provider
    row = dict(code="US.AAPL", order_id="o1", qty=10, price=10, trd_side="BUY",
               create_time="2026-03-09 10:00:00", updated_time="2026-03-10 11:00:00",
               dealt_qty=3, order_status="CANCELLED_PART")
    trade.history_order_list_query.return_value = (0, [row])
    result = adapter.orders([{"acc_id": 7, "trd_env": "REAL"}], "2026-03-09", "2026-03-09")
    assert result[0]["filled_quantity"] == 3
    assert result[0]["status"] == "CANCELLED_PART"
    assert result[0]["order_updated_at"] == "2026-03-10T11:00:00-04:00"
    trade.history_order_list_query.assert_called_once_with(start="2026-03-09", end="2026-03-09", acc_id=7, trd_env="REAL")
    assert {call[0] for call in trade.method_calls} == {"history_order_list_query", "close"}


def test_sdk_dataframe_rehab_column_and_symbol_normalization(provider):
    import pandas as pd
    adapter, quote, _, _ = provider
    quote.get_rehab.return_value = (0, pd.DataFrame([{"ex_div_date": "2026-01-02", "split_ratio": 2}]))
    quote.request_history_kline.return_value = (0, pd.DataFrame([bar("2026-01-02")]), None)
    assert adapter.history("AAPL", "2026-01-01", "2026-01-02")[0]["unsafe"]
    quote.get_rehab.assert_called_once_with("US.AAPL")


@pytest.mark.parametrize("symbol", ["HK.00700", "SH.600000", "SZ.000001"])
@pytest.mark.parametrize("kind,opening", [("WHOLE", "09:30"), ("MORNING", "09:30"), ("AFTERNOON", "13:00")])
def test_session_opening_times(provider, symbol, kind, opening):
    adapter, quote, _, _ = provider
    quote.request_trading_days.return_value = (0, [{"time": "2026-03-09", "trade_date_type": kind}])
    row = adapter.trading_sessions(symbol, "2026-03-09", "2026-03-09")[0]
    assert row["open_at"] == f"2026-03-09T{opening}:00+08:00"


def test_missing_order_update_time_fails_closed(provider):
    adapter, _, trade, _ = provider
    trade.history_order_list_query.return_value = (0, [dict(
        code="US.AAPL", order_id="o1", trd_side="BUY", create_time="2026-03-09 10:00:00",
    )])
    with pytest.raises(module.ReviewDataError, match="updated_time"):
        adapter.orders([{"acc_id": 7, "trd_env": "REAL"}], "2026-03-09", "2026-03-09")

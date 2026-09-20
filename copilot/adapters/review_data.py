"""Read-only Futu inputs for delayed review; no account discovery or trading.

Ranges are inclusive and bounded to 366 calendar days. Session timestamps are
conservative completion times (HK includes the closing auction). Callers must
compare close_at with their clock before consuming a daily bar.
"""

from __future__ import annotations

import math
import os
from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_symbol
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from ai_trading_copilot.copilot.adapters.futu_execution import (
    _create_trade_context,
    _load_futu,
)


class ReviewDataError(RuntimeError):
    """Data unavailable or unsafe to interpret; caller should defer review."""


_MARKETS = {
    "US": ("America/New_York", "16:00", "13:00"),
    "HK": ("Asia/Hong_Kong", "16:10", "12:10"),
    "SH": ("Asia/Shanghai", "15:00", "11:30"),
    "SZ": ("Asia/Shanghai", "15:00", "11:30"),
}


def _market(symbol):
    prefix, separator, suffix = symbol.partition(".")
    if not separator or not suffix or prefix not in _MARKETS:
        raise ReviewDataError(f"Unsupported explicit market code: {symbol}")
    return _MARKETS[prefix]


def _range(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if not 0 <= (last - first).days <= 365:
        raise ReviewDataError("Expected ordered range of at most 366 calendar days")


def _rows(data):
    return data.to_dict("records") if hasattr(data, "to_dict") else data


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ReviewDataError("Non-finite numeric data")
    return number


class FutuReviewDataAdapter:
    def __init__(self, *, host=None, port=None, security_firm=None):
        self.host = host or os.getenv("FUTU_OPEND_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("FUTU_OPEND_PORT", "11111"))
        self.security_firm = security_firm or os.getenv("FUTU_SECURITY_FIRM", "NONE")

    def _quote(self, sdk):
        return sdk.OpenQuoteContext(host=self.host, port=self.port)

    @staticmethod
    def _checked(sdk, result):
        ret, data = result
        if ret != sdk.RET_OK:
            raise ReviewDataError(str(data))
        return _rows(data)

    def trading_sessions(self, symbol, start_date, end_date) -> list[dict]:
        symbol = normalize_futu_symbol(symbol)
        zone, full_close, half_close = _market(symbol)
        _range(start_date, end_date)
        sdk = _load_futu()
        ctx = self._quote(sdk)
        try:
            rows = self._checked(sdk, ctx.request_trading_days(
                code=symbol, start=start_date, end=end_date,
            ))
            sessions = {}
            for row in rows:
                day = date.fromisoformat(row["time"]).isoformat()
                kind = row.get("trade_date_type")
                if kind not in {"WHOLE", "MORNING", "AFTERNOON"}:
                    raise ReviewDataError(f"Unknown trading session type: {kind}")
                closing = half_close if kind == "MORNING" else full_close
                opening = "13:00" if kind == "AFTERNOON" and symbol.split(".", 1)[0] in {"HK", "SH", "SZ"} else "09:30"
                open_at = datetime.combine(date.fromisoformat(day), time.fromisoformat(opening), ZoneInfo(zone))
                close_at = datetime.combine(date.fromisoformat(day), time.fromisoformat(closing), ZoneInfo(zone))
                if start_date <= day <= end_date:
                    sessions[day] = {"date": day, "open_at": open_at.isoformat(), "close_at": close_at.isoformat()}
            return [sessions[day] for day in sorted(sessions)]
        finally:
            ctx.close()

    def history(self, symbol, start_date, end_date) -> list[dict]:
        symbol = normalize_futu_symbol(symbol)
        _market(symbol)
        _range(start_date, end_date)
        sdk = _load_futu()
        ctx = self._quote(sdk)
        try:
            # Missing corporate-action permission is not evidence of no actions.
            try:
                actions = self._checked(sdk, ctx.get_rehab(symbol))
                action_dates = sorted({date.fromisoformat(str(r["ex_div_date"])[:10]).isoformat()
                                       for r in actions if start_date <= str(r["ex_div_date"])[:10] <= end_date})
                action_status = "detected" if action_dates else "clear"
            except Exception:
                action_dates, action_status = [], "unknown"
            bars, key, seen = {}, None, set()
            for _ in range(20):
                ret, data, next_key = ctx.request_history_kline(
                    symbol, start=start_date, end=end_date, ktype=sdk.KLType.K_DAY,
                    autype=sdk.AuType.NONE, max_count=1000, page_req_key=key,
                )
                for row in self._checked(sdk, (ret, data)):
                    day = date.fromisoformat(str(row["time_key"])[:10]).isoformat()
                    if not start_date <= day <= end_date:
                        continue
                    bar = {name: _number(row[name]) for name in ("open", "high", "low", "close", "volume")}
                    if min(bar[n] for n in ("open", "high", "low", "close")) <= 0 or bar["volume"] < 0:
                        raise ReviewDataError("Invalid OHLCV")
                    if not bar["low"] <= min(bar["open"], bar["close"]) <= max(bar["open"], bar["close"]) <= bar["high"]:
                        raise ReviewDataError("Inconsistent OHLC")
                    bar.update(date=day, adjustment="RAW", corporate_action_status=action_status,
                               corporate_action_dates=action_dates, unsafe=action_status != "clear")
                    if day in bars and bars[day] != bar:
                        raise ReviewDataError("Conflicting daily bars")
                    bars[day] = bar
                if next_key is None:
                    return [bars[day] for day in sorted(bars)]
                if next_key in seen:
                    raise ReviewDataError("Repeated history pagination cursor")
                seen.add(next_key)
                key = next_key
            raise ReviewDataError("History pagination exceeded bounded limit")
        finally:
            ctx.close()

    def fills(self, accounts: list[dict], start_date, end_date) -> list[dict]:
        return self._account_records(accounts, start_date, end_date, orders=False)

    def orders(self, accounts: list[dict], start_date, end_date) -> list[dict]:
        """Return order snapshots, including cancelled and partially filled orders.

        Quantities are cumulative order quantities, not additional executions.
        History is filtered by creation date; older open orders need separate
        persisted order tracking. Never infer fills from dealt_qty.
        """
        return self._account_records(accounts, start_date, end_date, orders=True)

    def _account_records(self, accounts, start_date, end_date, *, orders):
        _range(start_date, end_date)
        explicit = set()
        for account in accounts:
            acc_id, env = account.get("acc_id"), account.get("trd_env")
            if type(acc_id) is not int or acc_id <= 0 or env not in {"SIMULATE", "REAL"}:
                raise ReviewDataError("Each account needs positive acc_id and explicit SIMULATE/REAL")
            explicit.add((acc_id, env))
        if not explicit:
            return []
        sdk = _load_futu()
        if not hasattr(sdk.SecurityFirm, self.security_firm):
            raise ReviewDataError("Unknown security firm")
        ctx = _create_trade_context(sdk.OpenSecTradeContext, self.host, self.port,
                                    sdk.TrdMarket, getattr(sdk.SecurityFirm, self.security_firm))
        try:
            result = {}
            for acc_id, env in sorted(explicit):
                history_query = ctx.history_order_list_query if orders else ctx.history_deal_list_query
                response = history_query(
                    start=start_date, end=end_date, acc_id=acc_id, trd_env=getattr(sdk.TrdEnv, env),
                )
                coverage = "history"
                if response[0] != sdk.RET_OK and env == "SIMULATE":
                    # SDK accepts SIMULATE, but server/account capabilities vary.
                    # A current-day endpoint cannot backfill a historic interval.
                    today_in_all_markets = {
                        datetime.now(ZoneInfo(zone)).date().isoformat()
                        for zone, _, _ in _MARKETS.values()
                    }
                    if today_in_all_markets != {start_date} or start_date != end_date:
                        raise ReviewDataError("Waiting for historical SIMULATE coverage: " + str(response[1]))
                    query = ctx.order_list_query if orders else ctx.deal_list_query
                    response = query(acc_id=acc_id, trd_env=getattr(sdk.TrdEnv, env), refresh_cache=True)
                    coverage = "current_day_only"
                rows = self._checked(sdk, response)
                for row in rows:
                    zone, _, _ = _market(row["code"])
                    stamp = datetime.fromisoformat(row["create_time"])
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=ZoneInfo(zone))
                    if not start_date <= stamp.astimezone(ZoneInfo(zone)).date().isoformat() <= end_date:
                        continue
                    side = row["trd_side"]
                    if side not in {"BUY", "SELL", "BUY_BACK", "SELL_SHORT"}:
                        raise ReviewDataError(f"Unknown deal side: {side}")
                    if orders:
                        if row.get("order_id") in (None, ""):
                            raise ReviewDataError("Missing order_id")
                        # Missing update time must not fall back to creation time:
                        # that would let a later cancellation leak into an earlier review.
                        try:
                            updated = datetime.fromisoformat(row["updated_time"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ReviewDataError("Missing or invalid order updated_time") from exc
                        if updated.tzinfo is None:
                            updated = updated.replace(tzinfo=ZoneInfo(zone))
                        if updated < stamp:
                            raise ReviewDataError("Order update precedes creation")
                        order = dict(environment=env, account_id=acc_id, order_id=str(row["order_id"]),
                                     symbol=row["code"], side=side, quantity=_number(row["qty"]),
                                     price=_number(row["price"]), status=row["order_status"],
                                     filled_quantity=_number(row["dealt_qty"]),
                                     created_at=stamp.isoformat(), order_updated_at=updated.isoformat(), coverage=coverage)
                        if not 0 <= order["filled_quantity"] <= order["quantity"]:
                            raise ReviewDataError("Invalid cumulative filled quantity")
                        key = (env, acc_id, order["order_id"])
                        if key in result and result[key] != order:
                            raise ReviewDataError("Conflicting duplicate order")
                        result[key] = order
                        continue
                    if row.get("status", "OK") != "OK":
                        raise ReviewDataError("Non-final or corrected deal requires reconciliation")
                    for field in ("deal_id", "order_id"):
                        if row.get(field) in (None, ""):
                            raise ReviewDataError(f"Missing {field}")
                    fill = dict(environment=env, account_id=acc_id, deal_id=str(row["deal_id"]),
                                order_id=str(row["order_id"]), symbol=row["code"], side=side,
                                quantity=_number(row["qty"]), price=_number(row["price"]), executed_at=stamp.isoformat(),
                                coverage=coverage)
                    if fill["quantity"] <= 0 or fill["price"] <= 0:
                        raise ReviewDataError("Invalid deal quantity/price")
                    if row.get("fees") is not None:
                        fill["fees"] = _number(row["fees"])
                    key = (env, acc_id, fill["deal_id"])
                    if key in result and result[key] != fill:
                        raise ReviewDataError("Conflicting duplicate deal")
                    result[key] = fill
            timestamp, identifier = ("created_at", "order_id") if orders else ("executed_at", "deal_id")
            return sorted(result.values(), key=lambda row: (row[timestamp], row["environment"], row["account_id"], row[identifier]))
        finally:
            ctx.close()

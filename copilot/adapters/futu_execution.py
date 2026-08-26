"""Futu simulated trading adapter.

This adapter is intentionally scoped to ``TrdEnv.SIMULATE``. It never unlocks
trading and never submits real orders.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_symbol
from ai_trading_copilot.copilot.domain.models import (
    BrokerExecutionRequest,
    BrokerExecutionResult,
)


class FutuExecutionError(RuntimeError):
    """Raised when Futu OpenD cannot execute a simulated broker action."""


class FutuSimulatedExecutionAdapter:
    """Submit one-shot orders to a Futu simulated securities account."""

    trd_env = "SIMULATE"

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        acc_id: int | None = None,
        security_firm: str | None = None,
    ):
        self.host = host or os.getenv("FUTU_OPEND_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("FUTU_OPEND_PORT", "11111"))
        self.acc_id = acc_id if acc_id is not None else _optional_int(os.getenv("FUTU_ACC_ID"))
        self.security_firm = security_firm or os.getenv("FUTU_SECURITY_FIRM", "").strip()

    def place_order(self, request: BrokerExecutionRequest) -> BrokerExecutionResult:
        if request.trd_env != "SIMULATE":
            raise FutuExecutionError("FutuSimulatedExecutionAdapter only supports SIMULATE")

        futu = _load_futu()
        code = normalize_futu_symbol(request.symbol)
        side = _trd_side(futu.TrdSide, request.side)
        order_type = _order_type(futu.OrderType, request.order_type)
        price = float(request.price or 0)
        ctx = None
        try:
            ctx = _create_trade_context(
                futu.OpenSecTradeContext,
                self.host,
                self.port,
                futu.TrdMarket,
                _security_firm(futu.SecurityFirm, self.security_firm),
            )
            ret, data = ctx.place_order(
                price=price,
                qty=float(request.quantity),
                code=code,
                trd_side=side,
                order_type=order_type,
                trd_env=futu.TrdEnv.SIMULATE,
                acc_id=self.acc_id or 0,
                remark=request.idempotency_key[:64],
            )
        except Exception as exc:
            if isinstance(exc, FutuExecutionError):
                raise
            return BrokerExecutionResult(
                idempotency_key=request.idempotency_key,
                submitted=False,
                status="failed",
                message=f"Futu simulated order failed: {exc}",
            )
        finally:
            if ctx is not None:
                ctx.close()

        if ret != futu.RET_OK:
            return BrokerExecutionResult(
                idempotency_key=request.idempotency_key,
                submitted=False,
                status="failed",
                message=str(data),
                raw=_jsonable_records(data),
            )

        raw = _first_record_from_data(data)
        order_id = _first_present(raw.get("order_id"), raw.get("orderID"), raw.get("orderid"))
        return BrokerExecutionResult(
            idempotency_key=request.idempotency_key,
            submitted=True,
            order_id=None if order_id is None else str(order_id),
            status=str(raw.get("order_status") or raw.get("status") or "submitted"),
            message="Futu simulated order submitted.",
            raw=raw,
        )


def _load_futu():
    try:
        import futu
    except Exception as exc:  # pragma: no cover - depends on local SDK install
        raise FutuExecutionError(f"futu-api is not available: {exc}") from exc
    return futu


def _create_trade_context(
    context_cls,
    host: str,
    port: int,
    trd_market_cls,
    security_firm,
):
    kwargs = {
        "host": host,
        "port": port,
        "filter_trdmarket": getattr(trd_market_cls, "NONE", None),
    }
    if security_firm is not None:
        kwargs["security_firm"] = security_firm
    try:
        return context_cls(**kwargs)
    except TypeError:
        kwargs.pop("filter_trdmarket", None)
        return context_cls(**kwargs)


def _security_firm(security_firm_cls, configured: str):
    if not configured:
        return getattr(security_firm_cls, "NONE", None)
    return getattr(security_firm_cls, configured, getattr(security_firm_cls, "NONE", None))


def _trd_side(side_cls, value: str):
    side = value.strip().upper()
    if side == "BUY":
        return side_cls.BUY
    if side == "SELL":
        return side_cls.SELL
    raise FutuExecutionError(f"unsupported trade side: {value}")


def _order_type(order_type_cls, value: str):
    name = value.strip().upper() or "NORMAL"
    return getattr(order_type_cls, name, order_type_cls.NORMAL)


def _first_record_from_data(data: Any) -> Dict[str, Any]:
    if data is None:
        return {}
    if getattr(data, "empty", False):
        return {}
    if hasattr(data, "iloc"):
        return _clean_record(data.iloc[0].to_dict())
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return _clean_record(data[0])
    if isinstance(data, dict):
        return _clean_record(data)
    return {"value": str(data)}


def _jsonable_records(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return _clean_record(data)
    return _first_record_from_data(data)


def _clean_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


__all__ = ["FutuExecutionError", "FutuSimulatedExecutionAdapter"]

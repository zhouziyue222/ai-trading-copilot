"""Futu stock information adapter used by the copilot-local tools."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable


class FutuStockInfoError(RuntimeError):
    """Raised when Futu OpenD cannot return stock information."""


_FUTU_PREFIXES = {"US", "HK", "SH", "SZ", "SG"}


def normalize_futu_symbol(symbol: str) -> str:
    """Normalize common user tickers into Futu OpenD codes."""
    if not symbol or not symbol.strip():
        raise ValueError("symbol is required")

    raw = symbol.strip().upper().replace("-", ".")
    if "." in raw:
        left, right = raw.split(".", 1)
        if left in _FUTU_PREFIXES:
            if left == "HK" and right.isdigit():
                right = right.zfill(5)
            return f"{left}.{right}"
        if right == "HK" and left.isdigit():
            return f"HK.{left.zfill(5)}"
        if right in {"SS", "SH"} and left.isdigit():
            return f"SH.{left.zfill(6)}"
        if right == "SZ" and left.isdigit():
            return f"SZ.{left.zfill(6)}"

    if raw.isdigit():
        return f"HK.{raw.zfill(5)}" if len(raw) <= 5 else f"SH.{raw.zfill(6)}"
    if re.fullmatch(r"[A-Z][A-Z0-9.]*", raw):
        return f"US.{raw}"
    raise ValueError(f"Unsupported Futu ticker format: {symbol}")


def get_futu_stock_info(symbol: str) -> Dict[str, Any]:
    """Fetch quote and snapshot fields from Futu OpenD.

    This lives inside ai_trading_copilot so the base tradingagents package can
    remain read-only for this feature.
    """
    try:
        from futu import OpenQuoteContext, RET_OK, SubType
    except Exception as exc:  # pragma: no cover - depends on local SDK install
        raise FutuStockInfoError(f"futu-api is not available: {exc}") from exc

    host = os.getenv("FUTU_OPEND_HOST", "127.0.0.1")
    port = int(os.getenv("FUTU_OPEND_PORT", "11111"))
    code = normalize_futu_symbol(symbol)
    ctx = None
    try:
        ctx = _create_quote_context(OpenQuoteContext, host, port)
        _subscribe_quote(ctx, code, RET_OK, SubType)
        quote = _first_record(ctx.get_stock_quote([code]), RET_OK)
        snapshot = _first_record(ctx.get_market_snapshot([code]), RET_OK)
    except Exception as exc:
        if isinstance(exc, FutuStockInfoError):
            raise
        raise FutuStockInfoError(f"Futu OpenD stock info request failed for {code}: {exc}") from exc
    finally:
        if ctx is not None:
            ctx.close()

    merged = {**snapshot, **quote}
    merged["code"] = code
    merged["input_symbol"] = symbol.strip().upper()
    return merged


def format_stock_info(info: Dict[str, Any]) -> str:
    if not info:
        return "没有可用的股票信息。"
    labels = [
        ("code", "代码"),
        ("name", "名称"),
        ("last_price", "最新价"),
        ("open_price", "开盘价"),
        ("open", "开盘价"),
        ("high_price", "最高价"),
        ("high", "最高价"),
        ("low_price", "最低价"),
        ("low", "最低价"),
        ("prev_close_price", "前收盘价"),
        ("prev_close", "前收盘价"),
        ("volume", "成交量"),
        ("turnover", "成交额"),
        ("bid", "买价"),
        ("ask", "卖价"),
        ("listing_date", "上市日期"),
        ("sec_status", "证券状态"),
        ("market_val", "市值"),
    ]
    lines = ["# 富途股票信息", ""]
    for key, label in labels:
        if key in info:
            lines.append(f"- {label}: {info[key]}")
    return "\n".join(lines)


def _create_quote_context(context_cls, host: str, port: int):
    try:
        return context_cls(host=host, port=port, ai_type=1)
    except TypeError:
        return context_cls(host=host, port=port)


def _subscribe_quote(ctx: Any, code: str, ret_ok: Any, subtype_cls: Any) -> None:
    ret, data = ctx.subscribe([code], [subtype_cls.QUOTE], subscribe_push=False)
    if ret != ret_ok:
        raise FutuStockInfoError(str(data))


def _first_record(result: Iterable[Any], ret_ok: Any) -> Dict[str, Any]:
    ret, data = result
    if ret != ret_ok:
        raise FutuStockInfoError(str(data))
    if data is None or data.empty:
        return {}
    row = data.iloc[0]
    return {
        str(key): _clean_value(value)
        for key, value in row.to_dict().items()
    }


def _clean_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

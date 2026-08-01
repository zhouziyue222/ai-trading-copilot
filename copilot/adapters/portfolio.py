"""Futu portfolio adapter for copilot risk context."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

from ai_trading_copilot.copilot.domain.enums import ExecutionMode
from ai_trading_copilot.copilot.domain.localization import zh_label
from ai_trading_copilot.copilot.domain.models import PortfolioSnapshot


class FutuPortfolioError(RuntimeError):
    """Raised when Futu OpenD cannot return portfolio information."""


def get_futu_portfolio_snapshot(
    mode: ExecutionMode | str = ExecutionMode.SIMULATION,
) -> PortfolioSnapshot:
    """Fetch portfolio weights from Futu OpenD.

    The default source is the simulated account. ``ExecutionMode.LIVE`` maps to
    Futu's real trading environment, but this function only reads positions and
    account info; it never places orders.
    """
    try:
        from futu import (
            Currency,
            OpenSecTradeContext,
            RET_OK,
            SecurityFirm,
            TrdEnv,
            TrdMarket,
        )
    except Exception as exc:  # pragma: no cover - depends on local SDK install
        raise FutuPortfolioError(f"futu-api is not available: {exc}") from exc

    host = os.getenv("FUTU_OPEND_HOST", "127.0.0.1")
    port = int(os.getenv("FUTU_OPEND_PORT", "11111"))
    trd_env = _to_trd_env(mode, TrdEnv)
    security_firm = _security_firm(SecurityFirm)
    acc_id = _optional_int(os.getenv("FUTU_ACC_ID"))
    ctx = None
    try:
        ctx = _create_trade_context(
            OpenSecTradeContext,
            host,
            port,
            TrdMarket,
            security_firm,
        )
        account = _account_info(ctx, RET_OK, Currency, trd_env, acc_id)
        total_assets = _total_assets(account)
        positions = _records(
            ctx.position_list_query(
                trd_env=trd_env,
                acc_id=acc_id or 0,
                refresh_cache=True,
            ),
            RET_OK,
        )
    except Exception as exc:
        if isinstance(exc, FutuPortfolioError):
            raise
        raise FutuPortfolioError(f"Futu OpenD portfolio request failed: {exc}") from exc
    finally:
        if ctx is not None:
            ctx.close()

    return PortfolioSnapshot(
        current_drawdown=0.0,
        position_weights=_position_weights(positions, total_assets),
    )


def format_portfolio_snapshot(
    *,
    snapshot: PortfolioSnapshot,
    mode: ExecutionMode | str,
    error: str | None = None,
) -> str:
    lines = [
        "# 富途组合快照",
        "",
        f"- 来源：{zh_label(ExecutionMode(mode) if isinstance(mode, str) else mode)}",
        f"- 当前回撤：{snapshot.current_drawdown:.1%}",
        "- 持仓权重：",
    ]
    if snapshot.position_weights:
        for symbol, weight in sorted(snapshot.position_weights.items()):
            lines.append(f"  - {symbol}: {weight:.1%}")
    else:
        lines.append("  - 无")
    if error:
        lines.extend(["", f"获取错误：{error}"])
    return "\n".join(lines)


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


def _to_trd_env(mode: ExecutionMode | str, trd_env_cls):
    value = mode if isinstance(mode, ExecutionMode) else ExecutionMode(mode)
    if value == ExecutionMode.LIVE:
        return trd_env_cls.REAL
    return trd_env_cls.SIMULATE


def _security_firm(security_firm_cls):
    configured = os.getenv("FUTU_SECURITY_FIRM", "").strip()
    if not configured:
        return getattr(security_firm_cls, "NONE", None)
    return getattr(security_firm_cls, configured, getattr(security_firm_cls, "NONE", None))


def _account_info(ctx, ret_ok, currency_cls, trd_env, acc_id: int | None) -> Dict[str, Any]:
    kwargs = {
        "trd_env": trd_env,
        "acc_id": acc_id or 0,
        "refresh_cache": True,
    }
    currency = getattr(currency_cls, "USD", None)
    if currency is not None:
        kwargs["currency"] = currency
    ret, data = ctx.accinfo_query(**kwargs)
    if ret != ret_ok:
        raise FutuPortfolioError(str(data))
    if data is None or data.empty:
        return {}
    return _clean_record(data.iloc[0].to_dict())


def _records(result: Tuple[Any, Any], ret_ok) -> list[Dict[str, Any]]:
    ret, data = result
    if ret != ret_ok:
        raise FutuPortfolioError(str(data))
    if data is None or data.empty:
        return []
    return [_clean_record(row) for row in data.to_dict("records")]


def _total_assets(account: Dict[str, Any]) -> float:
    for key in ("total_assets", "net_assets", "market_val", "cash"):
        value = _float_or_none(account.get(key))
        if value and value > 0:
            return value
    return 0.0


def _position_weights(
    positions: Iterable[Dict[str, Any]],
    total_assets: float,
) -> Dict[str, float]:
    if total_assets <= 0:
        return {}
    weights: Dict[str, float] = {}
    for row in positions:
        symbol = str(row.get("code") or row.get("stock_name") or "").strip().upper()
        market_value = _float_or_none(
            row.get("market_val")
            or row.get("market_value")
            or row.get("nominal_price")
        )
        if not symbol or market_value is None or market_value <= 0:
            continue
        weights[symbol] = min(max(market_value / total_assets, 0.0), 1.0)
    return weights


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

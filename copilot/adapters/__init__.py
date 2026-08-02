"""Adapters that connect the copilot to market, portfolio, and reference data."""

from .market_data import (
    FutuMarketDataAdapter,
    FutuMarketDataError,
    get_stock_data_text,
    parse_price_bars_from_csv,
)
from .portfolio import (
    FutuPortfolioError,
    format_portfolio_snapshot,
    get_futu_portfolio_snapshot,
)
from .futu_execution import (
    FutuExecutionError,
    FutuSimulatedExecutionAdapter,
)
from .stock_info import (
    FutuStockInfoError,
    format_stock_info,
    get_futu_stock_info,
    normalize_futu_symbol,
)

__all__ = [
    "FutuPortfolioError",
    "FutuStockInfoError",
    "FutuMarketDataAdapter",
    "FutuMarketDataError",
    "FutuExecutionError",
    "FutuSimulatedExecutionAdapter",
    "format_portfolio_snapshot",
    "format_stock_info",
    "get_futu_portfolio_snapshot",
    "get_futu_stock_info",
    "get_stock_data_text",
    "normalize_futu_symbol",
    "parse_price_bars_from_csv",
]

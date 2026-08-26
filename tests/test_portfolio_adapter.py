from ai_trading_copilot.copilot.adapters.portfolio import _position_weights
from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_code_to_symbol


def test_futu_us_code_maps_to_internal_symbol_for_portfolio_weight():
    weights = _position_weights(
        [
            {"code": "US.AAPL", "market_val": 10_000},
            {"code": "US.MSFT", "market_val": 5_000},
        ],
        total_assets=100_000,
    )

    assert weights == {"AAPL": 0.10, "MSFT": 0.05}


def test_futu_internal_symbol_mapper_keeps_non_us_full_codes():
    assert normalize_futu_code_to_symbol("US.AAPL") == "AAPL"
    assert normalize_futu_code_to_symbol("HK.00700") == "HK.00700"


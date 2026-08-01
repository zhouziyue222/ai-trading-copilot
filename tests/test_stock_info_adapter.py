import sys
import types

import pandas as pd
import pytest

from ai_trading_copilot.copilot.adapters.stock_info import (
    FutuStockInfoError,
    get_futu_stock_info,
)


def test_get_futu_stock_info_subscribes_quote_before_fetch(monkeypatch):
    fake_futu, context_cls = _fake_futu_module()
    monkeypatch.setitem(sys.modules, "futu", fake_futu)

    info = get_futu_stock_info("CRCL")

    assert info["code"] == "US.CRCL"
    assert info["last_price"] == 113.67
    assert context_cls.instances[0].calls == [
        ("subscribe", ["US.CRCL"], ["QUOTE"], False),
        ("get_stock_quote", ["US.CRCL"]),
        ("get_market_snapshot", ["US.CRCL"]),
        ("close",),
    ]


def test_get_futu_stock_info_raises_when_quote_subscription_fails(monkeypatch):
    fake_futu, context_cls = _fake_futu_module(subscribe_ret=1, subscribe_msg="no Basic data")
    monkeypatch.setitem(sys.modules, "futu", fake_futu)

    with pytest.raises(FutuStockInfoError, match="no Basic data"):
        get_futu_stock_info("CRCL")

    assert context_cls.instances[0].calls == [
        ("subscribe", ["US.CRCL"], ["QUOTE"], False),
        ("close",),
    ]


def _fake_futu_module(subscribe_ret=0, subscribe_msg="subscribed"):
    class FakeSubType:
        QUOTE = "QUOTE"

    class FakeOpenQuoteContext:
        instances = []

        def __init__(self, host, port, ai_type=None):
            self.host = host
            self.port = port
            self.ai_type = ai_type
            self.calls = []
            self.instances.append(self)

        def subscribe(self, codes, subtypes, subscribe_push=True):
            self.calls.append(("subscribe", codes, subtypes, subscribe_push))
            return subscribe_ret, subscribe_msg

        def get_stock_quote(self, codes):
            self.calls.append(("get_stock_quote", codes))
            return 0, pd.DataFrame([{
                "code": codes[0],
                "name": "Circle",
                "last_price": 113.67,
            }])

        def get_market_snapshot(self, codes):
            self.calls.append(("get_market_snapshot", codes))
            return 0, pd.DataFrame([{
                "code": codes[0],
                "market_val": 1000000,
            }])

        def close(self):
            self.calls.append(("close",))

    fake_futu = types.SimpleNamespace(
        OpenQuoteContext=FakeOpenQuoteContext,
        RET_OK=0,
        SubType=FakeSubType,
    )
    return fake_futu, FakeOpenQuoteContext

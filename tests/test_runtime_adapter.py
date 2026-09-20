import os

from ai_trading_copilot.copilot.adapters import runtime
from yfinance.config import YfConfig


def test_yfinance_proxy_defaults_when_missing(monkeypatch):
    monkeypatch.delenv("YFINANCE_PROXY_URL", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    YfConfig.network.proxy = None

    runtime.ensure_yfinance_proxy()

    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert YfConfig.network.proxy == "http://127.0.0.1:7890"


def test_yfinance_proxy_preserves_existing_values(monkeypatch):
    monkeypatch.delenv("YFINANCE_PROXY_URL", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://existing-http:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://existing-https:8080")
    YfConfig.network.proxy = None

    runtime.ensure_yfinance_proxy()

    assert os.environ["HTTP_PROXY"] == "http://existing-http:8080"
    assert os.environ["HTTPS_PROXY"] == "http://existing-https:8080"
    assert YfConfig.network.proxy == "http://127.0.0.1:7890"

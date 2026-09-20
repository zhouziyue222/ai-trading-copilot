import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.agents.risk_agent import RiskAgent
from ai_trading_copilot.copilot.domain import (
    MarketRegime,
    PortfolioSnapshot,
    PriceBar,
    SubscriptionStatus,
    TradeDirection,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.graph.copilot_langgraph import (
    _merge_target_weight_overrides,
)
from ai_trading_copilot.copilot.services.risk_position_store import RiskPositionStore
from ai_trading_copilot.copilot.ui.app import UISettings, create_app


def _bars(closes, spread=0.02):
    return [
        PriceBar(
            date=f"2026-01-{index + 1:02d}",
            open=close,
            high=close * (1 + spread),
            low=close * (1 - spread),
            close=close,
            volume=1000,
        )
        for index, close in enumerate(closes)
    ]


def _plan(position_weight=0.20, stop_loss=90, reward_risk_ratio=3.0):
    return TradePlan(
        symbol="AAPL",
        subscription_status=SubscriptionStatus.ACTIONABLE,
        market_regime=MarketRegime.UPTREND,
        direction=TradeDirection.BUY,
        entry_logic="Buy near support.",
        support_level=100,
        stop_loss=stop_loss,
        targets=[110],
        reward_risk_ratio=reward_risk_ratio,
        position_weight=position_weight,
        holding_period="1-6 weeks",
        invalidation_conditions=["Close below support."],
        persona_fit_reason="Fits persona.",
    )


def test_risk_position_store_default_save_and_reset(tmp_path):
    path = tmp_path / "risk_position.user.json"
    store = RiskPositionStore(path)

    default = store.load()
    assert default["source"] == "default"
    assert default["persona"]["target_annual_volatility"] == 0.20
    assert default["persona"]["atr_window"] == 14
    assert default["target_weights"] == {}

    persona = UserPersonaConfig(**default["persona"])
    saved = store.save(
        persona.model_copy(update={"max_single_position_weight": 0.30}),
        {"aapl": 0.2},
    )
    assert saved["source"] == "user"
    assert saved["target_weights"] == {"AAPL": 0.2}
    assert json.loads(path.read_text(encoding="utf-8"))["persona"][
        "max_single_position_weight"
    ] == 0.30

    reset = store.reset()
    assert reset["source"] == "default"
    assert not path.exists()


def test_risk_position_api_get_put_reset(tmp_path):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "subscriptions.json",
            memory_database=tmp_path / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            risk_position_file=tmp_path / "risk_position.user.json",
            run_in_background=False,
        )
    )
    client = TestClient(app)

    default = client.get("/api/risk-position")
    assert default.status_code == 200
    assert default.json()["source"] == "default"

    persona = default.json()["persona"]
    persona["target_annual_volatility"] = 0.18
    persona["max_single_position_weight"] = 0.30
    response = client.put(
        "/api/risk-position",
        json={"persona": persona, "target_weights": {"aapl": 0.15}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "user"
    assert payload["target_weights"] == {"AAPL": 0.15}

    persisted = client.get("/api/risk-position").json()
    assert persisted["persona"]["target_annual_volatility"] == 0.18
    assert persisted["target_weights"]["AAPL"] == 0.15

    reset = client.post("/api/risk-position/reset")
    assert reset.status_code == 200
    assert reset.json()["source"] == "default"


def test_risk_position_api_rejects_invalid_target_weight(tmp_path):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "subscriptions.json",
            memory_database=tmp_path / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            risk_position_file=tmp_path / "risk_position.user.json",
            run_in_background=False,
        )
    )
    client = TestClient(app)
    persona = client.get("/api/risk-position").json()["persona"]
    response = client.put(
        "/api/risk-position",
        json={"persona": persona, "target_weights": {"AAPL": 1.5}},
    )
    assert response.status_code == 422


def test_risk_report_includes_mature_position_sizing_advice():
    closes = [100 + index * 0.4 for index in range(60)]
    closes.extend([95, 92, 98, 101, 96])
    bars = _bars(closes)
    result = RiskAgent().review_with_report(
        persona=UserPersonaConfig(),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        plan=_plan(stop_loss=90, reward_risk_ratio=3.0),
        price_history_by_symbol={"AAPL": bars},
    )

    assert result.report.startswith("# 风险管理报告")
    assert "## 仓位管理指标配置" in result.report
    assert "## 仓位管理建议（基于金融指标）" in result.report
    assert "年化波动率" in result.report
    assert "ATR%" in result.report
    assert "止损距离" in result.report
    assert "建议单票上限" in result.report


def test_risk_report_handles_missing_price_history():
    result = RiskAgent().review_with_report(
        persona=UserPersonaConfig(),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        plan=_plan(),
    )

    assert result.report.startswith("# 风险管理报告")
    assert "## 仓位管理建议（基于金融指标）" in result.report
    assert "止损距离" in result.report


def test_target_weight_overrides_only_apply_to_existing_plans():
    merged = _merge_target_weight_overrides(
        {"AAPL": 0.20, "MSFT": 0.10},
        {"aapl": 0.15, "NVDA": 0.40},
    )
    assert merged == {"AAPL": 0.15, "MSFT": 0.10}

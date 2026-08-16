import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.domain import BrokerExecutionResult, PortfolioSnapshot
from ai_trading_copilot.copilot.ui.app import UISettings, create_app


class FakeGraph:
    NODE_ORDER = ["Load", "Finish"]
    calls = []
    memory_agents = []

    def __init__(self, *, run_tracker, memory_agent):
        self.run_tracker = run_tracker
        self.__class__.memory_agents.append(memory_agent)

    def run(self, **kwargs):
        self.__class__.calls.append(kwargs)
        report_dir = Path(kwargs["report_output_dir"]) / "6_explanation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "run_explanation.md"
        report_path.write_text("# Summary\n\nUI run complete.\n", encoding="utf-8")
        self.run_tracker.record_report("run_explanation", report_path)
        return {
            "subscription_symbols": kwargs["subscription_symbols"],
            "agent_reports": {"run_explanation": str(report_path)},
            "analyst_reports": {},
            "radar_items": [],
            "technical_positions": {},
            "fundamental_analysis_by_symbol": {},
            "trade_plans": {},
            "risk_assessments": {},
            "execution_decisions": {},
            "explanations": {},
            "errors": [],
        }


class FakeFundamentalRetriever:
    def rag_status(self):
        return {
            "backend": "fundamental_chroma",
            "scope": "fundamental_only",
            "available": False,
            "document_count": 0,
        }

    def ingest_seed_knowledge(self):
        return {"added": 0, "skipped": 0, "errors": []}

    def ingest_text_knowledge(self, **kwargs):
        return {"added": 1, "skipped": 0, "errors": []}

    def ingest_online_fundamental_research(self, **kwargs):
        return {"added": 0, "skipped": 0, "errors": []}


class PendingOrderGraph:
    NODE_ORDER = ["Load", "Portfolio Manager"]
    portfolio_error = False

    def __init__(self, *, run_tracker, memory_agent=None, **kwargs):
        self.run_tracker = run_tracker

    def run(self, **kwargs):
        symbol = kwargs["subscription_symbols"][0]
        report_dir = Path(kwargs["report_output_dir"]) / "5_portfolio_manager"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "portfolio_manager.md"
        report_path.write_text("# Portfolio Manager\n\nPending simulated order.\n", encoding="utf-8")
        self.run_tracker.record_report("portfolio_manager", report_path)
        return {
            "subscription_symbols": kwargs["subscription_symbols"],
            "portfolio_mode": kwargs["portfolio_mode"],
            "execution_mode": kwargs["mode"],
            "agent_reports": {"portfolio_manager": str(report_path)},
            "analyst_reports": {},
            "radar_items": [{"symbol": symbol, "status": "actionable"}],
            "technical_positions": {},
            "fundamental_analysis_by_symbol": {},
            "trade_plans": {
                symbol: {
                    "symbol": symbol,
                    "subscription_status": "actionable",
                    "direction": "buy",
                    "entry_logic": "Buy near support.",
                }
            },
            "risk_assessments": {
                symbol: {
                    "symbol": symbol,
                    "approved": True,
                    "current_position_weight": 0.10,
                    "target_weight": 0.20,
                    "final_weight": 0.20,
                    "delta_weight": 0.10,
                    "portfolio_value": 100_000,
                    "estimated_trade_value": 10_000,
                    "clamped": False,
                }
            },
            "execution_decisions": {
                symbol: {
                    "symbol": symbol,
                    "mode": kwargs["mode"],
                    "status": "portfolio_decided",
                    "direction": "buy",
                    "approved_by_risk": True,
                    "requires_user_confirmation": False,
                    "message": "Portfolio Manager selected buy about 250 shares.",
                    "action": "buy",
                    "quantity": 250,
                    "current_weight": 0.10,
                    "target_weight": 0.20,
                    "final_weight": 0.20,
                    "delta_weight": 0.10,
                    "current_price": 40,
                    "estimated_trade_value": 10_000,
                    "pending_broker_order": True,
                    "broker_confirmation_required": True,
                    "submitted_to_broker": False,
                }
            },
            "explanations": {},
            "errors": ["portfolio_fetch_failed: boom"] if self.portfolio_error else [],
        }


class FailingPortfolioOrderGraph(PendingOrderGraph):
    portfolio_error = True


class RecordingBroker:
    def __init__(self):
        self.requests = []

    def place_order(self, request):
        self.requests.append(request)
        return BrokerExecutionResult(
            idempotency_key=request.idempotency_key,
            submitted=True,
            order_id="SIM-1",
            status="submitted",
            message="submitted",
        )


class RaisingBroker:
    def __init__(self):
        self.requests = []

    def place_order(self, request):
        self.requests.append(request)
        raise RuntimeError("OpenD unavailable")


@pytest.fixture()
def client(tmp_path):
    FakeGraph.calls = []
    FakeGraph.memory_agents = []
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_file=tmp_path / "config" / "memory.jsonl",
            reports_dir=tmp_path / "reports",
            graph_cls=FakeGraph,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
        )
    )
    return TestClient(app)


def _order_client(tmp_path, graph_cls=PendingOrderGraph, broker=None):
    broker = broker or RecordingBroker()
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_file=tmp_path / "config" / "memory.jsonl",
            reports_dir=tmp_path / "reports",
            graph_cls=graph_cls,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
            simulated_broker_factory=lambda: broker,
        )
    )
    return TestClient(app), broker


def _portfolio_client(tmp_path, portfolio_getter, *, timeout_seconds=5.0):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_file=tmp_path / "config" / "memory.jsonl",
            reports_dir=tmp_path / "reports",
            graph_cls=FakeGraph,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
            portfolio_snapshot_getter=portfolio_getter,
            portfolio_snapshot_timeout_seconds=timeout_seconds,
        )
    )
    return TestClient(app)


def test_subscription_list_loads_when_file_is_missing(client):
    response = client.get("/api/subscriptions")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_subscription_crud_persists_valid_json(client, tmp_path):
    create_response = client.post(
        "/api/subscriptions",
        json={
            "symbol": "aapl",
            "market_type": "US_STOCK",
            "reason": "pullback",
            "target_action": "watch support",
        },
    )
    assert create_response.status_code == 200
    assert create_response.json()["items"][0]["symbol"] == "AAPL"

    update_response = client.patch(
        "/api/subscriptions/AAPL",
        json={"status": "near_opportunity", "reason": "near support"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["items"][0]["status"] == "near_opportunity"

    delete_response = client.delete("/api/subscriptions/AAPL")
    assert delete_response.status_code == 200
    assert delete_response.json()["items"] == []

    payload = json.loads((tmp_path / "config" / "subscriptions.json").read_text(encoding="utf-8"))
    assert payload == {"items": []}


def test_create_run_constructs_graph_params_and_status(client):
    response = client.post(
        "/api/runs",
        json={
            "symbols": ["aapl"],
            "manual_symbols": "msft, CRCL",
            "selected_analysts": ["news_sentiment", "fundamental_analysis"],
            "trade_date": "2026-05-13",
            "look_back_days": 45,
            "portfolio_mode": "live",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"].startswith("run_UI_")
    assert payload["status"]["status"] == "succeeded"
    assert payload["reports"]["run_explanation"]["exists"] is True
    assert payload["status"]["decision_summary"]["metrics"]["symbol_count"] == 3
    assert payload["status"]["decision_summary"]["symbols"][0]["symbol"] == "AAPL"
    assert FakeGraph.calls[0]["subscription_symbols"] == ["AAPL", "MSFT", "CRCL"]
    assert [item.value for item in FakeGraph.calls[0]["selected_analysts"]] == [
        "news_sentiment",
        "fundamental_analysis",
    ]
    assert FakeGraph.calls[0]["mode"].value == "simulation"
    assert FakeGraph.calls[0]["portfolio_mode"].value == "live"
    assert FakeGraph.memory_agents[0].retrieve_context(symbol="AAPL") == []


def test_get_run_returns_status(client):
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = client.get(f"/api/runs/{created['run_id']}")

    assert response.status_code == 200
    assert response.json()["status"]["symbols"] == ["AAPL"]


def test_confirm_simulated_order_submits_once_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_SIMULATED_ORDER_MAX_QUANTITY", "100")
    order_client, broker = _order_client(tmp_path)
    created = order_client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()
    row = created["status"]["decision_summary"]["symbols"][0]

    assert row["pending_broker_order"] is True
    assert row["broker_confirmation_required"] is True
    assert broker.requests == []

    response = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(broker.requests) == 1
    request = broker.requests[0]
    assert request.symbol == "AAPL"
    assert request.side == "BUY"
    assert request.quantity == 100
    assert request.price == 40
    order = payload["order"]
    assert order["submitted_to_broker"] is True
    assert order["submitted_quantity"] == 100
    assert order["broker_order_id"] == "SIM-1"
    updated = payload["status"]["decision_summary"]["symbols"][0]
    assert updated["pending_broker_order"] is False
    assert updated["broker_confirmation_required"] is False
    assert payload["reports"]["simulated_broker_orders"]["exists"] is True
    report_response = order_client.get(
        f"/api/reports/{created['run_id']}/simulated_broker_orders"
    )
    assert report_response.status_code == 200
    assert "# Simulated Broker Orders" in report_response.text
    assert "SIM-1" in report_response.text

    repeated = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert repeated.status_code == 200
    assert len(broker.requests) == 1
    assert repeated.json()["order"]["broker_idempotency_key"] == order["broker_idempotency_key"]


def test_confirm_simulated_order_blocks_live_portfolio_mode(tmp_path):
    order_client, broker = _order_client(tmp_path)
    created = order_client.post(
        "/api/runs",
        json={"manual_symbols": "AAPL", "portfolio_mode": "live"},
    ).json()

    response = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert response.status_code == 400
    assert "simulated portfolio" in response.json()["detail"]
    assert broker.requests == []


def test_confirm_simulated_order_blocks_when_portfolio_fetch_failed(tmp_path):
    order_client, broker = _order_client(tmp_path, graph_cls=FailingPortfolioOrderGraph)
    created = order_client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert response.status_code == 400
    assert "Portfolio fetch failed" in response.json()["detail"]
    assert broker.requests == []


def test_confirm_simulated_order_records_broker_failure(tmp_path):
    broker = RaisingBroker()
    order_client, _ = _order_client(tmp_path, broker=broker)
    created = order_client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert response.status_code == 200
    assert len(broker.requests) == 1
    order = response.json()["order"]
    assert order["submitted_to_broker"] is False
    assert order["broker_status"] == "failed"
    assert "OpenD unavailable" in order["broker_message"]


def test_rag_status_endpoint_degrades_when_chroma_is_unavailable(client):
    response = client.get("/api/rag/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "fundamental_chroma"
    assert payload["scope"] == "fundamental_only"
    assert "available" in payload
    assert "document_count" in payload


def test_rag_ingest_text_endpoint_returns_ingest_result(client):
    response = client.post(
        "/api/rag/ingest-text",
        json={
            "title": "AAPL note",
            "text": "AAPL earnings guidance improved, but event risk remains.",
            "symbols": ["AAPL"],
            "tags": ["earnings"],
        },
    )

    assert response.status_code == 200
    assert set(response.json()) == {"added", "skipped", "errors"}


def test_simulated_portfolio_endpoint_returns_snapshot(tmp_path):
    def portfolio_getter(mode):
        assert mode.value == "simulation"
        return PortfolioSnapshot(
            total_value=100_000,
            cash=25_000,
            position_weights={"AAPL": 0.35, "MSFT": 0.20},
        )

    client = _portfolio_client(tmp_path, portfolio_getter)

    response = client.get("/api/portfolio/simulated")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["mode"] == "simulation"
    assert payload["total_value"] == 100_000
    assert payload["cash"] == 25_000
    assert payload["cash_weight"] == 0.25
    assert payload["position_weights"] == {"AAPL": 0.35, "MSFT": 0.20}
    assert payload["error"] is None
    assert payload["updated_at"]


def test_simulated_portfolio_endpoint_degrades_on_error(tmp_path):
    def portfolio_getter(mode):
        raise RuntimeError("OpenD unavailable")

    client = _portfolio_client(tmp_path, portfolio_getter)

    response = client.get("/api/portfolio/simulated")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["mode"] == "simulation"
    assert payload["total_value"] is None
    assert payload["cash"] is None
    assert payload["cash_weight"] is None
    assert payload["position_weights"] == {}
    assert "OpenD unavailable" in payload["error"]


def test_simulated_portfolio_endpoint_times_out_slow_snapshot(tmp_path):
    def portfolio_getter(mode):
        time.sleep(0.05)
        return PortfolioSnapshot(total_value=100_000)

    client = _portfolio_client(tmp_path, portfolio_getter, timeout_seconds=0.01)

    response = client.get("/api/portfolio/simulated")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "timed out" in payload["error"]


def test_report_endpoint_returns_markdown_and_rejects_unsafe_paths(client, tmp_path):
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = client.get(f"/api/reports/{created['run_id']}/run_explanation")
    assert response.status_code == 200
    assert "# Summary" in response.text

    run_dir = tmp_path / "reports" / "run_bad"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "reports": {
                    "leak": {
                        "path": str(outside),
                        "exists": True,
                        "bytes": 6,
                    }
                },
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    unsafe = client.get("/api/reports/run_bad/leak")
    assert unsafe.status_code == 400

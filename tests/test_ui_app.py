import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.domain import BrokerExecutionResult, PortfolioSnapshot
from ai_trading_copilot.copilot.ui import app as ui_app
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
    probe_values = []

    def rag_status(self, *, probe=False):
        self.__class__.probe_values.append(probe)
        return {
            "backend": "fundamental_chroma",
            "scope": "fundamental_only",
            "available": False,
            "document_count": 0,
            "error": "OPENAI_API_KEY or DASHSCOPE_API_KEY is not set",
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


class CancellableGraph:
    NODE_ORDER = ["Load", "Finish"]

    def __init__(self, *, run_tracker, memory_agent=None, cancellation_checker=None, **kwargs):
        self.run_tracker = run_tracker
        self.cancellation_checker = cancellation_checker or (lambda: False)

    def run(self, **kwargs):
        self.run_tracker.start_node("Load")
        self.run_tracker.succeed_node("Load")
        self.run_tracker.start_node("Finish")
        deadline = time.time() + 5
        while not self.cancellation_checker():
            if time.time() > deadline:
                raise TimeoutError("cancel signal was not received")
            time.sleep(0.01)
        raise ui_app.RunCancelled("Run cancelled by user.")


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


class FailingOnceBroker:
    def __init__(self):
        self.requests = []
        self._failed = False

    def place_order(self, request):
        self.requests.append(request)
        if not self._failed:
            self._failed = True
            raise RuntimeError("OpenD unavailable")
        return BrokerExecutionResult(
            idempotency_key=request.idempotency_key,
            submitted=True,
            order_id="SIM-2",
            status="submitted",
            message="submitted",
        )


@pytest.fixture()
def client(tmp_path):
    FakeGraph.calls = []
    FakeGraph.memory_agents = []
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            graph_cls=FakeGraph,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
        )
    )
    return TestClient(app)


def _order_client(tmp_path, graph_cls=PendingOrderGraph, broker=None, latest_price=40.0):
    broker = broker or RecordingBroker()
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            graph_cls=graph_cls,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
            simulated_broker_factory=lambda: broker,
            latest_price_getter=lambda symbol: latest_price,
        )
    )
    return TestClient(app), broker


def _portfolio_client(tmp_path, portfolio_getter, *, timeout_seconds=5.0):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            graph_cls=FakeGraph,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
            portfolio_snapshot_getter=portfolio_getter,
            portfolio_snapshot_timeout_seconds=timeout_seconds,
        )
    )
    return TestClient(app)


def _wait_for_status(client, run_id, predicate, *, timeout=5.0):
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        last_status = response.json()["status"]
        if predicate(last_status):
            return last_status
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for run status. Last status: {last_status}")


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


def test_web_run_can_disable_long_term_memory(client):
    response = client.post(
        "/api/runs",
        json={"manual_symbols": "AAPL", "long_term_memory_enabled": False},
    )

    assert response.status_code == 200
    assert FakeGraph.memory_agents[-1] is None


def test_runtime_config_reports_environment_default(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_LONG_TERM_MEMORY_ENABLED", "false")
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            graph_cls=FakeGraph,
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
        )
    )

    response = TestClient(app).get("/api/runtime-config")

    assert response.status_code == 200
    assert response.json()["long_term_memory_enabled"] is False


def test_get_run_returns_status(client):
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = client.get(f"/api/runs/{created['run_id']}")

    assert response.status_code == 200
    assert response.json()["status"]["symbols"] == ["AAPL"]


def test_cancel_completed_run_is_idempotent(client):
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = client.post(f"/api/runs/{created['run_id']}/cancel")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["status"] == "succeeded"
    assert payload["status"]["cancel_requested"] is False


def test_cancel_running_background_run_finishes_as_cancelled(tmp_path):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            graph_cls=CancellableGraph,
            run_in_background=True,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
        )
    )
    client = TestClient(app)
    created = client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()
    run_id = created["run_id"]

    _wait_for_status(client, run_id, lambda status: status["nodes"]["Finish"]["status"] == "running")
    cancel_response = client.post(f"/api/runs/{run_id}/cancel")

    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"]["cancel_requested"] is True
    final_status = _wait_for_status(
        client,
        run_id,
        lambda status: status["status"] == "cancelled"
        and status["reports"].get("run_audit", {}).get("exists") is True,
    )
    assert final_status["status"] == "cancelled"
    assert final_status["nodes"]["Load"]["status"] == "succeeded"
    assert final_status["nodes"]["Finish"]["status"] == "skipped"
    assert final_status["errors"] == []
    assert final_status["reports"]["run_audit"]["exists"] is True


def test_cancel_file_only_running_run_rejects_missing_worker(tmp_path):
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            run_in_background=False,
            fundamental_retriever_factory=lambda settings, auto_ingest_seed: FakeFundamentalRetriever(),
        )
    )
    client = TestClient(app)
    run_dir = tmp_path / "reports" / "run_file_only"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "run_id": "run_file_only",
                "symbols": ["AAPL"],
                "status": "running",
                "cancel_requested": False,
                "reports": {},
                "errors": [],
                "nodes": {},
            }
        ),
        encoding="utf-8",
    )

    response = client.post("/api/runs/run_file_only/cancel")

    assert response.status_code == 409
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert status["cancel_requested"] is False


def test_cancel_run_rejects_unsafe_run_id(client):
    response = client.post("/api/runs/%2E%2E/cancel")

    assert response.status_code == 400


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

    assert created["status"]["status"] == "degraded"

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
    assert order["broker_attempt_count"] == 1
    assert order["broker_retry_available"] is True
    updated = response.json()["status"]["decision_summary"]["symbols"][0]
    assert updated["pending_broker_order"] is True
    assert updated["broker_confirmation_required"] is True


def test_confirm_simulated_order_retries_after_transient_failure(tmp_path):
    broker = FailingOnceBroker()
    order_client, _ = _order_client(tmp_path, broker=broker)
    created = order_client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    first = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )
    second = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert first.json()["order"]["submitted_to_broker"] is False
    assert second.status_code == 200
    assert second.json()["order"]["submitted_to_broker"] is True
    assert second.json()["order"]["broker_order_id"] == "SIM-2"
    assert len(broker.requests) == 2


def test_confirm_simulated_order_blocks_on_price_drift(tmp_path):
    order_client, broker = _order_client(tmp_path, latest_price=50.0)
    created = order_client.post("/api/runs", json={"manual_symbols": "AAPL"}).json()

    response = order_client.post(
        f"/api/runs/{created['run_id']}/orders/AAPL/confirm-simulated"
    )

    assert response.status_code == 400
    assert "rerun before confirming" in response.json()["detail"]
    assert broker.requests == []


def test_rag_status_endpoint_degrades_when_chroma_is_unavailable(client):
    FakeFundamentalRetriever.probe_values = []

    response = client.get("/api/rag/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "fundamental_chroma"
    assert payload["scope"] == "fundamental_only"
    assert "available" in payload
    assert "document_count" in payload
    assert "OPENAI_API_KEY or DASHSCOPE_API_KEY is not set" in payload["error"]
    assert FakeFundamentalRetriever.probe_values == [True]


def test_observability_status_endpoint_reports_otel_configuration(client, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-copilot")
    monkeypatch.setenv("JAEGER_UI_URL", "http://127.0.0.1:16686")

    response = client.get("/api/observability/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["local_trace_enabled"] is True
    assert payload["otel_configured"] is True
    assert payload["otel_export_enabled"] is True
    assert payload["otlp_endpoint"] == "http://127.0.0.1:4317"
    assert payload["service_name"] == "test-copilot"
    assert payload["jaeger_ui_url"] == "http://127.0.0.1:16686"


def test_rag_status_endpoint_reports_missing_embedding_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(
        "ai_trading_copilot.copilot.ui.app.load_copilot_env",
        lambda: None,
    )
    app = create_app(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            run_in_background=False,
        )
    )
    client = TestClient(app)

    response = client.get("/api/rag/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "OPENAI_API_KEY or DASHSCOPE_API_KEY is not set" in payload["error"]


def test_rag_status_probe_uses_utf8_env_and_longer_timeout(tmp_path, monkeypatch):
    captured = {}

    class Completed:
        returncode = 0
        stdout = '{"available": true, "document_count": 7}\n'
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Completed()

    monkeypatch.setattr(ui_app.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONUTF8", "0")

    payload = ui_app._probe_rag_status_subprocess(
        UISettings(
            subscriptions_file=tmp_path / "config" / "subscriptions.json",
            memory_database=tmp_path / "config" / "memory.sqlite3",
            reports_dir=tmp_path / "reports",
            rag_chroma_dir=tmp_path / "config" / "rag_chroma",
        )
    )

    assert payload["available"] is True
    assert payload["document_count"] == 7
    assert captured["timeout"] == 30.0
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "strict"


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

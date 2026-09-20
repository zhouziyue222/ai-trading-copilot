from types import SimpleNamespace
import json
from threading import Event
import pytest

from fastapi.testclient import TestClient

from ai_trading_copilot.copilot import review
from ai_trading_copilot.copilot.ui import app as ui


class Repository:
    def __init__(self):
        self.association = None
        self.item = {"task": {"id": "r1", "run_id": "run", "symbol": "AAPL", "horizon_days": 5,
                              "status": "waiting_data", "reason": "missing bars", "next_check_at": None},
                     "snapshot": {}, "result": None}

    def list_reviews(self):
        return [self.item["task"]]

    def detail(self, review_id):
        return self.item if review_id == "r1" else None

    def retry(self, review_id):
        return review_id == "r1"

    def list_fills(self):
        return []

    def associate_fill(self, *args):
        if args[2] == "missing":
            raise KeyError("Fill not found")
        self.association = args


def service():
    return SimpleNamespace(repository=Repository(), run_due=lambda **kwargs: {"completed": 0}, register_run=lambda state: 3)


def test_review_api_and_validation(tmp_path, monkeypatch):
    instance = service()
    monkeypatch.setattr(ui, "create_review_service", lambda path: instance)
    client = TestClient(ui.create_app(ui.UISettings(reports_dir=tmp_path / "reports", memory_database=tmp_path / "m.db")))
    assert client.get("/reviews").status_code == 200
    assert client.get("/api/reviews").json()["items"][0]["id"] == "r1"
    assert client.get("/api/reviews/r1").json()["task"]["reason"] == "missing bars"
    assert client.get("/api/reviews/absent").status_code == 404
    assert client.post("/api/reviews/absent/retry").status_code == 404
    assert client.post("/api/reviews/r1/retry").json() == {"retried": True}
    assert client.post("/api/reviews/run-due").json() == {"completed": 0}
    assert client.get("/api/review-fills").json() == {"items": []}
    body = dict(environment="SIMULATE", account_id="123", deal_id="d1", run_id="run", symbol="AAPL")
    assert client.post("/api/review-fills/associate", json=body).status_code == 200
    assert instance.repository.association == tuple(body.values())
    assert client.post("/api/review-fills/associate", json={**body, "run_id": " "}).status_code == 422
    assert client.post("/api/review-fills/associate", json={**body, "deal_id": "missing"}).status_code == 404


def test_startup_catchup_does_not_block_lifespan(tmp_path, monkeypatch):
    started, release = Event(), Event()
    instance = service()
    def run_due(**kwargs):
        started.set()
        release.wait(5)
    instance.run_due = run_due
    monkeypatch.setattr(ui, "create_review_service", lambda path: instance)
    try:
        with TestClient(ui.create_app(ui.UISettings(reports_dir=tmp_path / "reports"))) as client:
            assert started.wait(2)
            assert client.get("/reviews").status_code == 200
        assert not release.is_set()
    finally:
        release.set()


def test_cli_dispatch_and_failures(tmp_path, monkeypatch, capsys):
    instance = service()
    monkeypatch.setattr(review, "create_review_service", lambda path: instance)
    for command in (["list"], ["show", "r1"], ["retry", "r1"], ["fills"], ["run-due"]):
        assert review.main(command) == 0
    assert review.main(["show", "missing"]) == 1
    assert "Review not found" in capsys.readouterr().err
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("[]")
    assert review.main(["import-snapshot", str(snapshot)]) == 1
    snapshot.write_text('{"run_id":"run"}')
    assert review.main(["import-snapshot", str(snapshot)]) == 1
    snapshot.write_text('{"run_id":"run", "generated_at":"2026-01-01T00:00:00+00:00", "trade_plans":[], "risk_assessments":{}, "execution_decisions":{}, "price_history_by_symbol":{"AAPL":[]}}')
    assert review.main(["import-snapshot", str(snapshot)]) == 1
    state = json.loads(snapshot.read_text())
    state["trade_plans"] = [dict(symbol="AAPL", direction="buy", subscription_status="actionable", market_regime="uptrend",
                                 entry_logic="support confirmed", holding_period="20 days", persona_fit_reason="fits")]
    snapshot.write_text(json.dumps(state))
    assert review.main(["import-snapshot", str(snapshot)]) == 0


@pytest.mark.parametrize("fail_registration,learning_enabled", [(False, True), (True, True), (False, False)])
def test_cli_persists_before_learning_even_with_learning_disabled(tmp_path, monkeypatch, fail_registration, learning_enabled):
    from ai_trading_copilot.copilot import run
    from ai_trading_copilot.copilot.services import delayed_review
    events, errors = [], []

    class Tracker:
        def __init__(self, **kwargs):
            pass
        def add_error(self, value):
            errors.append(value)
        def finish(self, **kwargs):
            pass
        def write_audit(self, **kwargs):
            return str(tmp_path / "audit.md")

    class Graph:
        NODE_ORDER = []
        def __init__(self, **kwargs):
            pass
        def run(self, **kwargs):
            return dict(run_id=kwargs['run_id'], trade_plans={'AAPL': {'symbol':'AAPL'}}, agent_reports={})

    def register(state):
        events.append('snapshot')
        assert state['memory_learning_enabled'] is learning_enabled
        if fail_registration:
            raise OSError('sqlite unavailable')
        return 3

    def learn(state):
        events.append('learning')
        return []

    monkeypatch.setattr(run, 'RunTracker', Tracker)
    monkeypatch.setattr(run, 'CopilotLangGraph', Graph)
    monkeypatch.setattr(run, '_enable_fundamental_rag', lambda: False)
    monkeypatch.setattr(run, 'create_default_memory_agent', lambda: SimpleNamespace(learn_from_run=learn))
    monkeypatch.setattr(delayed_review, 'create_review_service', lambda path: SimpleNamespace(register_run=register))
    flags = ['--long-term-memory'] if learning_enabled else ['--no-long-term-memory']
    assert run.main(['AAPL', '--output-dir', str(tmp_path), *flags]) == 0
    assert events == (['snapshot', 'learning'] if learning_enabled and not fail_registration else ['snapshot'])
    assert bool(errors) is fail_registration


@pytest.mark.parametrize("fail_registration,learning_enabled", [(False, True), (True, True), (False, False)])
def test_ui_persists_before_learning_even_with_learning_disabled(tmp_path, monkeypatch, fail_registration, learning_enabled):
    events, errors, final_errors = [], [], []
    tracker = SimpleNamespace(status={'status':'succeeded'}, add_error=errors.append)
    control = SimpleNamespace(tracker=tracker, cancel_event=Event(), params={'run_id':'ui-run'}, resume_snapshot=None, reason=None)
    runtime = SimpleNamespace(finalize=lambda control, **kwargs: final_errors.append(kwargs['error']), unregister=lambda control: None)

    class Graph:
        def __init__(self, **kwargs):
            pass
        def run(self, **kwargs):
            return dict(run_id=kwargs['run_id'], trade_plans={'AAPL':{'symbol':'AAPL'}})

    def register(state):
        events.append('snapshot')
        assert state['memory_learning_enabled'] is learning_enabled
        if fail_registration:
            raise OSError('sqlite unavailable')
        return 3

    def learn(state):
        events.append('learning')
        return []

    monkeypatch.setattr(ui, '_ui_enable_fundamental_rag', lambda: False)
    monkeypatch.setattr(ui, 'create_default_memory_agent', lambda path: SimpleNamespace(learn_from_run=learn))
    monkeypatch.setattr(ui, 'create_review_service', lambda path: SimpleNamespace(register_run=register))
    ui._run_copilot_job(Graph, control, runtime, tmp_path / 'memory.db', long_term_memory_enabled=learning_enabled)
    assert events == (['snapshot', 'learning'] if learning_enabled and not fail_registration else ['snapshot'])
    assert final_errors == [None]
    assert bool(errors) is fail_registration

from __future__ import annotations

from fastapi.testclient import TestClient

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryStatus,
    MemoryType,
)
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore
from ai_trading_copilot.copilot.ui.app import UISettings, create_app


def test_memory_review_api_stages_and_evaluates_candidate(tmp_path):
    memory_database = tmp_path / "memory.sqlite3"
    store = DistilledMemoryStore(memory_database)
    candidate = store.save_candidate(
        DistilledMemory(
            memory_type=MemoryType.STRATEGY_PERFORMANCE,
            lesson="Wait for support confirmation.",
            symbols=["AAPL"],
            confidence=0.8,
            metadata={"outcome_verified": True},
        )
    )
    client = TestClient(
        create_app(
            UISettings(
                subscriptions_file=tmp_path / "subscriptions.json",
                memory_database=memory_database,
                reports_dir=tmp_path / "reports",
                run_in_background=False,
            )
        )
    )

    listed = client.get("/api/memories?status=candidate")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["memory_id"] == candidate.memory_id

    staged = client.post(
        f"/api/memories/{candidate.memory_id}/transition",
        json={"status": "shadow", "actor": "tester", "reason": "review"},
    )
    assert staged.status_code == 200
    assert staged.json()["item"]["status"] == MemoryStatus.SHADOW.value

    evaluated = client.post(f"/api/memories/{candidate.memory_id}/evaluate")
    assert evaluated.status_code == 200
    assert evaluated.json()["evaluation"]["eligible"] is False


def test_memory_outcome_api_is_idempotent(tmp_path):
    client = TestClient(
        create_app(
            UISettings(
                subscriptions_file=tmp_path / "subscriptions.json",
                memory_database=tmp_path / "memory.sqlite3",
                reports_dir=tmp_path / "reports",
                run_in_background=False,
            )
        )
    )
    payload = {
        "run_id": "run-1",
        "symbol": "aapl",
        "horizon_days": 5,
        "realized_return": 0.03,
        "benchmark_return": 0.01,
        "max_drawdown": -0.04,
    }

    first = client.post("/api/memory-outcomes", json=payload)
    second = client.post("/api/memory-outcomes", json={**payload, "realized_return": 0.04})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["outcome"]["symbol"] == "AAPL"
    assert second.json()["outcome"]["realized_return"] == 0.04


def test_memory_jsonl_export_endpoint_is_removed(tmp_path):
    client = TestClient(
        create_app(
            UISettings(
                subscriptions_file=tmp_path / "subscriptions.json",
                memory_database=tmp_path / "memory.sqlite3",
                reports_dir=tmp_path / "reports",
                run_in_background=False,
            )
        )
    )

    assert client.post("/api/memories/export").status_code == 405

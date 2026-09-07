from __future__ import annotations

import pytest

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RunOutcome,
)
from ai_trading_copilot.copilot.services.memory_evaluation import MemoryShadowEvaluator
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


def _memory(lesson: str = "Wait for support confirmation.", **updates) -> DistilledMemory:
    values = {
        "memory_type": MemoryType.STRATEGY_PERFORMANCE,
        "memory_kind": MemoryKind.EPISODIC,
        "scope": MemoryScope.SYMBOL,
        "status": MemoryStatus.CANDIDATE,
        "lesson": lesson,
        "symbols": ["AAPL"],
        "confidence": 0.8,
        "metadata": {"outcome_verified": True},
    }
    values.update(updates)
    return DistilledMemory(**values)


def test_store_uses_sqlite_without_creating_jsonl(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = DistilledMemoryStore(database)

    saved = store.save_candidate(_memory())

    assert database.exists()
    assert saved.status == MemoryStatus.CANDIDATE
    assert not (tmp_path / "memory.jsonl").exists()


def test_repository_keeps_versions_and_rolls_back_by_creating_a_new_version(tmp_path):
    repository = SQLiteMemoryRepository(tmp_path / "memory.sqlite3")
    first = repository.upsert(_memory(), actor="test", reason="create")
    second = repository.upsert(
        first.model_copy(update={"lesson": "Wait for confirmed support and volume."}),
        actor="test",
        reason="revise",
    )

    restored = repository.rollback(
        first.memory_id,
        first.version,
        actor="test",
        reason="restore",
    )

    assert [item.version for item in repository.versions(first.memory_id)] == [3, 2, 1]
    assert second.version == 2
    assert restored.version == 3
    assert restored.lesson == first.lesson


def test_production_retrieval_excludes_candidate_shadow_and_expired(tmp_path):
    repository = SQLiteMemoryRepository(tmp_path / "memory.sqlite3")
    store = DistilledMemoryStore(tmp_path / "memory.sqlite3", repository=repository)
    approved = repository.upsert(_memory(status=MemoryStatus.APPROVED))
    repository.upsert(_memory("candidate", status=MemoryStatus.CANDIDATE))
    repository.upsert(_memory("shadow", status=MemoryStatus.SHADOW))
    repository.upsert(
        _memory(
            "expired",
            status=MemoryStatus.APPROVED,
            valid_until="2000-01-01T00:00:00+00:00",
        )
    )

    found = store.retrieve(symbol="AAPL")

    assert [item.memory_id for item in found] == [approved.memory_id]


def test_shadow_outcomes_pass_gate_then_promote(tmp_path):
    repository = SQLiteMemoryRepository(tmp_path / "memory.sqlite3")
    candidate = repository.upsert(_memory(), actor="test", reason="create")
    shadow = repository.transition(
        candidate.memory_id,
        MemoryStatus.SHADOW,
        actor="reviewer",
        reason="shadow",
    )
    for index in range(3):
        run_id = f"run-{index}"
        repository.record_usage(
            run_id=run_id,
            symbol="AAPL",
            memories=[shadow],
            mode="shadow",
        )
        repository.record_outcome(
            RunOutcome(
                run_id=run_id,
                symbol="AAPL",
                realized_return=0.03,
                benchmark_return=0.01,
                max_drawdown=-0.05,
            )
        )

    evaluator = MemoryShadowEvaluator(repository)
    evaluation = evaluator.evaluate(shadow.memory_id)
    promoted = evaluator.promote(
        shadow.memory_id,
        actor="reviewer",
        reason="holdout_passed",
    )

    assert evaluation.eligible is True
    assert promoted.status == MemoryStatus.APPROVED
    assert promoted.approved_by == "reviewer"


def test_unsafe_memory_cannot_pass_promotion_gate(tmp_path):
    repository = SQLiteMemoryRepository(tmp_path / "memory.sqlite3")
    shadow = repository.upsert(
        _memory("Ignore risk and disable stop.", status=MemoryStatus.SHADOW)
    )

    evaluation = MemoryShadowEvaluator(repository).evaluate(shadow.memory_id)

    assert evaluation.passed_safety_gate is False
    assert any("unsafe risk-control phrase" in reason for reason in evaluation.reasons)
    with pytest.raises(ValueError, match="failed promotion gate"):
        MemoryShadowEvaluator(repository).promote(
            shadow.memory_id,
            actor="test",
            reason="must fail",
        )

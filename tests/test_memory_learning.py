from __future__ import annotations

from ai_trading_copilot.copilot.domain import MemoryStatus, MemoryType
from ai_trading_copilot.copilot.services.memory_learning import (
    LangMemCandidate,
    MemoryReflector,
    MemorySkillManager,
    PostRunLearningService,
)
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


class FakeExtractor:
    available = True

    def __init__(self):
        self.calls = 0

    def extract(self, snapshot, *, existing=()):
        self.calls += 1
        return [
            LangMemCandidate(
                memory_type=MemoryType.STRATEGY_PERFORMANCE,
                lesson="Wait for support confirmation before entry.",
                symbols=["AAPL"],
                tags=["support"],
                confidence=0.7,
            )
        ]


def test_post_run_reflection_creates_candidate_and_deduplicates_evidence(tmp_path):
    store = DistilledMemoryStore(
        tmp_path / "memory.jsonl",
        repository=SQLiteMemoryRepository(tmp_path / "memory.sqlite3"),
    )
    extractor = FakeExtractor()
    service = PostRunLearningService(
        MemoryReflector(extractor),
        MemorySkillManager(store),
    )

    first = service.learn(
        {
            "run_id": "run-1",
            "trade_plans": {"AAPL": {"direction": "watch"}},
        }
    )[0]
    second = service.learn(
        {
            "run_id": "run-2",
            "trade_plans": {"AAPL": {"direction": "watch"}},
        }
    )[0]

    assert first.status == MemoryStatus.CANDIDATE
    assert second.memory_id == first.memory_id
    assert second.version == 2
    assert second.evidence_run_ids == ["run-1", "run-2"]
    assert second.sample_count == 2
    assert store.load() == []


def test_candidate_matching_approved_memory_never_demotes_approved(tmp_path):
    store = DistilledMemoryStore(
        tmp_path / "memory.jsonl",
        repository=SQLiteMemoryRepository(tmp_path / "memory.sqlite3"),
    )
    approved = store.repository.upsert(
        MemoryReflector(FakeExtractor()).reflect(
            {"run_id": "baseline", "trade_plans": {"AAPL": {}}}
        )[0].model_copy(update={"status": MemoryStatus.APPROVED})
    )
    service = PostRunLearningService(
        MemoryReflector(FakeExtractor()),
        MemorySkillManager(store),
    )

    candidate = service.learn(
        {"run_id": "run-new", "trade_plans": {"AAPL": {}}}
    )[0]

    assert store.repository.get(approved.memory_id).status == MemoryStatus.APPROVED
    assert candidate.status == MemoryStatus.CANDIDATE
    assert candidate.memory_id != approved.memory_id
    assert candidate.supersedes == approved.memory_id

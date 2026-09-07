from __future__ import annotations

import pytest

from ai_trading_copilot.copilot.domain import DistilledMemory, MemoryStatus, MemoryType
from ai_trading_copilot.copilot.services.memory_learning import (
    LangMemCandidate,
    LangMemCandidateExtractor,
    LangMemProposal,
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
            LangMemProposal(
                operation="insert",
                candidate=LangMemCandidate(
                    memory_type=MemoryType.STRATEGY_PERFORMANCE,
                    lesson="Wait for support confirmation before entry.",
                    symbols=["AAPL"],
                    tags=["support"],
                    confidence=0.7,
                ),
            )
        ]


def test_post_run_reflection_creates_candidate_and_deduplicates_evidence(tmp_path):
    store = DistilledMemoryStore(
        tmp_path / "memory.sqlite3",
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

    repeated = service.learn(
        {"run_id": "run-2", "trade_plans": {"AAPL": {"direction": "watch"}}}
    )[0]
    assert repeated.version == second.version
    assert repeated.evidence_run_ids == second.evidence_run_ids


def test_candidate_matching_approved_memory_never_demotes_approved(tmp_path):
    store = DistilledMemoryStore(
        tmp_path / "memory.sqlite3",
        repository=SQLiteMemoryRepository(tmp_path / "memory.sqlite3"),
    )
    approved = store.repository.upsert(
        MemoryReflector(FakeExtractor()).reflect(
            {"run_id": "baseline", "trade_plans": {"AAPL": {}}}
        )[0].memory.model_copy(update={"status": MemoryStatus.APPROVED})
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


def test_langmem_adapter_passes_existing_and_classifies_insert_and_update(monkeypatch):
    calls = {}

    class Extracted:
        def __init__(self, memory_id, content):
            self.id = memory_id
            self.content = content

    class Manager:
        def invoke(self, payload):
            calls["payload"] = payload
            existing = payload["existing"][0][1]
            return [
                Extracted(
                    "existing-1",
                    existing.model_copy(update={"lesson": "Updated support lesson."}),
                ),
                Extracted(
                    "langmem-generated-id",
                    LangMemCandidate(
                        memory_type=MemoryType.STRATEGY_PERFORMANCE,
                        lesson="New compact lesson.",
                        symbols=["AAPL"],
                    ),
                ),
            ]

    def create_manager(llm, **kwargs):
        calls["manager_kwargs"] = kwargs
        return Manager()

    monkeypatch.setattr("langmem.create_memory_manager", create_manager)
    extractor = LangMemCandidateExtractor(object())
    existing = DistilledMemory(
        memory_id="existing-1",
        memory_type=MemoryType.STRATEGY_PERFORMANCE,
        lesson="Original support lesson.",
        symbols=["AAPL"],
    )

    proposals = extractor.extract({"run_id": "run-1"}, existing=[existing])

    assert calls["manager_kwargs"]["enable_updates"] is True
    assert calls["manager_kwargs"]["enable_deletes"] is False
    assert calls["payload"]["existing"][0][0] == "existing-1"
    assert [item.operation for item in proposals] == ["update", "insert"]
    assert proposals[0].target_memory_id == "existing-1"
    assert proposals[1].target_memory_id is None


def test_langmem_adapter_without_llm_is_unavailable():
    extractor = LangMemCandidateExtractor(None)

    assert extractor.available is False
    assert extractor.extract({"run_id": "run-1"}) == []


def test_langmem_update_changes_candidate_in_place(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
    original = store.save_candidate(
        DistilledMemory(
            memory_type=MemoryType.STRATEGY_PERFORMANCE,
            lesson="Original support lesson.",
            symbols=["AAPL"],
            source_run_id="run-1",
            evidence_run_ids=["run-1"],
        )
    )
    reflected = MemoryReflector(FakeExtractor()).reflect(
        {"run_id": "run-2", "trade_plans": {"AAPL": {}}}
    )[0]
    reflected = reflected.model_copy(
        update={
            "operation": "update",
            "target_memory_id": original.memory_id,
            "memory": reflected.memory.model_copy(
                update={"lesson": "Updated support lesson."}
            ),
        }
    )

    updated = MemorySkillManager(store).ingest([reflected])[0]

    assert updated.memory_id == original.memory_id
    assert updated.version == 2
    assert updated.lesson == "Updated support lesson."
    assert updated.evidence_run_ids == ["run-1", "run-2"]


@pytest.mark.parametrize(
    "status",
    [
        MemoryStatus.SHADOW,
        MemoryStatus.APPROVED,
        MemoryStatus.REJECTED,
        MemoryStatus.DEPRECATED,
    ],
)
def test_langmem_update_never_mutates_frozen_memory(tmp_path, status):
    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
    frozen = store.repository.upsert(
        DistilledMemory(
            memory_id="frozen-memory",
            memory_type=MemoryType.STRATEGY_PERFORMANCE,
            status=status,
            lesson="Original frozen lesson.",
            symbols=["AAPL"],
        ),
        actor="test",
        reason="frozen_fixture",
    )
    reflected = MemoryReflector(FakeExtractor()).reflect(
        {"run_id": "run-2", "trade_plans": {"AAPL": {}}}
    )[0]
    proposal = reflected.model_copy(
        update={
            "operation": "update",
            "target_memory_id": frozen.memory_id,
            "memory": reflected.memory.model_copy(
                update={"lesson": "Proposed revised frozen lesson."}
            ),
        }
    )

    revision = MemorySkillManager(store).ingest([proposal])[0]

    unchanged = store.repository.get(frozen.memory_id)
    assert unchanged.status == status
    assert unchanged.version == frozen.version
    assert unchanged.lesson == frozen.lesson
    assert revision.status == MemoryStatus.CANDIDATE
    assert revision.memory_id != frozen.memory_id
    assert revision.supersedes == frozen.memory_id

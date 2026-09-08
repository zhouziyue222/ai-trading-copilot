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


@pytest.mark.parametrize("status", [s for s in MemoryStatus if s != MemoryStatus.CANDIDATE])
@pytest.mark.parametrize("operation", ["insert", "update"])
def test_frozen_revision_chain_and_replay(tmp_path, status, operation):
    from ai_trading_copilot.copilot.services.memory_learning import ReflectedMemoryProposal

    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
    frozen = store.repository.upsert(DistilledMemory(
        memory_id="original", memory_type=MemoryType.STRATEGY_PERFORMANCE,
        status=status, lesson="Wait for support confirmation before entry.", symbols=["AAPL"],
    ))
    manager = MemorySkillManager(store)

    def proposal(run):
        return ReflectedMemoryProposal(
            operation=operation,
            target_memory_id=frozen.memory_id if operation == "update" else None,
            memory=frozen.model_copy(update={
                "memory_id": "", "source_run_id": run, "evidence_run_ids": [run],
            }),
        )

    first = manager.ingest([proposal("run-1")])[0]
    assert first.supersedes == frozen.memory_id
    assert first.status == MemoryStatus.CANDIDATE
    first_frozen = store.repository.upsert(first.model_copy(update={"status": status}))
    # Explicitly targeting the original exercises traversal through an already
    # frozen revision, while insert exercises similarity selection.
    second = manager.ingest([proposal("run-2")])[0]
    assert second.supersedes == first.memory_id
    assert second.memory_id not in {frozen.memory_id, first.memory_id}
    assert second.status == MemoryStatus.CANDIDATE
    assert second.evidence_run_ids == ["run-2"]
    assert manager.ingest([proposal("run-2")])[0] == second
    assert store.repository.get(frozen.memory_id) == frozen
    assert store.repository.get(first.memory_id) == first_frozen


def test_same_run_correction_changes_content_but_replay_does_not_reweight(tmp_path):
    from ai_trading_copilot.copilot.services.memory_learning import ReflectedMemoryProposal

    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
    first = store.save_candidate(DistilledMemory(
        memory_type=MemoryType.STRATEGY_PERFORMANCE, lesson="Original support lesson.",
        symbols=["AAPL"], source_run_id="run-1", confidence=0.6,
    ))
    manager = MemorySkillManager(store)
    correction = ReflectedMemoryProposal(
        operation="update", target_memory_id=first.memory_id,
        memory=first.model_copy(update={"lesson": "Corrected support lesson.", "confidence": 0.9}),
    )
    corrected = manager.ingest([correction])[0]
    assert corrected.version == first.version + 1
    assert corrected.lesson == "Corrected support lesson."
    assert corrected.sample_count == 1
    assert corrected.confidence == first.confidence
    later = correction.model_copy(update={"memory": correction.memory.model_copy(update={
        "lesson": "Later independent evidence.", "source_run_id": "run-2",
        "evidence_run_ids": ["run-2"],
    })})
    updated = manager.ingest([later])[0]
    assert updated.confidence == pytest.approx(0.75)
    assert manager.ingest([correction])[0] == updated


def test_approval_between_matching_and_write_creates_revision(tmp_path, monkeypatch):
    from ai_trading_copilot.copilot.services.memory_learning import ReflectedMemoryProposal

    store = DistilledMemoryStore(tmp_path / "memory.sqlite3")
    original = store.save_candidate(DistilledMemory(
        memory_type=MemoryType.STRATEGY_PERFORMANCE, lesson="Original support lesson.",
        symbols=["AAPL"], source_run_id="run-1",
    ))
    other_connection = SQLiteMemoryRepository(store.database_path)
    save = store.repository.save_candidate
    frozen = []

    def approve_then_save(*args, **kwargs):
        frozen.append(other_connection.transition(
            original.memory_id, MemoryStatus.SHADOW, actor="reviewer", reason="concurrent review",
        ))
        return save(*args, **kwargs)

    monkeypatch.setattr(store.repository, "save_candidate", approve_then_save)
    result = MemorySkillManager(store).ingest([ReflectedMemoryProposal(
        operation="update", target_memory_id=original.memory_id,
        memory=original.model_copy(update={"lesson": "New lesson.", "source_run_id": "run-2"}),
    )])[0]
    assert store.repository.get(original.memory_id) == frozen[0]
    assert result.supersedes == original.memory_id
    assert result.status == MemoryStatus.CANDIDATE

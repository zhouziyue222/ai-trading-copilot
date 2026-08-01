import pytest
from pydantic import ValidationError

from ai_trading_copilot.copilot.domain import DistilledMemory, MemoryType
from ai_trading_copilot.copilot.services import DistilledMemoryStore, LocalVectorMemoryIndex


class SemanticTestEmbedder:
    dimensions = 2
    name = "semantic_test_embedder"

    def embed(self, text: str):
        lower = text.lower()
        if any(term in lower for term in ["dip", "pullback", "support", "retest"]):
            return [1.0, 0.0]
        if any(term in lower for term in ["earnings", "guidance", "event"]):
            return [0.0, 1.0]
        return [0.0, 0.0]


def _memory(lesson: str, symbol="AAPL", tags=None):
    return DistilledMemory(
        memory_type=MemoryType.STRATEGY_PERFORMANCE,
        lesson=lesson,
        symbols=[symbol],
        tags=tags or ["pullback"],
    )


def test_memory_store_loads_empty_when_file_missing(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")

    assert store.load() == []


def test_memory_store_appends_jsonl_entries(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")

    store.append(_memory("20-day average pullback worked better than chasing."))
    store.append(_memory("Avoid entries far above support.", symbol="SPY"))

    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].symbols == ["AAPL"]
    assert loaded[1].symbols == ["SPY"]


def test_memory_retrieval_prioritizes_symbol_matches(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    store.append(_memory("AAPL lesson.", symbol="AAPL"))
    store.append(_memory("SPY lesson.", symbol="SPY"))

    retrieved = store.retrieve(symbol="spy")

    assert retrieved[0].lesson == "SPY lesson."


def test_memory_retrieval_uses_tags_as_secondary_signal(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    store.append(_memory("Generic pullback lesson.", symbol="AAPL", tags=["pullback"]))
    store.append(_memory("False breakout lesson.", symbol="MSFT", tags=["breakout"]))

    retrieved = store.retrieve(tags=["breakout"])

    assert retrieved[0].lesson == "False breakout lesson."


def test_memory_retrieval_uses_query_terms(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    store.append(_memory("Support held after earnings gap.", symbol="AAPL"))
    store.append(_memory("Avoid chasing extended breakouts.", symbol="MSFT"))

    retrieved = store.retrieve(query="chasing breakout")

    assert retrieved[0].lesson == "Avoid chasing extended breakouts."


def test_memory_retrieval_prioritizes_tags_over_query_terms(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    store.append(_memory("Generic setup with breakout word.", symbol="AAPL", tags=["pullback"]))
    store.append(_memory("Tagged risk memory.", symbol="MSFT", tags=["breakout"]))

    retrieved = store.retrieve(tags=["breakout"], query="breakout")

    assert retrieved[0].lesson == "Tagged risk memory."


def test_memory_retrieval_caps_context_to_default_limit(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl", default_limit=5)
    for i in range(7):
        store.append(_memory(f"Lesson {i}.", symbol="AAPL"))

    retrieved = store.retrieve(symbol="AAPL", limit=7)

    assert len(retrieved) == 5


def test_memory_rejects_full_report_sized_lessons():
    with pytest.raises(ValidationError):
        DistilledMemory(
            memory_type=MemoryType.USER_BEHAVIOR,
            lesson="x" * 501,
            symbols=["AAPL"],
            tags=["behavior"],
        )


def test_memory_loads_optional_rag_metadata(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    store.append(
        DistilledMemory(
            memory_type=MemoryType.USER_BEHAVIOR,
            lesson="User prefers waiting for support confirmation.",
            symbols=["aapl"],
            tags=["behavior"],
            source_run_id="run_AAPL_20260514_090000",
            source_path="reports/run_AAPL_20260514_090000/run_audit.md",
            created_at="2026-05-14T09:00:00+08:00",
            confidence=0.8,
        )
    )

    loaded = store.load()[0]

    assert loaded.symbols == ["AAPL"]
    assert loaded.source_run_id == "run_AAPL_20260514_090000"
    assert loaded.source_path.endswith("run_audit.md")
    assert loaded.created_at == "2026-05-14T09:00:00+08:00"
    assert loaded.confidence == 0.8


def test_vector_memory_index_persists_sidecar_on_append(tmp_path):
    vector_path = tmp_path / "memory.vectors.json"
    store = DistilledMemoryStore(
        tmp_path / "memory.jsonl",
        vector_index=LocalVectorMemoryIndex(vector_path),
    )

    store.append(_memory("Wait for support confirmation before adding size."))

    assert vector_path.exists()
    assert "local_hashing_text_embedder" in vector_path.read_text(encoding="utf-8")


def test_vector_memory_retrieval_uses_injected_embedding_similarity(tmp_path):
    store = DistilledMemoryStore(
        tmp_path / "memory.jsonl",
        vector_index=LocalVectorMemoryIndex(
            tmp_path / "memory.vectors.json",
            embedder=SemanticTestEmbedder(),
        ),
    )
    store.append(_memory("Confirm a support retest before adding size.", symbol="AAPL"))
    store.append(
        _memory(
            "Avoid initiating positions before earnings.",
            symbol="MSFT",
            tags=["event"],
        )
    )

    retrieved = store.retrieve(query="buy the dip after confirmation")

    assert retrieved[0].lesson == "Confirm a support retest before adding size."

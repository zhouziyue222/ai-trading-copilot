import json

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from ai_trading_copilot.copilot.services.memory_retrieval import MemoryRetrievalSession
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


def _memory(memory_id: str, lesson: str, **updates) -> DistilledMemory:
    values = {
        "memory_id": memory_id,
        "memory_type": MemoryType.STRATEGY_PERFORMANCE,
        "memory_kind": MemoryKind.PROCEDURAL,
        "scope": MemoryScope.SYMBOL,
        "status": MemoryStatus.APPROVED,
        "lesson": lesson,
        "trigger": "AAPL is in an uptrend pullback near support.",
        "symbols": ["AAPL"],
        "tags": ["pullback", "trade_plan"],
        "market_regimes": ["uptrend"],
        "confidence": 0.8,
    }
    values.update(updates)
    return DistilledMemory(**values)


def test_retrieval_session_returns_approved_and_hides_paired_shadow(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    approved = store.repository.upsert(_memory("approved-aapl", "Wait for support confirmation."))
    shadow = store.repository.upsert(
        _memory(
            "shadow-aapl",
            "Shadow-only lesson must not enter prompts.",
            status=MemoryStatus.SHADOW,
        )
    )
    session = MemoryRetrievalSession(
        store,
        symbol="AAPL",
        consumer="trader",
        run_id="run-1",
        tags=["pullback"],
        market_regime="uptrend",
    )

    payload = json.loads(session.prefetch("AAPL uptrend pullback support"))

    assert payload["memories"][0]["id"] == f"{approved.memory_id}@{approved.version}"
    assert shadow.memory_id not in json.dumps(payload)
    assert session.records[0].shadow_ids == [f"{shadow.memory_id}@{shadow.version}"]
    assert store.repository.usage_for_memory(approved.memory_id, mode="trader")
    assert store.repository.usage_for_memory(shadow.memory_id, mode="shadow")


def test_retrieval_session_caps_optional_tool_calls_and_validates_citations(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    approved = store.repository.upsert(_memory("approved-aapl", "Wait for support confirmation."))
    session = MemoryRetrievalSession(
        store,
        symbol="AAPL",
        consumer="trader",
        run_id="run-2",
        market_regime="uptrend",
    )
    session.prefetch("AAPL support")
    tool = session.make_tools()[0]

    assert "symbol" not in tool.args
    tool.invoke({"query": "AAPL position sizing", "limit": 3})
    second = json.loads(tool.invoke({"query": "AAPL another query", "limit": 3}))
    valid = session.mark_cited([f"{approved.memory_id}@{approved.version}", "fake@1"])

    assert second["error"] == "optional memory search limit reached"
    assert valid == [f"{approved.memory_id}@{approved.version}"]
    assert store.repository.usage_for_memory(approved.memory_id, mode="trader_applied")


def test_retrieval_session_caps_combined_prompt_context_to_six(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    for index in range(3):
        store.repository.upsert(
            _memory(f"alpha-{index}", f"Alpha setup lesson {index}.")
        )
    for index in range(3):
        store.repository.upsert(
            _memory(f"beta-{index}", f"Beta setup lesson {index}.")
        )
    store.repository.upsert(_memory("gamma", "Gamma setup lesson."))
    session = MemoryRetrievalSession(
        store,
        symbol="AAPL",
        consumer="trader",
        market_regime="uptrend",
    )

    session.prefetch("alpha")
    session.make_tools()[0].invoke({"query": "beta", "limit": 3})

    assert len(session.approved_memories) == 6
    assert {item.memory_id for item in session.approved_memories} == {
        "alpha-0",
        "alpha-1",
        "alpha-2",
        "beta-0",
        "beta-1",
        "beta-2",
    }


def test_scored_retrieval_enforces_scope_and_minimum_relevance(tmp_path):
    store = DistilledMemoryStore(tmp_path / "memory.jsonl")
    matching = store.repository.upsert(_memory("matching", "AAPL support retest."))
    store.repository.upsert(
        _memory(
            "wrong-symbol",
            "MSFT support retest.",
            symbols=["MSFT"],
        )
    )
    store.repository.upsert(
        _memory(
            "wrong-regime",
            "Only use this in a downtrend.",
            scope=MemoryScope.MARKET_REGIME,
            symbols=[],
            market_regimes=["downtrend"],
        )
    )
    store.repository.upsert(
        _memory(
            "irrelevant-global",
            "Unrelated observation.",
            scope=MemoryScope.GLOBAL,
            symbols=[],
            tags=["unrelated"],
            market_regimes=[],
            trigger="",
        )
    )

    hits = store.retrieve_scored(
        symbol="AAPL",
        query="support retest",
        market_regime="uptrend",
        limit=3,
    )

    assert [hit.memory.memory_id for hit in hits] == [matching.memory_id]
    assert hits[0].matched_by[:2] == ["symbol", "market_regime"]

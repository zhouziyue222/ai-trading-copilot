import json
from pathlib import Path

from ai_trading_copilot.copilot.config import (
    DEFAULT_PERSONA_CONFIG,
    DEFAULT_PRODUCT_CONFIG,
)
from ai_trading_copilot.copilot.domain import UserPersonaConfig


ROOT = Path(__file__).resolve().parents[1]


def test_persona_default_json_loads_into_persona_model():
    payload = json.loads((ROOT / "config" / "persona.default.json").read_text())
    persona = UserPersonaConfig(**payload)

    assert persona.max_portfolio_drawdown == 0.25
    assert persona.stock_source == "user_subscription_list"


def test_product_default_json_matches_memory_limit():
    payload = json.loads((ROOT / "config" / "product.default.json").read_text())

    assert payload["learning_loop"]["memory_strategy"] == "distilled_retrieval_memory"
    assert (
        payload["learning_loop"]["retrieval_backend"]
        == "fundamental_chroma_structured_parent_child_bm25_rrf_rerank"
    )
    vector_index = payload["learning_loop"]["vector_index"]
    assert vector_index["scope"] == "fundamental_only"
    assert "trade_memory" not in vector_index["allowed_source_types"]
    assert vector_index["chunking"]["strategy"] == "structured_semantic_parent_child_v1"
    assert payload["learning_loop"]["vector_index"]["keyword_recall"] == "bm25"
    assert payload["learning_loop"]["vector_index"]["fusion"] == "rrf"
    assert payload["learning_loop"]["vector_index"]["rerank"]["enabled"] is True
    assert payload["learning_loop"]["max_retrieved_memories_per_run"] == 5


def test_python_defaults_match_json_configs():
    persona_json = json.loads((ROOT / "config" / "persona.default.json").read_text())
    product_json = json.loads((ROOT / "config" / "product.default.json").read_text())

    assert persona_json["stock_source"] == DEFAULT_PERSONA_CONFIG["stock_source"]
    assert product_json["agent_goal"] == DEFAULT_PRODUCT_CONFIG["agent_goal"]

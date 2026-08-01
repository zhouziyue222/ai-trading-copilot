import pytest
from pydantic import ValidationError

from ai_trading_copilot.copilot.config import (
    DEFAULT_PERSONA_CONFIG,
    DEFAULT_PRODUCT_CONFIG,
)
from ai_trading_copilot.copilot.domain import (
    ForbiddenInstrument,
    MarketType,
    Subscription,
    SubscriptionBook,
    SubscriptionStatus,
    UserPersonaConfig,
)


def test_default_persona_matches_prd_constraints():
    persona = UserPersonaConfig(**DEFAULT_PERSONA_CONFIG)

    assert persona.trading_style == "swing"
    assert persona.max_portfolio_drawdown == 0.25
    assert persona.max_single_position_weight == 0.50
    assert persona.default_position_weight_min == 0.10
    assert persona.default_position_weight_max == 0.25
    assert persona.allowed_markets == [MarketType.US_STOCK, MarketType.US_ETF]
    assert ForbiddenInstrument.LEVERAGE in persona.forbidden_instruments
    assert ForbiddenInstrument.OPTIONS in persona.forbidden_instruments
    assert persona.stock_source == "user_subscription_list"


def test_product_config_uses_distilled_retrieval_memory_limit():
    learning_loop = DEFAULT_PRODUCT_CONFIG["learning_loop"]

    assert DEFAULT_PRODUCT_CONFIG["stock_source"] == "user_subscription_list"
    assert learning_loop["enabled"] is True
    assert learning_loop["memory_strategy"] == "distilled_retrieval_memory"
    assert (
        learning_loop["retrieval_backend"]
        == "fundamental_chroma_structured_parent_child_bm25_rrf_rerank"
    )
    assert learning_loop["vector_index"]["scope"] == "fundamental_only"
    assert "earnings_report" in learning_loop["vector_index"]["allowed_source_types"]
    assert "trade_memory" not in learning_loop["vector_index"]["allowed_source_types"]
    assert learning_loop["vector_index"]["chunking"]["strategy"] == "structured_semantic_parent_child_v1"
    assert learning_loop["vector_index"]["chunking"]["child_chunk_size"] == 500
    assert learning_loop["vector_index"]["keyword_recall"] == "bm25"
    assert learning_loop["vector_index"]["fusion"] == "rrf"
    assert learning_loop["vector_index"]["rerank"]["model"] == "local_cross_feature_v1"
    assert learning_loop["max_retrieved_memories_per_run"] == 5


def test_subscription_symbols_are_normalized():
    book = SubscriptionBook(
        items=[
            Subscription(
                symbol=" spy ",
                market_type=MarketType.US_ETF,
                status=SubscriptionStatus.OBSERVING,
            )
        ]
    )

    assert book.symbols == ["SPY"]
    assert book.contains("spy")
    assert book.get("SPY").market_type == MarketType.US_ETF


def test_subscription_book_rejects_duplicates():
    with pytest.raises(ValidationError):
        SubscriptionBook(
            items=[
                Subscription(symbol="AAPL", market_type=MarketType.US_STOCK),
                Subscription(symbol="aapl", market_type=MarketType.US_STOCK),
            ]
        )


def test_persona_rejects_default_position_band_above_limit():
    with pytest.raises(ValidationError):
        UserPersonaConfig(
            max_single_position_weight=0.20,
            default_position_weight_min=0.10,
            default_position_weight_max=0.25,
        )

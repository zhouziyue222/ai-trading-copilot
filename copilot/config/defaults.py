"""Default product and persona configuration."""

from __future__ import annotations

from ai_trading_copilot.copilot.domain.enums import ForbiddenInstrument, MarketType


DEFAULT_PERSONA_CONFIG = {
    "persona_name": "swing_balanced_pullback_trader",
    "trading_style": "swing",
    "risk_profile": "balanced_plus",
    "max_portfolio_drawdown": 0.25,
    "allowed_markets": [MarketType.US_STOCK, MarketType.US_ETF],
    "forbidden_instruments": [
        ForbiddenInstrument.LEVERAGE,
        ForbiddenInstrument.OPTIONS,
    ],
    "max_single_position_weight": 0.50,
    "default_position_weight_min": 0.10,
    "default_position_weight_max": 0.25,
    "stock_source": "user_subscription_list",
    "preferred_market_regime": "bull_market_or_uptrend",
}


DEFAULT_PERSONA_MARKDOWN = """# User Persona

Swing balanced pullback trader.

- Market scope: US stocks and ETFs
- Trading horizon: several days to several weeks
- Risk profile: balanced plus
- Maximum portfolio drawdown: 25%
- Forbidden instruments: leverage and options
- Maximum single-symbol position: 50%; default suggested position: 10%-25%
- Symbol source: user subscription list
- Entry preference: pullbacks near support; avoid chasing extended moves
- Stop-loss discipline: no trade can proceed without a stop-loss price and invalidation conditions
"""


DEFAULT_PRODUCT_CONFIG = {
    "stock_source": "user_subscription_list",
    "agent_goal": "judge_entry_opportunities_for_subscribed_symbols",
    "news_sentiment_agent": "finnhub_news_sentiment_agent",
    "optional_analysts": [
        "news_sentiment",
        "technical_position",
        "fundamental_analysis",
    ],
    "post_trade_review_agent": "post_trade_review_learning_agent",
    "learning_loop": {
        "enabled": True,
        "memory_strategy": "distilled_retrieval_memory",
        "retrieval_backend": "fundamental_analyst_chroma_structured_parent_child_bm25_rrf_rerank",
        "vector_index": {
            "enabled": True,
            "chroma_path": "config/rag_chroma",
            "collection": "ai_trading_copilot_fundamentals",
            "scope": "fundamental_only",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-v4",
            "allowed_source_types": [
                "annual_report",
                "company_guidance",
                "earnings_call_transcript",
                "earnings_report",
                "fundamental_research_note",
                "investor_day",
                "sec_10k",
                "sec_10q",
            ],
            "chunking": {
                "strategy": "structured_semantic_parent_child_v1",
                "parent_chunk_size": 2200,
                "parent_chunk_overlap": 300,
                "child_chunk_size": 500,
                "child_chunk_overlap": 80,
            },
            "keyword_recall": "bm25",
            "fusion": "rrf",
            "rerank": {
                "enabled": True,
                "model": "local_cross_feature_v1",
                "stage": "post_rrf_candidate_rerank",
            },
            "fallback": "none",
        },
        "max_retrieved_memories_per_run": 5,
    },
}

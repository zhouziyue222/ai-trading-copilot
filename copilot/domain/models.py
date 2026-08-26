"""Pydantic models for the AI trading copilot domain."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .enums import (
    AnalystType,
    ExecutionMode,
    ExecutionStatus,
    ForbiddenInstrument,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    MemoryValidationTarget,
    MarketRegime,
    MarketType,
    SymbolTrendState,
    SubscriptionStatus,
    TradeDirection,
)
from .localization import zh_label


def normalize_symbol(symbol: str) -> str:
    """Normalize user-entered symbols without stripping exchange suffixes."""
    return symbol.strip().upper()


class UserPersonaConfig(BaseModel):
    """Trading constraints that every downstream agent must respect."""

    persona_name: str = "swing_balanced_pullback_trader"
    trading_style: str = "swing"
    risk_profile: str = "balanced_plus"
    max_portfolio_drawdown: float = Field(default=0.25, gt=0, le=1)
    allowed_markets: List[MarketType] = Field(
        default_factory=lambda: [MarketType.US_STOCK, MarketType.US_ETF]
    )
    forbidden_instruments: List[ForbiddenInstrument] = Field(
        default_factory=lambda: [
            ForbiddenInstrument.LEVERAGE,
            ForbiddenInstrument.OPTIONS,
        ]
    )
    max_single_position_weight: float = Field(default=0.50, gt=0, le=1)
    max_gross_exposure: float = Field(default=1.00, gt=0)
    default_position_weight_min: float = Field(default=0.10, ge=0, le=1)
    default_position_weight_max: float = Field(default=0.25, ge=0, le=1)
    stock_source: str = "user_subscription_list"
    preferred_market_regime: str = "bull_market_or_uptrend"

    @model_validator(mode="after")
    def validate_position_band(self) -> "UserPersonaConfig":
        if self.default_position_weight_min > self.default_position_weight_max:
            raise ValueError("default_position_weight_min cannot exceed max")
        if self.default_position_weight_max > self.max_single_position_weight:
            raise ValueError("default position band cannot exceed single-position limit")
        return self

    def allows_market(self, market_type: MarketType) -> bool:
        return market_type in self.allowed_markets


class Subscription(BaseModel):
    """A user-selected stock or ETF watched by the copilot."""

    symbol: str
    market_type: MarketType
    status: SubscriptionStatus = SubscriptionStatus.OBSERVING
    reason: str = ""
    target_action: str = ""

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class SubscriptionBook(BaseModel):
    """Subscription list is the only allowed source of opportunities."""

    items: List[Subscription] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_symbols(self) -> "SubscriptionBook":
        seen = set()
        for item in self.items:
            if item.symbol in seen:
                raise ValueError(f"duplicate subscription symbol: {item.symbol}")
            seen.add(item.symbol)
        return self

    @property
    def symbols(self) -> List[str]:
        return [item.symbol for item in self.items]

    def contains(self, symbol: str) -> bool:
        return normalize_symbol(symbol) in set(self.symbols)

    def get(self, symbol: str) -> Optional[Subscription]:
        normalized = normalize_symbol(symbol)
        for item in self.items:
            if item.symbol == normalized:
                return item
        return None


class PortfolioSnapshot(BaseModel):
    """Portfolio context used by risk limits and portfolio decisions."""

    current_drawdown: float = Field(default=0.0, ge=0, le=1)
    position_weights: Dict[str, float] = Field(default_factory=dict)
    cash: Optional[float] = Field(default=None, ge=0)
    total_value: Optional[float] = Field(default=None, gt=0)
    margin_requirement: float = Field(default=0.0, ge=0, le=1)

    @field_validator("position_weights")
    @classmethod
    def normalize_position_keys(cls, value: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for symbol, weight in value.items():
            if weight < -1 or weight > 1:
                raise ValueError("position weights must be between -1 and 1")
            normalized[normalize_symbol(symbol)] = weight
        return normalized


class PriceBar(BaseModel):
    """Daily OHLCV bar used by deterministic radar rules."""

    date: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_price_range(self) -> "PriceBar":
        if self.high < self.low:
            raise ValueError("high cannot be below low")
        if self.open > self.high or self.open < self.low:
            raise ValueError("open must be inside high/low range")
        if self.close > self.high or self.close < self.low:
            raise ValueError("close must be inside high/low range")
        return self


class MarketRegimeReport(BaseModel):
    regime: MarketRegime
    current_price: float
    moving_average_50: Optional[float] = None
    moving_average_200: Optional[float] = None
    reason: str


class TechnicalPosition(BaseModel):
    symbol: str
    current_price: float
    support_level: float
    recent_high: float
    moving_average_20: Optional[float] = None
    moving_average_50: Optional[float] = None
    distance_to_support_pct: float
    pullback_from_high_pct: float
    reward_risk_ratio: Optional[float] = None
    uptrend: bool


class TechnicalDimension(BaseModel):
    """Technical state for one scope: symbol, sector, or broad market."""

    scope: str
    symbol: str
    label: str = ""
    current_price: Optional[float] = None
    moving_average_20: Optional[float] = None
    moving_average_50: Optional[float] = None
    moving_average_200: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    trend_state: Optional[SymbolTrendState] = None
    reason: str = ""
    data_available: bool = True


class TechnicalContext(BaseModel):
    """Multi-dimensional technical context consumed by the Trader."""

    symbol: str
    stock: TechnicalDimension
    sector: Optional[TechnicalDimension] = None
    market: Optional[TechnicalDimension] = None
    summary: str = ""
    warnings: List[str] = Field(default_factory=list)
    decision_basis: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    downstream_summary: str = ""


class OpportunityRadarItem(BaseModel):
    symbol: str
    status: SubscriptionStatus
    status_label: str = ""
    trend_state: Optional[SymbolTrendState] = None
    trend_label: str = ""
    trend_reason: str = ""
    current_price: Optional[float] = None
    support_level: Optional[float] = None
    reward_risk_ratio: Optional[float] = None
    reason: str
    final_conclusion: str = ""
    review_reasons: List[str] = Field(default_factory=list)
    risk_points: List[str] = Field(default_factory=list)
    suggested_action: str = ""

    @model_validator(mode="after")
    def populate_display_fields(self) -> "OpportunityRadarItem":
        if not self.status_label:
            self.status_label = zh_label(self.status)
        if self.trend_state is not None and not self.trend_label:
            self.trend_label = zh_label(self.trend_state)
        return self


class FundamentalAnalysisReport(BaseModel):
    """Structured company-fundamental risk input for opportunity review."""

    symbol: str
    thesis_intact: bool = True
    material_risk: bool = False
    risk_flags: List[str] = Field(default_factory=list)
    summary: str = ""
    fundamental_score: Optional[float] = Field(default=None, ge=-1, le=1)
    sentiment_score: Optional[float] = Field(default=None, ge=-1, le=1)
    key_events: List[str] = Field(default_factory=list)
    data_availability: Dict[str, str] = Field(default_factory=dict)
    decision_basis: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    downstream_summary: str = ""

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class NewsSentimentReport(BaseModel):
    """Structured news, social sentiment, and event-risk signal."""

    symbol: str
    sentiment_score: float = Field(default=0.0, ge=-1, le=1)
    company_news_score: Optional[float] = Field(default=None, ge=-1, le=1)
    social_sentiment_score: Optional[float] = Field(default=None, ge=-1, le=1)
    earnings_event_score: Optional[float] = Field(default=None, ge=-1, le=1)
    material_risk: bool = False
    risk_flags: List[str] = Field(default_factory=list)
    key_events: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    news_references: List[Dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    data_availability: Dict[str, str] = Field(default_factory=dict)
    decision_basis: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    downstream_summary: str = ""

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class TradePlan(BaseModel):
    """Structured trade plan consumed by Risk Manager and Portfolio Manager."""

    symbol: str
    subscription_status: SubscriptionStatus
    market_regime: MarketRegime
    direction: TradeDirection
    entry_logic: str
    support_level: Optional[float] = Field(default=None, gt=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    targets: List[float] = Field(default_factory=list)
    reward_risk_ratio: Optional[float] = Field(default=None, ge=0)
    position_weight: float = Field(default=0.0, ge=0, le=1)
    holding_period: str
    invalidation_conditions: List[str] = Field(default_factory=list)
    persona_fit_reason: str
    uses_leverage: bool = False
    uses_options: bool = False
    is_chasing: bool = False
    breakout_confirmed: bool = False
    pullback_confirmed: bool = False

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class RiskLimits(BaseModel):
    """Portfolio-level limits used by the v2 Risk Manager."""

    max_position_pct: float = Field(default=0.20, gt=0, le=1)
    max_gross_exposure: float = Field(default=1.00, gt=0)


class ClampEvent(BaseModel):
    """One deterministic risk adjustment from requested to allowed weight."""

    symbol: str
    reason: str
    before: float
    after: float
    limit: float

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class RiskAssessment(BaseModel):
    """v2 risk result for one symbol after portfolio-level clamping."""

    symbol: str = ""
    approved: bool = True
    current_price: Optional[float] = Field(default=None, gt=0)
    current_position_weight: float = Field(default=0.0, ge=-1, le=1)
    target_weight: float = Field(default=0.0, ge=-1, le=1)
    final_weight: float = Field(default=0.0, ge=-1, le=1)
    delta_weight: float = 0.0
    portfolio_value: float = Field(default=0.0, ge=0)
    estimated_trade_value: float = 0.0
    clamped: bool = False
    clamps: List[ClampEvent] = Field(default_factory=list)
    risk_limits: Optional[RiskLimits] = None
    reasoning: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        return normalize_symbol(value) if value else ""


class BrokerExecutionRequest(BaseModel):
    """One simulated broker order request."""

    idempotency_key: str
    symbol: str
    side: str
    quantity: int = Field(gt=0)
    price: Optional[float] = Field(default=None, gt=0)
    order_type: str = "NORMAL"
    trd_env: str = "SIMULATE"


class BrokerExecutionResult(BaseModel):
    """Result returned by a broker adapter."""

    idempotency_key: str
    submitted: bool
    order_id: Optional[str] = None
    status: str = ""
    message: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


class ExecutionDecision(BaseModel):
    """Portfolio Manager final decision.

    The class name is retained for API compatibility. It no longer represents a
    broker execution handoff; it carries ai-hedge-fund-style portfolio actions.
    """

    symbol: str
    mode: ExecutionMode
    status: ExecutionStatus
    direction: TradeDirection
    approved_by_risk: bool
    requires_user_confirmation: bool
    message: str
    action: str = "hold"
    quantity: int = Field(default=0, ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    reasoning: str = ""
    current_weight: float = 0.0
    target_weight: float = 0.0
    final_weight: float = 0.0
    delta_weight: float = 0.0
    current_price: Optional[float] = Field(default=None, gt=0)
    estimated_trade_value: float = 0.0
    pending_broker_order: bool = False
    broker_confirmation_required: bool = False
    submitted_to_broker: bool = False
    submitted_quantity: int = Field(default=0, ge=0)
    broker_order_id: Optional[str] = None
    broker_status: str = ""
    broker_message: str = ""
    broker_idempotency_key: str = ""


class DistilledMemory(BaseModel):
    """Versioned, retrieval-oriented trading lesson.

    Legacy JSONL entries remain valid: they default to ``approved`` so a
    migration does not silently change existing production behaviour.
    Newly reflected memories must explicitly be created as ``candidate``.
    """

    memory_id: str = ""
    memory_type: MemoryType
    memory_kind: MemoryKind = MemoryKind.PROCEDURAL
    status: MemoryStatus = MemoryStatus.APPROVED
    scope: MemoryScope = MemoryScope.SYMBOL
    validation_target: MemoryValidationTarget = MemoryValidationTarget.OUTPERFORM
    lesson: str = Field(min_length=1, max_length=500)
    trigger: str = Field(default="", max_length=300)
    rationale: str = Field(default="", max_length=500)
    symbols: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    market_regimes: List[str] = Field(default_factory=list)
    timeframes: List[str] = Field(default_factory=list)
    source_run_id: Optional[str] = None
    source_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    last_validated_at: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    version: int = Field(default=1, ge=1)
    supersedes: Optional[str] = None
    evidence_run_ids: List[str] = Field(default_factory=list)
    counter_evidence_run_ids: List[str] = Field(default_factory=list)
    sample_count: int = Field(default=0, ge=0)
    outcome_metrics: Dict[str, float] = Field(default_factory=dict)
    created_by: str = "system"
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: List[str]) -> List[str]:
        return [normalize_symbol(symbol) for symbol in value if normalize_symbol(symbol)]

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: List[str]) -> List[str]:
        return sorted({tag.strip().lower() for tag in value if tag.strip()})

    @field_validator("market_regimes", "timeframes", "evidence_run_ids", "counter_evidence_run_ids")
    @classmethod
    def normalize_string_lists(cls, value: List[str]) -> List[str]:
        return sorted({item.strip().lower() for item in value if item.strip()})


class RunOutcome(BaseModel):
    """Delayed market outcome used to validate memories without hindsight leakage."""

    run_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    horizon_days: int = Field(default=5, gt=0)
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    realized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    benchmark_return: Optional[float] = None
    stopped_out: Optional[bool] = None
    source: str = "manual"
    notes: str = Field(default="", max_length=1000)
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("symbol")
    @classmethod
    def normalize_outcome_symbol(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class MemoryEvaluation(BaseModel):
    """Auditable shadow-evaluation result for a candidate memory."""

    memory_id: str
    memory_version: int
    eligible: bool = False
    passed_safety_gate: bool = False
    evidence_runs: int = 0
    outcome_samples: int = 0
    mean_realized_return: Optional[float] = None
    mean_excess_return: Optional[float] = None
    worst_drawdown: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RagDocument(BaseModel):
    """A retrieved knowledge chunk from the vector database."""

    id: str
    text: str = Field(min_length=1)
    source_type: str = "fundamental_research_note"
    source: str = ""
    title: str = ""
    symbols: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbols")
    @classmethod
    def normalize_rag_symbols(cls, value: List[str]) -> List[str]:
        return [normalize_symbol(symbol) for symbol in value if normalize_symbol(symbol)]

    @field_validator("tags")
    @classmethod
    def normalize_rag_tags(cls, value: List[str]) -> List[str]:
        return sorted({tag.strip().lower() for tag in value if tag.strip()})


class TraceEvent(BaseModel):
    """Node-level execution trace for user and developer explainability."""

    node_name: str
    symbol: Optional[str] = None
    input_summary: str = ""
    output_summary: str = ""
    route_reason: str = ""
    rule_hits: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SymbolExplanation(BaseModel):
    """User-facing explanation for one subscribed symbol."""

    symbol: str
    status: SubscriptionStatus
    summary: str
    key_evidence: List[str] = Field(default_factory=list)
    risk_notes: List[str] = Field(default_factory=list)
    execution_message: Optional[str] = None


class CopilotRunReport(BaseModel):
    """User-facing report generated from the same state used by trace output."""

    summary: str
    market_regime: Optional[MarketRegimeReport] = None
    symbols: List[SymbolExplanation] = Field(default_factory=list)
    risk_challenges: Dict[str, str] = Field(default_factory=dict)


__all__ = [
    "AnalystType",
    "BrokerExecutionRequest",
    "BrokerExecutionResult",
    "ClampEvent",
    "CopilotRunReport",
    "DistilledMemory",
    "MemoryEvaluation",
    "ExecutionDecision",
    "FundamentalAnalysisReport",
    "MarketRegimeReport",
    "NewsSentimentReport",
    "OpportunityRadarItem",
    "PortfolioSnapshot",
    "PriceBar",
    "RagDocument",
    "RiskAssessment",
    "RiskLimits",
    "RunOutcome",
    "Subscription",
    "SubscriptionBook",
    "SymbolTrendState",
    "SymbolExplanation",
    "TechnicalPosition",
    "TechnicalContext",
    "TechnicalDimension",
    "TraceEvent",
    "TradePlan",
    "UserPersonaConfig",
    "NewsSentimentReport",
    "ValidationError",
    "normalize_symbol",
]


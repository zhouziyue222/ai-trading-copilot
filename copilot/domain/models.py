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
    MemoryType,
    MarketRegime,
    MarketType,
    RiskRuleCode,
    RiskSeverity,
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
    """Portfolio context used by risk checks."""

    current_drawdown: float = Field(default=0.0, ge=0, le=1)
    position_weights: Dict[str, float] = Field(default_factory=dict)

    @field_validator("position_weights")
    @classmethod
    def normalize_position_keys(cls, value: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        for symbol, weight in value.items():
            if weight < 0 or weight > 1:
                raise ValueError("position weights must be between 0 and 1")
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


class FundamentalNewsReport(BaseModel):
    """Structured fundamental/news risk input for opportunity review."""

    symbol: str
    thesis_intact: bool = True
    material_risk: bool = False
    risk_flags: List[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("symbol")
    @classmethod
    def normalize_symbol_field(cls, value: str) -> str:
        normalized = normalize_symbol(value)
        if not normalized:
            raise ValueError("symbol is required")
        return normalized


class TradePlan(BaseModel):
    """Structured trade plan produced before hard risk approval."""

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


class RiskViolation(BaseModel):
    code: RiskRuleCode
    severity: RiskSeverity
    message: str


class RiskAssessment(BaseModel):
    """Result of hard risk validation."""

    approved: bool
    violations: List[RiskViolation] = Field(default_factory=list)

    @property
    def blocking_violations(self) -> List[RiskViolation]:
        return [v for v in self.violations if v.severity == RiskSeverity.BLOCK]

    @property
    def warnings(self) -> List[RiskViolation]:
        return [v for v in self.violations if v.severity == RiskSeverity.WARN]

    @classmethod
    def from_violations(cls, violations: List[RiskViolation]) -> "RiskAssessment":
        return cls(
            approved=not any(v.severity == RiskSeverity.BLOCK for v in violations),
            violations=violations,
        )


class ExecutionDecision(BaseModel):
    """Execution/alert manager output. It never places a real order by itself."""

    symbol: str
    mode: ExecutionMode
    status: ExecutionStatus
    direction: TradeDirection
    approved_by_risk: bool
    requires_user_confirmation: bool
    message: str


class DistilledMemory(BaseModel):
    """Compact, retrieval-oriented trading lesson."""

    memory_type: MemoryType
    lesson: str = Field(min_length=1, max_length=500)
    symbols: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    source_run_id: Optional[str] = None
    source_path: Optional[str] = None
    created_at: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: List[str]) -> List[str]:
        return [normalize_symbol(symbol) for symbol in value if normalize_symbol(symbol)]

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: List[str]) -> List[str]:
        return sorted({tag.strip().lower() for tag in value if tag.strip()})


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
    "CopilotRunReport",
    "DistilledMemory",
    "ExecutionDecision",
    "FundamentalNewsReport",
    "MarketRegimeReport",
    "OpportunityRadarItem",
    "PortfolioSnapshot",
    "PriceBar",
    "RagDocument",
    "RiskAssessment",
    "RiskViolation",
    "Subscription",
    "SubscriptionBook",
    "SymbolTrendState",
    "SymbolExplanation",
    "TechnicalPosition",
    "TraceEvent",
    "TradePlan",
    "UserPersonaConfig",
    "ValidationError",
    "normalize_symbol",
]

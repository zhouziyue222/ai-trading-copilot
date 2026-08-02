"""Stable domain enums for the AI trading copilot."""

from enum import Enum


class MarketType(str, Enum):
    US_STOCK = "US_STOCK"
    US_ETF = "US_ETF"


class ForbiddenInstrument(str, Enum):
    LEVERAGE = "LEVERAGE"
    OPTIONS = "OPTIONS"


class MemoryType(str, Enum):
    USER_BEHAVIOR = "user_behavior"
    STRATEGY_PERFORMANCE = "strategy_performance"
    SYMBOL_CHARACTERISTIC = "symbol_characteristic"


class AnalystType(str, Enum):
    OPPORTUNITY_RADAR = "opportunity_radar"
    TECHNICAL_POSITION = "technical_position"
    FUNDAMENTAL_NEWS = "fundamental_news"
    NEWS_SENTIMENT = "news_sentiment"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"


class MarketRegime(str, Enum):
    BULL_MARKET = "bull_market"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE_BOUND = "range_bound"
    REVERSAL_POINT = "reversal_point"
    TRADABLE_RANGE = "tradable_range"
    UNCLEAR = "unclear"
    WEAKENING = "weakening"
    BEAR_RISK = "bear_risk"


class SubscriptionStatus(str, Enum):
    OBSERVING = "observing"
    NEAR_OPPORTUNITY = "near_opportunity"
    ACTIONABLE = "actionable"
    RISK_ELEVATED = "risk_elevated"
    NOT_COMPATIBLE = "not_compatible"


class SymbolTrendState(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    UPTREND_PULLBACK = "uptrend_pullback"
    UNKNOWN = "unknown"


class TradeDirection(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    REDUCE = "reduce"
    SELL = "sell"
    WATCH = "watch"


class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    LIVE = "live"


class ExecutionStatus(str, Enum):
    BLOCKED_BY_RISK = "blocked_by_risk"
    ALERT_ONLY = "alert_only"
    SIMULATION_READY = "simulation_ready"
    SIMULATED_ORDER_SUBMITTED = "simulated_order_submitted"
    SIMULATED_ORDER_FAILED = "simulated_order_failed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    LIVE_READY = "live_ready"


class RiskSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


class RiskRuleCode(str, Enum):
    SUBSCRIPTION_REQUIRED = "subscription_required"
    MARKET_NOT_ALLOWED = "market_not_allowed"
    FORBIDDEN_INSTRUMENT = "forbidden_instrument"
    STOP_LOSS_REQUIRED = "stop_loss_required"
    INVALIDATION_REQUIRED = "invalidation_required"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    HIGH_POSITION_SIZE = "high_position_size"
    DRAWDOWN_CAUTION = "drawdown_caution"
    DRAWDOWN_DEFENSIVE = "drawdown_defensive"
    DRAWDOWN_LIMIT_REACHED = "drawdown_limit_reached"
    MARKET_REGIME_WEAK = "market_regime_weak"
    CHASE_CONFIRMATION_REQUIRED = "chase_confirmation_required"
    REWARD_RISK_TOO_LOW = "reward_risk_too_low"
    ACTIONABLE_STATUS_REQUIRED = "actionable_status_required"

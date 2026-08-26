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
    SHORT = "short"
    COVER = "cover"
    WATCH = "watch"


class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    LIVE = "live"


class ExecutionStatus(str, Enum):
    ALERT_ONLY = "alert_only"
    PORTFOLIO_DECIDED = "portfolio_decided"
    CONFIRMATION_REQUIRED = "confirmation_required"

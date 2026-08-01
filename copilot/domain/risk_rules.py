"""Hard risk rules for trade plan approval."""

from __future__ import annotations

from typing import List

from .enums import (
    ForbiddenInstrument,
    MarketRegime,
    RiskRuleCode,
    RiskSeverity,
    SubscriptionStatus,
    TradeDirection,
)
from .models import (
    PortfolioSnapshot,
    RiskAssessment,
    RiskViolation,
    SubscriptionBook,
    TradePlan,
    UserPersonaConfig,
)


MIN_REWARD_RISK_RATIO = 2.0
DRAWDOWN_CAUTION_LEVEL = 0.15
DRAWDOWN_DEFENSIVE_LEVEL = 0.20


def _violation(
    code: RiskRuleCode,
    severity: RiskSeverity,
    message: str,
) -> RiskViolation:
    return RiskViolation(code=code, severity=severity, message=message)


def evaluate_trade_plan(
    *,
    persona: UserPersonaConfig,
    subscriptions: SubscriptionBook,
    portfolio: PortfolioSnapshot,
    plan: TradePlan,
) -> RiskAssessment:
    """Apply deterministic gates before any trade plan can be actionable."""
    violations: List[RiskViolation] = []
    subscription = subscriptions.get(plan.symbol)

    if subscription is None:
        violations.append(
            _violation(
                RiskRuleCode.SUBSCRIPTION_REQUIRED,
                RiskSeverity.BLOCK,
                "交易机会只能为已订阅标的生成。",
            )
        )
    elif not persona.allows_market(subscription.market_type):
        violations.append(
            _violation(
                RiskRuleCode.MARKET_NOT_ALLOWED,
                RiskSeverity.BLOCK,
                f"{subscription.market_type.value} 不符合当前用户画像允许的市场范围。",
            )
        )

    if plan.uses_leverage and ForbiddenInstrument.LEVERAGE in persona.forbidden_instruments:
        violations.append(
            _violation(
                RiskRuleCode.FORBIDDEN_INSTRUMENT,
                RiskSeverity.BLOCK,
                "当前用户画像禁止使用杠杆。",
            )
        )

    if plan.uses_options and ForbiddenInstrument.OPTIONS in persona.forbidden_instruments:
        violations.append(
            _violation(
                RiskRuleCode.FORBIDDEN_INSTRUMENT,
                RiskSeverity.BLOCK,
                "当前用户画像禁止使用期权。",
            )
        )

    if plan.direction == TradeDirection.BUY:
        _validate_buy_plan(persona, portfolio, plan, violations)

    return RiskAssessment.from_violations(violations)


def _validate_buy_plan(
    persona: UserPersonaConfig,
    portfolio: PortfolioSnapshot,
    plan: TradePlan,
    violations: List[RiskViolation],
) -> None:
    if plan.subscription_status != SubscriptionStatus.ACTIONABLE:
        violations.append(
            _violation(
                RiskRuleCode.ACTIONABLE_STATUS_REQUIRED,
                RiskSeverity.BLOCK,
                "买入计划要求订阅标的处于可执行状态。",
            )
        )

    if plan.stop_loss is None:
        violations.append(
            _violation(
                RiskRuleCode.STOP_LOSS_REQUIRED,
                RiskSeverity.BLOCK,
                "买入计划必须包含止损位。",
            )
        )

    if not plan.invalidation_conditions:
        violations.append(
            _violation(
                RiskRuleCode.INVALIDATION_REQUIRED,
                RiskSeverity.BLOCK,
                "买入计划必须包含明确的失效条件。",
            )
        )

    if plan.position_weight > persona.max_single_position_weight:
        violations.append(
            _violation(
                RiskRuleCode.POSITION_LIMIT_EXCEEDED,
                RiskSeverity.BLOCK,
                "计划仓位超过单一标的上限。",
            )
        )
    elif plan.position_weight > persona.default_position_weight_max:
        violations.append(
            _violation(
                RiskRuleCode.HIGH_POSITION_SIZE,
                RiskSeverity.WARN,
                "计划仓位高于默认 10%-25% 区间，需要更高置信度。",
            )
        )

    if portfolio.current_drawdown >= persona.max_portfolio_drawdown:
        violations.append(
            _violation(
                RiskRuleCode.DRAWDOWN_LIMIT_REACHED,
                RiskSeverity.BLOCK,
                "组合回撤已达到最大阈值，应暂停新增买入。",
            )
        )
    elif portfolio.current_drawdown >= DRAWDOWN_DEFENSIVE_LEVEL:
        violations.append(
            _violation(
                RiskRuleCode.DRAWDOWN_DEFENSIVE,
                RiskSeverity.WARN,
                "组合回撤已处于防守警告区间。",
            )
        )
    elif portfolio.current_drawdown >= DRAWDOWN_CAUTION_LEVEL:
        violations.append(
            _violation(
                RiskRuleCode.DRAWDOWN_CAUTION,
                RiskSeverity.WARN,
                "组合回撤已处于警戒区间。",
            )
        )

    if plan.market_regime in {
        MarketRegime.UNCLEAR,
        MarketRegime.WEAKENING,
        MarketRegime.BEAR_RISK,
    }:
        violations.append(
            _violation(
                RiskRuleCode.MARKET_REGIME_WEAK,
                RiskSeverity.WARN,
                "市场环境不符合用户画像偏好的上升趋势条件。",
            )
        )

    if plan.is_chasing and not (plan.breakout_confirmed or plan.pullback_confirmed):
        violations.append(
            _violation(
                RiskRuleCode.CHASE_CONFIRMATION_REQUIRED,
                RiskSeverity.BLOCK,
                "追高只允许在突破确认或回调确认后进行。",
            )
        )

    if plan.is_chasing and (
        plan.reward_risk_ratio is None or plan.reward_risk_ratio < MIN_REWARD_RISK_RATIO
    ):
        violations.append(
            _violation(
                RiskRuleCode.REWARD_RISK_TOO_LOW,
                RiskSeverity.BLOCK,
                "追高要求收益风险比至少达到 2.0。",
            )
        )
    elif (
        plan.reward_risk_ratio is not None
        and plan.reward_risk_ratio < MIN_REWARD_RISK_RATIO
    ):
        violations.append(
            _violation(
                RiskRuleCode.REWARD_RISK_TOO_LOW,
                RiskSeverity.WARN,
                "收益风险比低于偏好阈值。",
            )
        )

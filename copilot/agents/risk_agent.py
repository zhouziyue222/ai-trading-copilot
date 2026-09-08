"""v2-style deterministic Risk Manager.

The Risk Manager does not choose trades. It receives requested portfolio
weights, clamps them to per-symbol and gross-exposure limits, and returns the
allowed final weights for the Portfolio Manager to turn into actions.
"""

from __future__ import annotations

from ai_trading_copilot.copilot.domain.localization import zh_label, zh_bool

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from ai_trading_copilot.copilot.domain.enums import ForbiddenInstrument, TradeDirection
from ai_trading_copilot.copilot.domain.models import (
    ClampEvent,
    PortfolioSnapshot,
    PriceBar,
    RiskAssessment,
    RiskLimits,
    SubscriptionBook,
    TradePlan,
    UserPersonaConfig,
    normalize_symbol,
)


DEFAULT_PORTFOLIO_VALUE = 100_000.0
EPSILON = 1e-9


@dataclass
class RiskReviewResult:
    assessment: RiskAssessment
    report: str
    risk_challenge: str


@dataclass
class RiskBookReviewResult:
    assessments: Dict[str, RiskAssessment]
    report: str
    risk_challenges: Dict[str, str]
    target_weights: Dict[str, float]
    final_weights: Dict[str, float]
    clamps: list[ClampEvent]


class RiskAgent:
    """Apply v2 risk limits to requested target weights."""

    def __init__(self, llm=None):
        self.llm = llm

    def review(
        self,
        *,
        persona: UserPersonaConfig,
        subscriptions: SubscriptionBook | None = None,
        portfolio: PortfolioSnapshot,
        plan: TradePlan,
        price_history_by_symbol: Mapping[str, Sequence[PriceBar]] | None = None,
        target_weight: float | None = None,
    ) -> RiskAssessment:
        symbol = normalize_symbol(plan.symbol)
        result = self.review_book(
            persona=persona,
            subscriptions=subscriptions,
            portfolio=portfolio,
            trade_plans={symbol: plan},
            target_weights={
                symbol: (
                    target_weight
                    if target_weight is not None
                    else _target_weight_from_plan(plan, portfolio)
                )
            },
            price_history_by_symbol=price_history_by_symbol,
        )
        return result.assessments[symbol]

    def review_with_report(
        self,
        *,
        persona: UserPersonaConfig,
        subscriptions: SubscriptionBook | None = None,
        portfolio: PortfolioSnapshot,
        plan: TradePlan,
        analyst_context: str = "",
        price_history_by_symbol: Mapping[str, Sequence[PriceBar]] | None = None,
        target_weight: float | None = None,
    ) -> RiskReviewResult:
        assessment = self.review(
            persona=persona,
            subscriptions=subscriptions,
            portfolio=portfolio,
            plan=plan,
            price_history_by_symbol=price_history_by_symbol,
            target_weight=target_weight,
        )
        challenge = _risk_challenge(assessment)
        return RiskReviewResult(
            assessment=assessment,
            report=_report_from_assessments(
                {assessment.symbol: assessment},
                note="基于既定限额核定仓位",
            ),
            risk_challenge=challenge,
        )

    def review_book(
        self,
        *,
        persona: UserPersonaConfig,
        subscriptions: SubscriptionBook | None = None,
        portfolio: PortfolioSnapshot,
        trade_plans: Mapping[str, TradePlan],
        target_weights: Mapping[str, float] | None = None,
        price_history_by_symbol: Mapping[str, Sequence[PriceBar]] | None = None,
    ) -> RiskBookReviewResult:
        price_history_by_symbol = price_history_by_symbol or {}
        plans = {
            normalize_symbol(symbol or plan.symbol): plan
            for symbol, plan in trade_plans.items()
        }
        requested = (
            _normalize_weights(target_weights)
            if target_weights is not None
            else {
                symbol: _target_weight_from_plan(plan, portfolio)
                for symbol, plan in plans.items()
            }
        )
        full_targets = _portfolio_current_weights(portfolio)
        full_targets.update(requested)

        eligibility_warnings: dict[str, list[str]] = {}
        for symbol, plan in plans.items():
            current_weight = portfolio.position_weights.get(symbol, 0.0)
            desired_weight = full_targets.get(symbol, current_weight)
            current_price = _current_price(symbol, plan, price_history_by_symbol)
            warnings = _eligibility_warnings(
                persona=persona,
                subscriptions=subscriptions,
                symbol=symbol,
                plan=plan,
                current_price=current_price,
                current_weight=current_weight,
                target_weight=desired_weight,
            )
            eligibility_warnings[symbol] = warnings
            if warnings:
                full_targets[symbol] = current_weight

        limits = RiskLimits(
            max_position_pct=persona.max_single_position_weight,
            max_gross_exposure=persona.max_gross_exposure,
        )
        final_weights, clamps = apply_limits(full_targets, limits)
        clamps_by_symbol = _clamps_by_symbol(clamps)
        portfolio_value = _portfolio_value(portfolio)

        assessments: Dict[str, RiskAssessment] = {}
        risk_challenges: Dict[str, str] = {}
        for symbol, plan in plans.items():
            current_weight = portfolio.position_weights.get(symbol, 0.0)
            target_weight = full_targets.get(symbol, current_weight)
            held_flat = bool(eligibility_warnings.get(symbol))
            final_weight = (
                current_weight
                if held_flat
                else final_weights.get(symbol, target_weight)
            )
            symbol_clamps = [] if held_flat else clamps_by_symbol.get(symbol, [])
            warnings = [
                *eligibility_warnings.get(symbol, []),
                *[_clamp_warning(event) for event in symbol_clamps],
            ]
            delta_weight = final_weight - current_weight
            assessment = RiskAssessment(
                symbol=symbol,
                approved=not eligibility_warnings.get(symbol),
                current_price=_current_price(symbol, plan, price_history_by_symbol),
                current_position_weight=round(current_weight, 6),
                target_weight=round(target_weight, 6),
                final_weight=round(final_weight, 6),
                delta_weight=round(delta_weight, 6),
                portfolio_value=round(portfolio_value, 2),
                estimated_trade_value=round(delta_weight * portfolio_value, 2),
                clamped=bool(symbol_clamps),
                clamps=symbol_clamps,
                risk_limits=limits,
                warnings=warnings,
                reasoning=_reasoning(
                    current_weight=current_weight,
                    target_weight=target_weight,
                    final_weight=final_weight,
                    limits=limits,
                    eligibility_warnings=eligibility_warnings.get(symbol, []),
                ),
            )
            assessments[symbol] = assessment
            risk_challenges[symbol] = _risk_challenge(assessment)

        return RiskBookReviewResult(
            assessments=assessments,
            report=_report_from_assessments(
                assessments,
                note="基于既定限额核定仓位",
            ),
            risk_challenges=risk_challenges,
            target_weights={symbol: round(weight, 6) for symbol, weight in full_targets.items()},
            final_weights={
                symbol: round(
                    portfolio.position_weights.get(symbol, weight)
                    if eligibility_warnings.get(symbol)
                    else weight,
                    6,
                )
                for symbol, weight in final_weights.items()
            },
            clamps=clamps,
        )


def apply_limits(
    weights: Mapping[str, float],
    limits: RiskLimits,
) -> tuple[Dict[str, float], list[ClampEvent]]:
    """Clamp signed target weights by single-name and gross-exposure limits."""

    final: Dict[str, float] = {}
    clamps: list[ClampEvent] = []

    for raw_symbol, raw_weight in weights.items():
        symbol = normalize_symbol(raw_symbol)
        weight = float(raw_weight)
        clamped = max(
            -limits.max_position_pct,
            min(limits.max_position_pct, weight),
        )
        if abs(clamped - weight) > EPSILON:
            clamps.append(
                ClampEvent(
                    symbol=symbol,
                    reason="max_position_pct",
                    before=weight,
                    after=clamped,
                    limit=limits.max_position_pct,
                )
            )
        final[symbol] = clamped

    gross = sum(abs(weight) for weight in final.values())
    if gross > limits.max_gross_exposure + EPSILON:
        scale = limits.max_gross_exposure / gross
        for symbol, weight in list(final.items()):
            scaled = weight * scale
            if abs(scaled - weight) > EPSILON:
                clamps.append(
                    ClampEvent(
                        symbol=symbol,
                        reason="max_gross_exposure",
                        before=weight,
                        after=scaled,
                        limit=limits.max_gross_exposure,
                    )
                )
            final[symbol] = scaled

    return final, clamps


def _target_weight_from_plan(plan: TradePlan, portfolio: PortfolioSnapshot) -> float:
    current_weight = portfolio.position_weights.get(plan.symbol, 0.0)
    planned_weight = float(plan.position_weight or 0.0)
    if plan.direction == TradeDirection.BUY:
        return planned_weight if planned_weight > 0 else current_weight
    if plan.direction in {TradeDirection.HOLD, TradeDirection.WATCH}:
        return current_weight
    if plan.direction == TradeDirection.REDUCE:
        return current_weight * 0.5
    if plan.direction in {TradeDirection.SELL, TradeDirection.COVER}:
        return 0.0
    if plan.direction == TradeDirection.SHORT:
        return -planned_weight
    return current_weight


def _normalize_weights(weights: Mapping[str, float] | None) -> Dict[str, float]:
    return {
        normalize_symbol(symbol): float(weight)
        for symbol, weight in (weights or {}).items()
        if normalize_symbol(symbol)
    }


def _portfolio_current_weights(portfolio: PortfolioSnapshot) -> Dict[str, float]:
    return {
        normalize_symbol(symbol): float(weight)
        for symbol, weight in portfolio.position_weights.items()
        if normalize_symbol(symbol)
    }


def _portfolio_value(portfolio: PortfolioSnapshot) -> float:
    if portfolio.total_value is not None and portfolio.total_value > 0:
        return float(portfolio.total_value)
    invested_weight = sum(abs(weight) for weight in portfolio.position_weights.values())
    if portfolio.cash is not None and invested_weight < 1:
        inferred = portfolio.cash / max(1 - invested_weight, 0.01)
        if inferred > 0:
            return float(inferred)
    return DEFAULT_PORTFOLIO_VALUE


def _current_price(
    symbol: str,
    plan: TradePlan,
    price_history_by_symbol: Mapping[str, Sequence[PriceBar]],
) -> float | None:
    bars = _bars_for_symbol(symbol, price_history_by_symbol)
    if bars:
        return float(bars[-1].close)
    if plan.support_level and plan.support_level > 0:
        return float(plan.support_level)
    if plan.targets:
        return float(plan.targets[0])
    return None


def _bars_for_symbol(
    symbol: str,
    price_history_by_symbol: Mapping[str, Sequence[PriceBar]],
) -> list[PriceBar]:
    normalized = normalize_symbol(symbol)
    for raw_symbol, bars in price_history_by_symbol.items():
        if normalize_symbol(raw_symbol) == normalized:
            return list(bars)
    return []


def _eligibility_warnings(
    *,
    persona: UserPersonaConfig,
    subscriptions: SubscriptionBook | None,
    symbol: str,
    plan: TradePlan,
    current_price: float | None,
    current_weight: float,
    target_weight: float,
) -> list[str]:
    warnings: list[str] = []
    subscription = subscriptions.get(symbol) if subscriptions is not None else None
    if subscriptions is not None and subscription is None:
        warnings.append("标的不在订阅清单内，维持当前仓位。")
    if subscription is not None and not persona.allows_market(subscription.market_type):
        warnings.append("该市场不符合用户配置，维持当前仓位。")
    if plan.uses_leverage and ForbiddenInstrument.LEVERAGE in persona.forbidden_instruments:
        warnings.append("用户配置禁止杠杆，维持当前仓位。")
    if plan.uses_options and ForbiddenInstrument.OPTIONS in persona.forbidden_instruments:
        warnings.append("用户配置禁止期权，维持当前仓位。")
    if abs(target_weight - current_weight) > EPSILON and current_price is None:
        warnings.append("【待补充】缺少当前价格，维持当前仓位。")
    return warnings


def _clamps_by_symbol(clamps: Sequence[ClampEvent]) -> dict[str, list[ClampEvent]]:
    output: dict[str, list[ClampEvent]] = {}
    for event in clamps:
        output.setdefault(event.symbol, []).append(event)
    return output


def _clamp_warning(event: ClampEvent) -> str:
    if event.reason == "max_position_pct":
        return (
            f"目标仓位从 {event.before:.2%} 限制至 {event.after:.2%}，"
            f"单标的仓位上限为 {event.limit:.2%}。"
        )
    return (
        f"目标仓位从 {event.before:.2%} 缩减至 {event.after:.2%}，"
        f"总敞口上限为 {event.limit:.2%}。"
    )


def _reasoning(
    *,
    current_weight: float,
    target_weight: float,
    final_weight: float,
    limits: RiskLimits,
    eligibility_warnings: Sequence[str],
) -> Dict[str, str]:
    return {
        "current_weight": f"Current portfolio weight is {current_weight:.2%}.",
        "target_weight": f"Requested target weight is {target_weight:.2%}.",
        "final_weight": f"Risk-adjusted final weight is {final_weight:.2%}.",
        "single_position_limit": (
            f"Each symbol is capped at +/-{limits.max_position_pct:.2%}."
        ),
        "gross_exposure_limit": (
            f"Gross target exposure is capped at {limits.max_gross_exposure:.2%}."
        ),
        "eligibility": (
            "Eligibility gate passed."
            if not eligibility_warnings
            else "Eligibility gate held the target flat before risk clamping."
        ),
    }


def _risk_challenge(assessment: RiskAssessment) -> str:
    if not assessment.approved:
        return (
            f"{assessment.symbol}：准入条件未通过，仓位维持在 "
            f"{assessment.current_position_weight:.2%}，组合经理不应交易。"
        )
    if assessment.clamped:
        return (
            f"{assessment.symbol}：目标仓位从 "
            f"{assessment.target_weight:.2%} 调整至 {assessment.final_weight:.2%}；"
            "被削减的敞口保留为现金。"
        )
    return (
        f"{assessment.symbol}：申请目标仓位 {assessment.target_weight:.2%} "
        "符合当前风控限额。"
    )


def _report_from_assessments(
    assessments: Mapping[str, RiskAssessment],
    *,
    note: str,
) -> str:
    lines = [
        "# 风险管理报告",
        "",
        f"- 说明： {note}",
        "",
        "| 标的 | 是否通过 | 当前仓位 | 申请仓位 | 最终仓位 | 仓位变化 | 是否受限 | 估算交易金额 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for symbol in sorted(assessments):
        assessment = assessments[symbol]
        lines.append(
            "| "
            + " | ".join(
                [
                    assessment.symbol,
                    zh_bool(assessment.approved),
                    f"{assessment.current_position_weight:.2%}",
                    f"{assessment.target_weight:.2%}",
                    f"{assessment.final_weight:.2%}",
                    f"{assessment.delta_weight:.2%}",
                    zh_bool(assessment.clamped),
                    f"{assessment.estimated_trade_value:.2f}",
                ]
            )
            + " |"
        )
    warnings = [
        f"{assessment.symbol}: {warning}"
        for assessment in assessments.values()
        for warning in assessment.warnings
    ]
    if warnings:
        lines.extend(["", "## 风险提示"])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## 风控结论"])
    lines.extend(f"- {_risk_challenge(assessment)}" for assessment in assessments.values())
    return "\n".join(lines) + "\n"


__all__ = [
    "RiskAgent",
    "RiskBookReviewResult",
    "RiskReviewResult",
    "apply_limits",
]

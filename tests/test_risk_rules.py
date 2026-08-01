from ai_trading_copilot.copilot.domain import (
    MarketRegime,
    MarketType,
    PortfolioSnapshot,
    RiskRuleCode,
    RiskSeverity,
    Subscription,
    SubscriptionBook,
    SubscriptionStatus,
    TradeDirection,
    TradePlan,
    UserPersonaConfig,
    evaluate_trade_plan,
)


def _persona() -> UserPersonaConfig:
    return UserPersonaConfig()


def _subscriptions() -> SubscriptionBook:
    return SubscriptionBook(
        items=[
            Subscription(
                symbol="AAPL",
                market_type=MarketType.US_STOCK,
                status=SubscriptionStatus.ACTIONABLE,
            ),
            Subscription(
                symbol="SPY",
                market_type=MarketType.US_ETF,
                status=SubscriptionStatus.NEAR_OPPORTUNITY,
            ),
        ]
    )


def _portfolio(drawdown: float = 0.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(current_drawdown=drawdown)


def _valid_buy_plan(**overrides) -> TradePlan:
    data = {
        "symbol": "AAPL",
        "subscription_status": SubscriptionStatus.ACTIONABLE,
        "market_regime": MarketRegime.UPTREND,
        "direction": TradeDirection.BUY,
        "entry_logic": "Pullback to support with trend intact.",
        "support_level": 180.0,
        "stop_loss": 174.0,
        "targets": [195.0, 205.0],
        "reward_risk_ratio": 2.5,
        "position_weight": 0.20,
        "holding_period": "1-6 weeks",
        "invalidation_conditions": ["Close below support for two sessions."],
        "persona_fit_reason": "Fits pullback preference and position limits.",
    }
    data.update(overrides)
    return TradePlan(**data)


def _codes(assessment):
    return {violation.code for violation in assessment.violations}


def test_valid_plan_is_approved():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(),
    )

    assert assessment.approved is True
    assert assessment.blocking_violations == []


def test_non_subscription_symbol_is_blocked():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(symbol="MSFT"),
    )

    assert assessment.approved is False
    assert RiskRuleCode.SUBSCRIPTION_REQUIRED in _codes(assessment)


def test_buy_without_stop_loss_is_blocked():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(stop_loss=None),
    )

    assert assessment.approved is False
    assert RiskRuleCode.STOP_LOSS_REQUIRED in _codes(assessment)


def test_buy_without_invalidation_conditions_is_blocked():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(invalidation_conditions=[]),
    )

    assert assessment.approved is False
    assert RiskRuleCode.INVALIDATION_REQUIRED in _codes(assessment)


def test_leverage_and_options_are_blocked():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(uses_leverage=True, uses_options=True),
    )

    assert assessment.approved is False
    forbidden = [
        v for v in assessment.violations
        if v.code == RiskRuleCode.FORBIDDEN_INSTRUMENT
    ]
    assert len(forbidden) == 2


def test_position_above_single_symbol_limit_is_blocked():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(position_weight=0.51),
    )

    assert assessment.approved is False
    assert RiskRuleCode.POSITION_LIMIT_EXCEEDED in _codes(assessment)


def test_position_above_default_band_warns_but_does_not_block():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(position_weight=0.30),
    )

    assert assessment.approved is True
    assert RiskRuleCode.HIGH_POSITION_SIZE in _codes(assessment)
    assert assessment.warnings[0].severity == RiskSeverity.WARN


def test_drawdown_warning_levels_and_limit():
    caution = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(0.15),
        plan=_valid_buy_plan(),
    )
    defensive = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(0.20),
        plan=_valid_buy_plan(),
    )
    limit = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(0.25),
        plan=_valid_buy_plan(),
    )

    assert caution.approved is True
    assert RiskRuleCode.DRAWDOWN_CAUTION in _codes(caution)
    assert defensive.approved is True
    assert RiskRuleCode.DRAWDOWN_DEFENSIVE in _codes(defensive)
    assert limit.approved is False
    assert RiskRuleCode.DRAWDOWN_LIMIT_REACHED in _codes(limit)


def test_weak_market_regime_warns():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(market_regime=MarketRegime.WEAKENING),
    )

    assert assessment.approved is True
    assert RiskRuleCode.MARKET_REGIME_WEAK in _codes(assessment)


def test_chasing_requires_confirmation_and_reward_risk():
    no_confirmation = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(is_chasing=True, reward_risk_ratio=2.5),
    )
    low_rr = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(
            is_chasing=True,
            breakout_confirmed=True,
            reward_risk_ratio=1.5,
        ),
    )
    confirmed = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(
            is_chasing=True,
            pullback_confirmed=True,
            reward_risk_ratio=2.5,
        ),
    )

    assert no_confirmation.approved is False
    assert RiskRuleCode.CHASE_CONFIRMATION_REQUIRED in _codes(no_confirmation)
    assert low_rr.approved is False
    assert RiskRuleCode.REWARD_RISK_TOO_LOW in _codes(low_rr)
    assert confirmed.approved is True


def test_buy_requires_actionable_subscription_status():
    assessment = evaluate_trade_plan(
        persona=_persona(),
        subscriptions=_subscriptions(),
        portfolio=_portfolio(),
        plan=_valid_buy_plan(
            symbol="SPY",
            subscription_status=SubscriptionStatus.NEAR_OPPORTUNITY,
        ),
    )

    assert assessment.approved is False
    assert RiskRuleCode.ACTIONABLE_STATUS_REQUIRED in _codes(assessment)


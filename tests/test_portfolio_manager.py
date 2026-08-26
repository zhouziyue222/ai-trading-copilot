from ai_trading_copilot.copilot.agents import PortfolioManager, RiskAgent
from ai_trading_copilot.copilot.domain import (
    ExecutionMode,
    ExecutionStatus,
    MarketRegime,
    PortfolioSnapshot,
    PriceBar,
    SubscriptionStatus,
    TradeDirection,
    TradePlan,
    UserPersonaConfig,
)


def _bars(closes):
    return [
        PriceBar(
            date=f"2026-05-{idx + 1:02d}",
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=1000,
        )
        for idx, close in enumerate(closes)
    ]


def _plan(direction=TradeDirection.BUY, **overrides):
    data = {
        "symbol": "AAPL",
        "subscription_status": SubscriptionStatus.ACTIONABLE,
        "market_regime": MarketRegime.UPTREND,
        "direction": direction,
        "entry_logic": "Buy near support.",
        "support_level": 100,
        "stop_loss": 97,
        "targets": [110],
        "reward_risk_ratio": 3,
        "position_weight": 0.2,
        "holding_period": "1-6 weeks",
        "invalidation_conditions": ["Close below support."],
        "persona_fit_reason": "Fits pullback persona.",
    }
    data.update(overrides)
    return TradePlan(**data)


def _risk(plan=None, portfolio=None, persona=None):
    plan = plan or _plan()
    portfolio = portfolio or PortfolioSnapshot(total_value=100_000, cash=80_000)
    return RiskAgent().review(
        persona=persona or UserPersonaConfig(),
        portfolio=portfolio,
        plan=plan,
        price_history_by_symbol={"AAPL": _bars([100, 101, 102, 103, 104])},
    )


def test_portfolio_manager_builds_target_weights_from_plan_direction():
    manager = PortfolioManager()
    portfolio = PortfolioSnapshot(position_weights={"AAPL": 0.20, "MSFT": -0.10})

    targets = manager.build_target_weights(
        trade_plans={
            "AAPL": _plan(direction=TradeDirection.REDUCE),
            "MSFT": _plan(symbol="MSFT", direction=TradeDirection.COVER),
            "TSLA": _plan(symbol="TSLA", direction=TradeDirection.SHORT, position_weight=0.15),
        },
        portfolio=portfolio,
    )

    assert targets == {"AAPL": 0.10, "MSFT": 0.0, "TSLA": -0.15}


def test_portfolio_manager_buys_to_risk_adjusted_final_weight():
    plan = _plan()
    risk = _risk(plan)
    decision = PortfolioManager().decide(
        plan=plan,
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
    )

    assert decision.status == ExecutionStatus.PORTFOLIO_DECIDED
    assert decision.action == "buy"
    assert decision.final_weight == 0.2
    assert decision.estimated_trade_value == 20_000
    assert decision.quantity == int(20_000 // 104)
    assert decision.pending_broker_order is True
    assert decision.broker_confirmation_required is True
    assert decision.submitted_to_broker is False


def test_portfolio_manager_live_action_requires_confirmation():
    plan = _plan()
    risk = _risk(plan)
    decision = PortfolioManager().decide(
        plan=plan,
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        mode=ExecutionMode.LIVE,
        user_confirmed=False,
    )

    assert decision.status == ExecutionStatus.CONFIRMATION_REQUIRED
    assert decision.requires_user_confirmation is True
    assert decision.pending_broker_order is False


def test_portfolio_manager_uses_clamped_weight_instead_of_requested_weight():
    plan = _plan(position_weight=0.30)
    risk = _risk(
        plan,
        persona=UserPersonaConfig(
            max_single_position_weight=0.10,
            default_position_weight_min=0.05,
            default_position_weight_max=0.10,
        ),
    )
    result = PortfolioManager().decide_with_report(
        plan=plan,
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
    )

    assert result.decision.action == "buy"
    assert result.decision.target_weight == 0.30
    assert result.decision.final_weight == 0.10
    assert result.decision.quantity == int(10_000 // 104)
    assert "Risk-adjusted final weight" in result.report


def test_portfolio_manager_holds_when_final_weight_is_unchanged():
    plan = _plan(direction=TradeDirection.HOLD)
    portfolio = PortfolioSnapshot(
        total_value=100_000,
        cash=80_000,
        position_weights={"AAPL": 0.20},
    )
    risk = _risk(plan, portfolio)
    decision = PortfolioManager().decide(
        plan=plan,
        risk_assessment=risk,
        portfolio=portfolio,
    )

    assert decision.action == "hold"
    assert decision.status == ExecutionStatus.ALERT_ONLY
    assert decision.quantity == 0
    assert decision.pending_broker_order is False

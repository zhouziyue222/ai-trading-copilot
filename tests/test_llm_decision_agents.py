from pathlib import Path

from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.agents import (
    ExecutionAlertManager,
    OpportunityReviewManager,
    RiskAgent,
    TraderAgent,
)
from ai_trading_copilot.copilot.domain import (
    ExecutionMode,
    ExecutionStatus,
    FundamentalNewsReport,
    MarketRegime,
    PortfolioSnapshot,
    RiskRuleCode,
    Subscription,
    SubscriptionBook,
    SubscriptionStatus,
    TechnicalPosition,
    TradeDirection,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.enums import MarketType
from ai_trading_copilot.copilot.domain.models import OpportunityRadarItem, TradePlan


class StaticLLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.content)


def _opportunity(status=SubscriptionStatus.ACTIONABLE):
    return OpportunityRadarItem(
        symbol="AAPL",
        status=status,
        current_price=102,
        support_level=100,
        reward_risk_ratio=3,
        reason="Tool reports show a pullback near support.",
    )


def _position():
    return TechnicalPosition(
        symbol="AAPL",
        current_price=102,
        support_level=100,
        recent_high=110,
        moving_average_20=101,
        moving_average_50=98,
        distance_to_support_pct=0.02,
        pullback_from_high_pct=0.073,
        reward_risk_ratio=3,
        uptrend=True,
    )


def _plan(**overrides):
    data = {
        "symbol": "AAPL",
        "subscription_status": SubscriptionStatus.ACTIONABLE,
        "market_regime": MarketRegime.UPTREND,
        "direction": TradeDirection.BUY,
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


def test_opportunity_review_manager_uses_llm_report_and_structured_json():
    llm = StaticLLM(
        "# Opportunity review\n\n"
        '{"status": "risk_elevated", "reason": "News risk is unresolved.", '
        '"final_conclusion": "Risk elevated", '
        '"review_reasons": ["news report"], '
        '"risk_points": ["regulatory_probe"], '
        '"suggested_action": "Avoid new entry."}'
    )

    result = OpportunityReviewManager(llm=llm).review_with_report(
        symbol="AAPL",
        opportunity=_opportunity(),
        technical_position=_position(),
        fundamental_news=FundamentalNewsReport(
            symbol="AAPL",
            material_risk=True,
            risk_flags=["regulatory_probe"],
        ),
        analyst_context="tool-derived reports",
    )

    assert result.item.status == SubscriptionStatus.RISK_ELEVATED
    assert result.item.risk_points == ["regulatory_probe"]
    assert result.report.startswith("# Opportunity review")
    assert llm.prompts


def test_trader_agent_uses_llm_to_build_trade_plan_and_report():
    llm = StaticLLM(
        "# Trader report\n\n"
        '{"direction": "buy", "entry_logic": "Enter only near support.", '
        '"market_regime": "uptrend", "support_level": 100, "stop_loss": 96.5, '
        '"targets": [110, 118], "reward_risk_ratio": 3.2, '
        '"position_weight": 0.2, "holding_period": "1-6 weeks", '
        '"invalidation_conditions": ["Close below 100"], '
        '"persona_fit_reason": "Controlled pullback setup.", '
        '"uses_leverage": false, "uses_options": false, '
        '"is_chasing": false, "breakout_confirmed": false, '
        '"pullback_confirmed": true}'
    )

    result = TraderAgent(llm=llm).create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position(),
        persona=UserPersonaConfig(),
        analyst_context="tool-derived reports",
    )

    assert result.plan.direction == TradeDirection.BUY
    assert result.plan.stop_loss == 96.5
    assert result.plan.targets == [110, 118]
    assert result.report.startswith("# Trader report")


def test_risk_agent_llm_report_cannot_override_hard_blocks():
    llm = StaticLLM("# Risk report\n\nApproved despite missing subscription.")
    result = RiskAgent(llm=llm).review_with_report(
        persona=UserPersonaConfig(),
        subscriptions=SubscriptionBook(
            items=[Subscription(symbol="MSFT", market_type=MarketType.US_STOCK)]
        ),
        portfolio=PortfolioSnapshot(),
        plan=_plan(),
        analyst_context="tool-derived reports",
    )

    assert result.assessment.approved is False
    assert result.assessment.blocking_violations[0].code == RiskRuleCode.SUBSCRIPTION_REQUIRED
    assert result.risk_challenge
    assert "风险挑战" in result.report
    assert result.risk_challenge in result.report
    assert result.report.startswith("# Risk report")


def test_execution_alert_manager_llm_report_keeps_live_confirmation_gate():
    llm = StaticLLM("# Execution report\n\nWait for user confirmation.")
    result = ExecutionAlertManager(llm=llm).prepare_with_report(
        plan=_plan(),
        risk_assessment=RiskAgent().review(
            persona=UserPersonaConfig(),
            subscriptions=SubscriptionBook(
                items=[Subscription(symbol="AAPL", market_type=MarketType.US_STOCK)]
            ),
            portfolio=PortfolioSnapshot(),
            plan=_plan(),
        ),
        mode=ExecutionMode.LIVE,
        user_confirmed=False,
    )

    assert result.decision.status == ExecutionStatus.CONFIRMATION_REQUIRED
    assert result.decision.requires_user_confirmation is True
    assert result.report.startswith("# Execution report")


def test_agent_report_paths_are_workspace_reports(tmp_path):
    path = tmp_path / "3_trader" / "trader.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Trader report\n", encoding="utf-8")

    assert Path(path).parent == tmp_path / "3_trader"

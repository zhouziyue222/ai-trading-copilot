from ai_trading_copilot.copilot.agents import (
    ExecutionAlertManager,
    FundamentalNewsAgent,
    OpportunityRadarAgent,
    OpportunityReviewManager,
    RiskAgent,
    TechnicalPositionAgent,
    TraderAgent,
)
from ai_trading_copilot.copilot.domain import (
    ExecutionStatus,
    FundamentalNewsReport,
    MarketType,
    PortfolioSnapshot,
    PriceBar,
    SubscriptionStatus,
    TradeDirection,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.workflow import SubscriptionOpportunityWorkflow


def _bars_from_closes(closes):
    bars = []
    for idx, close in enumerate(closes):
        bars.append(
            PriceBar(
                date=f"2026-03-{(idx % 28) + 1:02d}",
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1000,
            )
        )
    return bars


def _actionable_bars():
    bars = _bars_from_closes([80 + i * 0.5 for i in range(40)])
    for idx in range(19):
        bars.append(
            PriceBar(
                date=f"2026-04-{idx + 1:02d}",
                open=101,
                high=110 if idx == 0 else 103,
                low=100.5,
                close=101,
                volume=1000,
            )
        )
    bars.append(
        PriceBar(
            date="2026-04-20",
            open=101.5,
            high=103,
            low=100.5,
            close=102,
            volume=1000,
        )
    )
    return bars


class CapturingRiskAgent(RiskAgent):
    def __init__(self):
        self.seen_subscription_books = []

    def review(self, **kwargs):
        self.seen_subscription_books.append(kwargs["subscriptions"])
        return super().review(**kwargs)


def _workflow(risk_agent=None):
    return SubscriptionOpportunityWorkflow(
        graph=CopilotLangGraph(
            enable_default_llm=False,
            default_persona_config=UserPersonaConfig(),
            opportunity_radar_agent=OpportunityRadarAgent(),
            fundamental_news_agent=FundamentalNewsAgent(),
            opportunity_review_manager=OpportunityReviewManager(),
            technical_position_agent=TechnicalPositionAgent(),
            trader_agent=TraderAgent(),
            risk_agent=risk_agent or RiskAgent(),
            execution_manager=ExecutionAlertManager(),
        )
    )


def test_subscription_workflow_generates_simulated_buy_for_actionable_subscription():
    risk_agent = CapturingRiskAgent()
    run = _workflow(risk_agent=risk_agent).run(
        subscription_symbols=["AAPL", "SPY", "CRCL"],
        price_history_by_symbol={
            "AAPL": _actionable_bars(),
            "CRCL": _actionable_bars(),
            "MSFT": _actionable_bars(),
        },
        portfolio=PortfolioSnapshot(),
    )

    assert [item.radar_item.symbol for item in run.items] == ["AAPL", "SPY", "CRCL"]
    aapl = run.items[0]
    assert aapl.radar_item.status == SubscriptionStatus.ACTIONABLE
    assert aapl.trade_plan.direction == TradeDirection.BUY
    assert aapl.risk_assessment.approved is True
    assert aapl.execution_decision.status == ExecutionStatus.SIMULATION_READY
    crcl = run.items[2]
    assert crcl.radar_item.symbol == "CRCL"
    assert crcl.trade_plan.symbol == "CRCL"
    assert crcl.risk_assessment is not None
    assert crcl.execution_decision.symbol == "CRCL"
    assert crcl.execution_decision.status == ExecutionStatus.SIMULATION_READY
    assert risk_agent.seen_subscription_books
    assert risk_agent.seen_subscription_books[-1].get("CRCL").market_type == MarketType.US_STOCK


def test_subscription_workflow_keeps_missing_history_as_observation_only():
    run = _workflow().run(
        subscription_symbols=["AAPL", "SPY"],
        price_history_by_symbol={"AAPL": _actionable_bars()},
        portfolio=PortfolioSnapshot(),
    )

    spy = run.items[1]
    assert spy.radar_item.status == SubscriptionStatus.OBSERVING
    assert spy.trade_plan.direction == TradeDirection.WATCH
    assert spy.execution_decision.status == ExecutionStatus.ALERT_ONLY


def test_subscription_workflow_downgrades_material_news_risk():
    run = _workflow().run(
        subscription_symbols=["AAPL", "SPY"],
        price_history_by_symbol={"AAPL": _actionable_bars()},
        fundamental_news_by_symbol={
            "AAPL": FundamentalNewsReport(
                symbol="AAPL",
                material_risk=True,
                risk_flags=["earnings_gap_risk"],
                summary="Unresolved earnings risk.",
            )
        },
        portfolio=PortfolioSnapshot(),
    )

    aapl = run.items[0]
    assert aapl.radar_item.status == SubscriptionStatus.RISK_ELEVATED
    assert aapl.trade_plan.direction == TradeDirection.HOLD
    assert aapl.execution_decision.status == ExecutionStatus.ALERT_ONLY

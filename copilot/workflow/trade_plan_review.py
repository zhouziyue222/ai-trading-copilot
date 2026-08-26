"""First vertical slice: persona -> subscription -> risk limits -> portfolio decision."""

from __future__ import annotations

from pydantic import BaseModel

from ai_trading_copilot.copilot.agents import (
    PortfolioManager,
    RiskAgent,
)
from ai_trading_copilot.copilot.domain.enums import ExecutionMode
from ai_trading_copilot.copilot.domain.models import (
    ExecutionDecision,
    PortfolioSnapshot,
    RiskAssessment,
    Subscription,
    SubscriptionBook,
    TradePlan,
    UserPersonaConfig,
)


class TradePlanReview(BaseModel):
    plan: TradePlan
    risk_assessment: RiskAssessment
    execution_decision: ExecutionDecision


class TradePlanReviewWorkflow:
    """Review a proposed plan without placing a real order."""

    def __init__(
        self,
        *,
        persona_config: UserPersonaConfig,
        subscription_symbols: list[str],
        risk_agent: RiskAgent,
        portfolio_manager: PortfolioManager,
    ):
        self.persona_config = persona_config
        self.subscription_symbols = [symbol.strip().upper() for symbol in subscription_symbols]
        self.risk_agent = risk_agent
        self.portfolio_manager = portfolio_manager

    def review(
        self,
        *,
        plan: TradePlan,
        portfolio: PortfolioSnapshot,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
        run_id: str | None = None,
    ) -> TradePlanReview:
        from ai_trading_copilot.copilot.domain.enums import MarketType

        subscriptions = SubscriptionBook(
            items=[
                Subscription(symbol=symbol, market_type=MarketType.US_STOCK)
                for symbol in self.subscription_symbols
            ]
        )
        risk_assessment = self.risk_agent.review(
            persona=self.persona_config,
            subscriptions=subscriptions,
            portfolio=portfolio,
            plan=plan,
        )
        execution_decision = self.portfolio_manager.decide(
            plan=plan,
            risk_assessment=risk_assessment,
            portfolio=portfolio,
            mode=mode,
            user_confirmed=user_confirmed,
            run_id=run_id,
        )
        return TradePlanReview(
            plan=plan,
            risk_assessment=risk_assessment,
            execution_decision=execution_decision,
        )

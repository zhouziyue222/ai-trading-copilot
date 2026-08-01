"""Compatibility wrapper for the explainable subscription opportunity graph."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel

from ai_trading_copilot.copilot.domain.enums import AnalystType, ExecutionMode
from ai_trading_copilot.copilot.domain.models import (
    ExecutionDecision,
    FundamentalNewsReport,
    OpportunityRadarItem,
    PortfolioSnapshot,
    PriceBar,
    RiskAssessment,
    TechnicalPosition,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph


class SubscriptionOpportunityResult(BaseModel):
    radar_item: OpportunityRadarItem
    technical_position: Optional[TechnicalPosition] = None
    trade_plan: Optional[TradePlan] = None
    risk_assessment: Optional[RiskAssessment] = None
    execution_decision: Optional[ExecutionDecision] = None


class SubscriptionOpportunityRun(BaseModel):
    items: List[SubscriptionOpportunityResult]


class SubscriptionOpportunityWorkflow:
    """Runs the current MVP chain via CopilotLangGraph."""

    def __init__(self, graph: Optional[CopilotLangGraph] = None):
        self.graph = graph or CopilotLangGraph()

    def run(
        self,
        *,
        subscription_symbols: List[str],
        price_history_by_symbol: Optional[Dict[str, List[PriceBar]]] = None,
        portfolio: Optional[PortfolioSnapshot] = None,
        persona_markdown: Optional[str] = None,
        persona_config: Optional[UserPersonaConfig] = None,
        fundamental_news_by_symbol: Dict[str, FundamentalNewsReport] | None = None,
        selected_analysts: Optional[Iterable[AnalystType | str]] = None,
        report_output_dir: Optional[str | Path] = None,
        trade_date: Optional[str] = None,
        look_back_days: int = 90,
        mode: ExecutionMode = ExecutionMode.SIMULATION,
        portfolio_mode: ExecutionMode = ExecutionMode.SIMULATION,
        user_confirmed: bool = False,
    ) -> SubscriptionOpportunityRun:
        state = self.graph.run(
            subscription_symbols=subscription_symbols,
            price_history_by_symbol=price_history_by_symbol,
            portfolio=portfolio,
            persona_markdown=persona_markdown,
            persona_config=persona_config,
            fundamental_news_by_symbol=fundamental_news_by_symbol,
            selected_analysts=selected_analysts,
            report_output_dir=report_output_dir,
            trade_date=trade_date,
            look_back_days=look_back_days,
            mode=mode,
            portfolio_mode=portfolio_mode,
            user_confirmed=user_confirmed,
        )
        return SubscriptionOpportunityRun(
            items=[
                SubscriptionOpportunityResult(
                    radar_item=item,
                    technical_position=state.get("technical_positions", {}).get(item.symbol),
                    trade_plan=state.get("trade_plans", {}).get(item.symbol),
                    risk_assessment=state.get("risk_assessments", {}).get(item.symbol),
                    execution_decision=state.get("execution_decisions", {}).get(item.symbol),
                )
                for item in state.get("radar_items", [])
            ]
        )

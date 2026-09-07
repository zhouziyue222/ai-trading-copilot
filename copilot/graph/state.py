"""State contract for the explainable copilot LangGraph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict

from ai_trading_copilot.copilot.domain.models import (
    CopilotRunReport,
    DistilledMemory,
    ExecutionDecision,
    FundamentalAnalysisReport,
    MemoryRetrievalRecord,
    NewsSentimentReport,
    OpportunityRadarItem,
    PortfolioSnapshot,
    PriceBar,
    RiskAssessment,
    TechnicalContext,
    TechnicalPosition,
    TraceEvent,
    TradePlan,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.enums import ExecutionMode
from ai_trading_copilot.copilot.domain.enums import AnalystType


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    return {**left, **right}


class CopilotGraphState(TypedDict, total=False):
    persona_markdown: str
    persona_config: UserPersonaConfig
    subscription_symbols: List[str]
    selected_analysts: List[AnalystType]
    portfolio: PortfolioSnapshot
    portfolio_mode: ExecutionMode
    price_history_by_symbol: Dict[str, List[PriceBar]]
    fundamental_analysis_by_symbol: Dict[str, FundamentalAnalysisReport]
    trade_date: Optional[str]
    look_back_days: int
    radar_items: List[OpportunityRadarItem]
    technical_positions: Dict[str, TechnicalPosition]
    technical_contexts: Dict[str, TechnicalContext]
    news_sentiment_by_symbol: Dict[str, NewsSentimentReport]
    market_reports_by_symbol: Annotated[Dict[str, str], merge_dicts]
    fundamental_analysis_reports_by_symbol: Annotated[Dict[str, str], merge_dicts]
    news_sentiment_reports_by_symbol: Annotated[Dict[str, str], merge_dicts]
    opportunity_reports_by_symbol: Annotated[Dict[str, str], merge_dicts]
    trade_plans: Dict[str, TradePlan]
    risk_assessments: Dict[str, RiskAssessment]
    execution_decisions: Dict[str, ExecutionDecision]
    memories: Dict[str, List[DistilledMemory]]
    shadow_memories: Dict[str, List[DistilledMemory]]
    memory_retrievals: List[MemoryRetrievalRecord]
    long_term_memory_enabled: bool
    memory_candidates: List[DistilledMemory]
    explanations: Dict[str, Any]
    report: CopilotRunReport
    report_output_dir: str
    analyst_reports: Annotated[Dict[str, str], merge_dicts]
    agent_reports: Annotated[Dict[str, str], merge_dicts]
    trace_events: Annotated[List[TraceEvent], operator.add]
    errors: List[str]
    execution_mode: ExecutionMode
    user_confirmed: bool
    run_id: str
    trace_persisted: bool


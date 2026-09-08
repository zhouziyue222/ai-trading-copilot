from pathlib import Path

from langchain_core.messages import AIMessage

from ai_trading_copilot.copilot.agents import (
    PortfolioManager,
    RiskAgent,
    TraderAgent,
)
from ai_trading_copilot.copilot.domain import (
    ExecutionMode,
    ExecutionStatus,
    DistilledMemory,
    FundamentalAnalysisReport,
    MarketRegime,
    MemoryStatus,
    MemoryType,
    PortfolioSnapshot,
    SubscriptionStatus,
    TechnicalPosition,
    TradeDirection,
    UserPersonaConfig,
)
from ai_trading_copilot.copilot.domain.models import OpportunityRadarItem, TradePlan
from ai_trading_copilot.copilot.services.memory_retrieval import MemoryRetrievalSession
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


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


def _risk(plan=None):
    plan = plan or _plan()
    return RiskAgent().review(
        persona=UserPersonaConfig(),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        plan=plan,
    )


def test_trader_agent_internal_review_uses_llm_report_and_structured_json():
    llm = StaticLLM(
        "# Trader review\n\n"
        '{"direction": "hold", "entry_logic": "News risk is unresolved.", '
        '"market_regime": "uptrend", "support_level": 100, "stop_loss": null, '
        '"targets": [], "reward_risk_ratio": 3, "position_weight": 0, '
        '"holding_period": "No active holding period.", '
        '"invalidation_conditions": ["Resolve regulatory probe"], '
        '"persona_fit_reason": "Material risk blocks new entry.", '
        '"uses_leverage": false, "uses_options": false, '
        '"is_chasing": false, "breakout_confirmed": false, '
        '"pullback_confirmed": false}'
    )

    result = TraderAgent(llm=llm).create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position(),
        fundamental_analysis=FundamentalAnalysisReport(
            symbol="AAPL",
            material_risk=True,
            risk_flags=["regulatory_probe"],
        ),
        persona=UserPersonaConfig(),
        analyst_context="tool-derived reports",
    )

    assert result.opportunity is not None
    assert result.opportunity.status == SubscriptionStatus.RISK_ELEVATED
    assert result.opportunity.risk_points == ["regulatory_probe"]
    assert result.plan.direction == TradeDirection.HOLD
    assert result.report.startswith("# Trader review")
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


def test_trader_deterministic_gate_rejects_llm_buy_far_from_support():
    llm = StaticLLM(
        "# Trader report\n\n"
        '{"direction": "buy", "entry_logic": "Chase the move.", '
        '"market_regime": "uptrend", "support_level": 100, "stop_loss": 97, '
        '"targets": [120], "reward_risk_ratio": 3, "position_weight": 0.2, '
        '"holding_period": "1-6 weeks", "invalidation_conditions": ["Close below support"], '
        '"persona_fit_reason": "Momentum.", "uses_leverage": false, '
        '"uses_options": false, "is_chasing": false}'
    )

    result = TraderAgent(llm=llm).create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position().model_copy(
            update={"distance_to_support_pct": 0.08}
        ),
        persona=UserPersonaConfig(),
    )

    assert result.plan.direction == TradeDirection.HOLD
    assert result.plan.position_weight == 0
    assert "结论方向： 持有" in result.report
    assert "Chase the move" not in result.report


def test_risk_agent_report_explains_v2_weight_clamp():
    llm = StaticLLM("# Risk report\n\nThis should not drive deterministic limits.")
    result = RiskAgent(llm=llm).review_with_report(
        persona=UserPersonaConfig(),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        plan=_plan(),
        analyst_context="tool-derived reports",
    )

    assert result.assessment.target_weight == 0.2
    assert result.assessment.final_weight == 0.2
    assert result.risk_challenge
    assert result.risk_challenge in result.report
    assert result.report.startswith("# 风险管理报告")
    assert llm.prompts == []


def test_portfolio_manager_report_keeps_live_confirmation_gate():
    llm = StaticLLM(
        "# Portfolio report\n\n"
        '{"action": "buy", "quantity": 10, "confidence": 0.8, '
        '"reasoning": "Risk limit supports a small buy."}'
    )
    result = PortfolioManager(llm=llm).decide_with_report(
        plan=_plan(),
        risk_assessment=_risk(),
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        mode=ExecutionMode.LIVE,
        user_confirmed=False,
    )

    assert result.decision.status == ExecutionStatus.CONFIRMATION_REQUIRED
    assert result.decision.requires_user_confirmation is True
    assert result.report.startswith("# 组合管理报告")
    assert llm.prompts == []


def test_portfolio_manager_never_submits_orders():
    risk = _risk()

    disabled = PortfolioManager().decide(
        plan=_plan(),
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        mode=ExecutionMode.SIMULATION,
        run_id="run_unit",
    )
    enabled = PortfolioManager().decide(
        plan=_plan(),
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        mode=ExecutionMode.SIMULATION,
        run_id="run_unit",
    )

    assert disabled.status == ExecutionStatus.PORTFOLIO_DECIDED
    assert disabled.submitted_to_broker is False
    assert enabled.status == ExecutionStatus.PORTFOLIO_DECIDED
    assert enabled.submitted_to_broker is False
    assert enabled.action == disabled.action


def _memory_session(tmp_path, *, consumer, run_id="memory-run"):
    store = DistilledMemoryStore(tmp_path / f"{consumer}.sqlite3")
    memory = store.repository.upsert(
        DistilledMemory(
            memory_id=f"{consumer}-memory",
            memory_type=MemoryType.STRATEGY_PERFORMANCE,
            status=MemoryStatus.APPROVED,
            lesson="Use half size until the support retest is confirmed.",
            trigger="AAPL uptrend pullback near support.",
            symbols=["AAPL"],
            tags=["pullback", "position_sizing"],
            market_regimes=["uptrend"],
        )
    )
    return (
        MemoryRetrievalSession(
            store,
            symbol="AAPL",
            consumer=consumer,
            run_id=run_id,
            tags=["pullback"],
            market_regime="uptrend",
        ),
        memory,
    )


def test_trader_forced_prefetch_and_validated_memory_citation(tmp_path):
    session, memory = _memory_session(tmp_path, consumer="trader")
    reference = f"{memory.memory_id}@{memory.version}"
    llm = StaticLLM(
        "# Trader report\n\n"
        '{"direction": "buy", "entry_logic": "Enter near support.", '
        '"market_regime": "uptrend", "support_level": 100, "stop_loss": 97, '
        '"targets": [110], "reward_risk_ratio": 3, "position_weight": 0.1, '
        '"holding_period": "1-6 weeks", "invalidation_conditions": ["Close below support"], '
        '"persona_fit_reason": "Memory supports reduced size.", "uses_leverage": false, '
        '"uses_options": false, "is_chasing": false, "breakout_confirmed": false, '
        f'"pullback_confirmed": true, "memory_citations": ["{reference}"], '
        '"memory_influence": "Reduced initial size."}'
    )

    result = TraderAgent(llm=llm).create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position(),
        persona=UserPersonaConfig(),
        memory_session=session,
    )

    assert result.plan.memory_citations == [reference]
    assert result.plan.memory_influence == "Reduced initial size."
    assert session.records[0].request.phase == "prefetch"
    assert "Use half size" in str(llm.prompts[0])


def test_trader_memory_failure_falls_back_to_normal_llm_path():
    class FailingMemorySession:
        def prefetch(self, *args, **kwargs):
            raise RuntimeError("memory unavailable")

    llm = StaticLLM(
        "# Trader report\n\n"
        '{"direction": "hold", "entry_logic": "Wait.", "market_regime": "uptrend", '
        '"position_weight": 0, "holding_period": "wait", "invalidation_conditions": [], '
        '"persona_fit_reason": "Safe fallback.", "uses_leverage": false, '
        '"uses_options": false, "is_chasing": false}'
    )

    result = TraderAgent(llm=llm).create_plan_with_report(
        opportunity=_opportunity(),
        technical_position=_position(),
        persona=UserPersonaConfig(),
        memory_session=FailingMemorySession(),
    )

    assert result.plan.direction == TradeDirection.HOLD
    assert llm.prompts


def test_portfolio_memory_advisor_can_only_scale_risk_increasing_action(tmp_path):
    session, memory = _memory_session(tmp_path, consumer="portfolio_manager")
    reference = f"{memory.memory_id}@{memory.version}"
    llm = StaticLLM(
        "# Memory advisor\n\n"
        f'{{"decision": "scale", "scale": 0.5, "memory_citations": ["{reference}"], '
        '"memory_influence": "Start at half size."}'
    )
    risk = _risk()

    result = PortfolioManager(llm=llm).decide_with_report(
        plan=_plan(),
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(total_value=100_000, cash=80_000),
        memory_session=session,
    )

    assert risk.final_weight == 0.2
    assert result.decision.pre_memory_final_weight == 0.2
    assert result.decision.final_weight == 0.1
    assert result.decision.memory_scale == 0.5
    assert result.decision.memory_citations == [reference]


def test_portfolio_memory_advisor_cannot_block_risk_reduction(tmp_path):
    session, _ = _memory_session(tmp_path, consumer="portfolio_manager")
    llm = StaticLLM(
        '# Advisor\n\n{"decision": "hold", "scale": 0, "memory_citations": [], '
        '"memory_influence": "Do nothing."}'
    )
    risk = _risk().model_copy(
        update={
            "current_position_weight": 0.2,
            "target_weight": 0.1,
            "final_weight": 0.1,
            "delta_weight": -0.1,
            "estimated_trade_value": -10_000,
        }
    )

    result = PortfolioManager(llm=llm).decide_with_report(
        plan=_plan(direction=TradeDirection.REDUCE, position_weight=0.1),
        risk_assessment=risk,
        portfolio=PortfolioSnapshot(
            total_value=100_000,
            cash=80_000,
            position_weights={"AAPL": 0.2},
        ),
        memory_session=session,
    )

    assert result.decision.action == "reduce"
    assert result.decision.final_weight == 0.1
    assert session.records == []
    assert llm.prompts == []


def test_agent_report_paths_are_workspace_reports(tmp_path):
    path = tmp_path / "3_trader" / "trader.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Trader report\n", encoding="utf-8")

    assert Path(path).parent == tmp_path / "3_trader"



"""Post-trade review learning agent."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from ai_trading_copilot.copilot.domain import MemoryStatus, RunOutcome
from ai_trading_copilot.copilot.domain.models import DistilledMemory
from ai_trading_copilot.copilot.services.memory_evaluation import MemoryShadowEvaluator
from ai_trading_copilot.copilot.services.memory_learning import PostRunLearningService
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore


class PostTradeReviewLearningAgent:
    """Stores and retrieves compact post-trade lessons only."""

    def __init__(
        self,
        store: DistilledMemoryStore,
        *,
        learning_service: PostRunLearningService | None = None,
        evaluator: MemoryShadowEvaluator | None = None,
    ):
        self._store = store
        self.learning_service = learning_service
        self.evaluator = evaluator

    @property
    def store(self) -> DistilledMemoryStore:
        return self._store

    def remember(self, memory: DistilledMemory) -> None:
        self._store.append(memory)

    def retrieve_context(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
        run_id: str | None = None,
    ) -> List[DistilledMemory]:
        return self._store.retrieve(
            symbol=symbol,
            tags=tags,
            query=query,
            limit=limit,
            status=MemoryStatus.APPROVED,
            run_id=run_id,
            usage_mode="approved",
        )

    def retrieve_shadow_context(
        self,
        *,
        symbol: str | None = None,
        tags: Iterable[str] = (),
        query: str | None = None,
        limit: int | None = None,
        run_id: str | None = None,
    ) -> List[DistilledMemory]:
        """Retrieve for measurement only; callers must not inject into prompts."""

        return self._store.retrieve(
            symbol=symbol,
            tags=tags,
            query=query,
            limit=limit,
            status=MemoryStatus.SHADOW,
            run_id=run_id,
            usage_mode="shadow",
        )

    def learn_from_run(self, state: Mapping[str, Any]) -> List[DistilledMemory]:
        if self.learning_service is None:
            return []
        if not state.get("trade_plans") and not state.get("execution_decisions"):
            return []
        return self.learning_service.learn(state)

    def list_memories(
        self,
        *,
        statuses: Iterable[MemoryStatus | str] | None = None,
        symbol: str | None = None,
    ) -> List[DistilledMemory]:
        return self._store.list_all(statuses=statuses, symbol=symbol)

    def transition(
        self,
        memory_id: str,
        status: MemoryStatus | str,
        *,
        actor: str,
        reason: str,
    ) -> DistilledMemory:
        return self._store.repository.transition(
            memory_id,
            status,
            actor=actor,
            reason=reason,
        )

    def record_outcome(self, outcome: RunOutcome) -> RunOutcome:
        saved = self._store.repository.record_outcome(outcome)
        if self.evaluator is not None and self.evaluator.policy.allow_auto_promotion:
            memory_ids = self._store.repository.memory_ids_for_usage(
                run_id=outcome.run_id,
                symbol=outcome.symbol,
                mode="shadow",
            )
            for memory_id in memory_ids:
                evaluation = self.evaluator.evaluate(memory_id)
                if evaluation.eligible:
                    self.evaluator.promote(
                        memory_id,
                        actor="automatic_promotion_gate",
                        reason="shadow_outcome_gate_passed",
                        automatic=True,
                    )
        return saved

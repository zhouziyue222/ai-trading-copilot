"""Subscription status transition rules."""

from __future__ import annotations

from ai_trading_copilot.copilot.domain.enums import SubscriptionStatus
from ai_trading_copilot.copilot.domain.models import Subscription


ALLOWED_TRANSITIONS = {
    SubscriptionStatus.OBSERVING: {
        SubscriptionStatus.NEAR_OPPORTUNITY,
        SubscriptionStatus.ACTIONABLE,
        SubscriptionStatus.RISK_ELEVATED,
        SubscriptionStatus.NOT_COMPATIBLE,
    },
    SubscriptionStatus.NEAR_OPPORTUNITY: {
        SubscriptionStatus.OBSERVING,
        SubscriptionStatus.ACTIONABLE,
        SubscriptionStatus.RISK_ELEVATED,
        SubscriptionStatus.NOT_COMPATIBLE,
    },
    SubscriptionStatus.ACTIONABLE: {
        SubscriptionStatus.OBSERVING,
        SubscriptionStatus.NEAR_OPPORTUNITY,
        SubscriptionStatus.RISK_ELEVATED,
    },
    SubscriptionStatus.RISK_ELEVATED: {
        SubscriptionStatus.OBSERVING,
        SubscriptionStatus.NOT_COMPATIBLE,
    },
    SubscriptionStatus.NOT_COMPATIBLE: {
        SubscriptionStatus.OBSERVING,
        SubscriptionStatus.RISK_ELEVATED,
    },
}


def can_transition(
    current: SubscriptionStatus,
    target: SubscriptionStatus,
) -> bool:
    if current == target:
        return True
    return target in ALLOWED_TRANSITIONS[current]


def transition_subscription_status(
    subscription: Subscription,
    target: SubscriptionStatus,
) -> Subscription:
    if not can_transition(subscription.status, target):
        raise ValueError(
            f"Invalid subscription status transition: "
            f"{subscription.status.value} -> {target.value}"
        )
    return subscription.model_copy(update={"status": target})

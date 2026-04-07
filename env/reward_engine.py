"""Composite shaped reward computation for the DebtRecoveryEnv."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from env.models import (
    ActionType,
    BorrowerState,
    CollectionAction,
    ContactOutcome,
    RewardComponents,
    Sentiment,
)


_SENTIMENT_ORDER = [
    Sentiment.cooperative,
    Sentiment.avoidant,
    Sentiment.hostile,
    Sentiment.ghost,
]


def compute_step_reward(
    borrower_before: BorrowerState,
    borrower_after: BorrowerState,
    action: CollectionAction,
    contact_result: Dict[str, Any],
    portfolio_mean_outstanding: float,
    step: int,
    max_steps: int,
) -> RewardComponents:
    """Compute the composite shaped reward for a single step.

    Args:
        borrower_before: Borrower state before the action.
        borrower_after: Borrower state after the action.
        action: The action that was taken.
        contact_result: Dict with keys answered, ptp_made, payment_received_inr,
            complaint_filed, violation_triggered.
        portfolio_mean_outstanding: Mean outstanding across the portfolio.
        step: Current step number.
        max_steps: Maximum steps in the episode.

    Returns:
        ``RewardComponents`` with all individual signals.
    """
    action_type = ActionType(action.action_type)
    answered = contact_result.get("answered", False)
    ptp_made = contact_result.get("ptp_made", False)
    ptp_kept = contact_result.get("ptp_kept", False)
    payment = contact_result.get("payment_received_inr", 0.0)
    complaint_filed = contact_result.get("complaint_filed", False)
    violation_triggered = contact_result.get("violation_triggered", False)

    outstanding = borrower_before.outstanding_inr
    mean_out = max(portfolio_mean_outstanding, 1.0)  # avoid div-by-zero

    # ── Recovery signal ──────────────────────────────────────────────────
    recovery_signal = 0.0
    if payment > 0:
        if payment >= outstanding * 0.95:
            # Full payment
            recovery_signal = 1.0 * (outstanding / mean_out)
        elif action_type == ActionType.OFFER_OTS and ptp_kept:
            # OTS accepted and paid
            recovery_signal = 0.7 * (payment / mean_out)
        else:
            # Partial payment
            recovery_signal = 0.4 * (payment / outstanding)
    elif ptp_made and not ptp_kept:
        # PTP obtained but no payment yet
        recovery_signal = 0.15

    # ── Contact quality ──────────────────────────────────────────────────
    contact_quality = 0.0
    if answered:
        contact_quality = 0.05
        # Productive: sentiment improved
        sent_before_idx = _sentiment_idx(borrower_before.sentiment)
        sent_after_idx = _sentiment_idx(borrower_after.sentiment)
        if sent_after_idx < sent_before_idx:
            contact_quality = 0.08
    elif contact_result.get("contact_outcome") == ContactOutcome.voicemail:
        contact_quality = 0.01

    # 3rd+ failed attempt today penalty
    if not answered and borrower_before.contact_attempts_today >= 2:
        contact_quality = -0.03

    # ── Sentiment delta ──────────────────────────────────────────────────
    sentiment_delta = 0.0
    sent_before = Sentiment(borrower_before.sentiment)
    sent_after = Sentiment(borrower_after.sentiment)

    if sent_before != sent_after:
        before_idx = _sentiment_idx(sent_before)
        after_idx = _sentiment_idx(sent_after)

        if after_idx < before_idx:
            # Improvement
            if sent_before == Sentiment.hostile and sent_after == Sentiment.avoidant:
                sentiment_delta = 0.06
            elif sent_before == Sentiment.avoidant and sent_after == Sentiment.cooperative:
                sentiment_delta = 0.10
            else:
                sentiment_delta = 0.08
            # PTP active bonus
            if borrower_after.ptp_active:
                sentiment_delta = 0.12
        else:
            # Deterioration
            sentiment_delta = -0.08

    # ── Compliance score ─────────────────────────────────────────────────
    compliance_score = 0.02 if not violation_triggered else 0.0

    # ── Penalty: complaints ──────────────────────────────────────────────
    penalty_complaints = -0.50 if complaint_filed else 0.0

    # ── Penalty: violations ──────────────────────────────────────────────
    penalty_violations = 0.0
    if violation_triggered:
        penalty_violations -= 1.50

    # Legal escalation on DPD < 90
    if action_type == ActionType.ESCALATE_LEGAL and borrower_before.dpd < 90:
        penalty_violations -= 0.30

    # PTP broken after aggressive contacts
    if (
        borrower_before.ptp_active
        and not borrower_after.ptp_active
        and borrower_before.contact_attempts_today >= 3
    ):
        penalty_violations -= 0.25

    # ── Time decay ───────────────────────────────────────────────────────
    time_decay = 0.0
    if not borrower_after.is_resolved and borrower_after.dpd > 90:
        time_decay = -0.02

    return RewardComponents(
        recovery_signal=recovery_signal,
        contact_quality=contact_quality,
        sentiment_delta=sentiment_delta,
        compliance_score=compliance_score,
        penalty_complaints=penalty_complaints,
        penalty_violations=penalty_violations,
        time_decay=time_decay,
    )


def total_reward(components: RewardComponents) -> float:
    """Sum all reward components and clamp to [-2.0, 2.0]."""
    total = (
        components.recovery_signal
        + components.contact_quality
        + components.sentiment_delta
        + components.compliance_score
        + components.penalty_complaints
        + components.penalty_violations
        + components.time_decay
    )
    return float(np.clip(total, -2.0, 2.0))


def normalize_episode_reward(cumulative: float, max_steps: int) -> float:
    """Normalize cumulative episode reward to [0.0, 1.0] via min-max."""
    # Theoretical min: max_steps * -2.0, theoretical max: max_steps * 2.0
    min_r = max_steps * -2.0
    max_r = max_steps * 2.0
    if max_r == min_r:
        return 0.5
    return float(np.clip((cumulative - min_r) / (max_r - min_r), 0.0, 1.0))


def _sentiment_idx(sentiment) -> int:
    """Return index of sentiment in the ordered list."""
    s = Sentiment(sentiment)
    try:
        return _SENTIMENT_ORDER.index(s)
    except ValueError:
        return 2  # default to hostile

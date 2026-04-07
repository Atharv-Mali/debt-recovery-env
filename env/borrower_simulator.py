"""Probabilistic borrower response model with deterministic seeded RNG."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from env.models import (
    ActionType,
    BorrowerState,
    CollectionAction,
    ContactOutcome,
    EmploymentType,
    Sentiment,
)


# ── Constants ────────────────────────────────────────────────────────────────

_EMPLOYMENT_ANSWER_BASE: Dict[str, float] = {
    EmploymentType.salaried: 0.65,
    EmploymentType.self_employed: 0.50,
    EmploymentType.daily_wage: 0.35,
    EmploymentType.unemployed: 0.25,
}

_SENTIMENT_MULTIPLIER: Dict[str, float] = {
    Sentiment.cooperative: 1.3,
    Sentiment.avoidant: 0.6,
    Sentiment.hostile: 0.4,
    Sentiment.ghost: 0.05,
}

_CALL_TIME_MULTIPLIER: Dict[str, float] = {
    ActionType.CALL_MORNING: 1.1,
    ActionType.CALL_AFTERNOON: 0.9,
    ActionType.CALL_EVENING: 1.2,
}

_INCOME_PTP_KEPT_BASE: Dict[str, float] = {
    "low": 0.30,
    "low_mid": 0.50,
    "mid": 0.68,
    "high": 0.82,
}

_EMPLOYMENT_PTP_KEPT_MOD: Dict[str, float] = {
    EmploymentType.salaried: 1.0,
    EmploymentType.self_employed: 1.0,
    EmploymentType.daily_wage: 0.7,
    EmploymentType.unemployed: 0.4,
}

_SENTIMENT_ORDER = [Sentiment.cooperative, Sentiment.avoidant, Sentiment.hostile, Sentiment.ghost]

_CALL_ACTIONS = {ActionType.CALL_MORNING, ActionType.CALL_AFTERNOON, ActionType.CALL_EVENING}

_AGGRESSIVE_ACTIONS = {
    ActionType.SMS_WARNING,
    ActionType.ESCALATE_LEGAL,
    ActionType.FIELD_VISIT,
}

_EMPATHETIC_ACTIONS = {
    ActionType.EMAIL_EMPATHETIC,
    ActionType.GRANT_DEFERMENT,
    ActionType.OFFER_RESTRUCTURE,
}


# ── Public API ───────────────────────────────────────────────────────────────

def compute_answer_probability(
    borrower: BorrowerState,
    action: CollectionAction,
    rng: np.random.RandomState,
) -> float:
    """Compute probability that a borrower answers/responds to contact."""
    action_type = ActionType(action.action_type)

    # Non-contact actions → no answer
    if action_type not in _CALL_ACTIONS and action_type not in {
        ActionType.FIELD_VISIT,
        ActionType.WHATSAPP_NUDGE,
        ActionType.SMS_REMINDER,
        ActionType.SMS_WARNING,
        ActionType.SMS_SETTLEMENT,
        ActionType.EMAIL_FORMAL,
        ActionType.EMAIL_EMPATHETIC,
    }:
        return 0.0

    # DNC override
    emp = EmploymentType(borrower.employment_type)
    sent = Sentiment(borrower.sentiment)

    if borrower.dnc_status and action_type in (
        _CALL_ACTIONS | {ActionType.SMS_REMINDER, ActionType.SMS_WARNING, ActionType.SMS_SETTLEMENT}
    ):
        return 0.0

    base = _EMPLOYMENT_ANSWER_BASE.get(emp, 0.40)
    prob = base * _SENTIMENT_MULTIPLIER.get(sent, 0.5)

    # Time slot multiplier for calls
    if action_type in _CALL_TIME_MULTIPLIER:
        prob *= _CALL_TIME_MULTIPLIER[action_type]

    # Field visit has higher base
    if action_type == ActionType.FIELD_VISIT:
        prob = max(prob, 0.55) * 1.15

    # SMS/email/whatsapp have reduced engagement
    if action_type in {
        ActionType.SMS_REMINDER, ActionType.SMS_WARNING,
        ActionType.SMS_SETTLEMENT, ActionType.EMAIL_FORMAL,
        ActionType.EMAIL_EMPATHETIC, ActionType.WHATSAPP_NUDGE,
    }:
        prob *= 0.75

    # Contact frequency decay
    if borrower.contact_attempts_today >= 2:
        prob *= 0.4

    return float(np.clip(prob, 0.0, 0.95))


def compute_ptp_probability(
    borrower: BorrowerState,
    action: CollectionAction,
    rng: np.random.RandomState,
) -> float:
    """Compute probability that the borrower makes a Promise-To-Pay."""
    dpd = borrower.dpd

    if dpd < 30:
        base = 0.60
    elif dpd < 60:
        base = 0.45
    elif dpd < 90:
        base = 0.30
    elif dpd < 120:
        base = 0.18
    else:
        base = 0.08

    # Hardship penalty
    if borrower.hardship_flag:
        base *= 0.5

    # Broken PTP history
    ptp_made = borrower.ptp_active  # Use actual PTP history from profile
    history = _get_ptp_history_ratio(borrower)
    base *= (1.0 - history * 0.4)

    # OTS bonus
    action_type = ActionType(action.action_type)
    if action_type == ActionType.OFFER_OTS:
        sp = action.settlement_percentage or 1.0
        if sp < 0.8:
            base += 0.15

    return float(np.clip(base, 0.0, 0.85))


def compute_ptp_kept_probability(
    borrower: BorrowerState,
    rng: np.random.RandomState,
) -> float:
    """Compute probability that an active PTP is honored."""
    income = borrower.income_band
    base = _INCOME_PTP_KEPT_BASE.get(income, 0.50)

    emp = EmploymentType(borrower.employment_type)
    base *= _EMPLOYMENT_PTP_KEPT_MOD.get(emp, 1.0)

    return float(np.clip(base, 0.0, 0.90))


def compute_sentiment_transition(
    borrower: BorrowerState,
    action: CollectionAction,
    contact_result: Dict[str, Any],
    rng: np.random.RandomState,
) -> Sentiment:
    """Compute the borrower's new sentiment after an interaction."""
    current = Sentiment(borrower.sentiment)
    action_type = ActionType(action.action_type)
    current_idx = _SENTIMENT_ORDER.index(current)

    answered = contact_result.get("answered", False)

    # Empathetic action → improve sentiment one step
    if action_type in _EMPATHETIC_ACTIONS and answered:
        new_idx = max(0, current_idx - 1)
        return _SENTIMENT_ORDER[new_idx]

    # Aggressive contact on hostile → stay or worsen
    is_aggressive = (
        borrower.contact_attempts_today >= 3
        or action_type in _AGGRESSIVE_ACTIONS
    )

    if is_aggressive and current == Sentiment.hostile:
        if rng.random() < 0.3:
            return Sentiment.ghost
        return Sentiment.hostile

    # Pressure on hardship borrower → drop one step
    if is_aggressive and borrower.hardship_flag:
        new_idx = min(len(_SENTIMENT_ORDER) - 1, current_idx + 1)
        return _SENTIMENT_ORDER[new_idx]

    # No contact → decay toward ghost if DPD > 90
    if action_type == ActionType.NO_CONTACT and borrower.dpd > 90:
        if rng.random() < 0.35:
            new_idx = min(len(_SENTIMENT_ORDER) - 1, current_idx + 1)
            return _SENTIMENT_ORDER[new_idx]

    # Successful contact with positive outcome → slight improvement
    if answered and contact_result.get("ptp_made", False):
        new_idx = max(0, current_idx - 1)
        return _SENTIMENT_ORDER[new_idx]

    return current


def compute_complaint_probability(
    borrower: BorrowerState,
    action: CollectionAction,
    rng: np.random.RandomState,
) -> float:
    """Compute probability that a complaint is filed after this action."""
    base = 0.02
    action_type = ActionType(action.action_type)
    sentiment = Sentiment(borrower.sentiment)

    # Hostile sentiment
    if sentiment == Sentiment.hostile:
        base += 0.25

    # Too many contact attempts today
    if borrower.contact_attempts_today >= 3:
        base += 0.30

    # DNC violation
    if borrower.dnc_status and action_type in (
        _CALL_ACTIONS | {ActionType.SMS_REMINDER, ActionType.SMS_WARNING, ActionType.SMS_SETTLEMENT}
    ):
        base += 0.80

    # SMS_WARNING on hardship
    if action_type == ActionType.SMS_WARNING and borrower.hardship_flag:
        base += 0.15

    return float(np.clip(base, 0.0, 1.0))


def simulate_step(
    borrower: BorrowerState,
    action: CollectionAction,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Run a full simulation step and return contact results dict.

    Returns:
        {
            "answered": bool,
            "contact_outcome": ContactOutcome,
            "ptp_made": bool,
            "ptp_kept": bool,
            "payment_received_inr": float,
            "complaint_filed": bool,
            "violation_triggered": bool,
            "new_sentiment": Sentiment,
        }
    """
    action_type = ActionType(action.action_type)
    result: Dict[str, Any] = {
        "answered": False,
        "contact_outcome": ContactOutcome.not_attempted,
        "ptp_made": False,
        "ptp_kept": False,
        "payment_received_inr": 0.0,
        "complaint_filed": False,
        "violation_triggered": False,
        "new_sentiment": Sentiment(borrower.sentiment),
    }

    # Skip non-contact actions that don't involve communication
    if action_type in {ActionType.NO_CONTACT, ActionType.FLAG_WRITEOFF}:
        result["contact_outcome"] = ContactOutcome.not_attempted
        result["new_sentiment"] = compute_sentiment_transition(
            borrower, action, result, rng
        )
        return result

    # Check for DNC violation
    if borrower.dnc_status and action_type in (
        _CALL_ACTIONS | {ActionType.SMS_REMINDER, ActionType.SMS_WARNING, ActionType.SMS_SETTLEMENT}
    ):
        result["violation_triggered"] = True

    # Compute answer probability
    ans_prob = compute_answer_probability(borrower, action, rng)
    answered = rng.random() < ans_prob

    if answered:
        result["answered"] = True
        result["contact_outcome"] = ContactOutcome.answered

        # Check PTP
        if action_type in {
            ActionType.NEGOTIATE_PTP, ActionType.OFFER_OTS,
            ActionType.OFFER_RESTRUCTURE, ActionType.GRANT_DEFERMENT,
        } or action_type in _CALL_ACTIONS:
            ptp_prob = compute_ptp_probability(borrower, action, rng)
            if rng.random() < ptp_prob:
                result["ptp_made"] = True

                # Check if PTP is immediately kept (simplified: partial payment)
                kept_prob = compute_ptp_kept_probability(borrower, rng)
                if rng.random() < kept_prob:
                    result["ptp_kept"] = True
                    ptp_amount = action.ptp_amount_inr or borrower.outstanding_inr * 0.2
                    if action_type == ActionType.OFFER_OTS:
                        sp = action.settlement_percentage or 0.7
                        ptp_amount = borrower.outstanding_inr * sp
                    result["payment_received_inr"] = min(
                        ptp_amount, borrower.outstanding_inr
                    )
    else:
        # Determine contact outcome for failed attempts
        if action_type in _CALL_ACTIONS:
            outcomes = [ContactOutcome.voicemail, ContactOutcome.no_answer, ContactOutcome.refused]
            weights = [0.3, 0.5, 0.2]
            result["contact_outcome"] = rng.choice(outcomes, p=weights)
        else:
            result["contact_outcome"] = ContactOutcome.no_answer

    # Complaint probability
    comp_prob = compute_complaint_probability(borrower, action, rng)
    result["complaint_filed"] = bool(rng.random() < comp_prob)

    # Sentiment transition
    result["new_sentiment"] = compute_sentiment_transition(
        borrower, action, result, rng
    )

    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_ptp_history_ratio(borrower: BorrowerState) -> float:
    """Return the broken-to-made PTP ratio from implicit state."""
    # We don't store full history in BorrowerState, so estimate from
    # observable signals: if ptp_active but days_since_last_payment is high,
    # that suggests broken history.
    if borrower.days_since_last_payment > 90:
        return 0.7
    elif borrower.days_since_last_payment > 60:
        return 0.5
    elif borrower.days_since_last_payment > 30:
        return 0.3
    return 0.1

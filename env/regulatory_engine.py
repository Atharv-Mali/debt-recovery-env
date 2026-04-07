"""RBI Fair Practices Code — action masking and constraint enforcement."""

from __future__ import annotations

from typing import Dict, List, Any

from env.models import ActionType, BorrowerState, LegalStage, Sentiment


# ── Action categories ────────────────────────────────────────────────────────

CALL_ACTIONS = {
    ActionType.CALL_MORNING,
    ActionType.CALL_AFTERNOON,
    ActionType.CALL_EVENING,
}

SMS_ACTIONS = {
    ActionType.SMS_REMINDER,
    ActionType.SMS_WARNING,
    ActionType.SMS_SETTLEMENT,
}

CONTACT_ACTIONS = CALL_ACTIONS | SMS_ACTIONS

ALL_ACTIONS = list(ActionType)


def get_valid_actions(
    borrower: BorrowerState,
    session_state: Dict[str, Any],
    regulatory_audit_active: bool = False,
) -> List[ActionType]:
    """Return the list of actions the agent is allowed to take for this borrower.

    Implements RBI Fair Practices Code constraints as hard action masks.

    Args:
        borrower: Current state of the target borrower.
        session_state: Dict with keys ``session_calls_remaining``,
            ``session_visits_remaining``.
        regulatory_audit_active: If True, tighter constraints apply (Task 3 shock).

    Returns:
        List of valid ``ActionType`` values the agent may choose from.
    """
    valid: List[ActionType] = []
    legal_stage = LegalStage(borrower.legal_stage)

    # ── Written-off accounts: extremely limited actions ──────────────────
    if legal_stage == LegalStage.written_off:
        return [ActionType.FLAG_WRITEOFF, ActionType.NO_CONTACT]

    # ── Resolved accounts: no further action needed ──────────────────────
    if borrower.is_resolved:
        return [ActionType.NO_CONTACT]

    max_daily_calls = 3
    if regulatory_audit_active:
        max_daily_calls = 1

    calls_remaining = session_state.get("session_calls_remaining", 0)
    visits_remaining = session_state.get("session_visits_remaining", 0)

    for action in ALL_ACTIONS:
        action_type = ActionType(action)

        # ── DNC block: no calls or SMS ───────────────────────────────────
        if borrower.dnc_status and action_type in CONTACT_ACTIONS:
            continue

        # ── Daily call limit ─────────────────────────────────────────────
        if action_type in CALL_ACTIONS:
            if borrower.contact_attempts_today >= max_daily_calls:
                continue
            if calls_remaining <= 0:
                continue

        # ── Complaint cooling: block evening calls and SMS_WARNING ───────
        if borrower.complaint_count >= 2:
            if action_type in {ActionType.CALL_EVENING, ActionType.SMS_WARNING}:
                continue

        # ── ESCALATE_LEGAL only if DPD >= 90 ─────────────────────────────
        if action_type == ActionType.ESCALATE_LEGAL:
            if borrower.dpd < 90:
                continue

        # ── OFFER_OTS only if DPD >= 60 ──────────────────────────────────
        if action_type == ActionType.OFFER_OTS:
            if borrower.dpd < 60:
                continue

        # ── FIELD_VISIT needs remaining budget ───────────────────────────
        if action_type == ActionType.FIELD_VISIT:
            if visits_remaining <= 0:
                continue

        # ── FLAG_WRITEOFF only for written_off (handled above) ───────────
        if action_type == ActionType.FLAG_WRITEOFF:
            continue  # only valid for written_off stage

        valid.append(action_type)

    # Always allow NO_CONTACT
    if ActionType.NO_CONTACT not in valid:
        valid.append(ActionType.NO_CONTACT)

    return valid


def is_action_valid(
    action_type: ActionType,
    borrower: BorrowerState,
    session_state: Dict[str, Any],
    regulatory_audit_active: bool = False,
) -> bool:
    """Check if a specific action is valid for a borrower."""
    valid = get_valid_actions(borrower, session_state, regulatory_audit_active)
    return ActionType(action_type) in valid


def check_violation(
    action_type: ActionType,
    borrower: BorrowerState,
    session_state: Dict[str, Any],
    regulatory_audit_active: bool = False,
) -> Dict[str, Any]:
    """Return details about any regulatory violation triggered.

    Returns:
        Dict with ``violated`` (bool), ``violation_type`` (str), ``severity`` (str).
    """
    at = ActionType(action_type)
    result = {"violated": False, "violation_type": "", "severity": ""}

    # DNC violation
    if borrower.dnc_status and at in CONTACT_ACTIONS:
        return {
            "violated": True,
            "violation_type": "DNC_CONTACT",
            "severity": "critical",
        }

    # Over-contact
    max_calls = 1 if regulatory_audit_active else 3
    if at in CALL_ACTIONS and borrower.contact_attempts_today >= max_calls:
        return {
            "violated": True,
            "violation_type": "EXCESS_CONTACT",
            "severity": "high",
        }

    # Premature legal escalation
    if at == ActionType.ESCALATE_LEGAL and borrower.dpd < 90:
        return {
            "violated": True,
            "violation_type": "PREMATURE_LEGAL",
            "severity": "high",
        }

    # Premature OTS
    if at == ActionType.OFFER_OTS and borrower.dpd < 60:
        return {
            "violated": True,
            "violation_type": "PREMATURE_OTS",
            "severity": "medium",
        }

    return result

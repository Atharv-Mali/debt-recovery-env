"""Tests for the regulatory engine — action masking and constraint enforcement."""

from __future__ import annotations

import asyncio
import pytest

from env.models import (
    ActionType,
    BorrowerState,
    CollectionAction,
    ContactOutcome,
    EmploymentType,
    LegalStage,
    Sentiment,
)
from env.regulatory_engine import get_valid_actions, is_action_valid, check_violation
from env.environment import DebtRecoveryEnv


def _make_borrower(**overrides) -> BorrowerState:
    """Create a test borrower with sensible defaults."""
    defaults = {
        "account_id": "BRW_001",
        "outstanding_inr": 50000.0,
        "dpd": 45,
        "credit_score": 600,
        "employment_type": EmploymentType.salaried,
        "income_band": "mid",
        "city_tier": 2,
        "hardship_flag": False,
        "legal_stage": LegalStage.pre_legal,
        "dnc_status": False,
        "complaint_count": 0,
        "sentiment": Sentiment.cooperative,
        "contact_attempts_today": 0,
        "contact_attempts_week": 0,
        "last_contact_outcome": ContactOutcome.not_attempted,
        "ptp_active": False,
        "ptp_amount_inr": None,
        "ptp_due_days": None,
        "days_since_last_payment": 30,
        "partial_payment_received_inr": 0.0,
        "is_resolved": False,
    }
    defaults.update(overrides)
    return BorrowerState(**defaults)


def _session_state(calls: int = 30, visits: int = 3):
    return {"session_calls_remaining": calls, "session_visits_remaining": visits}


class TestDNCBlocking:
    """Test DNC constraints."""

    def test_dnc_blocks_calls(self):
        """DNC-registered borrower cannot receive call actions."""
        borrower = _make_borrower(dnc_status=True)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.CALL_MORNING not in valid
        assert ActionType.CALL_AFTERNOON not in valid
        assert ActionType.CALL_EVENING not in valid

    def test_dnc_blocks_sms(self):
        """DNC-registered borrower cannot receive SMS actions."""
        borrower = _make_borrower(dnc_status=True)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.SMS_REMINDER not in valid
        assert ActionType.SMS_WARNING not in valid
        assert ActionType.SMS_SETTLEMENT not in valid

    def test_dnc_allows_email_and_whatsapp(self):
        """DNC borrower can still receive emails and WhatsApp."""
        borrower = _make_borrower(dnc_status=True)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.EMAIL_FORMAL in valid
        assert ActionType.EMAIL_EMPATHETIC in valid
        assert ActionType.WHATSAPP_NUDGE in valid

    def test_dnc_violation_detected(self):
        """Calling a DNC borrower is detected as a violation."""
        borrower = _make_borrower(dnc_status=True)
        result = check_violation(
            ActionType.CALL_MORNING, borrower, _session_state()
        )
        assert result["violated"] is True
        assert result["violation_type"] == "DNC_CONTACT"


class TestMaxDailyCalls:
    """Test daily call limits."""

    def test_max_daily_calls(self):
        """4th call attempt in same day is blocked."""
        borrower = _make_borrower(contact_attempts_today=3)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.CALL_MORNING not in valid
        assert ActionType.CALL_AFTERNOON not in valid
        assert ActionType.CALL_EVENING not in valid

    def test_under_limit_allows_calls(self):
        """Under the limit, calls are allowed."""
        borrower = _make_borrower(contact_attempts_today=1)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.CALL_MORNING in valid

    def test_excess_contact_violation(self):
        """Exceeding contact limit is a violation."""
        borrower = _make_borrower(contact_attempts_today=3)
        result = check_violation(
            ActionType.CALL_MORNING, borrower, _session_state()
        )
        assert result["violated"] is True
        assert result["violation_type"] == "EXCESS_CONTACT"


class TestLegalEscalation:
    """Test legal escalation constraints."""

    def test_legal_escalation_threshold(self):
        """ESCALATE_LEGAL is invalid for DPD < 90."""
        borrower = _make_borrower(dpd=60)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.ESCALATE_LEGAL not in valid

    def test_legal_escalation_allowed_high_dpd(self):
        """ESCALATE_LEGAL is valid for DPD >= 90."""
        borrower = _make_borrower(dpd=95)
        valid = get_valid_actions(borrower, _session_state())

        assert ActionType.ESCALATE_LEGAL in valid

    def test_premature_legal_is_violation(self):
        """Escalating with DPD<90 is flagged."""
        borrower = _make_borrower(dpd=45)
        result = check_violation(
            ActionType.ESCALATE_LEGAL, borrower, _session_state()
        )
        assert result["violated"] is True
        assert result["violation_type"] == "PREMATURE_LEGAL"


class TestOTSConstraints:
    """Test OTS offer constraints."""

    def test_ots_requires_dpd_60(self):
        """OFFER_OTS is invalid for DPD < 60."""
        borrower = _make_borrower(dpd=30)
        valid = get_valid_actions(borrower, _session_state())
        assert ActionType.OFFER_OTS not in valid

    def test_ots_allowed_dpd_60_plus(self):
        """OFFER_OTS is valid for DPD >= 60."""
        borrower = _make_borrower(dpd=65)
        valid = get_valid_actions(borrower, _session_state())
        assert ActionType.OFFER_OTS in valid


class TestAuditShock:
    """Test regulatory audit shock in Task 3."""

    def test_audit_shock_reduces_quota(self):
        """Regulatory audit active reduces max daily calls to 1."""
        borrower = _make_borrower(contact_attempts_today=1)
        valid_normal = get_valid_actions(
            borrower, _session_state(), regulatory_audit_active=False
        )
        valid_audit = get_valid_actions(
            borrower, _session_state(), regulatory_audit_active=True
        )

        # Normal: can still call (under 3 limit)
        assert ActionType.CALL_MORNING in valid_normal
        # Audit: blocked (already at 1, limit is 1)
        assert ActionType.CALL_MORNING not in valid_audit

    def test_audit_shock_zero_attempts_allowed(self):
        """With audit active, first call is still allowed."""
        borrower = _make_borrower(contact_attempts_today=0)
        valid = get_valid_actions(
            borrower, _session_state(), regulatory_audit_active=True
        )
        assert ActionType.CALL_MORNING in valid


class TestWrittenOff:
    """Test written-off account constraints."""

    def test_written_off_limited_actions(self):
        """Written-off accounts only allow FLAG_WRITEOFF or NO_CONTACT."""
        borrower = _make_borrower(legal_stage=LegalStage.written_off)
        valid = get_valid_actions(borrower, _session_state())

        assert set(valid) == {ActionType.FLAG_WRITEOFF, ActionType.NO_CONTACT}


class TestFieldVisitBudget:
    """Test field visit budget constraints."""

    def test_field_visit_blocked_when_zero(self):
        """FIELD_VISIT is blocked when session_visits_remaining is 0."""
        borrower = _make_borrower()
        valid = get_valid_actions(borrower, _session_state(visits=0))
        assert ActionType.FIELD_VISIT not in valid

    def test_field_visit_allowed_with_budget(self):
        """FIELD_VISIT is allowed when budget remains."""
        borrower = _make_borrower()
        valid = get_valid_actions(borrower, _session_state(visits=2))
        assert ActionType.FIELD_VISIT in valid


class TestComplaintCooling:
    """Test complaint-based restrictions."""

    def test_high_complaints_block_evening_calls(self):
        """Borrower with 2+ complaints can't receive evening calls."""
        borrower = _make_borrower(complaint_count=2)
        valid = get_valid_actions(borrower, _session_state())
        assert ActionType.CALL_EVENING not in valid
        assert ActionType.SMS_WARNING not in valid

    def test_low_complaints_allow_all(self):
        """Borrower with 0 complaints has no complaint-based blocks."""
        borrower = _make_borrower(complaint_count=0)
        valid = get_valid_actions(borrower, _session_state())
        assert ActionType.CALL_EVENING in valid

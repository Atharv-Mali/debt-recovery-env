"""Pydantic v2 models for the DebtRecoveryEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class EmploymentType(str, Enum):
    salaried = "salaried"
    self_employed = "self_employed"
    daily_wage = "daily_wage"
    unemployed = "unemployed"


class LegalStage(str, Enum):
    pre_legal = "pre_legal"
    notice_sent = "notice_sent"
    sarfaesi = "sarfaesi"
    drt = "drt"
    written_off = "written_off"


class Sentiment(str, Enum):
    cooperative = "cooperative"
    avoidant = "avoidant"
    hostile = "hostile"
    ghost = "ghost"


class ContactOutcome(str, Enum):
    answered = "answered"
    voicemail = "voicemail"
    refused = "refused"
    no_answer = "no_answer"
    not_attempted = "not_attempted"


class ActionType(str, Enum):
    CALL_MORNING = "CALL_MORNING"
    CALL_AFTERNOON = "CALL_AFTERNOON"
    CALL_EVENING = "CALL_EVENING"
    SMS_REMINDER = "SMS_REMINDER"
    SMS_WARNING = "SMS_WARNING"
    SMS_SETTLEMENT = "SMS_SETTLEMENT"
    EMAIL_FORMAL = "EMAIL_FORMAL"
    EMAIL_EMPATHETIC = "EMAIL_EMPATHETIC"
    WHATSAPP_NUDGE = "WHATSAPP_NUDGE"
    FIELD_VISIT = "FIELD_VISIT"
    NO_CONTACT = "NO_CONTACT"
    NEGOTIATE_PTP = "NEGOTIATE_PTP"
    OFFER_RESTRUCTURE = "OFFER_RESTRUCTURE"
    OFFER_OTS = "OFFER_OTS"
    GRANT_DEFERMENT = "GRANT_DEFERMENT"
    ESCALATE_LEGAL = "ESCALATE_LEGAL"
    FLAG_WRITEOFF = "FLAG_WRITEOFF"


# ── Core Models ──────────────────────────────────────────────────────────────

class BorrowerState(BaseModel):
    """Observable state of a single borrower account."""

    model_config = {"use_enum_values": True}

    account_id: str
    outstanding_inr: float
    dpd: int
    credit_score: int
    employment_type: EmploymentType
    income_band: str
    city_tier: int
    hardship_flag: bool
    legal_stage: LegalStage
    dnc_status: bool
    complaint_count: int
    sentiment: Sentiment
    contact_attempts_today: int = 0
    contact_attempts_week: int = 0
    last_contact_outcome: ContactOutcome = ContactOutcome.not_attempted
    ptp_active: bool = False
    ptp_amount_inr: Optional[float] = None
    ptp_due_days: Optional[int] = None
    days_since_last_payment: int = 0
    partial_payment_received_inr: float = 0.0
    is_resolved: bool = False


class PortfolioObservation(BaseModel):
    """Full observation returned to the agent each step."""

    model_config = {"use_enum_values": True}

    accounts: List[BorrowerState]
    session_calls_remaining: int
    session_visits_remaining: int
    current_step: int
    max_steps: int
    regulatory_audit_active: bool = False
    portfolio_recovery_rate: float = 0.0
    task_id: str


class CollectionAction(BaseModel):
    """Action submitted by the agent each step."""

    model_config = {"use_enum_values": True}

    account_id: str
    action_type: ActionType
    settlement_percentage: Optional[float] = Field(
        default=None, ge=0.5, le=1.0, description="For OFFER_OTS: 0.5–1.0"
    )
    deferment_days: Optional[int] = Field(
        default=None, ge=1, le=90, description="For GRANT_DEFERMENT"
    )
    ptp_amount_inr: Optional[float] = Field(
        default=None, ge=0, description="For NEGOTIATE_PTP"
    )


class RewardComponents(BaseModel):
    """Individual reward signal components."""

    recovery_signal: float = 0.0
    contact_quality: float = 0.0
    sentiment_delta: float = 0.0
    compliance_score: float = 0.0
    penalty_complaints: float = 0.0
    penalty_violations: float = 0.0
    time_decay: float = 0.0


class CollectionReward(BaseModel):
    """Composite reward returned each step."""

    total: float
    components: RewardComponents
    episode_done: bool = False
    info: dict = Field(default_factory=dict)

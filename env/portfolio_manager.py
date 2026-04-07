"""Portfolio-level shared constraint and metric tracking."""

from __future__ import annotations

from typing import Dict, List, Any

from env.models import ActionType, BorrowerState


CALL_ACTIONS = {
    ActionType.CALL_MORNING,
    ActionType.CALL_AFTERNOON,
    ActionType.CALL_EVENING,
}


class PortfolioManager:
    """Tracks shared constraints across all accounts in a session.

    Manages:
        - Session call budget: ``num_accounts * 3``
        - Session visit budget: ``max(2, num_accounts // 10)``
        - Per-account payment tracking
        - Portfolio recovery rate
    """

    def __init__(self, accounts: List[BorrowerState]) -> None:
        self.num_accounts = len(accounts)
        self.session_calls_remaining = self.num_accounts * 3
        self.session_visits_remaining = max(2, self.num_accounts // 10)

        self._total_outstanding = sum(a.outstanding_inr for a in accounts)
        self._total_payments: float = 0.0
        self._resolved_accounts: set = set()
        self._payments_per_account: Dict[str, float] = {
            a.account_id: 0.0 for a in accounts
        }
        self._violations: List[Dict[str, Any]] = []
        self._complaints: List[Dict[str, Any]] = []

    @property
    def portfolio_recovery_rate(self) -> float:
        """Fraction of total outstanding that has been recovered."""
        if self._total_outstanding <= 0:
            return 0.0
        return self._total_payments / self._total_outstanding

    @property
    def total_payments(self) -> float:
        return self._total_payments

    @property
    def total_outstanding(self) -> float:
        return self._total_outstanding

    @property
    def mean_outstanding(self) -> float:
        if self.num_accounts <= 0:
            return 0.0
        return self._total_outstanding / self.num_accounts

    @property
    def resolved_count(self) -> int:
        return len(self._resolved_accounts)

    @property
    def violations(self) -> List[Dict[str, Any]]:
        return self._violations

    @property
    def complaints(self) -> List[Dict[str, Any]]:
        return self._complaints

    def get_session_state(self) -> Dict[str, Any]:
        """Return session-level state dict for use by regulatory engine."""
        return {
            "session_calls_remaining": self.session_calls_remaining,
            "session_visits_remaining": self.session_visits_remaining,
        }

    def consume_action(self, action_type: ActionType) -> None:
        """Decrement shared budgets based on the action taken."""
        at = ActionType(action_type)
        if at in CALL_ACTIONS:
            self.session_calls_remaining = max(0, self.session_calls_remaining - 1)
        elif at == ActionType.FIELD_VISIT:
            self.session_visits_remaining = max(0, self.session_visits_remaining - 1)

    def record_payment(self, account_id: str, amount: float) -> None:
        """Record a payment received from a borrower."""
        self._total_payments += amount
        self._payments_per_account[account_id] = (
            self._payments_per_account.get(account_id, 0.0) + amount
        )

    def mark_resolved(self, account_id: str) -> None:
        """Mark an account as resolved."""
        self._resolved_accounts.add(account_id)

    def is_resolved(self, account_id: str) -> bool:
        """Check if an account is resolved."""
        return account_id in self._resolved_accounts

    def record_violation(self, step: int, account_id: str, violation_type: str) -> None:
        """Log a regulatory violation."""
        self._violations.append({
            "step": step,
            "account_id": account_id,
            "violation_type": violation_type,
        })

    def record_complaint(self, step: int, account_id: str) -> None:
        """Log a complaint."""
        self._complaints.append({
            "step": step,
            "account_id": account_id,
        })

    def apply_audit_shock(self) -> None:
        """Apply regulatory audit shock — cut remaining call quota by 40%."""
        self.session_calls_remaining = int(self.session_calls_remaining * 0.6)

    def get_metrics(self) -> Dict[str, Any]:
        """Return all portfolio metrics."""
        return {
            "portfolio_recovery_rate": self.portfolio_recovery_rate,
            "total_payments": self._total_payments,
            "total_outstanding": self._total_outstanding,
            "resolved_accounts": len(self._resolved_accounts),
            "total_accounts": self.num_accounts,
            "session_calls_remaining": self.session_calls_remaining,
            "session_visits_remaining": self.session_visits_remaining,
            "violations_count": len(self._violations),
            "complaints_count": len(self._complaints),
            "payments_per_account": dict(self._payments_per_account),
        }

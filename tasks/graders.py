"""Deterministic grading functions for all three tasks.

All graders return a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List


def grade_task1(episode_log: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
    """Grade Task 1 - single cooperative borrower."""
    if not episode_log:
        return 0.0

    total_outstanding = 0.0
    total_payments = 0.0
    contact_established = False
    complaints = 0
    violations = 0

    for entry in episode_log:
        contact_result = entry.get("contact_result", {})
        borrower_before = entry.get("borrower_before", {})

        if total_outstanding == 0.0:
            total_outstanding = borrower_before.get("outstanding_inr", 0.0)

        if contact_result.get("answered", False):
            contact_established = True

        total_payments += contact_result.get("payment_received_inr", 0.0)

        if contact_result.get("complaint_filed", False):
            complaints += 1

        if entry.get("violation", {}).get("violated", False):
            violations += 1

    recovery_score = min(1.0, total_payments / max(total_outstanding, 1.0))
    contact_score = 1.0 if contact_established else 0.0
    compliance_score = 1.0 if (complaints == 0 and violations == 0) else 0.0

    score = 0.5 * recovery_score + 0.3 * contact_score + 0.2 * compliance_score
    return round(_clamp(score), 4)


def grade_task2(episode_log: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
    """Grade Task 2 - mixed portfolio of 10 accounts."""
    if not episode_log:
        return 0.0

    metrics = final_state.get("metrics", {})
    num_accounts = metrics.get("total_accounts", 10)
    total_outstanding = metrics.get("total_outstanding", 0.0)
    total_payments = metrics.get("total_payments", 0.0)
    resolved = metrics.get("resolved_accounts", 0)
    complaints_count = metrics.get("complaints_count", 0)

    portfolio_recovery = min(1.0, total_payments / max(total_outstanding, 1.0))
    compliance = max(0.0, 1.0 - (complaints_count / max(num_accounts, 1)))
    resolution_rate = resolved / max(num_accounts, 1)

    score = 0.5 * portfolio_recovery + 0.3 * compliance + 0.2 * resolution_rate
    return round(_clamp(score), 4)


def grade_task3(episode_log: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
    """Grade Task 3 - adversarial portfolio with a regulatory shock."""
    if not episode_log:
        return 0.0

    metrics = final_state.get("metrics", {})
    num_accounts = metrics.get("total_accounts", 25)
    total_outstanding = metrics.get("total_outstanding", 0.0)
    total_payments = metrics.get("total_payments", 0.0)
    resolved = metrics.get("resolved_accounts", 0)
    violations_count = metrics.get("violations_count", 0)

    unnecessary_escalations = 0
    post_audit_steps = 0
    post_audit_clean_steps = 0

    for entry in episode_log:
        action = entry.get("action", {})
        borrower_before = entry.get("borrower_before", {})
        violation = entry.get("violation", {})

        if action.get("action_type") == "ESCALATE_LEGAL" and borrower_before.get("dpd", 0) < 90:
            unnecessary_escalations += 1

        if entry.get("step", -1) >= 30:
            post_audit_steps += 1
            if not violation.get("violated", False):
                post_audit_clean_steps += 1

    portfolio_recovery = min(1.0, total_payments / max(total_outstanding, 1.0))
    compliance_component = max(0.0, 1.0 - (violations_count / max(num_accounts, 1)))
    legal_efficiency = max(0.0, 1.0 - (unnecessary_escalations / max(num_accounts, 1)))
    shock_adaptation = (
        post_audit_clean_steps / max(post_audit_steps, 1) if post_audit_steps else 1.0
    )
    resolution_rate = resolved / max(num_accounts, 1)

    raw_score = (
        0.35 * portfolio_recovery
        + 0.20 * compliance_component
        + 0.20 * legal_efficiency
        + 0.15 * shock_adaptation
        + 0.10 * resolution_rate
    )
    compliance_multiplier = 1.0 if violations_count == 0 else max(0.3, 1.0 - violations_count * 0.1)
    return round(_clamp(raw_score * compliance_multiplier), 4)


def grade(
    task_id: str,
    episode_log: List[Dict[str, Any]],
    final_state: Dict[str, Any],
) -> float:
    """Route to the correct grader based on task_id."""
    graders = {
        "task1_single_cooperative": grade_task1,
        "task2_portfolio_mixed": grade_task2,
        "task3_portfolio_adversarial": grade_task3,
    }
    grader = graders.get(task_id)
    if grader is None:
        raise ValueError(f"No grader found for task: {task_id}")

    return grader(episode_log, final_state)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))

"""Tests for the grading functions."""

from __future__ import annotations

import pytest

from tasks.graders import grade_task1, grade_task2, grade_task3, grade


class TestGradeTask1:
    """Test Task 1 grader."""

    def test_grade_task1_perfect(self):
        """Full payment with contact and no violations → near 1.0."""
        episode_log = [
            {
                "contact_result": {
                    "answered": True,
                    "payment_received_inr": 45000.0,
                    "complaint_filed": False,
                },
                "borrower_before": {"outstanding_inr": 45000.0},
                "violation": {"violated": False},
            }
        ]
        final_state = {}
        score = grade_task1(episode_log, final_state)
        assert 0.9 <= score < 1.0

    def test_grade_task1_zero(self):
        """No actions at all → score 0.0."""
        score = grade_task1([], {})
        assert 0.0 < score < 0.01

    def test_grade_task1_partial(self):
        """Partial payment with contact → mid-range score."""
        episode_log = [
            {
                "contact_result": {
                    "answered": True,
                    "payment_received_inr": 10000.0,
                    "complaint_filed": False,
                },
                "borrower_before": {"outstanding_inr": 45000.0},
                "violation": {"violated": False},
            }
        ]
        score = grade_task1(episode_log, {})
        assert 0.3 <= score <= 0.8

    def test_grade_task1_with_violation(self):
        """Payment but with violations → compliance drops to 0."""
        episode_log = [
            {
                "contact_result": {
                    "answered": True,
                    "payment_received_inr": 45000.0,
                    "complaint_filed": False,
                },
                "borrower_before": {"outstanding_inr": 45000.0},
                "violation": {"violated": True, "violation_type": "DNC_CONTACT"},
            }
        ]
        score = grade_task1(episode_log, {})
        # Should be lower due to compliance=0
        assert score <= 0.85


class TestGradeTask2:
    """Test Task 2 grader."""

    def test_grade_task2_uses_metrics(self):
        """Grader uses final_state metrics."""
        final_state = {
            "metrics": {
                "total_accounts": 10,
                "total_outstanding": 100000.0,
                "total_payments": 50000.0,
                "resolved_accounts": 5,
                "complaints_count": 0,
            }
        }
        score = grade_task2([{"dummy": True}], final_state)
        assert 0.0 < score < 1.0
        assert score > 0.3  # 50% recovery, full compliance, 50% resolution


class TestGradeTask3:
    """Test Task 3 grader with violations."""

    def test_grade_task3_violation_reduces(self):
        """Violations reduce score through compliance multiplier."""
        final_state_clean = {
            "metrics": {
                "total_accounts": 25,
                "total_outstanding": 1000000.0,
                "total_payments": 300000.0,
                "violations_count": 0,
            }
        }
        final_state_dirty = {
            "metrics": {
                "total_accounts": 25,
                "total_outstanding": 1000000.0,
                "total_payments": 300000.0,
                "violations_count": 3,
            }
        }
        log = [
            {
                "step": i,
                "action": {"action_type": "CALL_MORNING"},
                "borrower_before": {"dpd": 100},
                "violation": {"violated": False},
            }
            for i in range(60)
        ]
        score_clean = grade_task3(log, final_state_clean)
        score_dirty = grade_task3(log, final_state_dirty)
        assert score_clean > score_dirty


class TestAllGradersBounded:
    """Test that all graders return values strictly inside (0.0, 1.0)."""

    def test_all_graders_bounded(self):
        for grader_fn in [grade_task1, grade_task2, grade_task3]:
            # Empty
            score = grader_fn([], {})
            assert 0.0 < score < 1.0

            # Minimal
            log = [
                {
                    "step": 0,
                    "contact_result": {
                        "answered": False,
                        "payment_received_inr": 0.0,
                        "complaint_filed": False,
                    },
                    "borrower_before": {"outstanding_inr": 10000.0, "dpd": 30},
                    "action": {"action_type": "NO_CONTACT"},
                    "violation": {"violated": False},
                }
            ]
            state = {
                "metrics": {
                    "total_accounts": 10,
                    "total_outstanding": 100000.0,
                    "total_payments": 0.0,
                    "resolved_accounts": 0,
                    "complaints_count": 0,
                    "violations_count": 0,
                }
            }
            score = grader_fn(log, state)
            assert 0.0 < score < 1.0

    def test_grade_router(self):
        """Test the grade() routing function."""
        log = []
        state = {}
        for task_id in [
            "task1_single_cooperative",
            "task2_portfolio_mixed",
            "task3_portfolio_adversarial",
        ]:
            score = grade(task_id, log, state)
            assert 0.0 < score < 1.0

    def test_grade_invalid_task(self):
        with pytest.raises(ValueError):
            grade("invalid_task", [], {})

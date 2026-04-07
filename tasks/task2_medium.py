"""Task 2 — Mixed portfolio of 10 accounts, 30 steps.

Profile composition:
  4 cooperative (DPD 15–45)
  3 avoidant (DPD 45–75)
  2 hardship flagged (DPD 60–90, daily_wage)
  1 hostile with 1 existing complaint (DPD 75–90)
"""

TASK2_CONFIG = {
    "task_id": "task2_portfolio_mixed",
    "num_accounts": 10,
    "max_steps": 30,
    "difficulty": "medium",
    "description": "10 mixed accounts. Balance recovery, compliance, sentiment.",
    "profile_composition": [
        {"count": 4, "sentiments": ["cooperative"], "dpd_range": (15, 45)},
        {"count": 3, "sentiments": ["avoidant"], "dpd_range": (45, 75)},
        {
            "count": 2,
            "sentiments": ["cooperative", "avoidant"],
            "dpd_range": (60, 90),
            "hardship": True,
            "employments": ["daily_wage"],
        },
        {
            "count": 1,
            "sentiments": ["hostile"],
            "dpd_range": (75, 90),
            "min_complaints": 1,
        },
    ],
}

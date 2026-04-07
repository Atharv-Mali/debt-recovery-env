"""Task 3 — Adversarial 25-account portfolio, 60 steps.

Profile composition:
  6 cooperative, 7 avoidant, 4 hardship, 4 hostile, 4 ghost
  regulatory_audit_active = True at step 30 (reduces calls per day to 1)
"""

TASK3_CONFIG = {
    "task_id": "task3_portfolio_adversarial",
    "num_accounts": 25,
    "max_steps": 60,
    "difficulty": "hard",
    "description": "25 adversarial accounts with mid-episode regulatory shock.",
    "regulatory_audit_step": 30,
    "profile_composition": [
        {"count": 6, "sentiments": ["cooperative"], "dpd_range": (15, 60)},
        {"count": 7, "sentiments": ["avoidant"], "dpd_range": (30, 90)},
        {
            "count": 4,
            "sentiments": ["cooperative", "avoidant"],
            "dpd_range": (60, 100),
            "hardship": True,
        },
        {"count": 4, "sentiments": ["hostile"], "dpd_range": (60, 120)},
        {"count": 4, "sentiments": ["ghost"], "dpd_range": (120, 180)},
    ],
}

"""Task 1 — Single cooperative borrower, 10 steps.

Profile: DPD 15–30, sentiment=cooperative, salaried, no hardship,
pre-legal, answer probability ~0.80.
"""

TASK1_CONFIG = {
    "task_id": "task1_single_cooperative",
    "num_accounts": 1,
    "max_steps": 10,
    "difficulty": "easy",
    "description": "Single cooperative borrower. Establish contact and secure PTP.",
    "profile_filter": {
        "dpd_range": (15, 30),
        "sentiments": ["cooperative"],
        "employments": ["salaried"],
        "hardship": False,
        "legal_stages": ["pre_legal"],
    },
}

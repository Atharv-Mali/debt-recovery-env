"""Task definitions package."""

from tasks.task1_easy import TASK1_CONFIG
from tasks.task2_medium import TASK2_CONFIG
from tasks.task3_hard import TASK3_CONFIG
from tasks.graders import grade_task1, grade_task2, grade_task3, grade

__all__ = [
    "TASK1_CONFIG",
    "TASK2_CONFIG",
    "TASK3_CONFIG",
    "grade_task1",
    "grade_task2",
    "grade_task3",
    "grade",
]

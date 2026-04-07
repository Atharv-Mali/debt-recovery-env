"""DebtRecoveryEnv — main environment class (OpenEnv compliant)."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.models import (
    ActionType,
    BorrowerState,
    CollectionAction,
    CollectionReward,
    ContactOutcome,
    EmploymentType,
    LegalStage,
    PortfolioObservation,
    RewardComponents,
    Sentiment,
)
from env.borrower_simulator import simulate_step
from env.regulatory_engine import check_violation, get_valid_actions
from env.reward_engine import compute_step_reward, total_reward, normalize_episode_reward
from env.portfolio_manager import PortfolioManager


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Task configuration ───────────────────────────────────────────────────────

TASK_CONFIG = {
    "task1_single_cooperative": {
        "num_accounts": 1,
        "max_steps": 10,
        "regulatory_audit_step": None,
        "profile_filter": {
            "dpd_range": (15, 30),
            "sentiments": ["cooperative"],
            "employments": ["salaried"],
            "hardship": False,
            "legal_stages": ["pre_legal"],
        },
    },
    "task2_portfolio_mixed": {
        "num_accounts": 10,
        "max_steps": 30,
        "regulatory_audit_step": None,
        "profile_composition": [
            {"count": 4, "sentiments": ["cooperative"], "dpd_range": (15, 45)},
            {"count": 3, "sentiments": ["avoidant"], "dpd_range": (45, 75)},
            {"count": 2, "sentiments": ["cooperative", "avoidant"], "dpd_range": (60, 90),
             "hardship": True, "employments": ["daily_wage"]},
            {"count": 1, "sentiments": ["hostile"], "dpd_range": (75, 90),
             "min_complaints": 1},
        ],
    },
    "task3_portfolio_adversarial": {
        "num_accounts": 25,
        "max_steps": 60,
        "regulatory_audit_step": 30,
        "profile_composition": [
            {"count": 6, "sentiments": ["cooperative"], "dpd_range": (15, 60)},
            {"count": 7, "sentiments": ["avoidant"], "dpd_range": (30, 90)},
            {"count": 4, "sentiments": ["cooperative", "avoidant"], "dpd_range": (60, 100), "hardship": True},
            {"count": 4, "sentiments": ["hostile"], "dpd_range": (60, 120)},
            {"count": 4, "sentiments": ["ghost"], "dpd_range": (120, 180)},
        ],
    },
}


class DebtRecoveryEnv:
    """OpenEnv-compliant RL environment for Indian NBFC loan collections."""

    def __init__(self, task_id: str, seed: int = 42) -> None:
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASK_CONFIG.keys())}")

        self.task_id = task_id
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.config = TASK_CONFIG[task_id]
        self.max_steps = self.config["max_steps"]
        self.regulatory_audit_step = self.config.get("regulatory_audit_step")

        self.accounts: List[BorrowerState] = []
        self.portfolio: Optional[PortfolioManager] = None
        self.current_step = 0
        self.episode_log: List[Dict[str, Any]] = []
        self.cumulative_reward = 0.0
        self.regulatory_audit_active = False
        self._all_profiles: List[Dict[str, Any]] = []
        self._done = False

    async def reset(self) -> PortfolioObservation:
        """Reset the environment and return the initial observation."""
        self.rng = np.random.RandomState(self.seed)
        self.current_step = 0
        self.episode_log = []
        self.cumulative_reward = 0.0
        self.regulatory_audit_active = False
        self._done = False

        # Load all profiles
        profiles_path = DATA_DIR / "borrower_profiles.json"
        with open(profiles_path, "r", encoding="utf-8") as f:
            self._all_profiles = json.load(f)

        # Sample accounts based on task config
        self.accounts = self._sample_accounts()
        self.portfolio = PortfolioManager(self.accounts)

        return self._build_observation()

    async def step(
        self, action: CollectionAction
    ) -> Tuple[PortfolioObservation, CollectionReward]:
        """Execute one step in the environment.

        Args:
            action: The ``CollectionAction`` chosen by the agent.

        Returns:
            Tuple of (new observation, reward).
        """
        if self._done:
            return self._build_observation(), CollectionReward(
                total=0.0,
                components=RewardComponents(),
                episode_done=True,
                info={"message": "Episode already complete"},
            )

        # Find target account
        target_idx = self._find_account(action.account_id)
        if target_idx is None:
            # Invalid account — return penalty
            self.current_step += 1
            done = self.current_step >= self.max_steps
            self._done = done
            return self._build_observation(), CollectionReward(
                total=0.0,
                components=RewardComponents(),
                episode_done=done,
                info={"error": f"Account {action.account_id} not found"},
            )

        borrower_before = deepcopy(self.accounts[target_idx])
        action_type = ActionType(action.action_type)

        # Check regulatory audit shock
        if (
            self.regulatory_audit_step is not None
            and self.current_step >= self.regulatory_audit_step
            and not self.regulatory_audit_active
        ):
            self.regulatory_audit_active = True
            self.portfolio.apply_audit_shock()

        # 1. Check for violations
        session_state = self.portfolio.get_session_state()
        violation_info = check_violation(
            action_type, borrower_before, session_state, self.regulatory_audit_active
        )

        # 2. Run borrower simulator
        contact_result = simulate_step(borrower_before, action, self.rng)

        # Merge violation info
        if violation_info["violated"]:
            contact_result["violation_triggered"] = True
            self.portfolio.record_violation(
                self.current_step, action.account_id, violation_info["violation_type"]
            )

        # 3. Update borrower state
        borrower_after = self._update_borrower(
            self.accounts[target_idx], action, contact_result
        )
        self.accounts[target_idx] = borrower_after

        # 4. Update portfolio manager
        self.portfolio.consume_action(action_type)

        if contact_result["payment_received_inr"] > 0:
            self.portfolio.record_payment(
                action.account_id, contact_result["payment_received_inr"]
            )

        if contact_result.get("complaint_filed", False):
            self.portfolio.record_complaint(self.current_step, action.account_id)

        # Check if account is fully resolved
        if borrower_after.outstanding_inr <= 0 or borrower_after.is_resolved:
            self.portfolio.mark_resolved(action.account_id)

        # 5. Compute reward
        reward_components = compute_step_reward(
            borrower_before=borrower_before,
            borrower_after=borrower_after,
            action=action,
            contact_result=contact_result,
            portfolio_mean_outstanding=self.portfolio.mean_outstanding,
            step=self.current_step,
            max_steps=self.max_steps,
        )
        step_total = total_reward(reward_components)
        self.cumulative_reward += step_total

        # 6. Log step
        self.episode_log.append({
            "step": self.current_step,
            "action": action.model_dump(),
            "account_id": action.account_id,
            "contact_result": {
                k: v.value if hasattr(v, "value") else v
                for k, v in contact_result.items()
            },
            "reward_components": reward_components.model_dump(),
            "reward_total": step_total,
            "borrower_before": borrower_before.model_dump(),
            "borrower_after": borrower_after.model_dump(),
            "violation": violation_info,
        })

        # 7. Advance step and check done
        self.current_step += 1
        all_resolved = all(a.is_resolved for a in self.accounts)
        done = self.current_step >= self.max_steps or all_resolved
        self._done = done

        # 8. Build and return
        obs = self._build_observation()
        normalized = normalize_episode_reward(self.cumulative_reward, self.max_steps)

        reward = CollectionReward(
            total=round(normalized if done else step_total, 6),
            components=reward_components,
            episode_done=done,
            info={
                "step": self.current_step,
                "cumulative_reward": round(self.cumulative_reward, 6),
                "accounts_resolved": self.portfolio.resolved_count,
                "violations": len(self.portfolio.violations),
                "complaints": len(self.portfolio.complaints),
                "portfolio_recovery_rate": round(
                    self.portfolio.portfolio_recovery_rate, 6
                ),
                "contact_result": {
                    k: v.value if hasattr(v, "value") else v
                    for k, v in contact_result.items()
                },
            },
        )

        return obs, reward

    async def state(self) -> Dict[str, Any]:
        """Return full internal state including episode log."""
        return {
            "current_observation": self._build_observation().model_dump(),
            "episode_log": self.episode_log,
            "metrics": self.portfolio.get_metrics() if self.portfolio else {},
            "step_count": self.current_step,
            "cumulative_reward": self.cumulative_reward,
            "done": self._done,
            "task_id": self.task_id,
            "seed": self.seed,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_observation(self) -> PortfolioObservation:
        """Build the current portfolio observation."""
        return PortfolioObservation(
            accounts=self.accounts,
            session_calls_remaining=self.portfolio.session_calls_remaining
            if self.portfolio
            else 0,
            session_visits_remaining=self.portfolio.session_visits_remaining
            if self.portfolio
            else 0,
            current_step=self.current_step,
            max_steps=self.max_steps,
            regulatory_audit_active=self.regulatory_audit_active,
            portfolio_recovery_rate=self.portfolio.portfolio_recovery_rate
            if self.portfolio
            else 0.0,
            task_id=self.task_id,
        )

    def _find_account(self, account_id: str) -> Optional[int]:
        """Find account index by ID."""
        for i, a in enumerate(self.accounts):
            if a.account_id == account_id:
                return i
        return None

    def _sample_accounts(self) -> List[BorrowerState]:
        """Sample borrower accounts based on task configuration."""
        profiles = self._all_profiles
        accounts: List[BorrowerState] = []

        if "profile_filter" in self.config:
            # Simple filter mode (Task 1)
            filt = self.config["profile_filter"]
            candidates = self._filter_profiles(profiles, filt)
            self.rng.shuffle(candidates)
            selected = candidates[: self.config["num_accounts"]]
            for p in selected:
                accounts.append(self._profile_to_state(p))

        elif "profile_composition" in self.config:
            # Composition mode (Task 2, 3)
            for spec in self.config["profile_composition"]:
                count = spec["count"]
                candidates = self._filter_profiles(profiles, spec)
                self.rng.shuffle(candidates)

                # Ensure we have enough candidates
                if len(candidates) < count:
                    # Allow reuse with different IDs
                    while len(candidates) < count:
                        extra = profiles[self.rng.randint(0, len(profiles))]
                        candidates.append(extra)

                for p in candidates[:count]:
                    state = self._profile_to_state(p)

                    # Override DPD to match spec range
                    dpd_min, dpd_max = spec.get("dpd_range", (5, 180))
                    state.dpd = int(self.rng.randint(dpd_min, dpd_max + 1))

                    # Override sentiment if specified
                    if "sentiments" in spec:
                        state.sentiment = self.rng.choice(spec["sentiments"])

                    # Override hardship
                    if spec.get("hardship"):
                        state.hardship_flag = True

                    # Override employment
                    if "employments" in spec:
                        state.employment_type = self.rng.choice(spec["employments"])

                    # Override complaints
                    if "min_complaints" in spec:
                        state.complaint_count = max(
                            state.complaint_count, spec["min_complaints"]
                        )

                    accounts.append(state)

        # Assign unique IDs
        for i, acc in enumerate(accounts):
            acc.account_id = f"BRW_{i + 1:03d}"

        return accounts

    def _filter_profiles(
        self, profiles: List[Dict[str, Any]], filt: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter profiles based on criteria."""
        result = []
        for p in profiles:
            # DPD range
            if "dpd_range" in filt:
                lo, hi = filt["dpd_range"]
                if not (lo <= p["dpd"] <= hi):
                    continue

            # Sentiments
            if "sentiments" in filt:
                if p["sentiment"] not in filt["sentiments"]:
                    continue

            # Employments
            if "employments" in filt:
                if p["employment_type"] not in filt["employments"]:
                    continue

            # Hardship
            if "hardship" in filt:
                if p["hardship_flag"] != filt["hardship"]:
                    continue

            # Legal stages
            if "legal_stages" in filt:
                if p["legal_stage"] not in filt["legal_stages"]:
                    continue

            result.append(p)
        return result

    def _profile_to_state(self, profile: Dict[str, Any]) -> BorrowerState:
        """Convert a raw JSON profile to a BorrowerState."""
        return BorrowerState(
            account_id=profile["id"],
            outstanding_inr=float(profile["outstanding_inr"]),
            dpd=profile["dpd"],
            credit_score=profile["credit_score"],
            employment_type=profile["employment_type"],
            income_band=profile["income_band"],
            city_tier=profile["city_tier"],
            hardship_flag=profile["hardship_flag"],
            legal_stage=profile["legal_stage"],
            dnc_status=profile.get("dnc_registered", False),
            complaint_count=profile.get("complaint_count", 0),
            sentiment=profile["sentiment"],
            contact_attempts_today=0,
            contact_attempts_week=0,
            last_contact_outcome=ContactOutcome.not_attempted,
            ptp_active=False,
            ptp_amount_inr=None,
            ptp_due_days=None,
            days_since_last_payment=profile.get("days_since_last_payment", 0),
            partial_payment_received_inr=0.0,
            is_resolved=False,
        )

    def _update_borrower(
        self,
        borrower: BorrowerState,
        action: CollectionAction,
        contact_result: Dict[str, Any],
    ) -> BorrowerState:
        """Apply contact results to update borrower state in-place and return it."""
        action_type = ActionType(action.action_type)

        # Update contact counts for call/contact actions
        call_actions = {
            ActionType.CALL_MORNING,
            ActionType.CALL_AFTERNOON,
            ActionType.CALL_EVENING,
        }
        contact_actions = call_actions | {
            ActionType.SMS_REMINDER,
            ActionType.SMS_WARNING,
            ActionType.SMS_SETTLEMENT,
            ActionType.EMAIL_FORMAL,
            ActionType.EMAIL_EMPATHETIC,
            ActionType.WHATSAPP_NUDGE,
            ActionType.FIELD_VISIT,
        }

        if action_type in contact_actions:
            borrower.contact_attempts_today += 1
            borrower.contact_attempts_week += 1

        # Update last contact outcome
        outcome = contact_result.get("contact_outcome", ContactOutcome.not_attempted)
        if isinstance(outcome, str):
            outcome = ContactOutcome(outcome)
        borrower.last_contact_outcome = outcome

        # Update sentiment
        new_sentiment = contact_result.get("new_sentiment", borrower.sentiment)
        if isinstance(new_sentiment, str):
            new_sentiment = Sentiment(new_sentiment)
        borrower.sentiment = new_sentiment

        # Update complaint count
        if contact_result.get("complaint_filed", False):
            borrower.complaint_count += 1

        # Handle PTP
        if contact_result.get("ptp_made", False):
            borrower.ptp_active = True
            borrower.ptp_amount_inr = action.ptp_amount_inr or borrower.outstanding_inr * 0.2
            borrower.ptp_due_days = 7

        # Handle payment
        payment = contact_result.get("payment_received_inr", 0.0)
        if payment > 0:
            borrower.partial_payment_received_inr += payment
            borrower.outstanding_inr = max(0.0, borrower.outstanding_inr - payment)
            borrower.days_since_last_payment = 0

            if borrower.outstanding_inr <= 0:
                borrower.is_resolved = True
                borrower.ptp_active = False

            if contact_result.get("ptp_kept", False):
                borrower.ptp_active = False

        # Handle legal escalation
        if action_type == ActionType.ESCALATE_LEGAL:
            stage_order = [
                LegalStage.pre_legal,
                LegalStage.notice_sent,
                LegalStage.sarfaesi,
                LegalStage.drt,
            ]
            current_stage = LegalStage(borrower.legal_stage)
            if current_stage in stage_order:
                idx = stage_order.index(current_stage)
                if idx + 1 < len(stage_order):
                    borrower.legal_stage = stage_order[idx + 1]

        # Handle write-off
        if action_type == ActionType.FLAG_WRITEOFF:
            borrower.legal_stage = LegalStage.written_off
            borrower.is_resolved = True

        # Handle deferment
        if action_type == ActionType.GRANT_DEFERMENT:
            borrower.ptp_due_days = action.deferment_days or 30

        return borrower

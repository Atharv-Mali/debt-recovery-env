"""Baseline LLM agent using OpenAI API client."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from baseline.prompt_templates import SYSTEM_PROMPT, ACTION_PROMPT_TEMPLATE


class BaselineAgent:
    """GPT-4o-mini baseline agent for the DebtRecoveryEnv.

    Uses the OpenAI Python client to generate collection actions
    based on portfolio observations.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: list = []

    def reset(self) -> None:
        """Reset conversation history for a new episode."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def choose_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Choose an action given a portfolio observation.

        Args:
            observation: The PortfolioObservation as a dict.

        Returns:
            A dict representing a CollectionAction.
        """
        # Simplify observation for token efficiency
        simplified = self._simplify_observation(observation)
        obs_json = json.dumps(simplified, indent=2)

        prompt = ACTION_PROMPT_TEMPLATE.format(observation_json=obs_json)
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.3,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            self.conversation_history.append(
                {"role": "assistant", "content": content}
            )

            # Parse JSON from response
            action = self._parse_action(content)

            # Keep conversation short to avoid token bloat
            if len(self.conversation_history) > 10:
                self.conversation_history = (
                    self.conversation_history[:1] + self.conversation_history[-6:]
                )

            return action

        except Exception as e:
            print(f"  [Agent Error] {e}")
            # Fallback: pick first unresolved account and do NO_CONTACT
            return self._fallback_action(observation)

    def _simplify_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove verbose fields to reduce token count."""
        simplified = {
            "task_id": obs.get("task_id"),
            "current_step": obs.get("current_step"),
            "max_steps": obs.get("max_steps"),
            "session_calls_remaining": obs.get("session_calls_remaining"),
            "session_visits_remaining": obs.get("session_visits_remaining"),
            "regulatory_audit_active": obs.get("regulatory_audit_active"),
            "portfolio_recovery_rate": obs.get("portfolio_recovery_rate"),
            "accounts": [],
        }
        for acc in obs.get("accounts", []):
            simplified["accounts"].append({
                "account_id": acc.get("account_id"),
                "outstanding_inr": acc.get("outstanding_inr"),
                "dpd": acc.get("dpd"),
                "sentiment": acc.get("sentiment"),
                "employment_type": acc.get("employment_type"),
                "hardship_flag": acc.get("hardship_flag"),
                "dnc_status": acc.get("dnc_status"),
                "complaint_count": acc.get("complaint_count"),
                "contact_attempts_today": acc.get("contact_attempts_today"),
                "legal_stage": acc.get("legal_stage"),
                "ptp_active": acc.get("ptp_active"),
                "is_resolved": acc.get("is_resolved"),
                "days_since_last_payment": acc.get("days_since_last_payment"),
            })
        return simplified

    def _parse_action(self, content: str) -> Dict[str, Any]:
        """Parse a JSON action from LLM output, handling edge cases."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding any JSON object
        json_match = re.search(r"\{[^{}]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse action from: {content[:200]}")

    def _fallback_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a safe fallback action."""
        accounts = observation.get("accounts", [])
        for acc in accounts:
            if not acc.get("is_resolved", False):
                return {
                    "account_id": acc["account_id"],
                    "action_type": "NO_CONTACT",
                }
        # All resolved
        return {
            "account_id": accounts[0]["account_id"] if accounts else "BRW_001",
            "action_type": "NO_CONTACT",
        }

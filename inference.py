"""Submission inference entrypoint for DebtRecovery ENV.

This script launches the local environment server, runs one seeded episode per
task, and emits structured stdout logs using the required START/STEP/END tags.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from env.models import BorrowerState
from env.regulatory_engine import get_valid_actions


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

TASKS: List[Tuple[str, int]] = [
    ("task1_single_cooperative", 42),
    ("task2_portfolio_mixed", 1001),
    ("task3_portfolio_adversarial", 9999),
]

SYSTEM_PROMPT = """You are an RBI-compliant Indian NBFC collections agent.
Return exactly one JSON object describing the best next CollectionAction.

Rules:
- Use only the provided account_id.
- Use only one of the listed valid_actions.
- Respect DNC, complaint, and legal-escalation constraints.
- Prefer empathetic actions for hardship borrowers.
- For NEGOTIATE_PTP, include ptp_amount_inr.
- For OFFER_OTS, include settlement_percentage between 0.5 and 1.0.
- For GRANT_DEFERMENT, include deferment_days between 1 and 90.

Respond with JSON only. No markdown or explanation.
"""


def emit(tag: str, payload: Dict[str, Any]) -> None:
    print(
        f"[{tag}] {json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}",
        flush=True,
    )


def ensure_token() -> None:
    if HF_TOKEN:
        return
    raise RuntimeError(
        "HF_TOKEN is required. Set HF_TOKEN in the environment before running inference.py."
    )


def parse_json_object(content: str) -> Dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not parse JSON action from model output: {content[:200]}"
        )
    return json.loads(match.group(0))


def launch_server() -> Optional[subprocess.Popen[Any]]:
    with httpx.Client(timeout=2.0) as client:
        try:
            if client.get(f"{ENV_BASE_URL}/health").status_code == 200:
                return None
        except Exception:
            pass

    if not ENV_BASE_URL.startswith("http://127.0.0.1") and not ENV_BASE_URL.startswith(
        "http://localhost"
    ):
        raise RuntimeError(
            f"ENV_BASE_URL is unreachable: {ENV_BASE_URL}. "
            "Point ENV_BASE_URL to a healthy environment or leave it unset to launch the local server."
        )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "7860",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 30
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                if client.get(f"{ENV_BASE_URL}/health").status_code == 200:
                    return process
            except Exception:
                time.sleep(0.5)

    process.terminate()
    raise RuntimeError(
        "Timed out waiting for the local environment server to become healthy."
    )


def cleanup_server(process: Optional[subprocess.Popen[Any]]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def account_priority(account: Dict[str, Any]) -> float:
    if account.get("is_resolved", False):
        return float("-inf")

    score = 0.0
    score += float(account.get("dpd", 0)) * 1.8
    score += min(float(account.get("outstanding_inr", 0.0)) / 10000.0, 25.0)
    score += 12.0 if account.get("ptp_active", False) else 0.0
    score += 8.0 if account.get("hardship_flag", False) else 0.0
    score += 6.0 if account.get("sentiment") == "ghost" else 0.0
    score += 4.0 if account.get("sentiment") == "avoidant" else 0.0
    score -= 6.0 if account.get("complaint_count", 0) >= 2 else 0.0
    return score


def choose_account(observation: Dict[str, Any]) -> Dict[str, Any]:
    accounts = observation.get("accounts", [])
    unresolved = [
        account for account in accounts if not account.get("is_resolved", False)
    ]
    if not unresolved:
        return accounts[0]
    return max(unresolved, key=account_priority)


def valid_actions_for(account: Dict[str, Any], observation: Dict[str, Any]) -> List[str]:
    borrower = BorrowerState(**account)
    session_state = {
        "session_calls_remaining": observation.get("session_calls_remaining", 0),
        "session_visits_remaining": observation.get("session_visits_remaining", 0),
    }
    return [
        action.value
        for action in get_valid_actions(
            borrower,
            session_state,
            observation.get("regulatory_audit_active", False),
        )
    ]


def heuristic_action(account: Dict[str, Any], valid_actions: List[str]) -> Dict[str, Any]:
    outstanding = float(account.get("outstanding_inr", 0.0))
    dpd = int(account.get("dpd", 0))
    hardship = bool(account.get("hardship_flag", False))
    sentiment = account.get("sentiment")
    complaint_count = int(account.get("complaint_count", 0))

    if hardship and "OFFER_RESTRUCTURE" in valid_actions:
        return {
            "account_id": account["account_id"],
            "action_type": "OFFER_RESTRUCTURE",
        }

    if hardship and "GRANT_DEFERMENT" in valid_actions:
        return {
            "account_id": account["account_id"],
            "action_type": "GRANT_DEFERMENT",
            "deferment_days": 21,
        }

    if dpd >= 120 and complaint_count < 2 and "ESCALATE_LEGAL" in valid_actions:
        return {
            "account_id": account["account_id"],
            "action_type": "ESCALATE_LEGAL",
        }

    if dpd >= 60 and "OFFER_OTS" in valid_actions:
        return {
            "account_id": account["account_id"],
            "action_type": "OFFER_OTS",
            "settlement_percentage": 0.7 if dpd >= 120 else 0.8,
        }

    if sentiment == "cooperative" and "NEGOTIATE_PTP" in valid_actions:
        return {
            "account_id": account["account_id"],
            "action_type": "NEGOTIATE_PTP",
            "ptp_amount_inr": round(max(2000.0, outstanding * 0.2), 2),
        }

    for action_type in [
        "CALL_MORNING",
        "CALL_AFTERNOON",
        "EMAIL_EMPATHETIC",
        "SMS_REMINDER",
        "WHATSAPP_NUDGE",
    ]:
        if action_type in valid_actions:
            return {"account_id": account["account_id"], "action_type": action_type}

    return {"account_id": account["account_id"], "action_type": "NO_CONTACT"}


def sanitize_action(
    raw_action: Dict[str, Any],
    account: Dict[str, Any],
    valid_actions: List[str],
) -> Dict[str, Any]:
    action_type = raw_action.get("action_type")
    if action_type not in valid_actions:
        return heuristic_action(account, valid_actions)

    action: Dict[str, Any] = {
        "account_id": account["account_id"],
        "action_type": action_type,
    }

    if action_type == "NEGOTIATE_PTP":
        ptp_amount = raw_action.get("ptp_amount_inr")
        if not isinstance(ptp_amount, (int, float)) or ptp_amount <= 0:
            ptp_amount = max(
                2000.0, float(account.get("outstanding_inr", 0.0)) * 0.2
            )
        action["ptp_amount_inr"] = round(float(ptp_amount), 2)

    if action_type == "OFFER_OTS":
        settlement = raw_action.get("settlement_percentage")
        if not isinstance(settlement, (int, float)):
            settlement = 0.75
        action["settlement_percentage"] = min(
            1.0, max(0.5, round(float(settlement), 2))
        )

    if action_type == "GRANT_DEFERMENT":
        deferment_days = raw_action.get("deferment_days")
        if not isinstance(deferment_days, int):
            deferment_days = 21
        action["deferment_days"] = min(90, max(1, deferment_days))

    return action


def llm_action(
    client: OpenAI,
    account: Dict[str, Any],
    observation: Dict[str, Any],
    valid_actions: List[str],
) -> Dict[str, Any]:
    focused_observation = {
        "task_id": observation.get("task_id"),
        "current_step": observation.get("current_step"),
        "max_steps": observation.get("max_steps"),
        "regulatory_audit_active": observation.get("regulatory_audit_active"),
        "session_calls_remaining": observation.get("session_calls_remaining"),
        "session_visits_remaining": observation.get("session_visits_remaining"),
        "portfolio_recovery_rate": observation.get("portfolio_recovery_rate"),
        "selected_account": account,
        "valid_actions": valid_actions,
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(focused_observation, ensure_ascii=True),
            },
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    raw_action = parse_json_object(content)
    return sanitize_action(raw_action, account, valid_actions)


def run_episode(
    client: OpenAI, http_client: httpx.Client, task_id: str, seed: int
) -> Dict[str, Any]:
    emit("START", {"task_id": task_id, "seed": seed, "model_name": MODEL_NAME})

    reset_response = http_client.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30.0,
    )
    reset_response.raise_for_status()
    observation = reset_response.json()

    while True:
        account = choose_account(observation)
        valid_actions = valid_actions_for(account, observation)

        try:
            action = llm_action(client, account, observation, valid_actions)
        except Exception:
            action = heuristic_action(account, valid_actions)

        step_response = http_client.post(
            f"{ENV_BASE_URL}/step",
            json=action,
            timeout=30.0,
        )
        step_response.raise_for_status()
        result = step_response.json()

        reward = result["reward"]
        emit(
            "STEP",
            {
                "task_id": task_id,
                "seed": seed,
                "step": result["observation"]["current_step"],
                "account_id": action["account_id"],
                "action_type": action["action_type"],
                "reward": reward["total"],
                "episode_done": reward["episode_done"],
            },
        )

        observation = result["observation"]
        if reward["episode_done"]:
            break

    grade_response = http_client.get(f"{ENV_BASE_URL}/grade", timeout=30.0)
    grade_response.raise_for_status()
    grade_result = grade_response.json()

    emit(
        "END",
        {
            "task_id": task_id,
            "seed": seed,
            "score": grade_result["score"],
            "steps_taken": grade_result["steps_taken"],
            "done": grade_result["done"],
        },
    )
    return grade_result


def main() -> None:
    ensure_token()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    server_process = launch_server()
    atexit.register(cleanup_server, server_process)

    with httpx.Client(timeout=30.0) as http_client:
        for task_id, seed in TASKS:
            run_episode(client, http_client, task_id, seed)


if __name__ == "__main__":
    main()

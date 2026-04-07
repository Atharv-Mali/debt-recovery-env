"""Run baseline agent across all three tasks and print reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m baseline.run_baseline

Requires the FastAPI server to be running on localhost:7860.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import httpx

from baseline.baseline_agent import BaselineAgent


BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")

TASKS_SEEDS = {
    "task1_single_cooperative": [42, 137, 256],
    "task2_portfolio_mixed": [1001, 2048, 3333],
    "task3_portfolio_adversarial": [9999, 8421, 7070],
}


async def run_episode(
    client: httpx.AsyncClient,
    agent: BaselineAgent,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a single episode and return the result."""
    # Reset
    resp = await client.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30.0,
    )
    resp.raise_for_status()
    observation = resp.json()

    agent.reset()
    done = False
    steps = 0

    while not done:
        # Get action from LLM
        action = agent.choose_action(observation)

        # Step
        resp = await client.post(
            f"{BASE_URL}/step",
            json=action,
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()

        observation = result["observation"]
        reward_info = result["reward"]
        done = reward_info.get("episode_done", False)
        steps += 1

        if steps > 200:  # safety limit
            break

    # Grade
    resp = await client.get(f"{BASE_URL}/grade", timeout=30.0)
    resp.raise_for_status()
    grade_result = resp.json()

    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade_result["score"],
        "steps": steps,
    }


async def main() -> None:
    """Run all tasks and seeds, print results table."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Usage: set OPENAI_API_KEY=sk-... before running.")
        sys.exit(1)

    agent = BaselineAgent(api_key=api_key, model="gpt-4o-mini")
    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        # Check server health
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            resp.raise_for_status()
            print(f"✓ Server healthy: {resp.json()}\n")
        except Exception as e:
            print(f"ERROR: Cannot reach server at {BASE_URL}: {e}")
            print("Start the server first: uvicorn app:app --port 7860")
            sys.exit(1)

        # Run episodes
        print("=" * 65)
        print(f"{'Task':<35} {'Seed':<8} {'Score':<8} {'Steps':<6}")
        print("=" * 65)

        for task_id, seeds in TASKS_SEEDS.items():
            for seed in seeds:
                try:
                    result = await run_episode(client, agent, task_id, seed)
                    results.append(result)
                    print(
                        f"{result['task_id']:<35} "
                        f"{result['seed']:<8} "
                        f"{result['score']:<8.4f} "
                        f"{result['steps']:<6}"
                    )
                except Exception as e:
                    print(f"{task_id:<35} {seed:<8} ERROR: {e}")
                    results.append({
                        "task_id": task_id,
                        "seed": seed,
                        "score": 0.0,
                        "steps": 0,
                    })

    # Print summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    for task_id in TASKS_SEEDS:
        task_results = [r for r in results if r["task_id"] == task_id]
        if task_results:
            mean_score = sum(r["score"] for r in task_results) / len(task_results)
            print(f"Mean {task_id}: {mean_score:.4f}")

    overall_mean = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"\nOverall Mean: {overall_mean:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

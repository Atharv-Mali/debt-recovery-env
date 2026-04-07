"""Tests for the DebtRecoveryEnv environment."""

from __future__ import annotations

import asyncio
import pytest

from env.environment import DebtRecoveryEnv
from env.models import (
    ActionType,
    CollectionAction,
    CollectionReward,
    PortfolioObservation,
)


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run_async(coro):
    """Helper to run async functions in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEnvironmentReset:
    """Test environment reset behavior."""

    def test_reset_returns_observation(self):
        env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs = run_async(env.reset())

        assert isinstance(obs, PortfolioObservation)
        assert obs.task_id == "task1_single_cooperative"
        assert obs.current_step == 0
        assert obs.max_steps == 10
        assert len(obs.accounts) == 1
        assert obs.accounts[0].is_resolved is False

    def test_reset_task2(self):
        env = DebtRecoveryEnv(task_id="task2_portfolio_mixed", seed=1001)
        obs = run_async(env.reset())

        assert len(obs.accounts) == 10
        assert obs.max_steps == 30
        assert obs.session_calls_remaining == 30  # 10 * 3

    def test_reset_task3(self):
        env = DebtRecoveryEnv(task_id="task3_portfolio_adversarial", seed=9999)
        obs = run_async(env.reset())

        assert len(obs.accounts) == 25
        assert obs.max_steps == 60
        assert obs.session_calls_remaining == 75
        assert obs.regulatory_audit_active is False

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            DebtRecoveryEnv(task_id="nonexistent_task")


class TestEnvironmentStep:
    """Test environment step behavior."""

    def test_step_returns_reward(self):
        env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs = run_async(env.reset())

        action = CollectionAction(
            account_id=obs.accounts[0].account_id,
            action_type=ActionType.CALL_MORNING,
        )
        new_obs, reward = run_async(env.step(action))

        assert isinstance(new_obs, PortfolioObservation)
        assert isinstance(reward, CollectionReward)
        assert new_obs.current_step == 1

    def test_step_updates_state(self):
        env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs = run_async(env.reset())
        account_id = obs.accounts[0].account_id

        action = CollectionAction(
            account_id=account_id,
            action_type=ActionType.CALL_MORNING,
        )
        new_obs, _ = run_async(env.step(action))

        # Contact attempts should increment
        assert new_obs.accounts[0].contact_attempts_today >= 1

    def test_done_at_max_steps(self):
        env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs = run_async(env.reset())
        account_id = obs.accounts[0].account_id

        done = False
        for i in range(15):  # More than max_steps=10
            action = CollectionAction(
                account_id=account_id,
                action_type=ActionType.NO_CONTACT,
            )
            obs, reward = run_async(env.step(action))
            if reward.episode_done:
                done = True
                break

        assert done, "Episode should end at max_steps"

    def test_invalid_account_id(self):
        env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        run_async(env.reset())

        action = CollectionAction(
            account_id="INVALID_ID",
            action_type=ActionType.NO_CONTACT,
        )
        obs, reward = run_async(env.step(action))
        assert "error" in reward.info


class TestSeedReproducibility:
    """Test that same seed produces identical outcomes."""

    def test_seed_reproducibility(self):
        # Run episode 1
        env1 = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs1 = run_async(env1.reset())

        action = CollectionAction(
            account_id=obs1.accounts[0].account_id,
            action_type=ActionType.CALL_MORNING,
        )
        new_obs1, reward1 = run_async(env1.step(action))

        # Run episode 2 with same seed
        env2 = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs2 = run_async(env2.reset())

        action2 = CollectionAction(
            account_id=obs2.accounts[0].account_id,
            action_type=ActionType.CALL_MORNING,
        )
        new_obs2, reward2 = run_async(env2.step(action2))

        # Observations must match
        assert obs1.accounts[0].outstanding_inr == obs2.accounts[0].outstanding_inr
        assert obs1.accounts[0].dpd == obs2.accounts[0].dpd

        # Rewards must match
        assert reward1.total == reward2.total
        assert reward1.components.recovery_signal == reward2.components.recovery_signal

    def test_different_seeds_differ(self):
        env1 = DebtRecoveryEnv(task_id="task2_portfolio_mixed", seed=1001)
        obs1 = run_async(env1.reset())

        env2 = DebtRecoveryEnv(task_id="task2_portfolio_mixed", seed=2048)
        obs2 = run_async(env2.reset())

        # At least some accounts should differ
        dpds1 = [a.dpd for a in obs1.accounts]
        dpds2 = [a.dpd for a in obs2.accounts]
        assert dpds1 != dpds2


class TestEpisodeState:
    """Test state retrieval."""

    def test_state_returns_log(self):
        env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
        obs = run_async(env.reset())

        action = CollectionAction(
            account_id=obs.accounts[0].account_id,
            action_type=ActionType.CALL_MORNING,
        )
        run_async(env.step(action))

        state = run_async(env.state())
        assert "episode_log" in state
        assert len(state["episode_log"]) == 1
        assert state["step_count"] == 1
        assert "metrics" in state

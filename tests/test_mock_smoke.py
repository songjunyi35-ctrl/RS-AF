"""End-to-end deterministic CPU experiment tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from agentic_rl.config import CreditConfig, ExperimentConfig, PolicyConfig, RewardConfig
from agentic_rl.mock import run_mock_experiment
from agentic_rl.trajectory import Trajectory


class MockSmokeTests(unittest.TestCase):
    def test_all_credit_baselines_and_jsonl_round_trip(self) -> None:
        for method in ("terminal", "discounted_return", "progress_delta"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = ExperimentConfig(
                    policy=PolicyConfig(kind="rule", seed=42),
                    credit=CreditConfig(method=method, discount_factor=0.9),
                    trajectory_output=str(Path(directory) / "trace.jsonl"),
                )
                trajectory = run_mock_experiment(config, run_id=f"smoke-{method}")
                path = trajectory.write_jsonl(config.trajectory_output)
                restored = Trajectory.load_jsonl(path)
                self.assertEqual(restored.termination_reason, "success")
                self.assertEqual(restored.final_reward, 1.0)
                self.assertEqual(len(restored), 5)
                self.assertTrue(
                    all(row.assigned_credit is not None for row in restored.transitions)
                )

    def test_reward_progress_weight_is_applied(self) -> None:
        config = ExperimentConfig(
            policy=PolicyConfig(kind="rule", seed=42),
            reward=RewardConfig(progress_weight=2.0),
            credit=CreditConfig(method="discounted_return", discount_factor=1.0),
        )
        trajectory = run_mock_experiment(config, run_id="shaped-reward")
        self.assertAlmostEqual(sum(row.step_reward for row in trajectory.transitions), 2.0)
        self.assertAlmostEqual(trajectory.transitions[0].step_reward, 0.3)

    def test_rule_policy_clearly_outperforms_random_policy(self) -> None:
        base = ExperimentConfig()
        rule_successes = 0
        random_successes = 0
        for seed in range(20):
            rule = replace(base, policy=PolicyConfig(kind="rule", seed=seed))
            random = replace(base, policy=PolicyConfig(kind="random", seed=seed))
            rule_successes += run_mock_experiment(rule, run_id=f"rule-{seed}").final_reward == 1.0
            random_successes += run_mock_experiment(random, run_id=f"random-{seed}").final_reward == 1.0
        self.assertEqual(rule_successes, 20)
        self.assertGreaterEqual(rule_successes - random_successes, 10)


if __name__ == "__main__":
    unittest.main()

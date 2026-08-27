"""Unit tests for structured trajectories, rewards, verifiers, and credit."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import tempfile
import unittest

from agentic_rl.core import AgentState, MacroAction, VerifierResult
from agentic_rl.credit import (
    CounterfactualCreditAssigner,
    DiscountedReturn,
    ProgressDeltaCredit,
    TerminalRewardBroadcast,
)
from agentic_rl.rewards import SparseTerminalReward, VerifierScoreReward
from agentic_rl.trajectory import Trajectory, Transition
from agentic_rl.verifiers import ProgressVerifier


class _Marker(Enum):
    VALUE = "enum-value"


def _action(name: str = "generate") -> MacroAction:
    return MacroAction(operator_name=name, arguments={"attempt": 1}, metadata={})


def _result(progress: float, passed: bool = False) -> VerifierResult:
    return VerifierResult(
        score=progress,
        progress=progress,
        passed=passed,
        feedback="ok",
        metadata={},
    )


def _transition(step: int, progress: float, reward: float) -> Transition:
    return Transition(
        run_id="run-1",
        task_id="task-1",
        policy_name="rule",
        step=step,
        state_summary={"progress": progress},
        action=_action(),
        available_actions=["generate", "stop"],
        operator_observation={
            "text": "structured summary",
            "thought": "legacy private reasoning must not persist",
        },
        verifier_result=_result(progress, passed=progress >= 1.0),
        step_reward=reward,
        cost=1.0,
        metadata={"marker": _Marker.VALUE, "raw_reasoning": "must not persist"},
    )


def _trajectory() -> Trajectory:
    trajectory = Trajectory(run_id="run-1", task_id="task-1", policy_name="rule")
    trajectory.append(_transition(0, 0.2, 1.0))
    trajectory.append(_transition(1, 0.5, 2.0))
    trajectory.append(_transition(2, 1.0, 3.0))
    trajectory.finalize(4.0, "success")
    return trajectory


class TrajectoryTests(unittest.TestCase):
    def test_jsonl_round_trip_and_enum_safety(self) -> None:
        original = _trajectory()
        original.assign_credits([0.2, 0.3, 0.5])
        with tempfile.TemporaryDirectory() as directory:
            path = original.write_jsonl(Path(directory) / "trace.jsonl")
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("raw_reasoning", text)
            self.assertNotIn("legacy private reasoning", text)
            self.assertIn("enum-value", text)
            loaded = Trajectory.load_jsonl(path)

        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.final_reward, 4.0)
        self.assertEqual(loaded.termination_reason, "success")
        self.assertEqual(loaded.transitions[0].action, original.transitions[0].action)
        self.assertEqual(loaded.transitions[2].verifier_result, original.transitions[2].verifier_result)
        self.assertEqual([row.assigned_credit for row in loaded.transitions], [0.2, 0.3, 0.5])

    def test_append_rejects_mismatched_identity_and_credit_length(self) -> None:
        trajectory = Trajectory(run_id="run-1", task_id="task-1", policy_name="rule")
        wrong = _transition(0, 0.0, 0.0)
        wrong.run_id = "other"
        with self.assertRaises(ValueError):
            trajectory.append(wrong)
        trajectory.append(_transition(0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            trajectory.assign_credits([])

    def test_empty_trajectory_can_record_pre_action_termination(self) -> None:
        trajectory = Trajectory(run_id="run-1", task_id="task-1", policy_name="rule")
        trajectory.finalize(0.0, "max_steps")
        self.assertEqual(trajectory.final_reward, 0.0)
        self.assertEqual(trajectory.termination_reason, "max_steps")

        with tempfile.TemporaryDirectory() as directory:
            path = trajectory.write_jsonl(Path(directory) / "empty-trace.jsonl")
            self.assertIn("trajectory_summary", path.read_text(encoding="utf-8"))
            loaded = Trajectory.load_jsonl(path)
        self.assertEqual(len(loaded), 0)
        self.assertEqual(loaded.final_reward, 0.0)
        self.assertEqual(loaded.termination_reason, "max_steps")


class CreditTests(unittest.TestCase):
    def test_terminal_reward_broadcast_length_and_values(self) -> None:
        trajectory = _trajectory()
        self.assertEqual(TerminalRewardBroadcast().assign(trajectory), [4.0, 4.0, 4.0])

    def test_discounted_return_includes_separate_terminal_reward(self) -> None:
        trajectory = _trajectory()
        # [1 + .5*2 + .25*3 + .125*4, 2 + .5*3 + .25*4, 3 + .5*4]
        self.assertEqual(DiscountedReturn(gamma=0.5).assign(trajectory), [3.25, 4.5, 5.0])

    def test_discount_factor_validation(self) -> None:
        with self.assertRaises(ValueError):
            DiscountedReturn(gamma=1.01)

    def test_progress_delta(self) -> None:
        trajectory = _trajectory()
        credits = ProgressDeltaCredit(initial_progress=0.1).assign(trajectory)
        for actual, expected in zip(credits, [0.1, 0.3, 0.5]):
            self.assertAlmostEqual(actual, expected)

    def test_progress_delta_uses_first_previous_state_by_default(self) -> None:
        trajectory = Trajectory(run_id="run-1", task_id="task-1", policy_name="rule")
        transition = _transition(0, 0.8, 0.0)
        transition.state_summary["metadata"] = {"progress": 0.6}
        trajectory.append(transition)
        trajectory.finalize(0.0, "stop")
        self.assertAlmostEqual(ProgressDeltaCredit().assign(trajectory)[0], 0.2)

    def test_counterfactual_is_explicitly_unimplemented(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "alternative rollouts"):
            CounterfactualCreditAssigner().assign(_trajectory())


class VerifierAndRewardTests(unittest.TestCase):
    def test_progress_verifier(self) -> None:
        previous = AgentState(
            task="task", task_id="task-1", step=0, metadata={"progress": 0.0}
        )
        next_state = AgentState(
            task="task", task_id="task-1", step=1, metadata={"progress": 1.0}
        )
        result = ProgressVerifier().evaluate(previous, _action(), next_state)
        self.assertTrue(result.passed)
        self.assertEqual(result.progress, 1.0)

    def test_reward_functions_keep_step_and_terminal_signals_distinct(self) -> None:
        trajectory = _trajectory()
        previous = AgentState(task="task", task_id="task-1", step=0)
        next_state = AgentState(task="task", task_id="task-1", step=1)
        result = _result(0.75, passed=True)
        sparse = SparseTerminalReward(success_reward=2.0, failure_reward=-1.0)
        self.assertEqual(sparse.step_reward(previous, _action(), next_state, result), 0.0)
        self.assertEqual(sparse.final_reward(trajectory), 2.0)
        score = VerifierScoreReward(step_scale=2.0, final_scale=3.0)
        self.assertEqual(score.step_reward(previous, _action(), next_state, result), 1.5)
        self.assertEqual(score.final_reward(trajectory), 3.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for controller contracts, budgets, policies, and rollout safety."""

from __future__ import annotations

from enum import Enum
import unittest

from agentic_rl.adapters import AFlowOperatorAdapter, AFlowWorkflowAdapter
from agentic_rl.budget import BudgetManager
from agentic_rl.core import AgentState, MacroAction, OperatorResult
from agentic_rl.mock import MockVerifier, build_mock_registry, initial_mock_state
from agentic_rl.operators import OperatorRegistry, UnknownOperatorError
from agentic_rl.optimization import UnimplementedPolicyOptimizer
from agentic_rl.policies import RandomPolicy, RuleBasedPolicy, ScriptedPolicy
from agentic_rl.rewards import SparseTerminalReward
from agentic_rl.rollout import HierarchicalRolloutRunner


class _Mode(Enum):
    TEST = "test"


class _EchoOperator:
    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        return OperatorResult(
            observation={"action": action.operator_name},
            metadata={"state_updates": {"progress": 0.5}},
        )


class _FailingOperator:
    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        raise RuntimeError("expected test failure")


class _MutatingOperator:
    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        state.metadata["illicit_operator_mutation"] = True
        return OperatorResult(observation="done")


class _MutatingPolicy:
    name = "mutating"

    def select_action(self, state: AgentState, available_actions):
        state.metadata["illicit_policy_mutation"] = True
        return MacroAction("Stop")


class _FailingVerifier:
    def evaluate(self, previous_state, action, next_state):
        raise RuntimeError("expected verifier failure")


async def _legacy_async_operator(input: str):
    return {"response": input.upper(), "cost": 0.25, "token_cost": 7}


async def _legacy_code_workflow(problem: str, entry_point: str):
    return f"{problem}:{entry_point}", 2.5


class CoreSerializationTests(unittest.TestCase):
    def test_state_and_action_round_trip(self) -> None:
        state = AgentState(
            task="task",
            task_id="id",
            memory=[{"mode": _Mode.TEST}],
            metadata={"nested": _Mode.TEST},
        )
        action = MacroAction("Plan", {"mode": _Mode.TEST})
        restored_state = AgentState.from_dict(state.to_dict())
        restored_action = MacroAction.from_dict(action.to_dict())
        self.assertEqual(restored_state.memory, [{"mode": "test"}])
        self.assertEqual(restored_state.metadata, {"nested": "test"})
        self.assertEqual(restored_action.arguments, {"mode": "test"})


class RegistryAndBudgetTests(unittest.TestCase):
    def test_registry_registration_query_and_illegal_action(self) -> None:
        registry = OperatorRegistry()
        operator = _EchoOperator()
        registry.register("Echo", operator)
        self.assertIs(registry.query("Echo"), operator)
        self.assertEqual(registry.names(), ["Echo"])
        with self.assertRaises(UnknownOperatorError):
            registry.execute(
                AgentState(task="task", task_id="id"), MacroAction("Missing")
            )

    def test_budget_consumption_and_exhaustion(self) -> None:
        budget = BudgetManager(
            max_steps=2, max_operator_calls=2, token_budget=10, cost_budget=2.0
        )
        self.assertIsNone(budget.consume(steps=1, operator_calls=1, tokens=4, cost=0.5))
        self.assertEqual(budget.remaining()["steps"], 1)
        self.assertEqual(
            budget.consume(steps=1, operator_calls=1, tokens=6, cost=0.5),
            "max_steps",
        )
        self.assertTrue(budget.exceeded())

    def test_each_non_step_budget_terminates_independently(self) -> None:
        cases = [
            (BudgetManager(max_operator_calls=1), {"operator_calls": 1}, "max_operator_calls"),
            (BudgetManager(token_budget=3), {"tokens": 3}, "token_budget"),
            (BudgetManager(cost_budget=0.5), {"cost": 0.5}, "cost_budget"),
        ]
        for budget, consumption, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(budget.consume(**consumption), expected)

    def test_legacy_async_operator_adapter_has_no_sdk_dependency(self) -> None:
        adapter = AFlowOperatorAdapter("Generate", _legacy_async_operator)
        result = adapter.execute(
            AgentState(task="task", task_id="id"),
            MacroAction("Generate", {"input": "candidate"}),
        )
        self.assertEqual(result.observation, {"response": "CANDIDATE"})
        self.assertEqual(result.cost, 0.25)
        self.assertEqual(result.token_cost, 7)

    def test_legacy_workflow_adapter_extracts_cost_and_forwards_arguments(self) -> None:
        adapter = AFlowWorkflowAdapter(_legacy_code_workflow)
        result = adapter.execute(
            AgentState(task="repair", task_id="id"),
            MacroAction("AFlowWorkflow", {"entry_point": "solve"}),
        )
        self.assertEqual(result.observation, "repair:solve")
        self.assertEqual(result.cost, 2.5)

    def test_legacy_false_result_marks_operator_failure(self) -> None:
        adapter = AFlowOperatorAdapter("Test", lambda: {"result": False})
        result = adapter.execute(
            AgentState(task="task", task_id="id"), MacroAction("Test")
        )
        self.assertFalse(result.success)


class PolicyTests(unittest.TestCase):
    def test_random_policy_only_selects_legal_actions(self) -> None:
        policy = RandomPolicy(seed=42)
        state = AgentState(task="task", task_id="id")
        legal = ["Plan", "Generate", "Stop"]
        for _ in range(50):
            self.assertIn(policy.select_action(state, legal).operator_name, legal)

    def test_rule_policy_is_reproducible_and_does_not_mutate_state(self) -> None:
        state = AgentState(task="task", task_id="id")
        before = state.to_dict()
        legal = ["Plan", "Generate", "Review", "Stop"]
        first = RuleBasedPolicy(seed=7).select_action(state, legal)
        second = RuleBasedPolicy(seed=7).select_action(state, legal)
        self.assertEqual(first, second)
        self.assertEqual(first.operator_name, "Plan")
        self.assertEqual(state.to_dict(), before)

    def test_policy_optimizer_placeholder_is_explicit(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "PPO/GRPO"):
            UnimplementedPolicyOptimizer().update(RandomPolicy(seed=1), [])


class RolloutTests(unittest.TestCase):
    def _runner(self, actions, *, max_steps=8, registry=None):
        return HierarchicalRolloutRunner(
            registry=registry or build_mock_registry(
                ["Plan", "Generate", "Review", "Revise", "Test"]
            ),
            policy=ScriptedPolicy(actions),
            verifier=MockVerifier(),
            reward_function=SparseTerminalReward(),
            budget_manager=BudgetManager(
                max_steps=max_steps,
                max_operator_calls=max_steps,
                token_budget=1_000,
                cost_budget=100.0,
            ),
        )

    def test_normal_success_termination(self) -> None:
        runner = self._runner(["Plan", "Generate", "Review", "Revise", "Test"])
        trajectory = runner.run(initial_mock_state(), run_id="success")
        self.assertEqual(trajectory.termination_reason, "success")
        self.assertEqual(trajectory.final_reward, 1.0)
        self.assertEqual(len(trajectory), 5)

    def test_stop_action_terminates_without_operator_call(self) -> None:
        runner = self._runner(["Stop"])
        trajectory = runner.run(initial_mock_state(), run_id="stop")
        self.assertEqual(trajectory.termination_reason, "stop")
        self.assertEqual(runner.budget_manager.operator_calls, 0)
        self.assertTrue(trajectory.transitions[-1].operator_observation.success)

    def test_max_step_termination(self) -> None:
        runner = self._runner(["Plan", "Generate", "Review"], max_steps=2)
        trajectory = runner.run(initial_mock_state(), run_id="budget")
        self.assertEqual(trajectory.termination_reason, "max_steps")
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(trajectory.final_reward, 0.0)

    def test_operator_exception_is_recorded_and_safe(self) -> None:
        registry = OperatorRegistry({"Fail": _FailingOperator()})
        runner = self._runner(["Fail"], registry=registry)
        trajectory = runner.run(initial_mock_state(), run_id="exception")
        transition = trajectory.transitions[-1]
        self.assertEqual(trajectory.termination_reason, "operator_exception")
        self.assertFalse(transition.operator_observation.success)
        self.assertEqual(transition.metadata["exception_type"], "RuntimeError")

    def test_policy_and_operator_cannot_mutate_live_state(self) -> None:
        initial = initial_mock_state()
        policy_runner = HierarchicalRolloutRunner(
            registry=OperatorRegistry(),
            policy=_MutatingPolicy(),
            budget_manager=BudgetManager(max_steps=2),
        )
        policy_trace = policy_runner.run(initial, run_id="policy-mutation")
        self.assertNotIn("illicit_policy_mutation", initial.metadata)
        self.assertNotIn(
            "illicit_policy_mutation", policy_trace.transitions[0].state_summary["metadata"]
        )

        registry = OperatorRegistry({"Mutate": _MutatingOperator()})
        operator_runner = HierarchicalRolloutRunner(
            registry=registry,
            policy=ScriptedPolicy(["Mutate", "Stop"]),
            budget_manager=BudgetManager(max_steps=2, max_operator_calls=2),
        )
        operator_trace = operator_runner.run(initial_mock_state(), run_id="operator-mutation")
        final_metadata = operator_trace.metadata["final_state"]["metadata"]
        self.assertNotIn("illicit_operator_mutation", final_metadata)

    def test_stop_verifier_exception_is_recorded(self) -> None:
        runner = HierarchicalRolloutRunner(
            registry=OperatorRegistry(),
            policy=ScriptedPolicy(["Stop"]),
            verifier=_FailingVerifier(),
            budget_manager=BudgetManager(max_steps=2),
        )
        trajectory = runner.run(initial_mock_state(), run_id="stop-verifier-error")
        self.assertEqual(trajectory.termination_reason, "verifier_exception")
        self.assertEqual(
            trajectory.transitions[-1].metadata["verifier_exception"]["type"],
            "RuntimeError",
        )

    def test_initial_done_state_produces_valid_empty_trajectory(self) -> None:
        runner = self._runner(["Stop"])
        state = initial_mock_state()
        state.done = True
        trajectory = runner.run(state, run_id="already-done")
        self.assertEqual(len(trajectory), 0)
        self.assertEqual(trajectory.termination_reason, "state_done")


if __name__ == "__main__":
    unittest.main()

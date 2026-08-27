"""Deterministic CPU environment for hierarchical workflow experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .budget import BudgetManager
from .config import ExperimentConfig
from .core import AgentState, MacroAction, OperatorResult, VerifierResult
from .credit import DiscountedReturn, ProgressDeltaCredit, TerminalRewardBroadcast
from .operators import OperatorRegistry
from .policies import RandomPolicy, RuleBasedPolicy, ScriptedPolicy
from .rollout import HierarchicalRolloutRunner
from .trajectory import Trajectory


@dataclass(frozen=True)
class MockWorkflowOperator:
    """One frozen deterministic executor in a small code-repair state machine."""

    name: str
    token_cost: int = 20
    cost: float = 1.0

    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        if action.operator_name != self.name:
            raise ValueError(f"{self.name} cannot execute {action.operator_name}")

        phase = str(state.metadata.get("phase", "new"))
        progress = float(state.metadata.get("progress", 0.0))
        quality = float(state.metadata.get("quality", 0.0))
        updates: Dict[str, object]

        if self.name == "Plan":
            updates = {
                "phase": "planned",
                "progress": max(progress, 0.15),
                "recommended_action": "Generate",
            }
            observation = {"summary": "repair plan created", "milestone": "plan"}
        elif self.name == "Generate":
            planned = phase == "planned"
            quality = max(quality, 0.50 if planned else 0.15)
            updates = {
                "phase": "generated",
                "quality": quality,
                "progress": max(progress, 0.45 if planned else 0.20),
                "recommended_action": "Review",
            }
            observation = {"summary": "candidate patch generated", "quality": quality}
        elif self.name == "Review":
            has_candidate = phase in {"generated", "revised"}
            updates = {
                "phase": "reviewed" if has_candidate else phase,
                "progress": max(progress, 0.60 if has_candidate else progress),
                "defect_identified": has_candidate and quality < 0.8,
                "recommended_action": "Revise" if quality < 0.8 else "Test",
            }
            observation = {
                "summary": "candidate reviewed" if has_candidate else "nothing to review",
                "defect_identified": has_candidate and quality < 0.8,
            }
        elif self.name == "Revise":
            reviewed = phase == "reviewed"
            quality = 0.90 if reviewed else max(quality, 0.25)
            updates = {
                "phase": "revised",
                "quality": quality,
                "progress": max(progress, 0.85 if reviewed else 0.30),
                "recommended_action": "Test" if reviewed else "Review",
            }
            observation = {"summary": "candidate revised", "quality": quality}
        elif self.name == "Test":
            passed = quality >= 0.8 and phase in {"revised", "reviewed"}
            updates = {
                "phase": "passed" if passed else "test_failed",
                "test_passed": passed,
                "progress": 1.0 if passed else max(progress, 0.25),
                "recommended_action": "Stop" if passed else "Revise",
            }
            observation = {"summary": "tests passed" if passed else "tests failed", "passed": passed}
        elif self.name == "Tool":
            updates = {
                "phase": phase,
                "progress": max(progress, 0.10),
                "tool_context": True,
                "recommended_action": "Plan" if phase == "new" else "Generate",
            }
            observation = {"summary": "repository context inspected"}
        elif self.name == "Fail":
            raise RuntimeError("deterministic mock operator failure")
        else:  # Registry construction prevents this branch in normal use.
            raise ValueError(f"unsupported mock operator: {self.name}")

        return OperatorResult(
            observation=observation,
            memory_updates=[{"operator": self.name, "summary": observation["summary"]}],
            cost=self.cost,
            token_cost=self.token_cost,
            metadata={"state_updates": updates, "mock": True},
        )


class MockVerifier:
    """Expose state-machine progress without inspecting hidden reasoning."""

    def evaluate(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
    ) -> VerifierResult:
        del previous_state, action
        progress = float(next_state.metadata.get("progress", 0.0))
        passed = bool(next_state.metadata.get("test_passed", False))
        recommendation = next_state.metadata.get("recommended_action")
        return VerifierResult(
            score=1.0 if passed else progress,
            progress=progress,
            passed=passed,
            feedback={
                "summary": "success" if passed else "continue",
                "recommended_action": recommendation,
            },
            metadata={"environment": "deterministic_code_repair_v1"},
        )


@dataclass(frozen=True)
class MockRewardFunction:
    """Configurable progress shaping plus a separate pass/fail terminal reward."""

    success_reward: float = 1.0
    failure_reward: float = 0.0
    progress_weight: float = 0.0

    def step_reward(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
        verifier_result: VerifierResult,
    ) -> float:
        del action
        previous = float(previous_state.metadata.get("progress", 0.0))
        return self.progress_weight * (verifier_result.progress - previous)

    def final_reward(self, trajectory: Trajectory) -> float:
        if not trajectory.transitions:
            return float(self.failure_reward)
        result = trajectory.transitions[-1].verifier_result
        passed = (
            bool(result.get("passed", False))
            if isinstance(result, dict)
            else bool(getattr(result, "passed", False))
        )
        return float(self.success_reward if passed else self.failure_reward)


def build_mock_registry(operators: Iterable[str]) -> OperatorRegistry:
    supported = {"Plan", "Generate", "Review", "Revise", "Test", "Tool", "Fail"}
    registry = OperatorRegistry()
    for name in operators:
        if name == "Stop":
            continue
        if name not in supported:
            raise ValueError(f"unknown mock operator {name!r}; expected one of {sorted(supported)}")
        registry.register(name, MockWorkflowOperator(name))
    return registry


def build_credit_assigner(method: str, gamma: float = 0.95):
    normalized = method.lower().replace("-", "_")
    if normalized in {"terminal", "terminal_broadcast"}:
        return TerminalRewardBroadcast()
    if normalized in {"discounted", "discounted_return"}:
        return DiscountedReturn(gamma=gamma)
    if normalized in {"progress", "progress_delta"}:
        return ProgressDeltaCredit()
    raise ValueError(f"unsupported credit method: {method}")


def build_policy(kind: str, seed: int, script: Iterable[str] = ()):
    normalized = kind.lower().replace("-", "_")
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized in {"rule", "rule_based"}:
        return RuleBasedPolicy(seed=seed)
    if normalized in {"script", "scripted", "replay"}:
        return ScriptedPolicy(script)
    raise ValueError(f"unsupported policy: {kind}")


def initial_mock_state(task_id: str = "mock-code-repair-1") -> AgentState:
    return AgentState(
        task="Repair the deterministic mock program and make its tests pass.",
        task_id=task_id,
        metadata={"phase": "new", "progress": 0.0, "quality": 0.0},
    )


def build_mock_runner(
    config: ExperimentConfig,
    *,
    script: Iterable[str] = (),
) -> HierarchicalRolloutRunner:
    budget = BudgetManager(
        max_steps=config.budget.max_steps,
        max_operator_calls=config.budget.max_operator_calls,
        token_budget=config.budget.token_budget,
        cost_budget=config.budget.cost_budget,
    )
    return HierarchicalRolloutRunner(
        registry=build_mock_registry(config.operators),
        policy=build_policy(config.policy.kind, config.policy.seed, script),
        verifier=MockVerifier(),
        reward_function=MockRewardFunction(
            success_reward=config.reward.success_reward,
            failure_reward=config.reward.failure_reward,
            progress_weight=config.reward.progress_weight,
        ),
        credit_assigner=build_credit_assigner(
            config.credit.method, config.credit.discount_factor
        ),
        budget_manager=budget,
    )


def run_mock_experiment(
    config: ExperimentConfig,
    *,
    script: Iterable[str] = (),
    run_id: str = "mock-run",
) -> Trajectory:
    runner = build_mock_runner(config, script=script)
    return runner.run(initial_mock_state(), run_id=run_id)

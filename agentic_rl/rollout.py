"""Dynamic, budget-aware hierarchical rollout orchestration."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any
from uuid import uuid4

from .budget import BudgetManager
from .core import (
    AgentState,
    HighLevelPolicy,
    MacroAction,
    OperatorResult,
    Verifier,
    VerifierResult,
    to_serializable,
)
from .operators import OperatorRegistry
from .trajectory import Trajectory, Transition


StateUpdater = Callable[[AgentState, MacroAction, OperatorResult], AgentState]


def default_state_updater(
    state: AgentState, action: MacroAction, result: OperatorResult
) -> AgentState:
    """Return a new state containing an operator's explicit public updates.

    ``state_updates`` is namespaced under metadata.  This makes mock and real
    environments extensible without allowing an executor to silently overwrite
    core controller state such as the step or budget counters.
    """

    metadata = dict(state.metadata)
    updates = result.metadata.get("state_updates", {})
    if updates is not None:
        if not isinstance(updates, dict):
            raise TypeError("OperatorResult.metadata['state_updates'] must be a dict")
        metadata.update(updates)
    metadata.update(
        {
            "last_action": action.operator_name,
            "last_operator_success": result.success,
        }
    )
    return replace(
        state,
        step=state.step + 1,
        memory=[*state.memory, *result.memory_updates],
        last_observation=result.observation,
        metadata=metadata,
    )


class HierarchicalRolloutRunner:
    """Coordinate policy decisions and frozen operator execution on CPU.

    Reward and credit objects are intentionally duck-typed against the small
    public interfaces in ``agentic_rl.trajectory``.  The runner never imports an
    LLM SDK and policies never receive a mutable environment handle.
    """

    def __init__(
        self,
        registry: OperatorRegistry,
        policy: HighLevelPolicy,
        verifier: Verifier | None = None,
        reward_function: Any | None = None,
        credit_assigner: Any | None = None,
        budget_manager: BudgetManager | None = None,
        state_updater: StateUpdater | None = None,
        *,
        stop_action: str = "Stop",
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.verifier = verifier
        self.reward_function = reward_function
        self.credit_assigner = credit_assigner
        self.budget_manager = budget_manager or BudgetManager(max_steps=20)
        self.state_updater = state_updater or default_state_updater
        self.stop_action = stop_action

    @property
    def policy_name(self) -> str:
        return str(
            getattr(self.policy, "name", self.policy.__class__.__name__.lower())
        )

    def available_actions(self) -> list[str]:
        actions = self.registry.names()
        if not any(item.casefold() == self.stop_action.casefold() for item in actions):
            actions.append(self.stop_action)
        return actions

    @staticmethod
    def _state_summary(state: AgentState) -> dict[str, Any]:
        # Deliberately record memory size, not hidden/free-form model reasoning.
        return {
            "task": state.task,
            "task_id": state.task_id,
            "step": state.step,
            "memory_size": len(state.memory),
            "last_observation": to_serializable(state.last_observation),
            "verifier_feedback": to_serializable(state.verifier_feedback),
            "budget": to_serializable(state.budget),
            "done": state.done,
            "metadata": to_serializable(state.metadata),
        }

    @staticmethod
    def _normal_verifier_result(value: Any) -> VerifierResult:
        if isinstance(value, VerifierResult):
            return value
        if isinstance(value, dict):
            return VerifierResult.from_dict(value)
        raise TypeError(
            f"verifier returned {type(value).__name__}; expected VerifierResult"
        )

    def _evaluate(
        self, previous: AgentState, action: MacroAction, next_state: AgentState
    ) -> VerifierResult:
        if self.verifier is None:
            return VerifierResult()
        return self._normal_verifier_result(
            self.verifier.evaluate(
                deepcopy(previous), deepcopy(action), deepcopy(next_state)
            )
        )

    def _step_reward(
        self,
        previous: AgentState,
        action: MacroAction,
        next_state: AgentState,
        verifier_result: VerifierResult,
    ) -> float:
        if self.reward_function is None:
            return 0.0
        method = getattr(self.reward_function, "step_reward", None)
        if not callable(method):
            raise TypeError("reward_function must define step_reward(...) method")
        return float(
            method(
                deepcopy(previous),
                deepcopy(action),
                deepcopy(next_state),
                deepcopy(verifier_result),
            )
        )

    def _final_reward(self, trajectory: Trajectory) -> float:
        if self.reward_function is not None:
            method = getattr(self.reward_function, "final_reward", None)
            if not callable(method):
                raise TypeError("reward_function must define final_reward(trajectory)")
            return float(method(trajectory))
        if not trajectory.transitions:
            return 0.0
        last = trajectory.transitions[-1].verifier_result
        if isinstance(last, VerifierResult):
            return 1.0 if last.passed else last.score
        if isinstance(last, dict):
            return 1.0 if last.get("passed") else float(last.get("score", 0.0))
        return 0.0

    def _transition(
        self,
        *,
        run_id: str,
        previous: AgentState,
        action: MacroAction,
        available_actions: list[str],
        result: OperatorResult,
        verifier_result: VerifierResult,
        step_reward: float,
        termination_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Transition:
        return Transition(
            run_id=run_id,
            task_id=previous.task_id,
            policy_name=self.policy_name,
            step=previous.step,
            state_summary=self._state_summary(previous),
            action=action,
            available_actions=list(available_actions),
            operator_observation=result,
            verifier_result=verifier_result,
            step_reward=step_reward,
            cost=result.cost,
            termination_reason=termination_reason,
            metadata=dict(metadata or {}),
        )

    def run(
        self,
        initial_state: AgentState,
        *,
        run_id: str | None = None,
        trajectory: Trajectory | None = None,
    ) -> Trajectory:
        """Run until success, Stop, exhausted budget, or a safe failure."""

        self.budget_manager.reset()
        if trajectory is None:
            run_id = run_id or uuid4().hex
            trajectory = Trajectory(
                run_id=run_id,
                task_id=initial_state.task_id,
                policy_name=self.policy_name,
            )
        else:
            # When the caller supplies a trajectory its identity is authoritative.
            if run_id is not None and trajectory.run_id != run_id:
                raise ValueError("trajectory.run_id does not match run_id")
            run_id = trajectory.run_id

        state = replace(initial_state, budget=self.budget_manager.remaining())
        termination_reason: str | None = "state_done" if state.done else None

        while termination_reason is None:
            termination_reason = self.budget_manager.exceeded_reason()
            if termination_reason is not None:
                break

            available = self.available_actions()
            try:
                action = self.policy.select_action(deepcopy(state), list(available))
            except Exception as exc:
                termination_reason = "policy_exception"
                trajectory.metadata["policy_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                break
            if not isinstance(action, MacroAction):
                termination_reason = "invalid_policy_action"
                trajectory.metadata["policy_error"] = (
                    f"expected MacroAction, got {type(action).__name__}"
                )
                break
            if action.operator_name not in available:
                termination_reason = "invalid_policy_action"
                trajectory.metadata["policy_error"] = (
                    f"{action.operator_name!r} not in {available!r}"
                )
                break

            previous = state
            is_stop = action.operator_name.casefold() == self.stop_action.casefold()
            if is_stop:
                self.budget_manager.consume_step()
                result = OperatorResult(
                    observation={"stopped": True},
                    metadata={"termination_reason": "stop"},
                )
                state = self.state_updater(previous, action, result)
                if state.step <= previous.step:
                    state = replace(state, step=previous.step + 1)
                state = replace(
                    state,
                    budget=self.budget_manager.remaining(),
                    done=True,
                )
                failure_metadata: dict[str, Any] = {}
                try:
                    verifier_result = self._evaluate(previous, action, state)
                except Exception as exc:
                    verifier_result = VerifierResult(
                        feedback="verifier evaluation failed",
                        metadata={"exception_type": type(exc).__name__},
                    )
                    termination_reason = "verifier_exception"
                    failure_metadata["verifier_exception"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                try:
                    reward = self._step_reward(previous, action, state, verifier_result)
                except Exception as exc:
                    reward = 0.0
                    termination_reason = "reward_exception"
                    failure_metadata["reward_exception"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                trajectory.append(
                    self._transition(
                        run_id=run_id,
                        previous=previous,
                        action=action,
                        available_actions=available,
                        result=result,
                        verifier_result=verifier_result,
                        step_reward=reward,
                        termination_reason=termination_reason or "stop",
                        metadata=failure_metadata,
                    )
                )
                termination_reason = termination_reason or "stop"
                break

            try:
                # A decision and attempted operator invocation both consume budget,
                # even when the executor raises.
                self.budget_manager.consume(steps=1, operator_calls=1)
                result = self.registry.execute(deepcopy(previous), deepcopy(action))
                self.budget_manager.consume(
                    tokens=result.token_cost,
                    cost=result.cost,
                )
                state = self.state_updater(previous, action, result)
                if state.step <= previous.step:
                    state = replace(state, step=previous.step + 1)
                state = replace(state, budget=self.budget_manager.remaining())
            except Exception as exc:
                result = OperatorResult(
                    observation={
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    },
                    success=False,
                    metadata={"operator_exception": True},
                )
                state = default_state_updater(previous, action, result)
                state = replace(
                    state,
                    budget=self.budget_manager.remaining(),
                    done=True,
                )
                verifier_result = VerifierResult(
                    feedback="operator execution failed",
                    metadata={"exception_type": type(exc).__name__},
                )
                trajectory.append(
                    self._transition(
                        run_id=run_id,
                        previous=previous,
                        action=action,
                        available_actions=available,
                        result=result,
                        verifier_result=verifier_result,
                        step_reward=0.0,
                        termination_reason="operator_exception",
                        metadata={
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        },
                    )
                )
                termination_reason = "operator_exception"
                break

            try:
                verifier_result = self._evaluate(previous, action, state)
            except Exception as exc:
                verifier_result = VerifierResult(
                    feedback="verifier evaluation failed",
                    metadata={"exception_type": type(exc).__name__},
                )
                state = replace(state, done=True)
                termination_reason = "verifier_exception"
                trajectory.metadata["verifier_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            feedback = verifier_result.to_dict()
            state_metadata = dict(state.metadata)
            state_metadata.update(
                {
                    "verifier_score": verifier_result.score,
                    "verifier_progress": verifier_result.progress,
                    "verifier_passed": verifier_result.passed,
                }
            )
            state = replace(
                state,
                verifier_feedback=feedback,
                metadata=state_metadata,
                done=state.done or verifier_result.passed,
            )
            try:
                reward = self._step_reward(previous, action, state, verifier_result)
            except Exception as exc:
                reward = 0.0
                state = replace(state, done=True)
                termination_reason = "reward_exception"
                trajectory.metadata["reward_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

            if termination_reason is None:
                if verifier_result.passed:
                    termination_reason = "success"
                elif state.done:
                    termination_reason = str(
                        state.metadata.get("termination_reason", "state_done")
                    )
                else:
                    termination_reason = self.budget_manager.exceeded_reason()

            trajectory.append(
                self._transition(
                    run_id=run_id,
                    previous=previous,
                    action=action,
                    available_actions=available,
                    result=result,
                    verifier_result=verifier_result,
                    step_reward=reward,
                    termination_reason=termination_reason,
                )
            )

        try:
            final_reward = self._final_reward(trajectory)
        except Exception as exc:
            final_reward = 0.0
            termination_reason = "reward_exception"
            trajectory.metadata["reward_exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        trajectory.finalize(final_reward, termination_reason or "unknown")
        trajectory.metadata["final_state"] = self._state_summary(state)
        trajectory.metadata["budget"] = self.budget_manager.remaining()
        if self.credit_assigner is not None:
            try:
                credits = self.credit_assigner.assign(trajectory)
                trajectory.assign_credits(credits)
            except Exception as exc:
                trajectory.metadata["credit_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        return trajectory

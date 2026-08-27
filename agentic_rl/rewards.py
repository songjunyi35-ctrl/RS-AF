"""Reward interfaces, kept separate from verification and credit assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import AgentState, MacroAction, VerifierResult
from .trajectory import Trajectory


class RewardFunction(Protocol):
    """Compute observed step rewards and the rollout's terminal reward."""

    def step_reward(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
        verifier_result: VerifierResult,
    ) -> float:
        ...

    def final_reward(self, trajectory: Trajectory) -> float:
        ...


@dataclass(frozen=True)
class SparseTerminalReward:
    """Zero intermediate reward and a pass/fail terminal outcome."""

    success_reward: float = 1.0
    failure_reward: float = 0.0

    def step_reward(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
        verifier_result: VerifierResult,
    ) -> float:
        del previous_state, action, next_state, verifier_result
        return 0.0

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


@dataclass(frozen=True)
class VerifierScoreReward:
    """Expose verifier score as step reward and last score as final reward."""

    step_scale: float = 1.0
    final_scale: float = 1.0

    def step_reward(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
        verifier_result: VerifierResult,
    ) -> float:
        del previous_state, action, next_state
        return float(verifier_result.score) * self.step_scale

    def final_reward(self, trajectory: Trajectory) -> float:
        if not trajectory.transitions:
            return 0.0
        result = trajectory.transitions[-1].verifier_result
        score = (
            float(result.get("score", 0.0))
            if isinstance(result, dict)
            else float(getattr(result, "score", 0.0))
        )
        return score * self.final_scale


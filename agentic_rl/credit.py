"""Credit assignment interfaces and transparent CPU baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .trajectory import Trajectory


class CreditAssigner(Protocol):
    """Assign one scalar credit to every high-level policy decision."""

    def assign(self, trajectory: Trajectory) -> list[float]:
        ...


def _terminal_reward(trajectory: Trajectory) -> float:
    reward = trajectory.final_reward
    return float(reward) if reward is not None else 0.0


@dataclass(frozen=True)
class TerminalRewardBroadcast:
    """Give every valid high-level decision the terminal reward."""

    def assign(self, trajectory: Trajectory) -> list[float]:
        return [_terminal_reward(trajectory)] * len(trajectory.transitions)


@dataclass(frozen=True)
class DiscountedReturn:
    """Return-to-go over step rewards followed by the terminal reward.

    For ``T`` transitions, this computes
    ``G_t = sum_{k=t}^{T-1} gamma**(k-t) r_k + gamma**(T-t) R_final``.
    Thus final reward is a separate terminal signal and is not assumed to be
    included in the last step reward.
    """

    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")

    def assign(self, trajectory: Trajectory) -> list[float]:
        running = _terminal_reward(trajectory)
        returns = [0.0] * len(trajectory.transitions)
        for index in range(len(trajectory.transitions) - 1, -1, -1):
            transition = trajectory.transitions[index]
            running = float(transition.step_reward) + self.gamma * running
            returns[index] = running
        return returns


def _progress(transition: object) -> float:
    result = getattr(transition, "verifier_result", None)
    if isinstance(result, dict):
        return float(result.get("progress", 0.0))
    return float(getattr(result, "progress", 0.0))


@dataclass(frozen=True)
class ProgressDeltaCredit:
    """Credit each action with the observed change in verifier progress."""

    initial_progress: Optional[float] = None

    def assign(self, trajectory: Trajectory) -> list[float]:
        if self.initial_progress is not None:
            previous = float(self.initial_progress)
        elif trajectory.transitions:
            summary = trajectory.transitions[0].state_summary
            metadata = summary.get("metadata", {}) if isinstance(summary, dict) else {}
            previous = float(
                metadata.get("progress", metadata.get("verifier_progress", 0.0))
                if isinstance(metadata, dict)
                else 0.0
            )
        else:
            previous = 0.0
        credits: list[float] = []
        for transition in trajectory.transitions:
            current = _progress(transition)
            credits.append(current - previous)
            previous = current
        return credits


@dataclass(frozen=True)
class CounterfactualConfig:
    """Configuration reserved for a future counterfactual implementation."""

    num_alternatives: int = 4
    reuse_cached_execution: bool = True
    value_model_name: str | None = None

    def __post_init__(self) -> None:
        if self.num_alternatives < 1:
            raise ValueError("num_alternatives must be positive")


@dataclass(frozen=True)
class CounterfactualCreditAssigner:
    """Placeholder requiring alternative rollouts or a learned value model."""

    config: CounterfactualConfig = CounterfactualConfig()

    def assign(self, trajectory: Trajectory) -> list[float]:
        del trajectory
        raise NotImplementedError(
            "counterfactual credit requires alternative rollouts, cached executor "
            "results, or a learned value model; none is available in the CPU baseline"
        )

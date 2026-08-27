"""Extension contracts for future high-level policy optimization."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .core import HighLevelPolicy
from .trajectory import Trajectory


class PolicyOptimizer(Protocol):
    """Update a high-level policy from credited trajectories.

    No PPO/GRPO implementation is provided at this stage.  Future trainers can
    implement this interface while keeping rollout collection independent from
    GPU frameworks and distributed execution.
    """

    def update(
        self,
        policy: HighLevelPolicy,
        trajectories: Sequence[Trajectory],
    ) -> Mapping[str, Any]:
        """Perform one update and return structured training metrics."""


class UnimplementedPolicyOptimizer:
    """Explicit placeholder that cannot be mistaken for a training algorithm."""

    def update(
        self,
        policy: HighLevelPolicy,
        trajectories: Sequence[Trajectory],
    ) -> Mapping[str, Any]:
        del policy, trajectories
        raise NotImplementedError(
            "policy optimization requires a trainable controller and an explicit "
            "PPO/GRPO or supervised objective; the CPU scaffold only collects data"
        )


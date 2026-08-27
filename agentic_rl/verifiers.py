"""Verifier interfaces and deterministic CPU implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from .core import AgentState, MacroAction, VerifierResult


class Verifier(Protocol):
    """Extract an observable evaluation signal after an operator executes."""

    def evaluate(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
    ) -> VerifierResult:
        ...


def _metadata_progress(state: AgentState) -> float:
    metadata = getattr(state, "metadata", {})
    if not isinstance(metadata, Mapping):
        return 0.0
    return float(metadata.get("progress", 0.0))


@dataclass(frozen=True)
class ProgressVerifier:
    """Verify a state using a deterministic progress extraction function."""

    progress_fn: Callable[[AgentState], float] = _metadata_progress
    pass_threshold: float = 1.0

    def evaluate(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
    ) -> VerifierResult:
        del previous_state, action
        progress = float(self.progress_fn(next_state))
        passed = progress >= self.pass_threshold
        feedback = "target reached" if passed else f"progress={progress:.6g}"
        return VerifierResult(
            score=progress,
            progress=progress,
            passed=passed,
            feedback=feedback,
            metadata={"source": "progress_verifier"},
        )


@dataclass(frozen=True)
class DeterministicVerifier:
    """Verifier parameterized by a pure function, useful for mock experiments."""

    evaluate_fn: Callable[[AgentState, MacroAction, AgentState], VerifierResult]

    def evaluate(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
    ) -> VerifierResult:
        result = self.evaluate_fn(previous_state, action, next_state)
        if not isinstance(result, VerifierResult):
            raise TypeError("evaluate_fn must return VerifierResult")
        return result

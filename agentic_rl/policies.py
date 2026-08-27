"""CPU-only reference policies for high-level workflow control."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .core import AgentState, MacroAction


def _exact_available(name: str, available_actions: Sequence[str]) -> str | None:
    wanted = name.casefold()
    return next((item for item in available_actions if item.casefold() == wanted), None)


class RandomPolicy:
    """Uniform random policy with an isolated, reproducible RNG."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def select_action(
        self, state: AgentState, available_actions: list[str]
    ) -> MacroAction:
        del state
        if not available_actions:
            raise ValueError("RandomPolicy requires at least one available action")
        return MacroAction(self._rng.choice(available_actions))


class RuleBasedPolicy:
    """Deterministic workflow baseline for Plan/Generate/Review/Test loops.

    The policy first honors a structured ``recommended_action``/``next_action``
    from state metadata or verifier feedback.  Otherwise it applies a generic
    plan-generate-review-test-revise controller.  Every returned name is taken
    verbatim from ``available_actions``.
    """

    name = "rule_based"

    def __init__(self, seed: int | None = None) -> None:
        # Seed is accepted for configuration symmetry; this policy is deterministic.
        self.seed = seed

    def _recommendation(self, state: AgentState) -> str | None:
        for source in (state.metadata, state.verifier_feedback):
            if isinstance(source, Mapping):
                for key in ("recommended_action", "next_action", "action"):
                    if source.get(key):
                        return str(source[key])
        return None

    def select_action(
        self, state: AgentState, available_actions: list[str]
    ) -> MacroAction:
        if not available_actions:
            raise ValueError("RuleBasedPolicy requires at least one available action")

        def choose(*names: str) -> MacroAction | None:
            for name in names:
                exact = _exact_available(name, available_actions)
                if exact is not None:
                    return MacroAction(exact)
            return None

        recommendation = self._recommendation(state)
        if recommendation:
            selected = choose(recommendation)
            if selected:
                return selected

        feedback = state.verifier_feedback
        passed = bool(state.metadata.get("verifier_passed"))
        if isinstance(feedback, Mapping):
            passed = passed or bool(feedback.get("passed") or feedback.get("success"))
        if state.done or passed:
            return choose("Stop") or MacroAction(available_actions[0])

        last_action = str(state.metadata.get("last_action", ""))
        failed = state.metadata.get("last_operator_success") is False
        if isinstance(feedback, Mapping):
            failed = failed or bool(feedback.get("needs_revision"))
            failed = failed or feedback.get("success") is False

        if state.step == 0 and not state.memory:
            return choose("Plan", "Generate", "Tool") or MacroAction(available_actions[0])
        if failed or last_action.casefold() == "test":
            return choose("Revise", "Generate", "Tool", "Stop") or MacroAction(
                available_actions[0]
            )
        if last_action.casefold() == "plan":
            return choose("Generate", "Tool", "Test") or MacroAction(available_actions[0])
        if last_action.casefold() in {"generate", "tool", "revise"}:
            return choose("Review", "Test", "Stop") or MacroAction(available_actions[0])
        if last_action.casefold() == "review":
            return choose("Test", "Revise", "Stop") or MacroAction(available_actions[0])
        return choose("Test", "Review", "Generate", "Stop") or MacroAction(
            available_actions[0]
        )


class ScriptedPolicy:
    """Replay a finite sequence of action names or action objects."""

    name = "scripted"

    def __init__(
        self,
        actions: Iterable[str | MacroAction | Mapping[str, Any]],
        *,
        fallback_action: str = "Stop",
    ) -> None:
        self._actions = [self._coerce(action) for action in actions]
        self._index = 0
        self.fallback_action = fallback_action

    @staticmethod
    def _coerce(action: str | MacroAction | Mapping[str, Any]) -> MacroAction:
        if isinstance(action, MacroAction):
            return MacroAction.from_dict(action.to_dict())
        if isinstance(action, str):
            return MacroAction(action)
        if isinstance(action, Mapping):
            return MacroAction.from_dict(action)
        raise TypeError(f"unsupported scripted action: {type(action).__name__}")

    def reset(self) -> None:
        self._index = 0

    def select_action(
        self, state: AgentState, available_actions: list[str]
    ) -> MacroAction:
        del state
        if not available_actions:
            raise ValueError("ScriptedPolicy requires at least one available action")
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
        else:
            action = MacroAction(self.fallback_action)
        exact = _exact_available(action.operator_name, available_actions)
        if exact is None:
            raise ValueError(
                f"scripted action {action.operator_name!r} is not available; "
                f"expected one of {available_actions!r}"
            )
        return MacroAction(exact, action.arguments, action.metadata)


class ReplayPolicy(ScriptedPolicy):
    """Semantic alias for replaying actions extracted from stored trajectories."""

    name = "replay"

    def __init__(
        self,
        actions: Iterable[str | MacroAction | Mapping[str, Any] | Any],
        *,
        fallback_action: str = "Stop",
    ) -> None:
        extracted: list[str | MacroAction | Mapping[str, Any]] = []
        for item in actions:
            if hasattr(item, "action"):
                item = item.action
            elif isinstance(item, Mapping) and "action" in item:
                item = item["action"]
            extracted.append(item)
        super().__init__(extracted, fallback_action=fallback_action)

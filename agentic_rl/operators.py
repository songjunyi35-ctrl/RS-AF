"""Low-level operator registration and dispatch."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .core import AgentState, MacroAction, Operator, OperatorResult


class UnknownOperatorError(KeyError):
    """Raised when a policy asks for an unregistered operator."""


class DuplicateOperatorError(ValueError):
    """Raised when registration would silently replace an operator."""


class OperatorRegistry:
    """Insertion-ordered mapping from macro action names to executors."""

    def __init__(self, operators: Mapping[str, Operator] | None = None) -> None:
        self._operators: dict[str, Operator] = {}
        for name, operator in (operators or {}).items():
            self.register(name, operator)

    def register(
        self,
        name: str,
        operator: Operator | None = None,
        *,
        overwrite: bool = False,
    ) -> Operator | Callable[[Operator], Operator]:
        """Register an operator, also usable as ``@registry.register(name)``."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("operator name must be a non-empty string")

        def add(value: Operator) -> Operator:
            if not callable(getattr(value, "execute", None)):
                raise TypeError(f"operator {name!r} must define execute(state, action)")
            if name in self._operators and not overwrite:
                raise DuplicateOperatorError(f"operator {name!r} is already registered")
            self._operators[name] = value
            return value

        return add(operator) if operator is not None else add

    def query(self, name: str) -> Operator:
        try:
            return self._operators[name]
        except KeyError as exc:
            choices = ", ".join(self._operators) or "<none>"
            raise UnknownOperatorError(
                f"unknown operator {name!r}; registered operators: {choices}"
            ) from exc

    get = query

    def names(self) -> list[str]:
        return list(self._operators)

    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        result = self.query(action.operator_name).execute(state, action)
        if not isinstance(result, OperatorResult):
            raise TypeError(
                f"operator {action.operator_name!r} returned {type(result).__name__}; "
                "expected OperatorResult"
            )
        return result

    def __contains__(self, name: object) -> bool:
        return name in self._operators

    def __len__(self) -> int:
        return len(self._operators)

    def __iter__(self) -> Iterator[str]:
        return iter(self._operators)


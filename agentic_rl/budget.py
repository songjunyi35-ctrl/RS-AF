"""Budget accounting for hierarchical rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any


@dataclass
class BudgetManager:
    max_steps: int | None = None
    max_operator_calls: int | None = None
    token_budget: int | None = None
    cost_budget: float | None = None
    steps_used: int = 0
    operator_calls: int = 0
    tokens_used: int = 0
    cost_used: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_operator_calls", "token_budget"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
        if self.cost_budget is not None and self.cost_budget < 0:
            raise ValueError("cost_budget must be non-negative or None")
        self._validate_usage()

    def _validate_usage(self) -> None:
        if min(self.steps_used, self.operator_calls, self.tokens_used) < 0:
            raise ValueError("budget usage counters cannot be negative")
        if self.cost_used < 0:
            raise ValueError("cost_used cannot be negative")

    def reset(self) -> None:
        self.steps_used = 0
        self.operator_calls = 0
        self.tokens_used = 0
        self.cost_used = 0.0

    def consume(
        self,
        *,
        steps: int = 0,
        operator_calls: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> str | None:
        if min(steps, operator_calls, tokens) < 0 or cost < 0:
            raise ValueError("budget consumption cannot be negative")
        self.steps_used += int(steps)
        self.operator_calls += int(operator_calls)
        self.tokens_used += int(tokens)
        self.cost_used += float(cost)
        return self.exceeded_reason()

    def consume_step(self, count: int = 1) -> str | None:
        return self.consume(steps=count)

    def consume_operator(self, *, tokens: int = 0, cost: float = 0.0) -> str | None:
        return self.consume(operator_calls=1, tokens=tokens, cost=cost)

    @staticmethod
    def _remaining(limit: int | float | None, used: int | float) -> int | float | None:
        return None if limit is None else max(type(limit)(0), limit - used)

    def remaining(self) -> dict[str, int | float | None]:
        """Return a policy-facing snapshot; ``None`` means unlimited."""

        return {
            "steps": self._remaining(self.max_steps, self.steps_used),
            "operator_calls": self._remaining(
                self.max_operator_calls, self.operator_calls
            ),
            "tokens": self._remaining(self.token_budget, self.tokens_used),
            "cost": self._remaining(self.cost_budget, self.cost_used),
            "steps_used": self.steps_used,
            "operator_calls_used": self.operator_calls,
            "tokens_used": self.tokens_used,
            "cost_used": self.cost_used,
        }

    snapshot = remaining

    def exceeded_reason(self) -> str | None:
        """Return the first exhausted budget in deterministic priority order."""

        checks = (
            (self.max_steps, self.steps_used, "max_steps"),
            (self.max_operator_calls, self.operator_calls, "max_operator_calls"),
            (self.token_budget, self.tokens_used, "token_budget"),
            (self.cost_budget, self.cost_used, "cost_budget"),
        )
        for limit, used, reason in checks:
            if limit is not None and used >= limit:
                return reason
        return None

    def exceeded(self) -> bool:
        return self.exceeded_reason() is not None

    @property
    def total_cost(self) -> float:
        return self.cost_used


"""Optional compatibility adapters for existing AFlow executors.

This module deliberately does not import :mod:`scripts.operators`, so the new
CPU-only research path remains usable without model SDK configuration.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Mapping, Optional

from .core import AgentState, MacroAction, OperatorResult


ArgumentBuilder = Callable[[AgentState, MacroAction], Mapping[str, Any]]


def _run_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "AFlowOperatorAdapter.execute is synchronous and cannot run inside an "
        "active event loop; use the legacy operator directly from async code"
    )


class AFlowOperatorAdapter:
    """Expose a legacy callable/async AFlow Operator through the new protocol.

    ``argument_builder`` is intentionally explicit because old AFlow operators
    have heterogeneous signatures (``input``, ``problem``, ``solutions``, ...).
    No hidden prompt or reasoning is copied into the trajectory.
    """

    def __init__(
        self,
        name: str,
        legacy_operator: Callable[..., Any],
        argument_builder: Optional[ArgumentBuilder] = None,
        default_cost: float = 0.0,
        default_token_cost: int = 0,
    ) -> None:
        self.name = name
        self.legacy_operator = legacy_operator
        self.argument_builder = argument_builder or (
            lambda _state, action: dict(action.arguments)
        )
        self.default_cost = default_cost
        self.default_token_cost = default_token_cost

    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        if action.operator_name != self.name:
            raise ValueError(
                f"adapter for {self.name!r} cannot execute {action.operator_name!r}"
            )
        arguments = dict(self.argument_builder(state, action))
        raw_result = _run_awaitable(self.legacy_operator(**arguments))
        if isinstance(raw_result, OperatorResult):
            return raw_result

        metadata: Dict[str, Any] = {"adapter": "aflow", "legacy_operator": self.name}
        cost = self.default_cost
        token_cost = self.default_token_cost
        if isinstance(raw_result, Mapping):
            raw_mapping = dict(raw_result)
            cost = float(raw_mapping.pop("cost", cost))
            token_cost = int(raw_mapping.pop("token_cost", token_cost))
            observation: Any = raw_mapping
            legacy_success = raw_mapping.get("result", True)
            success = legacy_success if isinstance(legacy_success, bool) else True
        else:
            observation = raw_result
            success = True
        return OperatorResult(
            observation=observation,
            cost=cost,
            token_cost=token_cost,
            metadata=metadata,
            success=success,
        )


class AFlowWorkflowAdapter(AFlowOperatorAdapter):
    """Treat a complete legacy Workflow as one frozen low-level operator."""

    def __init__(self, workflow: Callable[..., Any], name: str = "AFlowWorkflow") -> None:
        super().__init__(
            name=name,
            legacy_operator=workflow,
            argument_builder=lambda state, action: {
                "problem": action.arguments.get("problem", state.task),
                **{
                    key: value
                    for key, value in action.arguments.items()
                    if key != "problem"
                },
            },
        )

    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        result = super().execute(state, action)
        observation = result.observation
        if (
            isinstance(observation, tuple)
            and len(observation) == 2
            and isinstance(observation[1], (int, float))
        ):
            return OperatorResult(
                observation=observation[0],
                memory_updates=result.memory_updates,
                cost=float(observation[1]),
                token_cost=result.token_cost,
                metadata=result.metadata,
                success=result.success,
            )
        return result

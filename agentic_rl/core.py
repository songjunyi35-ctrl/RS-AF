"""Core data contracts for hierarchical agent rollouts.

The module deliberately has no dependency on AFlow's LLM stack.  These values
are the boundary between a high-level policy, frozen operators, and trajectory
storage, so their dictionary representation is intentionally plain and stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, TypeVar, runtime_checkable


JSONValue = Any
_T = TypeVar("_T")


def to_serializable(value: Any) -> JSONValue:
    """Recursively turn common Python values into JSON-compatible values.

    Unknown values are represented with ``str`` rather than failing a rollout
    at persistence time.  Research integrations can therefore attach paths,
    enums, or small domain objects in metadata without coupling this package to
    their implementation.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return to_serializable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_serializable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_serializable(to_dict())
    return str(value)


def _dataclass_from_dict(cls: type[_T], data: Mapping[str, Any]) -> _T:
    """Construct ``cls`` while ignoring forward-compatible unknown fields."""

    known = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class AgentState:
    task: str
    task_id: str
    step: int = 0
    memory: list[Any] = field(default_factory=list)
    last_observation: Any = None
    verifier_feedback: Any = None
    budget: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep the documented constructor compact while preventing shared state.
        self.memory = list(self.memory or [])
        self.budget = dict(self.budget or {})
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentState":
        return _dataclass_from_dict(cls, data)


@dataclass
class MacroAction:
    operator_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.arguments = dict(self.arguments or {})
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MacroAction":
        return _dataclass_from_dict(cls, data)


@dataclass
class OperatorResult:
    observation: Any
    memory_updates: list[Any] = field(default_factory=list)
    cost: float = 0.0
    token_cost: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def __post_init__(self) -> None:
        self.memory_updates = list(self.memory_updates or [])
        self.cost = float(self.cost)
        self.token_cost = int(self.token_cost)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorResult":
        return _dataclass_from_dict(cls, data)


@dataclass
class VerifierResult:
    score: float = 0.0
    progress: float = 0.0
    passed: bool = False
    feedback: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        self.progress = float(self.progress)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerifierResult":
        return _dataclass_from_dict(cls, data)


@runtime_checkable
class Operator(Protocol):
    def execute(self, state: AgentState, action: MacroAction) -> OperatorResult:
        """Execute one frozen low-level operator."""


@runtime_checkable
class HighLevelPolicy(Protocol):
    def select_action(
        self, state: AgentState, available_actions: list[str]
    ) -> MacroAction:
        """Choose a macro action without mutating ``state``."""


@runtime_checkable
class Verifier(Protocol):
    def evaluate(
        self,
        previous_state: AgentState,
        action: MacroAction,
        next_state: AgentState,
    ) -> VerifierResult:
        """Extract a verifiable signal from a state transition."""

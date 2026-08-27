"""Trajectory records and stable, JSONL-based persistence.

The records in this module contain structured summaries only.  There is no
field for hidden chain-of-thought, and common raw-CoT metadata keys are omitted
when serializing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type, TypeVar

from .core import MacroAction, OperatorResult, VerifierResult


SCHEMA_VERSION = "1.0"
_PRIVATE_REASONING_KEYS = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "cot",
        "hidden_cot",
        "raw_chain_of_thought",
        "raw_reasoning",
        "reasoning",
        "thought",
    }
)
_DataclassT = TypeVar("_DataclassT")


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, and containers into JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVATE_REASONING_KEYS
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Observations should normally already be structured.  This fallback keeps
    # persistence robust for lightweight mock objects without pickling them.
    return repr(value)


def _construct_dataclass(cls: Type[_DataclassT], value: Any) -> Any:
    """Best-effort reconstruction while tolerating forward-compatible fields."""

    if not isinstance(value, Mapping):
        return value
    names = {item.name for item in fields(cls)}
    kwargs = {key: item for key, item in value.items() if key in names}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError):
        # A future schema may not map to the currently installed core type.  A
        # structured dictionary is more useful than making old traces unreadable.
        return dict(value)


@dataclass
class Transition:
    """One high-level policy decision and its observable consequences."""

    run_id: str
    task_id: str
    policy_name: str
    step: int
    state_summary: dict[str, Any]
    action: MacroAction | dict[str, Any]
    available_actions: list[str]
    operator_observation: OperatorResult | Any = None
    verifier_result: VerifierResult | dict[str, Any] | None = None
    step_reward: float = 0.0
    final_reward: Optional[float] = None
    assigned_credit: Optional[float] = None
    cost: float = 0.0
    termination_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        self.available_actions = list(self.available_actions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with a stable top-level schema."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "policy_name": self.policy_name,
            "step": self.step,
            "state_summary": _jsonable(self.state_summary),
            "action": _jsonable(self.action),
            "available_actions": _jsonable(self.available_actions),
            "operator_observation": _jsonable(self.operator_observation),
            "verifier_result": _jsonable(self.verifier_result),
            "step_reward": float(self.step_reward),
            "final_reward": self.final_reward,
            "assigned_credit": self.assigned_credit,
            "cost": float(self.cost),
            "termination_reason": self.termination_reason,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Transition":
        """Reconstruct a transition from its JSON representation."""

        required = {
            "run_id",
            "task_id",
            "policy_name",
            "step",
            "state_summary",
            "action",
            "available_actions",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"transition is missing required fields: {sorted(missing)}")
        verifier = data.get("verifier_result")
        observation = data.get("operator_observation")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            run_id=str(data["run_id"]),
            task_id=str(data["task_id"]),
            policy_name=str(data["policy_name"]),
            step=int(data["step"]),
            state_summary=dict(data["state_summary"]),
            action=_construct_dataclass(MacroAction, data["action"]),
            available_actions=list(data["available_actions"]),
            operator_observation=(
                _construct_dataclass(OperatorResult, observation)
                if observation is not None
                else None
            ),
            verifier_result=(
                _construct_dataclass(VerifierResult, verifier) if verifier is not None else None
            ),
            step_reward=float(data.get("step_reward", 0.0)),
            final_reward=(
                float(data["final_reward"]) if data.get("final_reward") is not None else None
            ),
            assigned_credit=(
                float(data["assigned_credit"])
                if data.get("assigned_credit") is not None
                else None
            ),
            cost=float(data.get("cost", 0.0)),
            termination_reason=data.get("termination_reason"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Trajectory:
    """An ordered set of transitions for one task rollout."""

    run_id: str
    task_id: str
    policy_name: str
    transitions: list[Transition] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def final_reward(self) -> Optional[float]:
        for transition in reversed(self.transitions):
            if transition.final_reward is not None:
                return float(transition.final_reward)
        value = self.metadata.get("final_reward")
        return float(value) if value is not None else None

    @property
    def termination_reason(self) -> Optional[str]:
        if self.transitions:
            return self.transitions[-1].termination_reason
        value = self.metadata.get("termination_reason")
        return str(value) if value is not None else None

    def append(self, transition: Transition) -> None:
        """Append a transition after checking rollout identity and ordering."""

        identity = (transition.run_id, transition.task_id, transition.policy_name)
        expected = (self.run_id, self.task_id, self.policy_name)
        if identity != expected:
            raise ValueError(f"transition identity {identity!r} does not match {expected!r}")
        if self.transitions and transition.step <= self.transitions[-1].step:
            raise ValueError("transition steps must be strictly increasing")
        self.transitions.append(transition)

    def finalize(self, final_reward: float, termination_reason: str) -> "Trajectory":
        """Record the terminal outcome on every row and reason on the last row."""

        reward = float(final_reward)
        # The rollout runner can legitimately terminate before its first action
        # (for example, an initially-done state or a zero-step budget).  Keep
        # that outcome on the trajectory even though there is no JSONL row.
        self.metadata["final_reward"] = reward
        self.metadata["termination_reason"] = termination_reason
        for transition in self.transitions:
            transition.final_reward = reward
        if self.transitions:
            self.transitions[-1].termination_reason = termination_reason
        return self

    def assign_credits(self, credits: Sequence[float]) -> "Trajectory":
        """Attach one credit value to each policy decision."""

        if len(credits) != len(self.transitions):
            raise ValueError(
                f"expected {len(self.transitions)} credits, received {len(credits)}"
            )
        for transition, credit in zip(self.transitions, credits):
            transition.assigned_credit = float(credit)
        return self

    def write_jsonl(self, path: str | Path, *, append: bool = False) -> Path:
        """Write transitions, or one summary record for a zero-decision rollout."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with destination.open(mode, encoding="utf-8", newline="\n") as handle:
            records = [transition.to_dict() for transition in self.transitions]
            if not records:
                # A rollout may terminate before the first decision (zero budget,
                # initial done state, policy error).  Persist an explicit envelope
                # rather than silently creating an unrecoverable empty file.
                records = [
                    {
                        "schema_version": self.schema_version,
                        "record_type": "trajectory_summary",
                        "run_id": self.run_id,
                        "task_id": self.task_id,
                        "policy_name": self.policy_name,
                        "metadata": _jsonable(self.metadata),
                    }
                ]
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        return destination

    # A short alias is convenient for callers and matches common serializer APIs.
    to_jsonl = write_jsonl

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "Trajectory":
        """Load a single-run JSONL trajectory and validate row consistency."""

        source = Path(path)
        transitions: list[Transition] = []
        summary: Optional[dict[str, Any]] = None
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        raise ValueError("JSONL record must be an object")
                    if record.get("record_type") == "trajectory_summary":
                        if transitions or summary is not None:
                            raise ValueError("trajectory summary must be the only JSONL record")
                        summary = record
                    else:
                        if summary is not None:
                            raise ValueError("trajectory summary cannot be mixed with transitions")
                        transitions.append(Transition.from_dict(record))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid transition at {source}:{line_number}: {exc}") from exc
        if summary is not None:
            return cls(
                run_id=str(summary["run_id"]),
                task_id=str(summary["task_id"]),
                policy_name=str(summary["policy_name"]),
                schema_version=str(summary.get("schema_version", SCHEMA_VERSION)),
                metadata=dict(summary.get("metadata", {})),
            )
        if not transitions:
            raise ValueError(f"trajectory file is empty: {source}")
        first = transitions[0]
        trajectory = cls(
            run_id=first.run_id,
            task_id=first.task_id,
            policy_name=first.policy_name,
            schema_version=first.schema_version,
        )
        for transition in transitions:
            trajectory.append(transition)
        return trajectory

    from_jsonl = load_jsonl

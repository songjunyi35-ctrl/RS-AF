"""Lightweight configuration for hierarchical policy experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for the high-level controller."""

    kind: str = "rule"
    seed: int = 42


@dataclass(frozen=True)
class BudgetConfig:
    """Limits visible to the policy and enforced by the rollout runner."""

    max_steps: int = 8
    max_operator_calls: int = 8
    token_budget: Optional[int] = 2_000
    cost_budget: Optional[float] = 10.0


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the deterministic mock task reward."""

    success_reward: float = 1.0
    failure_reward: float = 0.0
    progress_weight: float = 0.0


@dataclass(frozen=True)
class CreditConfig:
    """Credit assignment baseline and its common parameters."""

    method: str = "progress_delta"
    discount_factor: float = 0.95


@dataclass(frozen=True)
class ExperimentConfig:
    """Self-contained CPU rollout configuration.

    The original AFlow YAML is model-provider configuration.  Keeping research
    experiment settings separate avoids changing that legacy contract.
    """

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    operators: List[str] = field(
        default_factory=lambda: ["Plan", "Generate", "Review", "Revise", "Test", "Stop"]
    )
    trajectory_output: str = "runs/mock.jsonl"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            policy=PolicyConfig(**dict(data.get("policy", {}))),
            budget=BudgetConfig(**dict(data.get("budget", {}))),
            reward=RewardConfig(**dict(data.get("reward", {}))),
            credit=CreditConfig(**dict(data.get("credit", {}))),
            operators=list(data.get("operators", cls().operators)),
            trajectory_output=str(data.get("trajectory_output", cls().trajectory_output)),
        )

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        """Load JSON, or YAML when the repository's PyYAML is installed."""

        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional legacy dependency
                raise RuntimeError("YAML config requires PyYAML; JSON works without dependencies") from exc
            data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ValueError("experiment config must contain a mapping")
        return cls.from_dict(data)


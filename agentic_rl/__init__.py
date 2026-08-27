"""Hierarchical policy optimization research scaffold for AFlow."""

from .budget import BudgetManager
from .core import AgentState, MacroAction, OperatorResult, VerifierResult
from .credit import (
    CounterfactualConfig,
    CounterfactualCreditAssigner,
    DiscountedReturn,
    ProgressDeltaCredit,
    TerminalRewardBroadcast,
)
from .operators import OperatorRegistry
from .policies import RandomPolicy, ReplayPolicy, RuleBasedPolicy, ScriptedPolicy
from .rollout import HierarchicalRolloutRunner
from .trajectory import Trajectory, Transition

__all__ = [
    "AgentState",
    "BudgetManager",
    "CounterfactualConfig",
    "CounterfactualCreditAssigner",
    "DiscountedReturn",
    "HierarchicalRolloutRunner",
    "MacroAction",
    "OperatorRegistry",
    "OperatorResult",
    "ProgressDeltaCredit",
    "RandomPolicy",
    "ReplayPolicy",
    "RuleBasedPolicy",
    "ScriptedPolicy",
    "TerminalRewardBroadcast",
    "Trajectory",
    "Transition",
    "VerifierResult",
]


"""Run a deterministic hierarchical rollout without models or network access."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Optional, Sequence

from ..config import ExperimentConfig
from ..mock import run_mock_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="optional JSON/YAML experiment config")
    parser.add_argument("--policy", choices=["rule", "random", "scripted"])
    parser.add_argument(
        "--credit-method",
        choices=["terminal", "discounted_return", "progress_delta"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-operator-calls", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--cost-budget", type=float)
    parser.add_argument("--discount-factor", type=float)
    parser.add_argument(
        "--operators",
        help="comma-separated mock operators; Stop is automatically available",
    )
    parser.add_argument(
        "--script",
        default="Plan,Generate,Review,Revise,Test",
        help="comma-separated actions used only by --policy scripted",
    )
    return parser


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    policy = replace(
        config.policy,
        kind=args.policy if args.policy is not None else config.policy.kind,
        seed=args.seed if args.seed is not None else config.policy.seed,
    )
    budget = replace(
        config.budget,
        max_steps=args.max_steps if args.max_steps is not None else config.budget.max_steps,
        max_operator_calls=(
            args.max_operator_calls
            if args.max_operator_calls is not None
            else config.budget.max_operator_calls
        ),
        token_budget=(
            args.token_budget if args.token_budget is not None else config.budget.token_budget
        ),
        cost_budget=(
            args.cost_budget if args.cost_budget is not None else config.budget.cost_budget
        ),
    )
    credit = replace(
        config.credit,
        method=(
            args.credit_method if args.credit_method is not None else config.credit.method
        ),
        discount_factor=(
            args.discount_factor
            if args.discount_factor is not None
            else config.credit.discount_factor
        ),
    )
    operators = (
        [item.strip() for item in args.operators.split(",") if item.strip()]
        if args.operators
        else config.operators
    )
    return replace(
        config,
        policy=policy,
        budget=budget,
        credit=credit,
        operators=operators,
        trajectory_output=args.output or config.trajectory_output,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = _build_config(args)
    script = [item.strip() for item in args.script.split(",") if item.strip()]
    run_id = f"mock-{config.policy.kind}-seed-{config.policy.seed}"
    trajectory = run_mock_experiment(config, script=script, run_id=run_id)
    output = trajectory.write_jsonl(config.trajectory_output)
    summary = {
        "run_id": trajectory.run_id,
        "policy": trajectory.policy_name,
        "steps": len(trajectory),
        "final_reward": trajectory.final_reward,
        "termination_reason": trajectory.termination_reason,
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


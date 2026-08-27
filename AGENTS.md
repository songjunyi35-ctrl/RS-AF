# RS-AF Project Instructions

These instructions apply to the repository root and every subdirectory.

## Project identity

RS-AF is a research scaffold for **Hierarchical Policy Optimization for
Long-Horizon LLM Agents**, built additively on the original AFlow workflow
generation baseline.

The main research target is high-level, long-horizon credit assignment. The
high-level controller selects macro actions such as Plan, Generate, Review,
Revise, Test, Tool, Ensemble, and Stop. Low-level LLM executors remain frozen or
mocked unless the user explicitly requests otherwise.

## Repository map

- `agentic_rl/`: CPU-first hierarchical policy, rollout, trajectory, reward,
  credit, budget, adapter, and mock experiment code.
- `tests/`: standard-library `unittest` coverage for the research scaffold.
- `docs/hierarchical_policy_optimization.md`: architecture and extension guide.
- `scripts/`, `benchmarks/`, `workspace/`, `run.py`: original AFlow baseline.
- `config/hierarchical_mock.example.yaml`: deterministic mock configuration.
- `runs/mock.jsonl`: example structured trajectory.

Read the architecture document before making cross-module design changes.

## Development principles

1. Prefer additive changes under `agentic_rl/`. Do not move, delete, or broadly
   rewrite the original AFlow baseline.
2. Keep high-level policy and low-level executor decoupled. A policy selects a
   `MacroAction`; it must not mutate live environment state or depend directly
   on an LLM SDK.
3. Keep `Verifier`, `RewardFunction`, and `CreditAssigner` semantically separate.
4. Keep core rollout and mock paths CPU-only and usable without GPU, model
   downloads, network access, API keys, or private environment variables.
5. Use typed, small interfaces based on dataclasses and Protocols. Avoid heavy
   frameworks and speculative abstractions.
6. Preserve deterministic behavior under fixed seeds. Mock costs and token
   usage must be explicit and reproducible.
7. Record structured summaries only. Never require or persist hidden
   chain-of-thought; filter raw reasoning fields from trajectories.
8. Treat `CounterfactualCreditAssigner` and `PolicyOptimizer` as explicit
   extension points until they have correct alternative-rollout/value-model or
   training semantics. Do not disguise heuristics as PPO, GRPO, or
   counterfactual estimation.
9. Enforce step, operator-call, token, and cost budgets through `BudgetManager`.
10. Preserve backward compatibility through adapters when integrating existing
    AFlow Operators or Workflows.

## Expected workflow

- Inspect `git status` before editing and preserve unrelated user changes.
- Make reasonable defaults and continue autonomously unless genuinely blocked.
- Use sub-agents only when the user explicitly requests delegation or parallel
  agent work.
- Add or update tests for behavioral changes and failure boundaries.
- Do not download models, install CUDA dependencies, call external LLM APIs, or
  start distributed training unless explicitly requested.
- Do not commit, force-push, or rewrite Git history unless explicitly requested.

## Validation commands

Run from the repository root:

```bash
python -m unittest discover -s tests -v
python -m compileall -q agentic_rl tests
python -m agentic_rl.examples.run_mock_rollout \
  --policy rule \
  --credit-method progress_delta \
  --seed 42 \
  --output runs/mock.jsonl
```

The mock rollout must complete without GPU, network, external APIs, or model
downloads. When checking legacy AFlow, distinguish existing dependency/data/API
failures from regressions introduced by the current change.

## Current research limits

- No complete PPO/GRPO trainer.
- No joint high-level and low-level model training.
- No distributed rollout system.
- No real counterfactual estimator without alternative executions, a valid
  cache, or a learned value model.
- The default runner is synchronous; legacy async executors use adapters.


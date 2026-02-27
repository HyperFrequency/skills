---
name: adaptive-walk-forward-epoch-selection
description: >
  Adaptive walk-forward analysis with intelligent epoch selection.
  Use when validating strategies with time-series cross-validation or selecting optimal training windows.
version: "1.0.0"
allowed-tools: Read, Write, Edit, Bash
---

# Adaptive Walk-Forward Epoch Selection

Source: https://mcpmarket.com/tools/skills/adaptive-walk-forward-epoch-selection

## Use When

- You need to apply this capability as part of trading research workflows (data, features, backtests, ML, reporting).
- You want a reproducible output that can be committed to this repo (code, configs, docs).

## Inputs To Ask For

- Objective: what success looks like (metric, constraints, time horizon).
- Data: symbols, timeframe, sampling, data sources, and leakage risks.
- Constraints: compute budget, latency, interpretability, and deployment requirements.

## Outputs

- A concrete plan (steps + checks).
- A minimal implementation sketch (files to create/change) and verification steps.
- If applicable: a risk checklist (leakage, overfitting, evaluation pitfalls).

## Workflow

1. Restate the task in measurable terms.
2. Enumerate required artifacts (datasets, features, configs, scripts, reports).
3. Propose a default approach and 1-2 alternatives.
4. Add validation gates (walk-forward, Monte Carlo, sanity checks).
5. Produce repo-ready deliverables (code + docs) and a run command.

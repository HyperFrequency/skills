#!/usr/bin/env markdown
# Backtest-Optimize (Codex-Native)

Codex-first backtesting + optimization workflow intended for Vast (multi-GPU) while continuously enforcing utilization SLOs and writing durable artifacts.

Canonical skill package path (workspace root):
- `/Users/DanBot/Desktop/alpha_gen/skills/backtest-optimize/SKILL.md`

## Canonical Defaults

- Defaults/config: `config/workflow_defaults.yaml`
- Model policy (advisory): `config/model_policy.yaml`
- Runbook: `docs/guides/SWARM_OPTIMIZATION_RUNBOOK.md`
- Hook contracts: `hooks/pipeline-hooks.md`
- Optional multi-provider orchestration (PAL MCP): `docs/reference/PAL_MCP_INTEGRATION.md`
- Optional vendor graph ingest (VectorBT PRO): `docs/reference/VECTORBT_GRAPH_INGEST.md`
- Vendored HyperFrequency references (additive): `docs/reference/hyperfrequency/`

## Entry Points

Control plane (always-on tmux loops):

```bash
bash scripts/start_swarm_watchdogs.sh
```

Unified wrapper (starts control plane, then launches work):

```bash
scripts/backtest-optimize --parallel
```

Work plane (multi-GPU, multi-symbol):

```bash
cd WORKFLOW
./launch_parallel.sh
```

Single-symbol pipeline:

```bash
scripts/backtest-optimize --single --symbol SOL --engine native --prescreen 50000 --paths 1000 --by-regime
```

Tearsheets:
- `scripts/tearsheet/strategy_tearsheet.py` (preferred, requires `pnl` in trades CSV)
- `scripts/tearsheet/generate_tearsheet.py` (wrapper that can approximate `pnl` from pct columns)
- Command wrapper (slash-command ergonomics): `scripts/generate-tearsheet STRATEGY --trades trades.csv --capital 10000 --output ./tearsheets`

Docs: `docs/guides/TEARSHEET_GENERATION.md`

## SLOs / Guardrails

- CPU utilization target: `>= 70%`
- GPU utilization target: `>= 70%`
- No silent GPU fallback for GPU sweeps

Enforced by:
- `hooks/hardware_capacity_watchdog.py`
- `scripts/process_auditor.py`

## Optional: Multi-Provider Orchestration (PAL MCP)

If you want research/consensus across multiple model providers (OpenRouter/OpenAI/Anthropic/xAI/local), attach PAL as an MCP server and keep secrets out of git:

- Config template: `config/mcp/pal.mcp.json.example`
- Docs: `docs/reference/PAL_MCP_INTEGRATION.md`
- Installer: `scripts/install_pal_mcp_server.sh` (creates `~/.venvs/pal-mcp313`)

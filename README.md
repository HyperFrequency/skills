# HyperFrequency Skills

Claude Code skills for algorithmic trading, backtesting, and quantitative analysis.

## Installation

```bash
# Copy all skills
cp -r ./* ~/.claude/skills/

# Or copy specific skills
cp -r strategy-workflow ~/.claude/skills/
cp -r ml-pipeline ~/.claude/skills/
```

## Core Trading Skills

| Skill | Description |
|-------|-------------|
| [strategy-workflow](./strategy-workflow/) | Strategy development, backtesting, Optuna optimization, walk-forward validation |
| [order-flow-opt](./order-flow-opt/) | MAE optimization, L2 exhaustion detection, 33-feature extraction |
| [nautilus-trader](./nautilus-trader/) | NautilusTrader platform + Hyperliquid live trading |
| [tearsheet-generator](./tearsheet-generator/) | Professional tearsheets with [QuantStats](https://github.com/ranaroussi/quantstats) SVG visualizations |

## ML & Analysis Skills

| Skill | Description |
|-------|-------------|
| [ml-pipeline](./ml-pipeline/) | Feature engineering, AutoML, deep learning, financial RL |
| [continuous-learning-pattern-extraction](./continuous-learning-pattern-extraction/) | Pattern learning and knowledge curation |
| [strategy-translator](./strategy-translator/) | Translate strategies between frameworks |

## Reference Skills

| Skill | Description |
|-------|-------------|
| [research-documentation](./research-documentation/) | API docs, Context7, arXiv research |
| [arxiv-research-search](./arxiv-research-search/) | Search and analyze arXiv papers for quant research |
| [context7-documentation-lookup](./context7-documentation-lookup/) | Context7 documentation and API reference lookup |

## Skill Format

Each skill uses YAML frontmatter:

```yaml
---
name: skill-name
description: What the skill does and when to use it
version: "1.0.0"
allowed-tools: Read, Write, Edit, Bash
---
```

## License

MIT

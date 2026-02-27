# HyperFrequency Skills

Claude Code skills for algorithmic trading, backtesting, and quantitative analysis.

## Installation

Copy skills to your Claude Code skills directory:

```bash
# Copy all skills
cp -r ./* ~/.claude/skills/

# Or copy specific skills
cp -r nautilus-trader ~/.claude/skills/
cp -r tearsheet-generator ~/.claude/skills/
```

## Available Skills

### Core Trading Skills

| Skill | Description |
|-------|-------------|
| [nautilus-trader](./nautilus-trader/) | NautilusTrader platform for strategy development and Hyperliquid live trading |
| [tearsheet-generator](./tearsheet-generator/) | Professional tearsheets with SVG visualizations using [QuantStats](https://github.com/ranaroussi/quantstats) |
| [order-flow-opt](./order-flow-opt/) | Order flow MAE optimization with exhaustion-timed entries |
| [backtest-optimize](./backtest-optimize/) | Comprehensive backtesting and optimization workflow |

### ML & Analysis Skills

| Skill | Description |
|-------|-------------|
| [ml-feature-engineering](./ml-feature-engineering/) | Machine learning feature engineering toolkit |
| [strategy-translator](./strategy-translator/) | Translate strategies between frameworks |
| [continuous-learning-pattern-extraction](./continuous-learning-pattern-extraction/) | Extract and learn patterns from trading data |

## Skill Format

Each skill follows the Claude Code skill format with YAML frontmatter:

```yaml
---
name: skill-name
description: What the skill does and when to use it
version: "1.0.0"
allowed-tools: Read, Write, Edit, Bash
---

# Skill Name

## Overview
...
```

## Contributing

1. Fork this repository
2. Create your skill in a new directory
3. Include proper YAML frontmatter
4. Submit a pull request

## License

MIT

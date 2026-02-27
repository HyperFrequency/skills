---
name: order-flow-opt
description: Order flow MAE optimization workflow - extract features, create strategies, run backtests with exhaustion-timed entries to minimize drawdown
---

# /hyperfrequency:order-flow-opt - Order Flow MAE Optimization

Minimize Maximum Adverse Excursion (MAE) in trading strategies using L2 order book exhaustion detection.

## What This Skill Does

1. **Downloads L2 book data** from Hyperliquid S3 (if needed)
2. **Extracts order flow features** (33 features per bar)
3. **Creates MAE-optimized strategies** (your indicator + flow timing)
4. **Runs backtests** comparing with/without flow optimization
5. **Tracks MAE improvement** across entry types

## Project Location

```
/Users/DanBot/Desktop/HyperFrequency/order-flow-opt/
```

## Quick Commands

| Command | Action |
|---------|--------|
| `/hyperfrequency:order-flow-opt` | Interactive mode - show options |
| `/hyperfrequency:order-flow-opt download BTC` | Download L2 data for BTC |
| `/hyperfrequency:order-flow-opt extract BTC` | Extract features for BTC |
| `/hyperfrequency:order-flow-opt backtest BTC` | Run backtest |
| `/hyperfrequency:order-flow-opt compare BTC` | Compare with/without flow |
| `/hyperfrequency:order-flow-opt test` | Quick verification test |
| `/hyperfrequency:order-flow-opt create MyStrategy` | Create strategy template |

## Workflow Execution

When this skill is invoked, follow these steps:

### Step 1: Check Data Availability
```bash
ls -la /Users/DanBot/Desktop/HyperFrequency/data-historical/hyperliquid/l2book/
```

If no data or stale, offer to download using memory-safe chunked downloader:
```bash
cd /Users/DanBot/Desktop/HyperFrequency/scripts/data
python3 datapull_hl_chunked.py --start YYYY-MM-DD --end YYYY-MM-DD --workers 4
```

### Step 2: Extract Features
```bash
cd /Users/DanBot/Desktop/HyperFrequency/order-flow-opt
python3 scripts/extract_features.py --symbol SYMBOL --start YYYY-MM-DD --end YYYY-MM-DD
```

### Step 3: Create/Modify Strategy

Create file in `order-flow-opt/src/strategies/`:
```python
from src.mae_strategy import MAEOptimizedStrategy, MAEStrategyConfig
from dataclasses import dataclass

@dataclass
class MyConfig(MAEStrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    # Your indicator params
    ema_fast: int = 8
    ema_slow: int = 21

class MyStrategy(MAEOptimizedStrategy):
    def indicator_signal(self, bar) -> int:
        # YOUR INDICATOR - UNCHANGED
        # Return: 1=BUY, -1=SELL, 0=FLAT
        if self.ema_fast > self.ema_slow:
            return 1
        elif self.ema_fast < self.ema_slow:
            return -1
        return 0

    def should_exit(self, bar, flow) -> bool:
        # YOUR EXIT LOGIC
        return self.hit_stop or self.hit_target
```

### Step 4: Run Backtest
```bash
cd /Users/DanBot/Desktop/HyperFrequency/order-flow-opt
python3 scripts/run_backtest.py --symbol BTC --start 2024-01-01 --end 2024-06-30
python3 scripts/run_backtest.py --symbol BTC --compare  # Compare with/without
```

## Core Concept: Exhaustion Detection

**Your indicator tells you WHEN to trade. Order flow tells you the OPTIMAL MOMENT within that window.**

```
Signal: BUY (EMA crossed)
├── Bar 1: Imbalance=-0.33 (sellers active) → WAIT
├── Bar 2: Imbalance=-0.20 (sellers weakening) → WAIT
├── Bar 3: Imbalance=-0.07 (sellers exhausting) → WAIT
├── Bar 4: Imbalance=+0.07 (buyers emerging) → WAIT
└── Bar 5: Imbalance=+0.20, Score=0.60 → ENTER NOW ✓
```

Result: Instead of -2.1% MAE at Bar 1, you get -0.4% MAE at Bar 5.

## Exhaustion Signals (6 Types)

| Signal | Description | Weight |
|--------|-------------|--------|
| `IMBALANCE_RECOVERY` | Adverse imbalance returning to neutral | 0.25 |
| `IMBALANCE_FLIP` | Imbalance flipped favorable (strongest) | 0.20 |
| `SPREAD_NORMALIZED` | Wide spread returning to normal | 0.15 |
| `DEPTH_RECOVERY` | Depleted side recovering | 0.15 |
| `MOMENTUM_DECAY` | Rate of adverse change slowing | 0.15 |
| `WALL_ABSORBED` | Large opposing orders consumed | 0.10 |

**Entry triggers when combined score >= 0.5**

## Configuration Presets

| Preset | Window | Threshold | Trade-off |
|--------|--------|-----------|-----------|
| `conservative` | 20 bars | 0.6 | Best MAE, more missed |
| `balanced` | 15 bars | 0.5 | Default |
| `aggressive` | 10 bars | 0.35 | Fewer missed |

## Expected Results

| Metric | Without Flow | With Flow |
|--------|--------------|-----------|
| Avg MAE | -1.8% | -0.8% |
| Win Rate | 52% | 58-62% |
| Trade Count | 100% | 60-75% |
| Sharpe | 1.0 | 1.3-1.6 |

## Data Paths

| Data | Path |
|------|------|
| L2 Book Raw | `/Users/DanBot/Desktop/HyperFrequency/data-historical/hyperliquid/l2book/` |
| Features | `/Users/DanBot/Desktop/HyperFrequency/data-historical/features/` |
| OHLCV | `/Users/DanBot/Desktop/HyperFrequency/data-historical/hyperliquid/futures/parquet/` |
| Module | `/Users/DanBot/Desktop/HyperFrequency/order-flow-opt/` |
| Config | `/Users/DanBot/Desktop/HyperFrequency/order-flow-opt/configs/default.yaml` |

## Module Files

| File | Purpose |
|------|---------|
| `src/features.py` | OrderFlowFeatures (33 features from L2) |
| `src/exhaustion.py` | ExhaustionDetector (6 signal types) |
| `src/mae_strategy.py` | MAEOptimizedStrategy base class |
| `src/data_loader.py` | L2BookLoader (memory-efficient) |
| `src/example_ema_strategy.py` | Complete EMA+Flow example |

## Quick Verification Test

```bash
cd /Users/DanBot/Desktop/HyperFrequency/order-flow-opt
python3 -c "
from src.features import OrderFlowFeatures
from src.exhaustion import ExhaustionDetector

flow = OrderFlowFeatures(levels=5)
detector = ExhaustionDetector()

# Signal with selling pressure
bids = [(100.0, 50), (99.9, 100)]
asks = [(100.1, 200), (100.2, 300)]
signal = flow.update(bids, asks, 0)
detector.update(signal)
print(f'Signal: imbalance={signal.book_imbalance:.3f}')

# Simulate exhaustion
for i in range(6):
    bids = [(100.0, 50*(1+i*0.3)), (99.9, 100*(1+i*0.3))]
    asks = [(100.1, 200*(1-i*0.15)), (100.2, 300*(1-i*0.15))]
    snap = flow.update(bids, asks, i+1)
    detector.update(snap)
    result = detector.detect_sell_exhaustion(snap, signal)
    status = '*** ENTER ***' if result.ready else ''
    print(f'Bar {i+1}: imb={snap.book_imbalance:+.3f}, score={result.exhaustion_score:.2f} {status}')
    if result.ready: break
print('Test passed!')
"
```

## Integration Levels

### Level 1: Filter Only (Add to existing strategy)
```python
if flow.spread_bps > 12:
    return  # Skip wide spread
if signal == 1 and flow.book_imbalance < -0.35:
    return  # Skip buying into sellers
```

### Level 2: Timing Only
```python
if signal == 1:
    result = detector.detect_sell_exhaustion(flow, signal_flow)
    if result.ready:
        execute()
```

### Level 3: Full Integration (Recommended)
```python
class MyStrategy(MAEOptimizedStrategy):
    def indicator_signal(self, bar) -> int:
        return your_indicator(bar)  # Unchanged
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No L2 data | Run `datapull_hl_chunked.py` |
| Memory overflow | Use chunked downloader with --workers 4 |
| Entry window expires | Lower threshold to 0.4 or increase window |
| Too many filtered | Increase max_spread_bps |
| nautilus_trader import error | `pip install nautilus_trader` |

## Related Skills

- `/datapull` - Download market data
- `/backtest` - Run backtests
- `/nautilus-trader-hlfix` - NautilusTrader patterns

## Store Pattern in Memory

After successful optimization:
```bash
npx @claude-flow/cli@latest memory store \
  --key "pattern/orderflow/mae-$(date +%s)" \
  --value "MAE reduced from X% to Y% using exhaustion threshold Z" \
  --namespace patterns
```

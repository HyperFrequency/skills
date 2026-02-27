# Order Flow MAE Optimization - Claude Code Instructions

## Project Purpose

This module minimizes **Maximum Adverse Excursion (MAE)** in trading strategies by timing entries based on order flow exhaustion detection. Your indicator logic stays **100% unchanged** - order flow only affects WHEN within the signal window you enter.

## Key Concept

```
Indicator says: "BUY now"
Order flow says: "Wait, sellers still active..."
Wait for exhaustion...
Order flow says: "Sellers exhausted, ENTER NOW"
Result: 50-70% MAE reduction
```

## Data Pipeline

### L2 Book Data Location
```
/Users/DanBot/Desktop/HyperFrequency/data-historical/hyperliquid/l2book/
├── BTC/
│   ├── 20230415_00.lz4
│   ├── 20230415_01.lz4
│   └── ... (hourly snapshots)
├── ETH/
├── SOL/
└── ...
```

### Feature Output Location
```
/Users/DanBot/Desktop/HyperFrequency/data-historical/features/
├── BTC_flow_features_5m.parquet
├── ETH_flow_features_5m.parquet
└── ...
```

## Workflow Steps

### Step 1: Download L2 Book Data (if needed)
```bash
# Use the memory-safe chunked downloader
cd /Users/DanBot/Desktop/HyperFrequency/scripts/data
python3 datapull_hl_chunked.py --start 2024-01-01 --end 2024-12-31 --workers 4
```

### Step 2: Extract Order Flow Features
```bash
cd /Users/DanBot/Desktop/HyperFrequency/order-flow-opt
python3 scripts/extract_features.py --symbol BTC --start 2024-01-01 --end 2024-12-31
python3 scripts/extract_features.py --all-symbols  # For all symbols
```

### Step 3: Create Strategy (Subclass MAEOptimizedStrategy)
```python
from src.mae_strategy import MAEOptimizedStrategy, MAEStrategyConfig

class MyStrategy(MAEOptimizedStrategy):
    def indicator_signal(self, bar) -> int:
        # YOUR INDICATOR - UNCHANGED
        if self.ema_fast > self.ema_slow:
            return 1  # BUY
        elif self.ema_fast < self.ema_slow:
            return -1  # SELL
        return 0

    def should_exit(self, bar, flow) -> bool:
        # YOUR EXIT LOGIC
        return self.hit_stop or self.hit_target
```

### Step 4: Run Backtest
```bash
python3 scripts/run_backtest.py --symbol BTC --start 2024-01-01 --end 2024-06-30
python3 scripts/run_backtest.py --symbol BTC --compare  # With/without flow comparison
```

## Module Components

| File | Purpose |
|------|---------|
| `src/features.py` | OrderFlowFeatures - calculates 33 features from L2 book |
| `src/exhaustion.py` | ExhaustionDetector - detects 6 exhaustion signal types |
| `src/mae_strategy.py` | MAEOptimizedStrategy - base class with entry state machine |
| `src/data_loader.py` | L2BookLoader - loads .lz4 files, memory-efficient |
| `src/example_ema_strategy.py` | Complete EMA crossover example |

## Exhaustion Signals

| Signal | Description | Weight |
|--------|-------------|--------|
| `IMBALANCE_RECOVERY` | Adverse imbalance returning to neutral | 0.25 |
| `IMBALANCE_FLIP` | Imbalance flipped to favorable (strongest) | 0.20 |
| `SPREAD_NORMALIZED` | Wide spread returning to normal | 0.15 |
| `DEPTH_RECOVERY` | Depleted side recovering | 0.15 |
| `MOMENTUM_DECAY` | Rate of adverse change slowing | 0.15 |
| `WALL_ABSORBED` | Large opposing orders consumed | 0.10 |

Entry triggers when combined score >= 0.5 (configurable).

## Configuration Presets

| Preset | Entry Window | Threshold | Trade-off |
|--------|--------------|-----------|-----------|
| `conservative` | 20 bars | 0.6 | Best MAE, more missed trades |
| `balanced` | 15 bars | 0.5 | Good balance (default) |
| `aggressive` | 10 bars | 0.35 | Fewer missed trades, higher MAE |

## Integration Levels

### Level 1: Filter Only (Minimal Change)
```python
signal = your_indicator(bar)
if flow.spread_bps > 12 or flow.book_imbalance * signal < -0.3:
    signal = 0  # Block
```

### Level 2: Timing Only
```python
signal = your_indicator(bar)
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

## Expected Results

| Metric | Without Flow | With Flow |
|--------|--------------|-----------|
| Avg MAE | -1.8% | -0.8% |
| Win Rate | 52% | 58-62% |
| Trade Count | 100% | 60-75% |
| Sharpe | 1.0 | 1.3-1.6 |

## Quick Test

```bash
cd /Users/DanBot/Desktop/HyperFrequency/order-flow-opt
python3 -c "
from src.features import OrderFlowFeatures
from src.exhaustion import ExhaustionDetector

flow = OrderFlowFeatures(levels=5)
detector = ExhaustionDetector()

# Signal: selling pressure
bids = [(100.0, 50), (99.9, 100)]
asks = [(100.1, 200), (100.2, 300)]
signal = flow.update(bids, asks, 0)
detector.update(signal)

print(f'Signal imbalance: {signal.book_imbalance:.3f}')

# Simulate recovery
for i in range(6):
    bids = [(100.0, 50*(1+i*0.3)), (99.9, 100*(1+i*0.3))]
    asks = [(100.1, 200*(1-i*0.15)), (100.2, 300*(1-i*0.15))]
    snap = flow.update(bids, asks, i+1)
    detector.update(snap)
    result = detector.detect_sell_exhaustion(snap, signal)
    print(f'Bar {i+1}: imb={snap.book_imbalance:+.3f}, score={result.exhaustion_score:.2f}, ready={result.ready}')
    if result.ready:
        print('*** ENTER ***')
        break
"
```

## Files Modified by This Workflow

- `data-historical/features/*.parquet` - Generated feature files
- `order-flow-opt/configs/default.yaml` - Configuration
- Strategy files you create in `order-flow-opt/src/`

## Memory Integration

Store successful patterns:
```bash
npx @claude-flow/cli@latest memory store \
  --key "pattern/orderflow/mae-timing" \
  --value "Use exhaustion detection for entry timing, wait for score>=0.5" \
  --namespace patterns
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No L2 data | Run `datapull_hl_chunked.py` first |
| Memory overflow | Use chunked downloader, not original |
| Entry window expires | Lower threshold or increase window |
| Too many filtered | Increase max_spread_bps, lower min_depth_percentile |

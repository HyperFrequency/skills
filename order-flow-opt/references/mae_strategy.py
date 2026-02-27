"""
MAE-Optimized Strategy Base for NautilusTrader

Base strategy class that implements MAE-optimized entry timing
using order flow exhaustion detection.

Features:
- Pending entry state machine (IDLE -> PENDING -> POSITIONED)
- Exhaustion-based entry timing
- MAE tracking and logging
- Order flow filtering
- Position sizing based on flow conditions

Usage:
    class MyStrategy(MAEOptimizedStrategy):
        def indicator_signal(self, bar) -> int:
            # Your indicator logic here
            return 1  # BUY, -1 SELL, 0 FLAT

        def should_exit(self, bar, flow) -> bool:
            # Your exit logic here
            return False
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, List
import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.data import Bar, OrderBookDeltas, OrderBookDepth10
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.orders import MarketOrder

from .features import OrderFlowFeatures, OrderFlowSnapshot
from .exhaustion import ExhaustionDetector, ExhaustionConfig, ExhaustionSignal


class EntryState(Enum):
    """State machine for entry timing."""
    IDLE = "idle"                  # No signal, waiting
    PENDING_LONG = "pending_long"  # Have BUY signal, waiting for sell exhaustion
    PENDING_SHORT = "pending_short"  # Have SELL signal, waiting for buy exhaustion
    POSITIONED = "positioned"      # In position


@dataclass
class MAEStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for MAE-optimized strategy.

    Indicator params should be added in subclass config.
    """
    # Instrument
    instrument_id: str

    # Entry window
    entry_window_bars: int = 15          # Max bars to wait for exhaustion
    require_exhaustion: bool = True       # If False, enter immediately on signal

    # Exhaustion thresholds
    exhaustion_threshold: float = 0.5     # Score needed to trigger entry
    immediate_entry_imbalance: float = 0.25  # Skip waiting if flow already favorable

    # Position sizing
    base_position_usd: float = 10000.0
    max_position_usd: float = 50000.0
    min_position_usd: float = 1000.0
    use_flow_sizing: bool = True          # Scale size by flow conditions

    # Flow filters
    max_spread_bps: float = 15.0          # Skip if spread too wide
    min_depth_percentile: float = 0.2     # Skip if book too thin
    block_opposing_flow: bool = True      # Skip if flow strongly opposes signal
    opposing_flow_threshold: float = 0.35  # Threshold for blocking

    # Order flow feature settings
    book_levels: int = 10
    flow_buffer_size: int = 100

    # MAE logging
    log_mae: bool = True


class MAEOptimizedStrategy(Strategy):
    """
    Base strategy with MAE-optimized entry timing.

    Subclass this and implement:
    - indicator_signal(bar) -> int: Your indicator logic
    - should_exit(bar, flow) -> bool: Your exit logic

    The base class handles:
    - Order flow feature calculation
    - Exhaustion detection and timing
    - Entry state machine
    - Position sizing
    - MAE tracking
    """

    def __init__(self, config: MAEStrategyConfig):
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.instrument: Optional[Instrument] = None

        # Config
        self.entry_window_bars = config.entry_window_bars
        self.require_exhaustion = config.require_exhaustion
        self.exhaustion_threshold = config.exhaustion_threshold
        self.immediate_entry_imbalance = config.immediate_entry_imbalance

        self.base_position_usd = config.base_position_usd
        self.max_position_usd = config.max_position_usd
        self.min_position_usd = config.min_position_usd
        self.use_flow_sizing = config.use_flow_sizing

        self.max_spread_bps = config.max_spread_bps
        self.min_depth_percentile = config.min_depth_percentile
        self.block_opposing_flow = config.block_opposing_flow
        self.opposing_flow_threshold = config.opposing_flow_threshold

        self.log_mae = config.log_mae

        # Order flow components
        self.flow = OrderFlowFeatures(
            levels=config.book_levels,
            buffer_size=config.flow_buffer_size
        )
        self.exhaustion = ExhaustionDetector(
            config=ExhaustionConfig(exhaustion_threshold=config.exhaustion_threshold)
        )

        # State
        self.state = EntryState.IDLE
        self.pending_signal: Optional[int] = None
        self.signal_bar_count: int = 0
        self.signal_flow: Optional[OrderFlowSnapshot] = None
        self.bar_count: int = 0

        # Current position tracking
        self.entry_price: float = 0.0
        self.entry_bar: int = 0
        self.position_direction: int = 0
        self.mae: float = 0.0  # Most negative excursion

        # MAE log
        self.mae_log: List[Dict] = []

        # Latest flow snapshot
        self._latest_flow: Optional[OrderFlowSnapshot] = None

    def on_start(self):
        """Strategy startup - subscribe to data."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument: {self.instrument_id}")
            self.stop()
            return

        # Subscribe to order book (for flow features)
        # Note: Adjust based on your data type (depth10, deltas, etc.)
        self.subscribe_order_book_deltas(self.instrument_id)

        self.log.info(f"MAE Strategy started for {self.instrument_id}")
        self.log.info(f"  Entry window: {self.entry_window_bars} bars")
        self.log.info(f"  Exhaustion threshold: {self.exhaustion_threshold}")
        self.log.info(f"  Require exhaustion: {self.require_exhaustion}")

    def on_order_book_deltas(self, deltas: OrderBookDeltas):
        """Update flow features from order book deltas."""
        # Get current book state from cache
        book = self.cache.order_book(self.instrument_id)
        if book is None:
            return

        # Extract bids and asks
        bids = [(float(level.price), float(level.size)) for level in book.bids()]
        asks = [(float(level.price), float(level.size)) for level in book.asks()]

        # Update flow features
        self._latest_flow = self.flow.update(
            bids=bids,
            asks=asks,
            timestamp_ns=deltas.ts_event
        )

        # Update exhaustion detector
        self.exhaustion.update(self._latest_flow)

    def on_bar(self, bar: Bar):
        """Main strategy logic on bar close."""
        self.bar_count += 1

        # Get flow snapshot (use latest or calculate from book)
        flow = self._latest_flow
        if flow is None:
            return  # No flow data yet

        # Update position MAE tracking
        if self.state == EntryState.POSITIONED:
            self._update_mae(bar)

        # State machine
        if self.state == EntryState.IDLE:
            self._handle_idle_state(bar, flow)

        elif self.state == EntryState.PENDING_LONG:
            self._handle_pending_long(bar, flow)

        elif self.state == EntryState.PENDING_SHORT:
            self._handle_pending_short(bar, flow)

        elif self.state == EntryState.POSITIONED:
            self._handle_positioned(bar, flow)

    def _handle_idle_state(self, bar: Bar, flow: OrderFlowSnapshot):
        """Check for new indicator signal."""
        signal = self.indicator_signal(bar)

        if signal == 0:
            return

        # Apply flow filter
        if not self._flow_filter_pass(signal, flow):
            self.log.debug(f"Signal {signal} blocked by flow filter")
            return

        # Check if we can enter immediately
        if not self.require_exhaustion:
            self._execute_entry(signal, bar, flow, "immediate_no_exhaustion")
            return

        # Check if flow already favorable (skip exhaustion wait)
        if signal == 1 and flow.book_imbalance > self.immediate_entry_imbalance:
            self._execute_entry(signal, bar, flow, "immediate_favorable")
            return
        elif signal == -1 and flow.book_imbalance < -self.immediate_entry_imbalance:
            self._execute_entry(signal, bar, flow, "immediate_favorable")
            return

        # Queue for exhaustion-timed entry
        if signal == 1:
            self.state = EntryState.PENDING_LONG
            self.log.info(f"BUY signal queued, waiting for sell exhaustion. "
                         f"Imbalance: {flow.book_imbalance:.3f}")
        else:
            self.state = EntryState.PENDING_SHORT
            self.log.info(f"SELL signal queued, waiting for buy exhaustion. "
                         f"Imbalance: {flow.book_imbalance:.3f}")

        self.pending_signal = signal
        self.signal_bar_count = self.bar_count
        self.signal_flow = flow

    def _handle_pending_long(self, bar: Bar, flow: OrderFlowSnapshot):
        """Wait for sell exhaustion before long entry."""
        bars_waited = self.bar_count - self.signal_bar_count

        # Check window expiry
        if bars_waited > self.entry_window_bars:
            self.log.info(f"Entry window expired after {bars_waited} bars")
            self._reset_pending()
            return

        # Check for exhaustion
        result = self.exhaustion.detect_sell_exhaustion(flow, self.signal_flow)

        if result.ready:
            signals_str = ", ".join(s.value for s in result.signals)
            self.log.info(f"Sell exhaustion detected after {bars_waited} bars. "
                         f"Score: {result.exhaustion_score:.2f}, Signals: [{signals_str}]")
            self._execute_entry(1, bar, flow, "exhaustion_timed", bars_waited)
        else:
            if bars_waited % 5 == 0:
                self.log.debug(f"Waiting: {bars_waited}/{self.entry_window_bars} bars, "
                              f"score: {result.exhaustion_score:.2f}")

    def _handle_pending_short(self, bar: Bar, flow: OrderFlowSnapshot):
        """Wait for buy exhaustion before short entry."""
        bars_waited = self.bar_count - self.signal_bar_count

        if bars_waited > self.entry_window_bars:
            self.log.info(f"Entry window expired after {bars_waited} bars")
            self._reset_pending()
            return

        result = self.exhaustion.detect_buy_exhaustion(flow, self.signal_flow)

        if result.ready:
            signals_str = ", ".join(s.value for s in result.signals)
            self.log.info(f"Buy exhaustion detected after {bars_waited} bars. "
                         f"Score: {result.exhaustion_score:.2f}, Signals: [{signals_str}]")
            self._execute_entry(-1, bar, flow, "exhaustion_timed", bars_waited)
        else:
            if bars_waited % 5 == 0:
                self.log.debug(f"Waiting: {bars_waited}/{self.entry_window_bars} bars, "
                              f"score: {result.exhaustion_score:.2f}")

    def _handle_positioned(self, bar: Bar, flow: OrderFlowSnapshot):
        """Manage open position."""
        if self.should_exit(bar, flow):
            self._execute_exit(bar, flow)

    def _flow_filter_pass(self, signal: int, flow: OrderFlowSnapshot) -> bool:
        """Check if flow conditions allow trade."""
        # Spread check
        if flow.spread_bps > self.max_spread_bps:
            self.log.debug(f"Spread too wide: {flow.spread_bps:.1f} > {self.max_spread_bps}")
            return False

        # Depth check
        depth_pct = self.flow.get_depth_percentile()
        if depth_pct < self.min_depth_percentile:
            self.log.debug(f"Book too thin: {depth_pct:.2f} < {self.min_depth_percentile}")
            return False

        # Opposing flow check
        if self.block_opposing_flow:
            if signal == 1 and flow.book_imbalance < -self.opposing_flow_threshold:
                self.log.debug(f"Strong opposing flow for BUY: {flow.book_imbalance:.3f}")
                return False
            if signal == -1 and flow.book_imbalance > self.opposing_flow_threshold:
                self.log.debug(f"Strong opposing flow for SELL: {flow.book_imbalance:.3f}")
                return False

        return True

    def _calculate_position_size(self, signal: int, flow: OrderFlowSnapshot) -> float:
        """Calculate position size based on flow conditions."""
        if not self.use_flow_sizing:
            return self.base_position_usd

        size = self.base_position_usd

        # Liquidity scaling
        depth_pct = self.flow.get_depth_percentile()
        liq_mult = min(1.5, max(0.5, depth_pct * 2))
        size *= liq_mult

        # Spread penalty
        if flow.spread_bps > 5:
            spread_mult = max(0.5, 1 - (flow.spread_bps - 5) / 30)
            size *= spread_mult

        # Flow alignment bonus
        if signal == 1 and flow.book_imbalance > 0.2:
            size *= 1.0 + min(flow.book_imbalance, 0.5)
        elif signal == -1 and flow.book_imbalance < -0.2:
            size *= 1.0 + min(abs(flow.book_imbalance), 0.5)
        elif signal == 1 and flow.book_imbalance < -0.15:
            size *= 0.6  # Reduce size if slight opposing flow
        elif signal == -1 and flow.book_imbalance > 0.15:
            size *= 0.6

        # Clamp
        return max(self.min_position_usd, min(self.max_position_usd, size))

    def _execute_entry(
        self,
        direction: int,
        bar: Bar,
        flow: OrderFlowSnapshot,
        entry_type: str,
        bars_waited: int = 0
    ):
        """Execute entry order."""
        size_usd = self._calculate_position_size(direction, flow)
        price = float(flow.mid_price)
        quantity = self.instrument.make_qty(Decimal(str(size_usd / price)))

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY if direction == 1 else OrderSide.SELL,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
        )

        self.submit_order(order)

        # Update state
        self.state = EntryState.POSITIONED
        self.entry_price = price
        self.entry_bar = self.bar_count
        self.position_direction = direction
        self.mae = 0.0

        self.log.info(f"ENTRY {['SHORT', '', 'LONG'][direction + 1]} | "
                     f"Price: {price:.4f} | Size: ${size_usd:,.0f} | "
                     f"Type: {entry_type} | Waited: {bars_waited} bars | "
                     f"Imbalance: {flow.book_imbalance:.3f}")

        # Store entry info for MAE log
        self._entry_info = {
            'direction': direction,
            'entry_price': price,
            'entry_type': entry_type,
            'bars_waited': bars_waited,
            'entry_bar': self.bar_count,
            'imbalance_at_signal': self.signal_flow.book_imbalance if self.signal_flow else None,
            'imbalance_at_entry': flow.book_imbalance,
            'spread_at_entry': flow.spread_bps,
            'size_usd': size_usd,
        }

        self._reset_pending()

    def _execute_exit(self, bar: Bar, flow: OrderFlowSnapshot):
        """Execute exit order."""
        position = self.portfolio.net_position(self.instrument_id)
        if position == 0:
            self.state = EntryState.IDLE
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL if position > 0 else OrderSide.BUY,
            quantity=self.instrument.make_qty(abs(position)),
            time_in_force=TimeInForce.IOC,
        )

        self.submit_order(order)

        # Calculate PnL
        exit_price = float(flow.mid_price)
        if self.position_direction == 1:
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - exit_price) / self.entry_price

        bars_held = self.bar_count - self.entry_bar

        self.log.info(f"EXIT | Price: {exit_price:.4f} | PnL: {pnl_pct*100:.2f}% | "
                     f"MAE: {self.mae*100:.2f}% | Held: {bars_held} bars")

        # Log MAE data
        if self.log_mae and hasattr(self, '_entry_info'):
            self.mae_log.append({
                **self._entry_info,
                'exit_price': exit_price,
                'exit_bar': self.bar_count,
                'bars_held': bars_held,
                'pnl_pct': pnl_pct,
                'mae': self.mae,
            })

        self.state = EntryState.IDLE
        self.position_direction = 0

    def _update_mae(self, bar: Bar):
        """Track maximum adverse excursion during position."""
        if self.position_direction == 0:
            return

        if self.position_direction == 1:  # Long
            excursion = (float(bar.low) - self.entry_price) / self.entry_price
        else:  # Short
            excursion = (self.entry_price - float(bar.high)) / self.entry_price

        self.mae = min(self.mae, excursion)

    def _reset_pending(self):
        """Reset pending entry state."""
        self.pending_signal = None
        self.signal_bar_count = 0
        self.signal_flow = None

    # === Methods to override in subclass ===

    def indicator_signal(self, bar: Bar) -> int:
        """
        Override this with your indicator logic.

        Returns:
            1 for BUY signal
            -1 for SELL signal
            0 for no signal
        """
        raise NotImplementedError("Subclass must implement indicator_signal()")

    def should_exit(self, bar: Bar, flow: OrderFlowSnapshot) -> bool:
        """
        Override this with your exit logic.

        Args:
            bar: Current bar
            flow: Current order flow snapshot

        Returns:
            True to exit position, False to hold
        """
        raise NotImplementedError("Subclass must implement should_exit()")

    # === Analysis methods ===

    def get_mae_summary(self) -> Dict:
        """Get summary statistics of MAE log."""
        if not self.mae_log:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.mae_log)

        summary = {
            'total_trades': len(df),
            'avg_mae': df['mae'].mean(),
            'median_mae': df['mae'].median(),
            'worst_mae': df['mae'].min(),
            'avg_pnl': df['pnl_pct'].mean(),
            'win_rate': (df['pnl_pct'] > 0).mean(),
        }

        # By entry type
        if 'entry_type' in df.columns:
            by_type = df.groupby('entry_type').agg({
                'mae': ['mean', 'median', 'min', 'count'],
                'pnl_pct': ['mean', lambda x: (x > 0).mean()]
            })
            summary['by_entry_type'] = by_type.to_dict()

        return summary

    def on_stop(self):
        """Log MAE summary on strategy stop."""
        if self.log_mae and self.mae_log:
            summary = self.get_mae_summary()
            self.log.info("=" * 50)
            self.log.info("MAE SUMMARY")
            self.log.info("=" * 50)
            self.log.info(f"Total trades: {summary.get('total_trades', 0)}")
            self.log.info(f"Avg MAE: {summary.get('avg_mae', 0)*100:.2f}%")
            self.log.info(f"Median MAE: {summary.get('median_mae', 0)*100:.2f}%")
            self.log.info(f"Worst MAE: {summary.get('worst_mae', 0)*100:.2f}%")
            self.log.info(f"Win Rate: {summary.get('win_rate', 0)*100:.1f}%")

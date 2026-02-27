"""
Order Flow Feature Extraction

Calculates real-time features from L2 order book data for:
- Book imbalance (directional pressure)
- Microstructure metrics (spread, depth, microprice)
- Volume profile analysis
- Wall detection
- Temporal dynamics
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque
from collections import deque
from decimal import Decimal
import numpy as np


@dataclass
class OrderFlowSnapshot:
    """
    Point-in-time snapshot of order flow features.
    All features derived from L2 book state.
    """
    timestamp_ns: int

    # Microstructure
    mid_price: float
    spread: float
    spread_bps: float
    microprice: float
    microprice_dev: float  # Deviation from mid as fraction of spread

    # Depth
    bid_depth: float       # Total bid volume (levels 1-N)
    ask_depth: float       # Total ask volume (levels 1-N)
    total_depth: float
    bid_depth_usd: float   # Dollar value of bid depth
    ask_depth_usd: float

    # Imbalance
    book_imbalance: float          # (bid - ask) / (bid + ask), range [-1, 1]
    tob_imbalance: float           # Top-of-book only
    weighted_imbalance: float      # Price-weighted (closer levels matter more)
    imbalance_3: float             # Levels 1-3
    imbalance_5: float             # Levels 1-5
    imbalance_10: float            # Levels 1-10

    # Volume concentration
    vol_concentration_3: float     # Top 3 levels as % of total
    vol_concentration_5: float

    # Wall detection
    bid_wall_level: int            # Level with largest bid (1-indexed)
    ask_wall_level: int
    bid_wall_strength: float       # Wall size / avg size
    ask_wall_strength: float
    bid_wall_distance_bps: float   # Distance from mid to wall
    ask_wall_distance_bps: float

    # Raw data (for advanced analysis)
    best_bid: float = 0.0
    best_ask: float = 0.0
    best_bid_size: float = 0.0
    best_ask_size: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in self.__dict__.items()}


class OrderFlowFeatures:
    """
    Real-time order flow feature calculator.

    Maintains rolling buffers for temporal features and
    calculates all metrics from order book updates.

    Usage:
        flow = OrderFlowFeatures(levels=10, buffer_size=100)

        # On each book update
        snapshot = flow.update(bids, asks, timestamp_ns)

        # Access features
        print(f"Imbalance: {snapshot.book_imbalance:.3f}")
        print(f"Spread: {snapshot.spread_bps:.1f} bps")

        # Get temporal features
        temporal = flow.get_temporal_features(windows=[5, 15, 60])
    """

    def __init__(self, levels: int = 10, buffer_size: int = 100):
        """
        Args:
            levels: Number of order book levels to analyze
            buffer_size: Rolling buffer size for temporal features
        """
        self.levels = levels
        self.buffer_size = buffer_size

        # Rolling buffers for temporal analysis
        self._imbalance_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._spread_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._depth_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._mid_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._microprice_buffer: Deque[float] = deque(maxlen=buffer_size)

        # Latest snapshot
        self._latest: Optional[OrderFlowSnapshot] = None

        # Statistics for percentile calculations
        self._depth_history: Deque[float] = deque(maxlen=1000)
        self._spread_history: Deque[float] = deque(maxlen=1000)

    def update(
        self,
        bids: List[tuple],  # [(price, size), ...] best to worst
        asks: List[tuple],  # [(price, size), ...] best to worst
        timestamp_ns: int
    ) -> OrderFlowSnapshot:
        """
        Update features with new order book state.

        Args:
            bids: List of (price, size) tuples, best bid first
            asks: List of (price, size) tuples, best ask first
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            OrderFlowSnapshot with all calculated features
        """
        # Ensure we have enough levels
        bids = bids[:self.levels] if len(bids) >= self.levels else bids
        asks = asks[:self.levels] if len(asks) >= self.levels else asks

        if not bids or not asks:
            return self._empty_snapshot(timestamp_ns)

        # Extract prices and sizes
        bid_prices = np.array([float(b[0]) for b in bids])
        bid_sizes = np.array([float(b[1]) for b in bids])
        ask_prices = np.array([float(a[0]) for a in asks])
        ask_sizes = np.array([float(a[1]) for a in asks])

        # === Microstructure ===
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0

        # Microprice (volume-weighted fair value)
        best_bid_size = bid_sizes[0]
        best_ask_size = ask_sizes[0]
        total_tob = best_bid_size + best_ask_size

        if total_tob > 0:
            microprice = (best_bid * best_ask_size + best_ask * best_bid_size) / total_tob
        else:
            microprice = mid_price

        microprice_dev = (microprice - mid_price) / spread if spread > 0 else 0

        # === Depth ===
        bid_depth = float(np.sum(bid_sizes))
        ask_depth = float(np.sum(ask_sizes))
        total_depth = bid_depth + ask_depth

        # Dollar depth
        bid_depth_usd = float(np.sum(bid_prices * bid_sizes))
        ask_depth_usd = float(np.sum(ask_prices * ask_sizes))

        # === Imbalance ===
        book_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        tob_imbalance = (best_bid_size - best_ask_size) / total_tob if total_tob > 0 else 0

        # Weighted imbalance (1/level weighting)
        n_levels = min(len(bid_sizes), len(ask_sizes))
        if n_levels > 0:
            weights = 1.0 / np.arange(1, n_levels + 1)
            weighted_bid = np.sum(bid_sizes[:n_levels] * weights)
            weighted_ask = np.sum(ask_sizes[:n_levels] * weights)
            weighted_total = weighted_bid + weighted_ask
            weighted_imbalance = (weighted_bid - weighted_ask) / weighted_total if weighted_total > 0 else 0
        else:
            weighted_imbalance = 0

        # Multi-level imbalances
        imbalance_3 = self._calc_level_imbalance(bid_sizes, ask_sizes, 3)
        imbalance_5 = self._calc_level_imbalance(bid_sizes, ask_sizes, 5)
        imbalance_10 = self._calc_level_imbalance(bid_sizes, ask_sizes, 10)

        # === Volume Concentration ===
        vol_concentration_3 = (np.sum(bid_sizes[:3]) + np.sum(ask_sizes[:3])) / total_depth if total_depth > 0 else 0
        vol_concentration_5 = (np.sum(bid_sizes[:5]) + np.sum(ask_sizes[:5])) / total_depth if total_depth > 0 else 0

        # === Wall Detection ===
        bid_wall_level, bid_wall_strength = self._detect_wall(bid_sizes)
        ask_wall_level, ask_wall_strength = self._detect_wall(ask_sizes)

        # Wall distance from mid
        if bid_wall_level > 0 and bid_wall_level <= len(bid_prices):
            bid_wall_distance_bps = (mid_price - bid_prices[bid_wall_level - 1]) / mid_price * 10000
        else:
            bid_wall_distance_bps = 0

        if ask_wall_level > 0 and ask_wall_level <= len(ask_prices):
            ask_wall_distance_bps = (ask_prices[ask_wall_level - 1] - mid_price) / mid_price * 10000
        else:
            ask_wall_distance_bps = 0

        # Create snapshot
        snapshot = OrderFlowSnapshot(
            timestamp_ns=timestamp_ns,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            microprice=microprice,
            microprice_dev=microprice_dev,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd,
            book_imbalance=book_imbalance,
            tob_imbalance=tob_imbalance,
            weighted_imbalance=weighted_imbalance,
            imbalance_3=imbalance_3,
            imbalance_5=imbalance_5,
            imbalance_10=imbalance_10,
            vol_concentration_3=vol_concentration_3,
            vol_concentration_5=vol_concentration_5,
            bid_wall_level=bid_wall_level,
            ask_wall_level=ask_wall_level,
            bid_wall_strength=bid_wall_strength,
            ask_wall_strength=ask_wall_strength,
            bid_wall_distance_bps=bid_wall_distance_bps,
            ask_wall_distance_bps=ask_wall_distance_bps,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
        )

        # Update buffers
        self._imbalance_buffer.append(book_imbalance)
        self._spread_buffer.append(spread_bps)
        self._depth_buffer.append(total_depth)
        self._mid_buffer.append(mid_price)
        self._microprice_buffer.append(microprice)

        # Update history for percentiles
        self._depth_history.append(total_depth)
        self._spread_history.append(spread_bps)

        self._latest = snapshot
        return snapshot

    def _calc_level_imbalance(self, bid_sizes: np.ndarray, ask_sizes: np.ndarray, levels: int) -> float:
        """Calculate imbalance for specific number of levels."""
        bid_sum = np.sum(bid_sizes[:levels])
        ask_sum = np.sum(ask_sizes[:levels])
        total = bid_sum + ask_sum
        return (bid_sum - ask_sum) / total if total > 0 else 0

    def _detect_wall(self, sizes: np.ndarray) -> tuple:
        """
        Detect price wall (unusually large order).
        Returns (level, strength) where strength = size / avg_size.
        """
        if len(sizes) == 0:
            return 0, 0.0

        avg_size = np.mean(sizes)
        if avg_size == 0:
            return 0, 0.0

        max_idx = np.argmax(sizes)
        max_size = sizes[max_idx]
        strength = max_size / avg_size

        # Only count as wall if significantly larger than average
        if strength >= 2.0:
            return int(max_idx + 1), float(strength)
        return 0, 0.0

    def _empty_snapshot(self, timestamp_ns: int) -> OrderFlowSnapshot:
        """Return empty snapshot when no data available."""
        return OrderFlowSnapshot(
            timestamp_ns=timestamp_ns,
            mid_price=0, spread=0, spread_bps=0,
            microprice=0, microprice_dev=0,
            bid_depth=0, ask_depth=0, total_depth=0,
            bid_depth_usd=0, ask_depth_usd=0,
            book_imbalance=0, tob_imbalance=0, weighted_imbalance=0,
            imbalance_3=0, imbalance_5=0, imbalance_10=0,
            vol_concentration_3=0, vol_concentration_5=0,
            bid_wall_level=0, ask_wall_level=0,
            bid_wall_strength=0, ask_wall_strength=0,
            bid_wall_distance_bps=0, ask_wall_distance_bps=0,
        )

    def get_temporal_features(self, windows: List[int] = None) -> Dict[str, float]:
        """
        Calculate temporal/rolling features.

        Args:
            windows: List of lookback periods (in updates)

        Returns:
            Dictionary of temporal features
        """
        if windows is None:
            windows = [5, 15, 60]

        features = {}
        imb = list(self._imbalance_buffer)
        spread = list(self._spread_buffer)
        depth = list(self._depth_buffer)

        for w in windows:
            if len(imb) >= w:
                # Imbalance stats
                imb_window = imb[-w:]
                features[f'imbalance_ma_{w}'] = np.mean(imb_window)
                features[f'imbalance_std_{w}'] = np.std(imb_window)
                features[f'imbalance_mom_{w}'] = imb[-1] - imb[-w] if len(imb) >= w else 0

                # Z-score
                std = features[f'imbalance_std_{w}']
                if std > 0:
                    features[f'imbalance_zscore_{w}'] = (imb[-1] - features[f'imbalance_ma_{w}']) / std
                else:
                    features[f'imbalance_zscore_{w}'] = 0

                # Spread stats
                spread_window = spread[-w:]
                features[f'spread_ma_{w}'] = np.mean(spread_window)
                features[f'spread_std_{w}'] = np.std(spread_window)

                # Depth change
                depth_window = depth[-w:]
                if depth_window[0] > 0:
                    features[f'depth_change_{w}'] = (depth[-1] - depth_window[0]) / depth_window[0]
                else:
                    features[f'depth_change_{w}'] = 0
            else:
                # Not enough data
                for suffix in ['ma', 'std', 'mom', 'zscore']:
                    features[f'imbalance_{suffix}_{w}'] = 0
                features[f'spread_ma_{w}'] = 0
                features[f'spread_std_{w}'] = 0
                features[f'depth_change_{w}'] = 0

        return features

    def get_depth_percentile(self) -> float:
        """Get current depth as percentile of recent history."""
        if not self._depth_history or not self._latest:
            return 0.5

        current = self._latest.total_depth
        history = list(self._depth_history)
        return sum(1 for d in history if d <= current) / len(history)

    def get_spread_percentile(self) -> float:
        """Get current spread as percentile of recent history."""
        if not self._spread_history or not self._latest:
            return 0.5

        current = self._latest.spread_bps
        history = list(self._spread_history)
        return sum(1 for s in history if s <= current) / len(history)

    @property
    def latest(self) -> Optional[OrderFlowSnapshot]:
        """Get most recent snapshot."""
        return self._latest

    def reset(self):
        """Clear all buffers and state."""
        self._imbalance_buffer.clear()
        self._spread_buffer.clear()
        self._depth_buffer.clear()
        self._mid_buffer.clear()
        self._microprice_buffer.clear()
        self._depth_history.clear()
        self._spread_history.clear()
        self._latest = None

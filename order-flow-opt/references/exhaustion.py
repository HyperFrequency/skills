"""
Exhaustion Detection for MAE Optimization

Detects when adverse order flow pressure is exhausting,
signaling optimal entry timing to minimize Maximum Adverse Excursion.

Exhaustion Patterns:
- Imbalance recovery (adverse imbalance returning to neutral/favorable)
- Spread normalization (wide spread returning to normal)
- Depth recovery (depleted side recovering)
- Momentum decay (rate of adverse change slowing)
- Wall absorption (large orders being consumed)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Set
from collections import deque
from enum import Enum
import numpy as np

from .features import OrderFlowSnapshot


class ExhaustionType(Enum):
    """Types of exhaustion signals detected."""
    IMBALANCE_RECOVERY = "imbalance_recovery"
    SPREAD_NORMALIZED = "spread_normalized"
    DEPTH_RECOVERY = "depth_recovery"
    MOMENTUM_DECAY = "momentum_decay"
    IMBALANCE_FLIP = "imbalance_flip"
    WALL_ABSORBED = "wall_absorbed"
    TOB_FLIP = "tob_flip"


@dataclass
class ExhaustionSignal:
    """
    Result of exhaustion detection.

    Attributes:
        exhaustion_score: 0.0 to 1.0, higher = more exhausted
        ready: True if score exceeds threshold (safe to enter)
        signals: Set of exhaustion types detected
        details: Additional signal details
        bars_analyzed: Number of bars in analysis window
    """
    exhaustion_score: float
    ready: bool
    signals: Set[ExhaustionType]
    details: Dict[str, float]
    bars_analyzed: int

    def __repr__(self):
        sig_names = [s.value for s in self.signals]
        return f"ExhaustionSignal(score={self.exhaustion_score:.2f}, ready={self.ready}, signals={sig_names})"


@dataclass
class ExhaustionConfig:
    """Configuration for exhaustion detection thresholds."""

    # Score threshold to consider exhaustion complete
    exhaustion_threshold: float = 0.5

    # Imbalance recovery
    imbalance_recovery_min: float = 0.15  # Min change to count as recovery
    imbalance_flip_threshold: float = 0.1  # Threshold for flip detection

    # Spread normalization
    spread_elevated_mult: float = 1.3  # Spread > this * avg = elevated
    spread_normal_mult: float = 1.1    # Spread < this * avg = normalized

    # Depth recovery
    depth_recovery_min: float = 1.1  # Depth must recover to this * signal_depth

    # Momentum decay
    momentum_decay_factor: float = 0.5  # Recent momentum < this * older = decaying

    # Wall absorption
    wall_absorption_factor: float = 0.7  # Wall < this * signal_wall = absorbed

    # Weights for scoring
    weight_imbalance_recovery: float = 0.25
    weight_spread_normalized: float = 0.15
    weight_depth_recovery: float = 0.15
    weight_momentum_decay: float = 0.15
    weight_imbalance_flip: float = 0.20
    weight_wall_absorbed: float = 0.10


class ExhaustionDetector:
    """
    Detects when adverse order flow pressure is exhausting.

    For BUY signals: detects selling pressure exhaustion
    For SELL signals: detects buying pressure exhaustion

    Usage:
        detector = ExhaustionDetector()

        # When indicator generates signal, capture flow state
        signal_flow = flow_features.latest

        # On each subsequent bar, check for exhaustion
        for bar in bars:
            flow = flow_features.update(book_data)
            detector.update(flow)

            if signal_direction == 1:  # BUY
                result = detector.detect_sell_exhaustion(flow, signal_flow)
            else:  # SELL
                result = detector.detect_buy_exhaustion(flow, signal_flow)

            if result.ready:
                execute_entry()
                break
    """

    def __init__(self, config: ExhaustionConfig = None, buffer_size: int = 30):
        """
        Args:
            config: Exhaustion detection thresholds
            buffer_size: Rolling buffer size for temporal analysis
        """
        self.config = config or ExhaustionConfig()
        self.buffer_size = buffer_size

        # Rolling buffers
        self._imbalance_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._spread_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._depth_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._tob_imbalance_buffer: Deque[float] = deque(maxlen=buffer_size)

        # Snapshot history for wall tracking
        self._snapshot_buffer: Deque[OrderFlowSnapshot] = deque(maxlen=buffer_size)

    def update(self, snapshot: OrderFlowSnapshot):
        """
        Update internal buffers with new snapshot.
        Call this on every bar/update.
        """
        self._imbalance_buffer.append(snapshot.book_imbalance)
        self._spread_buffer.append(snapshot.spread_bps)
        self._depth_buffer.append(snapshot.total_depth)
        self._tob_imbalance_buffer.append(snapshot.tob_imbalance)
        self._snapshot_buffer.append(snapshot)

    def detect_sell_exhaustion(
        self,
        current: OrderFlowSnapshot,
        signal_snapshot: OrderFlowSnapshot
    ) -> ExhaustionSignal:
        """
        Detect selling pressure exhaustion for BUY entry timing.

        We're waiting for sellers to exhaust before entering long.

        Signals checked:
        1. Imbalance was negative, now recovering toward zero/positive
        2. Spread was wide (panic), now normalizing
        3. Bid depth was depleted, now recovering
        4. Selling momentum decaying (rate of change slowing)
        5. Ask wall being absorbed (sellers getting filled)
        6. Top-of-book imbalance flipping

        Args:
            current: Current order flow snapshot
            signal_snapshot: Snapshot when indicator generated signal

        Returns:
            ExhaustionSignal with score, ready flag, and detected signals
        """
        cfg = self.config
        score = 0.0
        signals: Set[ExhaustionType] = set()
        details: Dict[str, float] = {}

        imb_at_signal = signal_snapshot.book_imbalance
        imb_now = current.book_imbalance

        # 1. Imbalance Recovery
        # Was negative (selling), now less negative or positive
        imb_recovery = imb_now - imb_at_signal
        details['imbalance_at_signal'] = imb_at_signal
        details['imbalance_now'] = imb_now
        details['imbalance_recovery'] = imb_recovery

        if imb_at_signal < 0 and imb_recovery > cfg.imbalance_recovery_min:
            score += cfg.weight_imbalance_recovery
            signals.add(ExhaustionType.IMBALANCE_RECOVERY)

        # 2. Spread Normalization
        spread_at_signal = signal_snapshot.spread_bps
        spread_now = current.spread_bps
        spread_ma = np.mean(self._spread_buffer) if self._spread_buffer else spread_now
        details['spread_at_signal'] = spread_at_signal
        details['spread_now'] = spread_now
        details['spread_ma'] = spread_ma

        if spread_at_signal > spread_ma * cfg.spread_elevated_mult:  # Was elevated
            if spread_now < spread_ma * cfg.spread_normal_mult:  # Now normal
                score += cfg.weight_spread_normalized
                signals.add(ExhaustionType.SPREAD_NORMALIZED)

        # 3. Bid Depth Recovery (for longs, we want bid side healthy)
        bid_depth_at_signal = signal_snapshot.bid_depth
        bid_depth_now = current.bid_depth
        details['bid_depth_at_signal'] = bid_depth_at_signal
        details['bid_depth_now'] = bid_depth_now

        if bid_depth_at_signal > 0:
            depth_recovery = bid_depth_now / bid_depth_at_signal
            details['bid_depth_recovery_ratio'] = depth_recovery
            if depth_recovery > cfg.depth_recovery_min:
                score += cfg.weight_depth_recovery
                signals.add(ExhaustionType.DEPTH_RECOVERY)

        # 4. Momentum Decay
        # Rate of imbalance change slowing (selling losing steam)
        if len(self._imbalance_buffer) >= 5:
            imb_list = list(self._imbalance_buffer)
            recent_mom = imb_list[-1] - imb_list[-3]  # Recent 2-bar change
            older_mom = imb_list[-3] - imb_list[-5]   # Older 2-bar change
            details['recent_momentum'] = recent_mom
            details['older_momentum'] = older_mom

            # If older momentum was negative (selling) and recent is less negative
            if older_mom < -0.03 and recent_mom > older_mom * cfg.momentum_decay_factor:
                score += cfg.weight_momentum_decay
                signals.add(ExhaustionType.MOMENTUM_DECAY)

        # 5. Imbalance Flip (strongest signal)
        # Imbalance actually flipped from negative to positive
        if imb_at_signal < -cfg.imbalance_flip_threshold and imb_now > cfg.imbalance_flip_threshold:
            score += cfg.weight_imbalance_flip
            signals.add(ExhaustionType.IMBALANCE_FLIP)

        # 6. Ask Wall Absorption
        # Large ask wall getting consumed (aggressive buyers)
        ask_wall_at_signal = signal_snapshot.ask_wall_strength
        ask_wall_now = current.ask_wall_strength
        details['ask_wall_at_signal'] = ask_wall_at_signal
        details['ask_wall_now'] = ask_wall_now

        if ask_wall_at_signal > 2.0:  # There was a wall
            if ask_wall_now < ask_wall_at_signal * cfg.wall_absorption_factor:
                score += cfg.weight_wall_absorbed
                signals.add(ExhaustionType.WALL_ABSORBED)

        # 7. Top-of-book flip (bonus)
        tob_at_signal = signal_snapshot.tob_imbalance
        tob_now = current.tob_imbalance
        if tob_at_signal < -0.1 and tob_now > 0.1:
            score += 0.05  # Small bonus
            signals.add(ExhaustionType.TOB_FLIP)

        # Clamp score
        score = min(1.0, score)

        return ExhaustionSignal(
            exhaustion_score=score,
            ready=score >= cfg.exhaustion_threshold,
            signals=signals,
            details=details,
            bars_analyzed=len(self._imbalance_buffer)
        )

    def detect_buy_exhaustion(
        self,
        current: OrderFlowSnapshot,
        signal_snapshot: OrderFlowSnapshot
    ) -> ExhaustionSignal:
        """
        Detect buying pressure exhaustion for SELL entry timing.

        We're waiting for buyers to exhaust before entering short.
        Mirror logic of sell_exhaustion.

        Args:
            current: Current order flow snapshot
            signal_snapshot: Snapshot when indicator generated signal

        Returns:
            ExhaustionSignal with score, ready flag, and detected signals
        """
        cfg = self.config
        score = 0.0
        signals: Set[ExhaustionType] = set()
        details: Dict[str, float] = {}

        imb_at_signal = signal_snapshot.book_imbalance
        imb_now = current.book_imbalance

        # 1. Imbalance Recovery (reversed for shorts)
        # Was positive (buying), now less positive or negative
        imb_recovery = imb_at_signal - imb_now  # Reversed
        details['imbalance_at_signal'] = imb_at_signal
        details['imbalance_now'] = imb_now
        details['imbalance_recovery'] = imb_recovery

        if imb_at_signal > 0 and imb_recovery > cfg.imbalance_recovery_min:
            score += cfg.weight_imbalance_recovery
            signals.add(ExhaustionType.IMBALANCE_RECOVERY)

        # 2. Spread Normalization (same logic)
        spread_at_signal = signal_snapshot.spread_bps
        spread_now = current.spread_bps
        spread_ma = np.mean(self._spread_buffer) if self._spread_buffer else spread_now
        details['spread_at_signal'] = spread_at_signal
        details['spread_now'] = spread_now
        details['spread_ma'] = spread_ma

        if spread_at_signal > spread_ma * cfg.spread_elevated_mult:
            if spread_now < spread_ma * cfg.spread_normal_mult:
                score += cfg.weight_spread_normalized
                signals.add(ExhaustionType.SPREAD_NORMALIZED)

        # 3. Ask Depth Recovery (for shorts, we want ask side healthy)
        ask_depth_at_signal = signal_snapshot.ask_depth
        ask_depth_now = current.ask_depth
        details['ask_depth_at_signal'] = ask_depth_at_signal
        details['ask_depth_now'] = ask_depth_now

        if ask_depth_at_signal > 0:
            depth_recovery = ask_depth_now / ask_depth_at_signal
            details['ask_depth_recovery_ratio'] = depth_recovery
            if depth_recovery > cfg.depth_recovery_min:
                score += cfg.weight_depth_recovery
                signals.add(ExhaustionType.DEPTH_RECOVERY)

        # 4. Momentum Decay (reversed)
        if len(self._imbalance_buffer) >= 5:
            imb_list = list(self._imbalance_buffer)
            recent_mom = imb_list[-1] - imb_list[-3]
            older_mom = imb_list[-3] - imb_list[-5]
            details['recent_momentum'] = recent_mom
            details['older_momentum'] = older_mom

            # If older momentum was positive (buying) and recent is less positive
            if older_mom > 0.03 and recent_mom < older_mom * cfg.momentum_decay_factor:
                score += cfg.weight_momentum_decay
                signals.add(ExhaustionType.MOMENTUM_DECAY)

        # 5. Imbalance Flip (reversed)
        if imb_at_signal > cfg.imbalance_flip_threshold and imb_now < -cfg.imbalance_flip_threshold:
            score += cfg.weight_imbalance_flip
            signals.add(ExhaustionType.IMBALANCE_FLIP)

        # 6. Bid Wall Absorption
        bid_wall_at_signal = signal_snapshot.bid_wall_strength
        bid_wall_now = current.bid_wall_strength
        details['bid_wall_at_signal'] = bid_wall_at_signal
        details['bid_wall_now'] = bid_wall_now

        if bid_wall_at_signal > 2.0:
            if bid_wall_now < bid_wall_at_signal * cfg.wall_absorption_factor:
                score += cfg.weight_wall_absorbed
                signals.add(ExhaustionType.WALL_ABSORBED)

        # 7. TOB flip
        tob_at_signal = signal_snapshot.tob_imbalance
        tob_now = current.tob_imbalance
        if tob_at_signal > 0.1 and tob_now < -0.1:
            score += 0.05
            signals.add(ExhaustionType.TOB_FLIP)

        score = min(1.0, score)

        return ExhaustionSignal(
            exhaustion_score=score,
            ready=score >= cfg.exhaustion_threshold,
            signals=signals,
            details=details,
            bars_analyzed=len(self._imbalance_buffer)
        )

    def reset(self):
        """Clear all buffers."""
        self._imbalance_buffer.clear()
        self._spread_buffer.clear()
        self._depth_buffer.clear()
        self._tob_imbalance_buffer.clear()
        self._snapshot_buffer.clear()

    def get_imbalance_trend(self, periods: int = 5) -> float:
        """
        Get recent imbalance trend direction.
        Returns: positive = improving, negative = worsening
        """
        if len(self._imbalance_buffer) < periods:
            return 0.0

        imb_list = list(self._imbalance_buffer)
        return imb_list[-1] - imb_list[-periods]

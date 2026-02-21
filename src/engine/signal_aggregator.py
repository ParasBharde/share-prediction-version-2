"""
Signal Aggregator

Purpose:
    Collects signals from all strategies and merges them per symbol.
    Calculates weighted confidence scores based on strategy priority.
    Resolves conflicting signals (BUY vs SELL) via weighted voting.

Dependencies:
    - strategies.base_strategy for TradingSignal
    - utils.constants for SignalType

Logging:
    - Aggregation summary at INFO
    - Conflict resolution at DEBUG
    - Merge failures at ERROR

Fallbacks:
    Conflicting signals are resolved via majority voting.
    If tied, the higher-confidence signal wins.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.monitoring.logger import get_logger
from src.monitoring.metrics import signal_generated_counter
from src.strategies.base_strategy import TradingSignal
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)

# Default weight map for strategy-level weighting
DEFAULT_STRATEGY_WEIGHTS: Dict[str, float] = {
    "Mother Candle V2": 0.40,
    "Momentum Breakout": 0.35,
    "Volume Surge": 0.35,
    "Mean Reversion": 0.30,
    # Intraday strategies
    "Intraday Momentum": 0.35,
    "Intraday Volume Surge": 0.35,
    "Intraday Mean Reversion": 0.30,
    # Options strategies
    "Options OI Breakout": 0.40,
    "Options VWAP Supertrend": 0.35,
    "Options PCR Sentiment": 0.30,
    # ── BTST Suite ────────────────────────────────────────────────────
    # Flag Pattern has the highest weight — strong momentum setup with
    # a measured-move target and well-defined risk.
    "Flag Pattern": 0.42,
    # Darvas Box — proven institutional accumulation pattern.
    "Darvas Box": 0.38,
    # Triangle and Channel are pattern-regression strategies; slightly
    # lower weight due to subjectivity in pivot selection.
    "Symmetrical Triangle": 0.35,
    "Descending Channel": 0.33,
}


@dataclass
class AggregatedSignal:
    """A merged signal combining input from multiple strategies."""

    symbol: str
    company_name: str
    signal_type: SignalType
    weighted_confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    priority: AlertPriority
    contributing_strategies: List[str] = field(default_factory=list)
    strategy_count: int = 0
    conflict_resolved: bool = False
    individual_signals: List[Dict[str, Any]] = field(
        default_factory=list
    )
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated signal to dictionary."""
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "signal_type": self.signal_type.value,
            "weighted_confidence": round(self.weighted_confidence, 4),
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "priority": self.priority.value,
            "contributing_strategies": self.contributing_strategies,
            "strategy_count": self.strategy_count,
            "conflict_resolved": self.conflict_resolved,
            "generated_at": self.generated_at.isoformat(),
        }


def aggregate_signals(
    all_signals: List[TradingSignal],
    strategy_weights: Optional[Dict[str, float]] = None,
) -> List[AggregatedSignal]:
    """
    Aggregate signals from multiple strategies per symbol.

    Groups all signals by symbol, then merges signals for the
    same symbol by calculating weighted confidence and resolving
    any conflicting signal types through voting.

    Args:
        all_signals: Flat list of TradingSignal objects from all
            strategies and all stocks.
        strategy_weights: Optional mapping of strategy name to
            weight (0.0-1.0). Defaults to built-in weights.

    Returns:
        List of AggregatedSignal objects, one per symbol.
    """
    weights = strategy_weights or DEFAULT_STRATEGY_WEIGHTS

    if not all_signals:
        logger.info("No signals to aggregate")
        return []

    # Group signals by symbol
    signals_by_symbol: Dict[str, List[TradingSignal]] = defaultdict(list)
    for signal in all_signals:
        signals_by_symbol[signal.symbol].append(signal)

    logger.info(
        f"Aggregating {len(all_signals)} signals across "
        f"{len(signals_by_symbol)} symbols",
        extra={
            "total_signals": len(all_signals),
            "unique_symbols": len(signals_by_symbol),
        },
    )

    aggregated: List[AggregatedSignal] = []

    for symbol, signals in signals_by_symbol.items():
        try:
            merged = _merge_signals_for_symbol(symbol, signals, weights)
            if merged is not None:
                aggregated.append(merged)
        except Exception as e:
            logger.error(
                f"{symbol}: Failed to aggregate signals: {e}",
                exc_info=True,
                extra={"symbol": symbol, "signal_count": len(signals)},
            )

    logger.info(
        f"Aggregation complete: {len(aggregated)} merged signals "
        f"from {len(all_signals)} raw signals",
        extra={
            "aggregated_count": len(aggregated),
            "raw_count": len(all_signals),
        },
    )

    return aggregated


def _merge_signals_for_symbol(
    symbol: str,
    signals: List[TradingSignal],
    weights: Dict[str, float],
) -> Optional[AggregatedSignal]:
    """
    Merge all signals for a single symbol into one aggregated signal.

    If all signals agree on direction (BUY/SELL), they are merged
    directly. If there are conflicts, weighted voting determines
    the final direction.

    Args:
        symbol: The stock symbol being merged.
        signals: List of signals for this symbol from various strategies.
        weights: Strategy name to weight mapping.

    Returns:
        AggregatedSignal or None if merge produces invalid output.
    """
    if len(signals) == 1:
        sig = signals[0]
        weight = weights.get(sig.strategy_name, 0.5)
        return AggregatedSignal(
            symbol=sig.symbol,
            company_name=sig.company_name,
            signal_type=sig.signal_type,
            weighted_confidence=sig.confidence * weight,
            entry_price=sig.entry_price,
            target_price=sig.target_price,
            stop_loss=sig.stop_loss,
            priority=sig.priority,
            contributing_strategies=[sig.strategy_name],
            strategy_count=1,
            conflict_resolved=False,
            individual_signals=[sig.to_dict()],
            generated_at=sig.generated_at,
        )

    # Check for conflicting signal types
    signal_types = {sig.signal_type for sig in signals}
    has_conflict = len(signal_types) > 1

    if has_conflict:
        resolved_signals, resolved_type = _merge_conflicting(
            signals, weights
        )
        logger.debug(
            f"{symbol}: Conflict resolved via voting -> "
            f"{resolved_type.value} ({len(resolved_signals)} strategies)",
            extra={
                "symbol": symbol,
                "original_types": [st.value for st in signal_types],
                "resolved_type": resolved_type.value,
            },
        )
        signals_to_merge = resolved_signals
        final_type = resolved_type
        conflict_resolved = True
    else:
        signals_to_merge = signals
        final_type = signals[0].signal_type
        conflict_resolved = False

    # Calculate weighted confidence
    weighted_confidence = _calculate_weighted_confidence(
        signals_to_merge, weights
    )

    # Average the price levels from agreeing strategies
    entry_price = sum(s.entry_price for s in signals_to_merge) / len(
        signals_to_merge
    )
    target_price = sum(s.target_price for s in signals_to_merge) / len(
        signals_to_merge
    )
    stop_loss = sum(s.stop_loss for s in signals_to_merge) / len(
        signals_to_merge
    )

    # Highest priority among contributing signals
    priority_order = {
        AlertPriority.CRITICAL: 0,
        AlertPriority.HIGH: 1,
        AlertPriority.MEDIUM: 2,
        AlertPriority.LOW: 3,
    }
    best_priority = min(
        (s.priority for s in signals_to_merge),
        key=lambda p: priority_order.get(p, 99),
    )

    contributing = [s.strategy_name for s in signals_to_merge]

    return AggregatedSignal(
        symbol=symbol,
        company_name=signals_to_merge[0].company_name,
        signal_type=final_type,
        weighted_confidence=weighted_confidence,
        entry_price=round(entry_price, 2),
        target_price=round(target_price, 2),
        stop_loss=round(stop_loss, 2),
        priority=best_priority,
        contributing_strategies=contributing,
        strategy_count=len(signals_to_merge),
        conflict_resolved=conflict_resolved,
        individual_signals=[s.to_dict() for s in signals],
        generated_at=max(s.generated_at for s in signals_to_merge),
    )


def _merge_conflicting(
    signals: List[TradingSignal],
    weights: Dict[str, float],
) -> tuple:
    """
    Resolve conflicting signals via weighted voting.

    Each signal casts a vote for its signal type, weighted by
    (strategy_weight * confidence). The signal type with the
    highest total weighted vote wins. Signals of the losing
    type are excluded from the merged result.

    Args:
        signals: List of conflicting signals for the same symbol.
        weights: Strategy name to weight mapping.

    Returns:
        Tuple of (winning signals list, winning SignalType).
    """
    # Tally weighted votes per signal type
    type_votes: Dict[SignalType, float] = defaultdict(float)
    type_signals: Dict[SignalType, List[TradingSignal]] = defaultdict(list)

    for sig in signals:
        strategy_weight = weights.get(sig.strategy_name, 0.5)
        vote = strategy_weight * sig.confidence
        type_votes[sig.signal_type] += vote
        type_signals[sig.signal_type].append(sig)

    # Pick the type with the highest weighted vote
    winning_type = max(type_votes, key=type_votes.get)

    logger.debug(
        f"Conflict vote tally: "
        f"{', '.join(f'{t.value}={v:.3f}' for t, v in type_votes.items())} "
        f"-> winner: {winning_type.value}"
    )

    return type_signals[winning_type], winning_type


def _calculate_weighted_confidence(
    signals: List[TradingSignal],
    weights: Dict[str, float],
) -> float:
    """
    Calculate a combined weighted confidence score.

    Each signal's confidence is multiplied by its strategy weight
    and the results are averaged. Multi-strategy agreement
    receives a bonus multiplier.

    Args:
        signals: List of signals contributing to the score.
        weights: Strategy name to weight mapping.

    Returns:
        Combined weighted confidence score between 0.0 and 1.0.
    """
    if not signals:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for sig in signals:
        w = weights.get(sig.strategy_name, 0.5)
        weighted_sum += sig.confidence * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    base_confidence = weighted_sum / total_weight

    # Multi-strategy agreement bonus (up to 15% boost)
    agreement_bonus = min(0.15, 0.05 * (len(signals) - 1))
    final_confidence = min(1.0, base_confidence + agreement_bonus)

    logger.debug(
        f"Weighted confidence: base={base_confidence:.3f}, "
        f"bonus={agreement_bonus:.3f}, final={final_confidence:.3f}"
    )

    return round(final_confidence, 4)

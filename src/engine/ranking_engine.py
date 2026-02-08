"""
Multi-Factor Ranking Engine

Purpose:
    Ranks aggregated signals using multiple factors to produce
    a prioritized watchlist. Applies recency penalty, sector
    diversification limits, and composite scoring.

Dependencies:
    - signal_aggregator for AggregatedSignal
    - datetime for recency calculations

Logging:
    - Ranking pipeline start/end at INFO
    - Scoring details at DEBUG
    - Filter removals at DEBUG

Fallbacks:
    Missing metadata fields default to neutral values.
    Signals with invalid data are ranked last, not dropped.
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.engine.signal_aggregator import AggregatedSignal
from src.monitoring.logger import get_logger
from src.monitoring.metrics import signal_generated_counter
from src.utils.config_loader import load_config, get_nested

logger = get_logger(__name__)

# Default ranking weights
DEFAULT_CONFIDENCE_WEIGHT = 0.40
DEFAULT_RECENCY_WEIGHT = 0.20
DEFAULT_RISK_REWARD_WEIGHT = 0.25
DEFAULT_STRATEGY_COUNT_WEIGHT = 0.15

# Recency window
DEFAULT_RECENCY_WINDOW_DAYS = 7

# Sector diversification
DEFAULT_MAX_PER_SECTOR = 2


def rank_signals(
    signals: List[AggregatedSignal],
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
) -> List[AggregatedSignal]:
    """
    Rank aggregated signals using a multi-factor scoring pipeline.

    The pipeline applies the following stages in order:
    1. Calculate a composite score for each signal.
    2. Apply a recency penalty to down-weight stale signals.
    3. Sort by composite score descending.
    4. Apply sector diversification to cap exposure per sector.

    Args:
        signals: List of AggregatedSignal objects to rank.
        max_per_sector: Maximum signals allowed per sector
            in the final output. Defaults to 2.

    Returns:
        Ranked and filtered list of AggregatedSignal objects,
        ordered from highest to lowest composite score.
    """
    if not signals:
        logger.info("No signals to rank")
        return []

    config = load_config("system")
    ranking_config = config.get("ranking", {})

    max_sector = get_nested(
        ranking_config, "max_per_sector", max_per_sector
    )

    logger.info(
        f"Ranking {len(signals)} signals "
        f"(max_per_sector={max_sector})",
        extra={"signal_count": len(signals)},
    )

    # Stage 1: Compute composite scores
    scored_signals = []
    for signal in signals:
        try:
            score = _calculate_composite_score(signal, ranking_config)
            scored_signals.append((signal, score))
        except Exception as e:
            logger.warning(
                f"{signal.symbol}: Scoring failed, assigning 0: {e}",
                extra={"symbol": signal.symbol, "error": str(e)},
            )
            scored_signals.append((signal, 0.0))

    # Stage 2: Apply recency penalty
    scored_signals = _apply_recency_penalty(scored_signals, ranking_config)

    # Stage 3: Sort by composite score descending
    scored_signals.sort(key=lambda pair: pair[1], reverse=True)

    logger.debug(
        "Score ranking: "
        + ", ".join(
            f"{s.symbol}={score:.4f}"
            for s, score in scored_signals[:10]
        )
    )

    # Stage 4: Sector diversification
    ranked = _apply_sector_diversification(
        scored_signals, max_per_sector=max_sector
    )

    logger.info(
        f"Ranking complete: {len(ranked)} signals after filtering "
        f"(from {len(signals)} input)",
        extra={
            "ranked_count": len(ranked),
            "input_count": len(signals),
        },
    )

    return ranked


def _calculate_composite_score(
    signal: AggregatedSignal,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calculate a composite ranking score for a signal.

    The composite score is a weighted sum of:
    - Weighted confidence from aggregation.
    - Risk-reward ratio implied by entry/target/stop-loss.
    - Strategy agreement count (more strategies = higher score).

    Args:
        signal: The aggregated signal to score.
        config: Optional ranking config for custom weights.

    Returns:
        Composite score as a float (higher is better).
    """
    cfg = config or {}
    w_confidence = get_nested(
        cfg, "weight_confidence", DEFAULT_CONFIDENCE_WEIGHT
    )
    w_risk_reward = get_nested(
        cfg, "weight_risk_reward", DEFAULT_RISK_REWARD_WEIGHT
    )
    w_strategy_count = get_nested(
        cfg, "weight_strategy_count", DEFAULT_STRATEGY_COUNT_WEIGHT
    )

    # Confidence component (0-1)
    confidence_score = signal.weighted_confidence

    # Risk-reward component (normalized to 0-1 via sigmoid-like)
    risk = signal.entry_price - signal.stop_loss
    reward = signal.target_price - signal.entry_price
    if risk > 0 and reward > 0:
        rr_ratio = reward / risk
        # Normalize: RR of 2.0 maps to ~0.73, RR of 3.0 to ~0.82
        rr_score = 1.0 - (1.0 / (1.0 + rr_ratio))
    else:
        rr_score = 0.0

    # Strategy agreement component (normalized: 1->0.33, 2->0.67, 3->1.0)
    strategy_score = min(1.0, signal.strategy_count / 3.0)

    composite = (
        w_confidence * confidence_score
        + w_risk_reward * rr_score
        + w_strategy_count * strategy_score
    )

    logger.debug(
        f"{signal.symbol}: composite={composite:.4f} "
        f"(conf={confidence_score:.3f}*{w_confidence}, "
        f"rr={rr_score:.3f}*{w_risk_reward}, "
        f"strat={strategy_score:.3f}*{w_strategy_count})",
    )

    return composite


def _apply_recency_penalty(
    scored_signals: List[tuple],
    config: Optional[Dict[str, Any]] = None,
) -> List[tuple]:
    """
    Apply a time-decay penalty to signal scores.

    Signals older than the recency window receive a linearly
    decaying multiplier. Signals within the window receive
    no penalty. The window defaults to 7 days.

    Args:
        scored_signals: List of (AggregatedSignal, score) tuples.
        config: Optional ranking config with recency window setting.

    Returns:
        Updated list of (AggregatedSignal, adjusted_score) tuples.
    """
    cfg = config or {}
    window_days = get_nested(
        cfg, "recency_window_days", DEFAULT_RECENCY_WINDOW_DAYS
    )
    recency_weight = get_nested(
        cfg, "weight_recency", DEFAULT_RECENCY_WEIGHT
    )

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    adjusted = []
    for signal, score in scored_signals:
        signal_time = signal.generated_at
        # Ensure timezone-aware comparison
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)

        if signal_time >= cutoff:
            # Within window: scale from 1.0 (just now) to 0.5 (at cutoff)
            age_fraction = (now - signal_time).total_seconds() / (
                window_days * 86400
            )
            recency_factor = 1.0 - (0.5 * age_fraction)
        else:
            # Beyond window: heavy penalty
            days_old = (now - signal_time).days
            recency_factor = max(0.1, 0.5 * math.exp(
                -0.1 * (days_old - window_days)
            ))

        # Blend recency into the score
        adjusted_score = (
            score * (1.0 - recency_weight)
            + score * recency_factor * recency_weight
        )

        if recency_factor < 0.9:
            logger.debug(
                f"{signal.symbol}: Recency penalty applied "
                f"(factor={recency_factor:.3f}, "
                f"score {score:.4f} -> {adjusted_score:.4f})"
            )

        adjusted.append((signal, adjusted_score))

    return adjusted


def _apply_sector_diversification(
    scored_signals: List[tuple],
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
) -> List[AggregatedSignal]:
    """
    Limit the number of signals per sector for diversification.

    Iterates through signals in score order and caps each sector
    at the configured maximum. Excess signals are dropped.

    Args:
        scored_signals: Sorted list of (AggregatedSignal, score)
            tuples, highest score first.
        max_per_sector: Maximum signals to keep per sector.

    Returns:
        Filtered list of AggregatedSignal objects respecting
        sector caps.
    """
    sector_counts: Dict[str, int] = defaultdict(int)
    result: List[AggregatedSignal] = []
    dropped = 0

    for signal, score in scored_signals:
        # Extract sector from individual signals metadata
        sector = _extract_sector(signal)

        if sector and sector_counts[sector] >= max_per_sector:
            logger.debug(
                f"{signal.symbol}: Dropped for sector diversification "
                f"(sector={sector}, count={sector_counts[sector]})"
            )
            dropped += 1
            continue

        if sector:
            sector_counts[sector] += 1
        result.append(signal)

    if dropped > 0:
        logger.info(
            f"Sector diversification: dropped {dropped} signals "
            f"(max {max_per_sector} per sector)",
            extra={
                "dropped": dropped,
                "sector_distribution": dict(sector_counts),
            },
        )

    return result


def _extract_sector(signal: AggregatedSignal) -> Optional[str]:
    """
    Extract the sector from an aggregated signal's metadata.

    Looks through the individual signal details for a sector
    field in metadata or company_info.

    Args:
        signal: The aggregated signal to inspect.

    Returns:
        Sector string or None if not available.
    """
    for individual in signal.individual_signals:
        metadata = individual.get("metadata", {})
        sector = metadata.get("sector")
        if sector:
            return sector

    return None

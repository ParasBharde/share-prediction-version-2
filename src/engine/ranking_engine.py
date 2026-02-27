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

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.engine.signal_aggregator import AggregatedSignal
from src.monitoring.logger import get_logger
from src.monitoring.metrics import signal_generated_counter
from src.utils.config_loader import load_config, get_nested

logger = get_logger(__name__)

# Default ranking weights
# Note: weights must sum to 1.0.  The RS component is a boost applied on
# top of the composite score (not a weight in the linear sum) so the base
# weights below are unchanged.
DEFAULT_CONFIDENCE_WEIGHT    = 0.45   # raised from 0.40 (most predictive)
DEFAULT_RISK_REWARD_WEIGHT   = 0.30   # raised from 0.25
DEFAULT_STRATEGY_COUNT_WEIGHT = 0.15
DEFAULT_GAP_RISK_WEIGHT      = 0.10   # replaces recency (irrelevant for same-day scans)

# Gap-risk: a stock that gapped up strongly today has less room to run
# (high gap = less upside potential in BTST holding period)
# Score = 1 - min(gap_pct / MAX_ACCEPTABLE_GAP, 1.0)
DEFAULT_MAX_ACCEPTABLE_GAP_PCT = 5.0   # >5% open gap = highest gap risk

# ── Relative Strength (RS) component ──────────────────────────────────────
# Boost applied to the final composite score when a stock's 20-day return
# outperforms the Nifty 50 index over the same period.  The boost is a
# flat additive bonus (not a multiplier) so it cannot overwhelm a bad
# pattern signal, but provides a meaningful tie-breaker between equal
# pattern setups.
DEFAULT_RS_BOOST             = 0.05   # +5 pp composite boost for RS leaders
# Lookback used to compute each signal's return (stored in metadata)
RS_RETURN_KEY                = "return_20d"

# Sector diversification
DEFAULT_MAX_PER_SECTOR = 2
# Sector placeholder: when sector metadata is unavailable, skip the cap
_UNKNOWN_SECTOR = "Unknown"


def rank_signals(
    signals: List[AggregatedSignal],
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    nifty_return_20d: float = 0.0,
) -> List[AggregatedSignal]:
    """
    Rank aggregated signals using a multi-factor scoring pipeline.

    The pipeline applies the following stages in order:
    1. Calculate a composite score for each signal.
    2. Apply Relative Strength (RS) boost for stocks outperforming Nifty 50.
    3. Apply gap-risk adjustment to down-weight stocks with large open gaps.
    4. Sort by composite score descending.
    5. Apply sector diversification to cap exposure per sector.

    Args:
        signals: List of AggregatedSignal objects to rank.
        max_per_sector: Maximum signals allowed per sector
            in the final output. Defaults to 2.
        nifty_return_20d: Nifty 50's 20-day price return (as a percentage,
            e.g. 3.5 means +3.5%).  When provided, signals whose metadata
            carries a ``return_20d`` value exceeding this benchmark receive
            a composite score boost (DEFAULT_RS_BOOST).  Pass 0.0 (default)
            to disable the RS component when Nifty data is unavailable.

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
        f"(max_per_sector={max_sector}, nifty_20d={nifty_return_20d:.2f}%)",
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

    # Stage 2: Relative Strength boost — reward stocks that are already
    # outperforming the Nifty 50 index over the past 20 days.  This
    # surfaces institutional-grade momentum leaders from within the pattern
    # set rather than ranking purely on signal quality.
    scored_signals = _apply_rs_boost(
        scored_signals, nifty_return_20d, ranking_config
    )

    # Stage 3: Apply gap-risk adjustment (replaces useless recency penalty)
    # For BTST: a stock that gapped up heavily has less upside in the holding period.
    scored_signals = _apply_gap_risk_adjustment(scored_signals, ranking_config)

    # Stage 4: Sort by composite score descending
    scored_signals.sort(key=lambda pair: pair[1], reverse=True)

    logger.debug(
        "Score ranking: "
        + ", ".join(
            f"{s.symbol}={score:.4f}"
            for s, score in scored_signals[:10]
        )
    )

    # Stage 5: Sector diversification
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
    risk = abs(signal.entry_price - signal.stop_loss)
    reward = abs(signal.target_price - signal.entry_price)
    if risk > 0 and reward > 0:
        rr_ratio = reward / risk
        # Normalize: RR of 2.0 maps to ~0.67, RR of 3.0 to ~0.75
        rr_score = 1.0 - (1.0 / (1.0 + rr_ratio))
    else:
        rr_score = 0.0

    # Strategy agreement component (normalized: 1->0.33, 2->0.67, 3->1.0)
    strategy_score = min(1.0, signal.strategy_count / 3.0)

    # Note: gap-risk component is applied in the separate adjustment step
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


def _apply_rs_boost(
    scored_signals: List[tuple],
    nifty_return_20d: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> List[tuple]:
    """
    Apply a Relative Strength (RS) score boost for Nifty outperformers.

    A stock that has already risen more than the Nifty 50 index over the
    past 20 trading days is showing genuine institutional buying interest
    — not just a sector rotation or index lift.  Rewarding these stocks
    in the ranking surfaces high-conviction momentum breakouts.

    Logic
    -----
    Each signal's ``return_20d`` is extracted from its individual signal
    metadata.  If ``return_20d > nifty_return_20d``, a flat additive boost
    (DEFAULT_RS_BOOST, 0.05 by default) is added to the composite score.

    The ``return_20d`` value must be pre-computed by the caller (e.g.
    the scan script) and stored in each signal's metadata dict:

        signal.metadata["return_20d"] = (close_today / close_20d_ago - 1) * 100

    When the metadata key is absent the signal's score is left unchanged —
    the function is a safe no-op when RS data is unavailable.

    Args:
        scored_signals: List of (AggregatedSignal, score) tuples.
        nifty_return_20d: Nifty 50 20-day return in percent (e.g. 3.5 = +3.5%).
            Set to 0.0 to disable the RS boost.
        config: Optional ranking config for custom boost magnitude.

    Returns:
        Updated list of (AggregatedSignal, adjusted_score) tuples.
    """
    cfg = config or {}
    rs_boost = get_nested(cfg, "rs_boost", DEFAULT_RS_BOOST)

    # No-op when Nifty return is not provided or RS boost is disabled
    if nifty_return_20d == 0.0 and rs_boost == 0.0:
        return scored_signals

    boosted = []
    for signal, score in scored_signals:
        # Extract return_20d from the first individual signal that has it
        stock_return_20d: Optional[float] = None
        for ind_sig in signal.individual_signals:
            meta = ind_sig.get("metadata", {})
            if RS_RETURN_KEY in meta:
                stock_return_20d = float(meta[RS_RETURN_KEY])
                break

        if stock_return_20d is not None and stock_return_20d > nifty_return_20d:
            boosted_score = score + rs_boost
            logger.debug(
                f"{signal.symbol}: RS boost applied "
                f"(stock_20d={stock_return_20d:.1f}% > "
                f"nifty_20d={nifty_return_20d:.1f}%, "
                f"score {score:.4f} -> {boosted_score:.4f})"
            )
            boosted.append((signal, boosted_score))
        else:
            boosted.append((signal, score))

    return boosted


def _apply_gap_risk_adjustment(
    scored_signals: List[tuple],
    config: Optional[Dict[str, Any]] = None,
) -> List[tuple]:
    """
    Adjust scores based on intraday gap risk.

    A stock that opened with a large gap-up has already captured
    much of its potential move, leaving less upside for a BTST
    holding overnight. Gap risk is computed as:

        gap_pct = (open - prev_close) / prev_close * 100

    If gap_pct metadata is available in the signal, a gap-risk
    score of (1 - gap_pct / max_gap) is blended in.

    When gap metadata is unavailable (most cases currently), this
    function is a no-op — it simply returns the scores unchanged.

    Args:
        scored_signals: List of (AggregatedSignal, score) tuples.
        config: Optional ranking config.

    Returns:
        Updated list of (AggregatedSignal, adjusted_score) tuples.
    """
    cfg = config or {}
    gap_risk_weight = get_nested(
        cfg, "weight_gap_risk", DEFAULT_GAP_RISK_WEIGHT
    )
    max_gap = get_nested(
        cfg, "max_acceptable_gap_pct", DEFAULT_MAX_ACCEPTABLE_GAP_PCT
    )

    adjusted = []
    for signal, score in scored_signals:
        # Try to extract gap data from individual signal metadata
        gap_pct: Optional[float] = None
        for ind_sig in signal.individual_signals:
            meta = ind_sig.get("metadata", {})
            if "gap_pct" in meta:
                gap_pct = float(meta["gap_pct"])
                break

        if gap_pct is not None and gap_pct > 0:
            # Higher gap = higher risk = lower gap_risk_score
            gap_risk_score = max(0.0, 1.0 - gap_pct / max_gap)
            adjusted_score = (
                score * (1.0 - gap_risk_weight)
                + gap_risk_score * gap_risk_weight
            )
            if gap_pct > 1.0:
                logger.debug(
                    f"{signal.symbol}: Gap-risk adjustment "
                    f"(gap={gap_pct:.1f}%, "
                    f"score {score:.4f} -> {adjusted_score:.4f})"
                )
        else:
            # No gap data available — leave score unchanged
            adjusted_score = score

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

        # Skip diversification cap for "Unknown" sector — applying a cap
        # when sector data is unavailable would incorrectly group all stocks
        # together and block legitimate signals from different real sectors.
        if (
            sector
            and sector != _UNKNOWN_SECTOR
            and sector_counts[sector] >= max_per_sector
        ):
            logger.debug(
                f"{signal.symbol}: Dropped for sector diversification "
                f"(sector={sector}, count={sector_counts[sector]})"
            )
            dropped += 1
            continue

        if sector and sector != _UNKNOWN_SECTOR:
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

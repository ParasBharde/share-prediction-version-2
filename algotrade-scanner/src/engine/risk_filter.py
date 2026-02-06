"""
Portfolio Risk Filter

Purpose:
    Applies portfolio-level risk checks to candidate signals
    before they become actionable. Enforces position size limits,
    sector concentration caps, correlation thresholds, and
    market condition checks.

Dependencies:
    - signal_aggregator for AggregatedSignal
    - numpy for correlation calculations

Logging:
    - Filter pipeline summary at INFO
    - Individual filter results at DEBUG
    - Rejections at WARNING

Fallbacks:
    If portfolio state is unavailable, conservative defaults apply.
    VIX check is skipped when market data is unavailable.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.engine.signal_aggregator import AggregatedSignal
from src.monitoring.logger import get_logger
from src.monitoring.metrics import signal_generated_counter
from src.utils.config_loader import load_config, get_nested

logger = get_logger(__name__)

# Default risk thresholds
DEFAULT_MAX_CORRELATION = 0.70
DEFAULT_MAX_POSITION_SIZE_PCT = 20.0  # percent of portfolio
DEFAULT_MAX_SECTOR_CONCENTRATION_PCT = 30.0  # percent of portfolio
DEFAULT_VIX_THRESHOLD = 30.0  # above this = high volatility regime


@dataclass
class PortfolioState:
    """Current portfolio state for risk assessment."""

    total_value: float = 0.0
    cash_available: float = 0.0
    positions: List[Dict[str, Any]] = field(default_factory=list)
    sector_allocations: Dict[str, float] = field(default_factory=dict)
    position_returns: Dict[str, List[float]] = field(
        default_factory=dict
    )
    current_vix: Optional[float] = None

    @property
    def invested_value(self) -> float:
        """Total currently invested."""
        return self.total_value - self.cash_available

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)


@dataclass
class RiskCheckResult:
    """Result of running risk filters on a signal."""

    signal: AggregatedSignal
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.signal.symbol,
            "passed": self.passed,
            "checks": self.checks,
            "reasons": self.reasons,
        }


def filter_signals(
    signals: List[AggregatedSignal],
    portfolio_state: Optional[PortfolioState] = None,
) -> List[AggregatedSignal]:
    """
    Apply portfolio risk filters to a list of signals.

    Runs each signal through the following checks:
    1. Position size limit (max 20% of portfolio per position).
    2. Sector concentration (max 30% in any single sector).
    3. Correlation check (new position should not be >0.7
       correlated with existing holdings).
    4. Market condition (VIX-based volatility regime check).

    Signals that fail any check are excluded from the result.

    Args:
        signals: List of AggregatedSignal objects to filter.
        portfolio_state: Current portfolio state. If None,
            conservative defaults are used.

    Returns:
        Filtered list of AggregatedSignal objects that pass
        all risk checks.
    """
    config = load_config("scanner")
    risk_config = config.get("risk", {})

    portfolio = portfolio_state or PortfolioState()

    logger.info(
        f"Running risk filters on {len(signals)} signals "
        f"(portfolio_value={portfolio.total_value:.2f}, "
        f"positions={portfolio.position_count})",
        extra={
            "signal_count": len(signals),
            "portfolio_value": portfolio.total_value,
            "position_count": portfolio.position_count,
        },
    )

    # Check market conditions first (applies to all signals)
    market_ok = _check_market_condition(portfolio, risk_config)
    if not market_ok:
        logger.warning(
            "Market conditions unfavorable (high VIX), "
            "rejecting all new signals",
            extra={"vix": portfolio.current_vix},
        )
        return []

    approved: List[AggregatedSignal] = []
    rejected_count = 0

    for signal in signals:
        try:
            result = _evaluate_signal(signal, portfolio, risk_config)

            if result.passed:
                approved.append(signal)
                logger.debug(
                    f"{signal.symbol}: Passed all risk checks",
                    extra=result.to_dict(),
                )
            else:
                rejected_count += 1
                logger.warning(
                    f"{signal.symbol}: Rejected by risk filter "
                    f"({', '.join(result.reasons)})",
                    extra=result.to_dict(),
                )
        except Exception as e:
            rejected_count += 1
            logger.error(
                f"{signal.symbol}: Risk check failed with error: {e}",
                exc_info=True,
                extra={"symbol": signal.symbol},
            )

    logger.info(
        f"Risk filtering complete: {len(approved)} approved, "
        f"{rejected_count} rejected",
        extra={
            "approved": len(approved),
            "rejected": rejected_count,
        },
    )

    return approved


def _evaluate_signal(
    signal: AggregatedSignal,
    portfolio: PortfolioState,
    risk_config: Dict[str, Any],
) -> RiskCheckResult:
    """
    Run all risk checks on a single signal.

    Args:
        signal: The signal to evaluate.
        portfolio: Current portfolio state.
        risk_config: Risk configuration dictionary.

    Returns:
        RiskCheckResult with pass/fail and details.
    """
    result = RiskCheckResult(signal=signal, passed=True)

    # Check 1: Position size
    pos_ok = _check_position_size(signal, portfolio, risk_config)
    result.checks["position_size"] = pos_ok
    if not pos_ok:
        result.passed = False
        result.reasons.append("position_size_exceeded")

    # Check 2: Sector concentration
    sector_ok = _check_sector_concentration(
        signal, portfolio, risk_config
    )
    result.checks["sector_concentration"] = sector_ok
    if not sector_ok:
        result.passed = False
        result.reasons.append("sector_concentration_exceeded")

    # Check 3: Correlation
    corr_ok = _check_correlation(signal, portfolio, risk_config)
    result.checks["correlation"] = corr_ok
    if not corr_ok:
        result.passed = False
        result.reasons.append("high_correlation_with_holdings")

    return result


def _check_position_size(
    signal: AggregatedSignal,
    portfolio: PortfolioState,
    risk_config: Dict[str, Any],
) -> bool:
    """
    Check if the signal would exceed position size limits.

    A single position must not exceed the configured maximum
    percentage of total portfolio value.

    Args:
        signal: Signal being evaluated.
        portfolio: Current portfolio state.
        risk_config: Risk configuration with position size limits.

    Returns:
        True if position size is within limits.
    """
    max_pct = get_nested(
        risk_config,
        "max_position_size_pct",
        DEFAULT_MAX_POSITION_SIZE_PCT,
    )

    if portfolio.total_value <= 0:
        logger.debug(
            f"{signal.symbol}: No portfolio value, "
            f"position size check skipped"
        )
        return True

    # Estimate position value from entry price
    # (Actual quantity would be determined by the order manager)
    max_position_value = portfolio.total_value * (max_pct / 100.0)

    # Check if any existing position in same symbol already exists
    existing_value = 0.0
    for pos in portfolio.positions:
        if pos.get("symbol") == signal.symbol:
            qty = pos.get("quantity", 0)
            price = pos.get("current_price", 0)
            existing_value += qty * price

    available_for_position = max_position_value - existing_value

    if available_for_position <= 0:
        logger.debug(
            f"{signal.symbol}: Position size limit reached "
            f"(existing={existing_value:.2f}, "
            f"max={max_position_value:.2f})",
        )
        return False

    logger.debug(
        f"{signal.symbol}: Position size OK "
        f"(available={available_for_position:.2f}, "
        f"max={max_position_value:.2f})"
    )
    return True


def _check_sector_concentration(
    signal: AggregatedSignal,
    portfolio: PortfolioState,
    risk_config: Dict[str, Any],
) -> bool:
    """
    Check if adding this signal would exceed sector concentration.

    Total allocation to any single sector must not exceed the
    configured maximum percentage of portfolio value.

    Args:
        signal: Signal being evaluated.
        portfolio: Current portfolio state.
        risk_config: Risk configuration with sector limits.

    Returns:
        True if sector concentration is within limits.
    """
    max_pct = get_nested(
        risk_config,
        "max_sector_concentration_pct",
        DEFAULT_MAX_SECTOR_CONCENTRATION_PCT,
    )

    if portfolio.total_value <= 0:
        return True

    # Extract sector from signal metadata
    sector = None
    for individual in signal.individual_signals:
        metadata = individual.get("metadata", {})
        sector = metadata.get("sector")
        if sector:
            break

    if not sector:
        logger.debug(
            f"{signal.symbol}: No sector info, "
            f"sector concentration check skipped"
        )
        return True

    current_sector_value = portfolio.sector_allocations.get(sector, 0.0)
    current_sector_pct = (
        (current_sector_value / portfolio.total_value) * 100.0
        if portfolio.total_value > 0
        else 0.0
    )

    if current_sector_pct >= max_pct:
        logger.debug(
            f"{signal.symbol}: Sector '{sector}' at "
            f"{current_sector_pct:.1f}% (max {max_pct}%)",
        )
        return False

    logger.debug(
        f"{signal.symbol}: Sector '{sector}' at "
        f"{current_sector_pct:.1f}% OK (max {max_pct}%)"
    )
    return True


def _check_correlation(
    signal: AggregatedSignal,
    portfolio: PortfolioState,
    risk_config: Dict[str, Any],
) -> bool:
    """
    Check if the signal is too correlated with existing holdings.

    Computes Pearson correlation between the candidate stock's
    return series and each existing position's return series.
    If any correlation exceeds the threshold, the check fails.

    Args:
        signal: Signal being evaluated.
        portfolio: Current portfolio state with position_returns.
        risk_config: Risk configuration with correlation threshold.

    Returns:
        True if correlation with all holdings is below threshold.
    """
    max_corr = get_nested(
        risk_config,
        "max_correlation",
        DEFAULT_MAX_CORRELATION,
    )

    candidate_returns = portfolio.position_returns.get(signal.symbol)

    if not candidate_returns or not portfolio.position_returns:
        logger.debug(
            f"{signal.symbol}: No return data available, "
            f"correlation check skipped"
        )
        return True

    candidate_arr = np.array(candidate_returns)

    for held_symbol, held_returns in portfolio.position_returns.items():
        if held_symbol == signal.symbol:
            continue

        if not held_returns:
            continue

        held_arr = np.array(held_returns)

        # Align lengths
        min_len = min(len(candidate_arr), len(held_arr))
        if min_len < 5:
            continue

        try:
            corr_matrix = np.corrcoef(
                candidate_arr[:min_len], held_arr[:min_len]
            )
            correlation = abs(corr_matrix[0, 1])

            if np.isnan(correlation):
                continue

            if correlation > max_corr:
                logger.debug(
                    f"{signal.symbol}: High correlation with "
                    f"{held_symbol} ({correlation:.3f} > {max_corr})",
                )
                return False

        except Exception as e:
            logger.debug(
                f"{signal.symbol}: Correlation calc error "
                f"with {held_symbol}: {e}"
            )
            continue

    logger.debug(f"{signal.symbol}: Correlation check passed")
    return True


def _check_market_condition(
    portfolio: PortfolioState,
    risk_config: Dict[str, Any],
) -> bool:
    """
    Check overall market conditions via VIX level.

    When VIX exceeds the threshold, market is considered to be
    in a high-volatility regime and new positions should be
    avoided.

    Args:
        portfolio: Portfolio state containing current VIX reading.
        risk_config: Risk configuration with VIX threshold.

    Returns:
        True if market conditions are acceptable for new positions.
    """
    vix_threshold = get_nested(
        risk_config,
        "vix_threshold",
        DEFAULT_VIX_THRESHOLD,
    )

    if portfolio.current_vix is None:
        logger.debug(
            "VIX data unavailable, skipping market condition check"
        )
        return True

    if portfolio.current_vix > vix_threshold:
        logger.warning(
            f"VIX at {portfolio.current_vix:.1f} exceeds "
            f"threshold {vix_threshold:.1f}",
            extra={
                "vix": portfolio.current_vix,
                "threshold": vix_threshold,
            },
        )
        return False

    logger.debug(
        f"VIX at {portfolio.current_vix:.1f} within threshold "
        f"({vix_threshold:.1f})"
    )
    return True

"""
Input Validation Functions

Purpose:
    Validates inputs across the system.
    Used at system boundaries for data integrity.

Dependencies:
    None (utility module)

Logging:
    Validation failures at WARNING level.

Fallbacks:
    Returns False or raises ValueError on invalid input.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def validate_symbol(symbol: str) -> bool:
    """
    Validate NSE stock symbol.

    Args:
        symbol: Stock symbol string.

    Returns:
        True if valid symbol format.
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # NSE symbols: uppercase letters, digits, hyphens, max 20 chars
    cleaned = symbol.strip().upper()
    if len(cleaned) > 20 or len(cleaned) == 0:
        return False

    valid_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-&"
    )
    return all(c in valid_chars for c in cleaned)


def validate_ohlcv_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate OHLCV DataFrame for data quality.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].

    Returns:
        Dictionary with validation results:
            - valid: bool
            - issues: list of issue descriptions
            - rows_affected: number of problematic rows
    """
    issues: List[str] = []
    rows_affected = 0

    required_columns = {"open", "high", "low", "close", "volume"}
    actual_columns = set(df.columns.str.lower())

    missing = required_columns - actual_columns
    if missing:
        issues.append(f"Missing columns: {missing}")
        return {
            "valid": False,
            "issues": issues,
            "rows_affected": len(df),
        }

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Check for empty DataFrame
    if df.empty:
        issues.append("DataFrame is empty")
        return {"valid": False, "issues": issues, "rows_affected": 0}

    # Check for negative prices
    for col in ["open", "high", "low", "close"]:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            issues.append(f"Negative values in {col}: {neg_count} rows")
            rows_affected += neg_count

    # Check OHLC consistency: high >= low
    inconsistent = (df["high"] < df["low"]).sum()
    if inconsistent > 0:
        issues.append(
            f"High < Low in {inconsistent} rows"
        )
        rows_affected += inconsistent

    # Check OHLC consistency: high >= open and high >= close
    high_issues = (
        (df["high"] < df["open"]) | (df["high"] < df["close"])
    ).sum()
    if high_issues > 0:
        issues.append(
            f"High not highest in {high_issues} rows"
        )
        rows_affected += high_issues

    # Check OHLC consistency: low <= open and low <= close
    low_issues = (
        (df["low"] > df["open"]) | (df["low"] > df["close"])
    ).sum()
    if low_issues > 0:
        issues.append(
            f"Low not lowest in {low_issues} rows"
        )
        rows_affected += low_issues

    # Check for zero volume
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        issues.append(f"Zero volume in {zero_vol} rows")

    # Check for NaN values
    nan_count = df[list(required_columns)].isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values found: {nan_count} total")
        rows_affected += nan_count

    if issues:
        logger.warning(
            "OHLCV data quality issues detected",
            extra={"issues": issues, "rows_affected": rows_affected},
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "rows_affected": rows_affected,
    }


def validate_price_change(
    current_price: float,
    previous_close: float,
    max_change_percent: float = 20.0,
) -> bool:
    """
    Validate price change is within reasonable bounds.

    Args:
        current_price: Current stock price.
        previous_close: Previous day's closing price.
        max_change_percent: Maximum allowed percentage change.

    Returns:
        True if price change is within bounds.
    """
    if previous_close <= 0:
        return False

    change_pct = abs(
        (current_price - previous_close) / previous_close * 100
    )
    if change_pct > max_change_percent:
        logger.warning(
            f"Price change {change_pct:.2f}% exceeds "
            f"threshold {max_change_percent}%",
            extra={
                "current_price": current_price,
                "previous_close": previous_close,
                "change_percent": change_pct,
            },
        )
        return False

    return True


def validate_date_range(
    start_date: date, end_date: date
) -> bool:
    """
    Validate date range for data fetching.

    Args:
        start_date: Start date.
        end_date: End date.

    Returns:
        True if range is valid.
    """
    if start_date > end_date:
        return False

    # Don't allow future dates
    if end_date > date.today():
        return False

    # Don't allow too old dates (max 5 years)
    max_lookback = date.today().replace(
        year=date.today().year - 5
    )
    if start_date < max_lookback:
        return False

    return True


def validate_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate strategy configuration structure.

    Args:
        config: Strategy config dictionary.

    Returns:
        Dictionary with validation results.
    """
    issues: List[str] = []

    # Required top-level keys
    required_keys = [
        "strategy",
        "indicators",
        "signal_generation",
        "risk_management",
    ]
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key: {key}")

    # Validate strategy section
    strategy = config.get("strategy", {})
    if not strategy.get("name"):
        issues.append("Strategy name is required")

    # Validate indicators
    indicators = config.get("indicators", [])
    if not indicators:
        issues.append("At least one indicator required")

    total_weight = sum(
        ind.get("weight", 0) for ind in indicators
    )
    if abs(total_weight - 1.0) > 0.01:
        issues.append(
            f"Indicator weights sum to {total_weight}, "
            f"should be 1.0"
        )

    # Validate signal generation
    sig_gen = config.get("signal_generation", {})
    min_met = sig_gen.get("min_conditions_met", 0)
    if min_met > len(indicators):
        issues.append(
            f"min_conditions_met ({min_met}) > "
            f"number of indicators ({len(indicators)})"
        )

    threshold = sig_gen.get("confidence_threshold", 0)
    if not 0 <= threshold <= 1:
        issues.append(
            f"confidence_threshold must be 0-1, got {threshold}"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


def validate_portfolio_constraints(
    position_value: float,
    portfolio_value: float,
    sector: str,
    sector_allocations: Dict[str, float],
    max_position_pct: float = 20.0,
    max_sector_pct: float = 30.0,
) -> Dict[str, Any]:
    """
    Validate portfolio constraints before placing an order.

    Args:
        position_value: Value of proposed position.
        portfolio_value: Total portfolio value.
        sector: Sector of the stock.
        sector_allocations: Current sector allocation dict.
        max_position_pct: Maximum single position percentage.
        max_sector_pct: Maximum sector allocation percentage.

    Returns:
        Dictionary with validation results and reasons.
    """
    issues: List[str] = []

    if portfolio_value <= 0:
        return {"valid": False, "issues": ["Invalid portfolio value"]}

    # Position size check
    position_pct = (position_value / portfolio_value) * 100
    if position_pct > max_position_pct:
        issues.append(
            f"Position size {position_pct:.1f}% exceeds "
            f"limit {max_position_pct}%"
        )

    # Sector allocation check
    current_sector_pct = sector_allocations.get(sector, 0)
    new_sector_pct = current_sector_pct + (
        position_value / portfolio_value * 100
    )
    if new_sector_pct > max_sector_pct:
        issues.append(
            f"Sector allocation {new_sector_pct:.1f}% exceeds "
            f"limit {max_sector_pct}%"
        )

    return {"valid": len(issues) == 0, "issues": issues}

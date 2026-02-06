"""
Moving Average Indicators

Purpose:
    Calculates various moving average indicators.
    SMA, EMA, WMA, and crossover signals.

Dependencies:
    - pandas, numpy

Logging:
    Calculation errors at WARNING.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        SMA series.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        EMA series.
    """
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        WMA series.
    """
    weights = np.arange(1, period + 1)

    def weighted_avg(x):
        return np.average(x, weights=weights[-len(x):])

    return series.rolling(window=period).apply(
        weighted_avg, raw=True
    )


def ema_crossover(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
) -> pd.DataFrame:
    """
    EMA Crossover signal.

    Args:
        series: Price series.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.

    Returns:
        DataFrame with fast_ema, slow_ema, signal columns.
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)

    signal = pd.Series(0, index=series.index)
    signal[fast_ema > slow_ema] = 1   # Bullish
    signal[fast_ema < slow_ema] = -1  # Bearish

    return pd.DataFrame(
        {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "signal": signal,
            "crossover": signal.diff().abs() > 0,
        }
    )


def ema_alignment(
    df: pd.DataFrame,
    periods: Optional[list] = None,
) -> pd.Series:
    """
    Check EMA alignment (bullish: short > medium > long).

    Args:
        df: DataFrame with 'close' column.
        periods: List of EMA periods [short, medium, long].

    Returns:
        Series: 1 = bullish aligned, -1 = bearish, 0 = mixed.
    """
    if periods is None:
        periods = [20, 50, 200]

    emas = {}
    for period in periods:
        col_name = f"ema_{period}"
        emas[col_name] = ema(df["close"], period)

    ema_names = [f"ema_{p}" for p in periods]

    result = pd.Series(0, index=df.index)

    # Bullish: short > medium > long
    bullish = True
    for i in range(len(ema_names) - 1):
        bullish_check = emas[ema_names[i]] > emas[ema_names[i + 1]]
        if i == 0:
            bullish_cond = bullish_check
        else:
            bullish_cond = bullish_cond & bullish_check

    result[bullish_cond] = 1

    # Bearish: short < medium < long
    for i in range(len(ema_names) - 1):
        bearish_check = emas[ema_names[i]] < emas[ema_names[i + 1]]
        if i == 0:
            bearish_cond = bearish_check
        else:
            bearish_cond = bearish_cond & bearish_check

    result[bearish_cond] = -1

    return result


def price_vs_ma(
    df: pd.DataFrame,
    period: int = 200,
    ma_type: str = "ema",
) -> pd.DataFrame:
    """
    Calculate price position relative to moving average.

    Args:
        df: DataFrame with 'close' column.
        period: MA period.
        ma_type: 'sma' or 'ema'.

    Returns:
        DataFrame with ma value and distance percentage.
    """
    if ma_type == "sma":
        ma_values = sma(df["close"], period)
    else:
        ma_values = ema(df["close"], period)

    distance_pct = (
        (df["close"] - ma_values) / ma_values * 100
    )

    return pd.DataFrame(
        {
            f"{ma_type}_{period}": ma_values,
            f"distance_pct_{period}": distance_pct,
            f"above_{ma_type}_{period}": df["close"] > ma_values,
        }
    )

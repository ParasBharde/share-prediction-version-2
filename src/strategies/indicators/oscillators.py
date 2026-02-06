"""
Oscillator Indicators

Purpose:
    Calculates RSI, MACD, Stochastic, and other oscillators.

Dependencies:
    - pandas, numpy

Logging:
    Calculation errors at WARNING.
"""

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series (typically close).
        period: RSI lookback period.

    Returns:
        RSI series (0-100).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(
        com=period - 1, min_periods=period
    ).mean()
    avg_loss = loss.ewm(
        com=period - 1, min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Args:
        series: Price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        DataFrame with macd_line, signal_line, histogram.
    """
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }
    )


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        k_period: %K lookback period.
        d_period: %D smoothing period.

    Returns:
        DataFrame with %K and %D.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)

    k_pct = (close - lowest_low) / denominator * 100
    d_pct = k_pct.rolling(window=d_period).mean()

    return pd.DataFrame(
        {
            "stoch_k": k_pct,
            "stoch_d": d_pct,
        }
    )


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period.

    Returns:
        Williams %R series (-100 to 0).
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)

    return -100 * (highest_high - close) / denominator


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period.

    Returns:
        CCI series.
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    # Avoid division by zero
    mean_dev = mean_dev.replace(0, np.nan)

    return (typical_price - sma_tp) / (0.015 * mean_dev)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period.

    Returns:
        DataFrame with ADX, +DI, -DI.
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smooth with EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = (
        100
        * plus_dm.ewm(span=period, adjust=False).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = (
        100
        * minus_dm.ewm(span=period, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    # ADX
    dx = (
        abs(plus_di - minus_di)
        / (plus_di + minus_di).replace(0, np.nan)
        * 100
    )
    adx_val = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame(
        {
            "adx": adx_val,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )

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


def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    ATR using Wilder's recursive smoothing (the correct method).

    Wilder's formula: ATR[i] = (ATR[i-1] * (n-1) + TR[i]) / n
    This differs from EWM (span=period) which underestimates ATR.
    Using EWM produces tighter stop-losses than the strategy intends.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR lookback period (default 14).

    Returns:
        ATR series using Wilder's smoothing.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing: seed with simple mean of first `period` TRs,
    # then apply recursive: ATR[i] = (ATR[i-1] * (n-1) + TR[i]) / n
    atr_values = np.full(len(tr), np.nan)
    if len(tr) < period:
        return pd.Series(atr_values, index=close.index)

    atr_values[period - 1] = tr.iloc[:period].mean()
    for i in range(period, len(tr)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr.iloc[i]) / period

    return pd.Series(atr_values, index=close.index)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend indicator using Wilder's ATR and array-based computation.

    Fixes:
    - Uses wilder_atr() instead of EWM (correct band width)
    - Avoids pandas .iloc assignment (ChainedAssignmentError in pandas 2.x)
      by computing band adjustment and supertrend via numpy arrays

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period for band calculation.
        multiplier: ATR multiplier for band width.

    Returns:
        DataFrame with supertrend, direction (+1 bullish, -1 bearish),
        upper_band, and lower_band.
    """
    atr = wilder_atr(high, low, close, period)

    hl2 = (high + low) / 2.0
    # Raw bands (before Supertrend adjustment)
    raw_upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    raw_lower = (hl2 - multiplier * atr).to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)

    n = len(close_arr)
    upper_band  = raw_upper.copy()
    lower_band  = raw_lower.copy()
    direction   = np.ones(n, dtype=int)
    st_val      = np.full(n, np.nan)

    for i in range(1, n):
        # Adjust lower band: only trail upward (prevent band from dropping)
        if raw_lower[i] > lower_band[i - 1] or close_arr[i - 1] < lower_band[i - 1]:
            lower_band[i] = raw_lower[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # Adjust upper band: only trail downward (prevent band from rising)
        if raw_upper[i] < upper_band[i - 1] or close_arr[i - 1] > upper_band[i - 1]:
            upper_band[i] = raw_upper[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # Direction flip logic
        if direction[i - 1] == -1 and close_arr[i] > upper_band[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close_arr[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        # Supertrend value = lower band when bullish, upper band when bearish
        st_val[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    idx = close.index
    return pd.DataFrame(
        {
            "supertrend": pd.Series(st_val, index=idx),
            "direction":  pd.Series(direction, index=idx),
            "upper_band": pd.Series(upper_band, index=idx),
            "lower_band": pd.Series(lower_band, index=idx),
        }
    )


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

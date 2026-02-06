"""
Volume Indicators

Purpose:
    Calculates volume-based technical indicators.
    OBV, VWAP, Volume SMA, Money Flow Index, etc.

Dependencies:
    - pandas, numpy

Logging:
    Calculation errors at WARNING.
"""

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def volume_sma(
    volume: pd.Series, period: int = 20
) -> pd.Series:
    """
    Volume Simple Moving Average.

    Args:
        volume: Volume series.
        period: Lookback period.

    Returns:
        Volume SMA series.
    """
    return volume.rolling(window=period, min_periods=period).mean()


def volume_ratio(
    volume: pd.Series, period: int = 20
) -> pd.Series:
    """
    Volume ratio relative to average.

    Args:
        volume: Volume series.
        period: Average lookback period.

    Returns:
        Volume ratio (current / average).
    """
    avg = volume_sma(volume, period)
    return volume / avg.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        OBV series.
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Volume Weighted Average Price (cumulative intraday).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.

    Returns:
        VWAP series.
    """
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback period.

    Returns:
        MFI series (0-100).
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    # Positive and negative money flow
    positive = pd.Series(0.0, index=close.index)
    negative = pd.Series(0.0, index=close.index)

    tp_diff = typical_price.diff()
    positive[tp_diff > 0] = raw_money_flow[tp_diff > 0]
    negative[tp_diff < 0] = raw_money_flow[tp_diff < 0]

    positive_sum = positive.rolling(window=period).sum()
    negative_sum = negative.rolling(window=period).sum()

    money_ratio = positive_sum / negative_sum.replace(
        0, np.nan
    )

    return 100 - (100 / (1 + money_ratio))


def accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Accumulation/Distribution Line.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.

    Returns:
        A/D line series.
    """
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)

    clv = ((close - low) - (high - close)) / hl_range
    ad = (clv * volume).cumsum()

    return ad


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Chaikin Money Flow.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback period.

    Returns:
        CMF series.
    """
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)

    clv = ((close - low) - (high - close)) / hl_range
    mf_volume = clv * volume

    cmf = (
        mf_volume.rolling(window=period).sum()
        / volume.rolling(window=period).sum().replace(0, np.nan)
    )

    return cmf


def force_index(
    close: pd.Series,
    volume: pd.Series,
    period: int = 13,
) -> pd.Series:
    """
    Force Index.

    Args:
        close: Close price series.
        volume: Volume series.
        period: EMA smoothing period.

    Returns:
        Force Index series.
    """
    raw_fi = close.diff() * volume
    return raw_fi.ewm(span=period, adjust=False).mean()

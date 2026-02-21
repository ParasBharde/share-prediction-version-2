"""
Bull Flag Pattern Strategy

Purpose:
    Detects Bull Flag patterns — a strong upward 'Pole' followed by a
    tight rectangular 'Flag' consolidation — and signals BUY on a
    high-volume breakout above the flag high.

Algorithm:
    1. Trend filter   : last close > 50 EMA.
    2. Flag scan      : look at last 3–10 candles for tight consolidation
                        (flag_range < max_flag_range_pct).
    3. Pole detection : the N candles before the flag must show a gain of
                        >= min_pole_gain_pct and have above-average volume.
    4. Retracement    : flag pullback <= 50% of pole height.
    5. Breakout       : last close > flag high.
    6. Volume         : breakout volume >= 1.3x 20-day average.
    7. Target         : measured move = flag_high + pole_height  (or fixed %).
    8. SL             : flag low (or fixed %).

Metadata for ChartVisualizer:
    pole_start_idx   – negative index of pole start
    pole_end_idx     – negative index of pole top / flag start
    flag_start_idx   – negative index of first flag candle
    flag_end_idx     – always -2
    flag_high        – flag resistance price
    flag_low         – flag support price
    pole_height      – absolute height of the pole

Dependencies:
    - numpy, pandas
    - src.strategies.base_strategy

Logging:
    Signals at INFO; rejections at DEBUG.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class FlagPatternStrategy(BaseStrategy):
    """
    Bull Flag breakout strategy optimized for BTST setups.

    Fires when a sharp pole is followed by a tight flag and the stock
    breaks out on elevated volume — signalling momentum resumption.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        params = self.strategy_config.get("params", {})

        # Pole params
        self.min_pole_gain_pct = params.get("min_pole_gain_pct", 5.0)
        self.min_pole_days = params.get("min_pole_days", 3)
        self.max_pole_days = params.get("max_pole_days", 10)

        # Flag params
        self.min_flag_days = params.get("min_flag_days", 3)
        self.max_flag_days = params.get("max_flag_days", 10)
        self.max_flag_range_pct = params.get("max_flag_range_pct", 8.0)
        self.max_flag_retrace_pct = params.get(
            "max_flag_retrace_pct", 50.0
        )

        # Volume
        self.breakout_vol_multiplier = params.get(
            "breakout_vol_multiplier", 1.3
        )
        self.pole_vol_multiplier = params.get("pole_vol_multiplier", 1.2)
        self.vol_avg_period = params.get("vol_avg_period", 20)

        # Trend filter
        self.require_uptrend = params.get("require_uptrend", True)
        self.trend_ema_period = params.get("trend_ema_period", 50)

        # Risk
        self.use_measured_move_target = params.get(
            "use_measured_move_target", True
        )
        self.target_pct = params.get("target_pct", 8.0)
        self.stop_loss_pct = params.get("stop_loss_pct", 3.0)
        self.max_stop_loss_pct = params.get("max_stop_loss_pct", 5.0)
        self.use_flag_low_sl = params.get("use_flag_low_sl", True)

        self._scan_stats: Dict[str, int] = {
            "total": 0,
            "pre_filter_rejected": 0,
            "insufficient_data": 0,
            "trend_rejected": 0,
            "no_pattern": 0,
            "volume_rejected": 0,
            "sl_too_wide": 0,
            "low_confidence": 0,
            "signals": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scan_stats(self) -> Dict[str, int]:
        """Return diagnostic counters for this scan run."""
        return dict(self._scan_stats)

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        """
        Scan a single stock for a Bull Flag breakout.

        Args:
            symbol: NSE stock symbol.
            df: OHLCV DataFrame sorted ascending, date-indexed.
            company_info: Company metadata dict.

        Returns:
            TradingSignal or None.
        """
        self._scan_stats["total"] += 1

        if not self.apply_pre_filters(company_info):
            self._scan_stats["pre_filter_rejected"] += 1
            return None

        min_data = (
            self.trend_ema_period
            + self.max_flag_days
            + self.max_pole_days
            + 10
        )
        if len(df) < min_data:
            self._scan_stats["insufficient_data"] += 1
            return None

        # ── 1. Trend filter ──────────────────────────────────────────────
        ema_series = df["close"].ewm(
            span=self.trend_ema_period, adjust=False
        ).mean()
        last_close = float(df["close"].iloc[-1])
        last_ema = float(ema_series.iloc[-1])

        if self.require_uptrend and last_close < last_ema:
            self._scan_stats["trend_rejected"] += 1
            logger.debug(
                f"{symbol}: Trend rejected — close {last_close:.2f} "
                f"< EMA{self.trend_ema_period} {last_ema:.2f}"
            )
            return None

        # ── 2. Pattern detection ──────────────────────────────────────────
        pattern = self._find_flag_pattern(df)
        if pattern is None:
            self._scan_stats["no_pattern"] += 1
            return None

        # ── 3. Volume confirmation ────────────────────────────────────────
        vol_avg = float(
            df["volume"]
            .iloc[-(self.vol_avg_period + 1) : -1]
            .mean()
        )
        breakout_vol = float(df["volume"].iloc[-1])
        vol_ratio = breakout_vol / vol_avg if vol_avg > 0 else 0.0

        if vol_ratio < self.breakout_vol_multiplier:
            self._scan_stats["volume_rejected"] += 1
            logger.debug(
                f"{symbol}: Volume rejected — {vol_ratio:.2f}x "
                f"< {self.breakout_vol_multiplier}x"
            )
            return None

        # ── 4. Entry / Target / SL ────────────────────────────────────────
        entry_price = round(last_close, 2)
        flag_high = pattern["flag_high"]
        flag_low = pattern["flag_low"]
        pole_height = pattern["pole_height"]

        if self.use_measured_move_target:
            target_price = round(flag_high + pole_height, 2)
        else:
            target_price = round(
                entry_price * (1 + self.target_pct / 100), 2
            )

        flag_low_sl = round(flag_low, 2)
        fixed_sl = round(
            entry_price * (1 - self.stop_loss_pct / 100), 2
        )

        if self.use_flag_low_sl and flag_low_sl > fixed_sl:
            stop_loss = flag_low_sl
            sl_method = "flag_low"
        else:
            stop_loss = fixed_sl
            sl_method = f"fixed_{self.stop_loss_pct}pct"

        sl_distance_pct = (
            abs(entry_price - stop_loss) / entry_price * 100
        )
        if sl_distance_pct > self.max_stop_loss_pct:
            self._scan_stats["sl_too_wide"] += 1
            return None

        risk = entry_price - stop_loss
        if risk <= 0:
            self._scan_stats["sl_too_wide"] += 1
            return None
        rr_ratio = round((target_price - entry_price) / risk, 2)

        # ── 5. Confidence ─────────────────────────────────────────────────
        confidence = 0.70
        if pattern["pole_gain_pct"] >= 10.0:
            confidence += 0.06
        if vol_ratio >= 2.0:
            confidence += 0.06
        if pattern["flag_range_pct"] <= 4.0:
            confidence += 0.05  # Very tight flag = higher quality
        if rr_ratio >= 2.0:
            confidence += 0.05
        price_vs_ema_pct = (last_close - last_ema) / last_ema * 100
        if price_vs_ema_pct > 5:
            confidence += 0.04
        confidence = min(round(confidence, 4), 1.0)

        # ── 6. Build signal ───────────────────────────────────────────────
        indicator_details = {
            "flag_pole": {
                "passed": True,
                "pole_gain_pct": round(pattern["pole_gain_pct"], 2),
                "pole_days": pattern["pole_days"],
                "pole_start_price": round(pattern["pole_start_price"], 2),
            },
            "flag_consolidation": {
                "passed": True,
                "flag_high": round(flag_high, 2),
                "flag_low": round(flag_low, 2),
                "flag_range_pct": round(pattern["flag_range_pct"], 2),
                "flag_days": pattern["flag_days"],
                "retrace_pct": round(pattern["retrace_pct"], 2),
            },
            "breakout": {
                "passed": True,
                "breakout_close": round(last_close, 2),
                "above_flag_by_pct": round(
                    (last_close - flag_high) / flag_high * 100, 2
                ),
            },
            "volume_confirmation": {
                "passed": True,
                "vol_ratio": round(vol_ratio, 2),
                "threshold": self.breakout_vol_multiplier,
            },
        }

        signal = TradingSignal(
            symbol=symbol,
            company_name=company_info.get("name", symbol),
            strategy_name=self.name,
            signal_type=SignalType.BUY,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            priority=AlertPriority.HIGH,
            indicators_met=4,
            total_indicators=4,
            indicator_details=indicator_details,
            metadata={
                "timeframe": "1D",
                "mode": "daily",
                # Chart drawing coordinates (negative indices)
                "pole_start_idx": pattern["pole_start_idx"],
                "pole_end_idx": pattern["pole_end_idx"],
                "flag_start_idx": pattern["flag_start_idx"],
                "flag_end_idx": -2,
                "flag_high": round(flag_high, 2),
                "flag_low": round(flag_low, 2),
                "pole_height": round(pole_height, 2),
                "pole_start_price": round(pattern["pole_start_price"], 2),
                # Risk metrics
                "vol_ratio": round(vol_ratio, 2),
                "sl_method": sl_method,
                "sl_distance_pct": round(sl_distance_pct, 2),
                "rr_ratio": rr_ratio,
                "target_pct": round(
                    (target_price - entry_price) / entry_price * 100, 2
                ),
                "trend_ema": round(last_ema, 2),
            },
        )

        logger.info(
            f"SIGNAL: {self.name} — {symbol} "
            f"| Pole: +{pattern['pole_gain_pct']:.1f}% "
            f"({pattern['pole_days']}d) "
            f"| Flag: {pattern['flag_days']}d "
            f"tight={pattern['flag_range_pct']:.1f}% "
            f"| Entry: {entry_price} "
            f"| Target: {target_price} "
            f"| Vol: {vol_ratio:.2f}x "
            f"| R:R 1:{rr_ratio} "
            f"| Conf: {confidence:.0%}"
        )

        self._scan_stats["signals"] += 1
        return signal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_flag_pattern(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a Bull Flag pattern ending at the last candle.

        Iterates over valid flag lengths (min_flag_days to max_flag_days),
        then checks for a qualifying pole immediately before the flag.
        The last candle must close above the flag high (breakout).

        Args:
            df: Full OHLCV DataFrame.

        Returns:
            Dict describing the pattern, or None if not found.
        """
        n = len(df)
        breakout_close = float(df["close"].iloc[-1])

        for flag_len in range(self.min_flag_days, self.max_flag_days + 1):
            # The flag consists of candles from flag_start to n-2 (inclusive)
            # The last candle (n-1) is the breakout bar
            flag_end_abs = n - 2           # last consolidation candle
            flag_start_abs = flag_end_abs - flag_len + 1

            if flag_start_abs < 1:
                continue

            flag_slice = df.iloc[flag_start_abs : flag_end_abs + 1]
            flag_high = float(flag_slice["high"].max())
            flag_low = float(flag_slice["low"].min())

            # Flag range must be tight
            flag_range_pct = (flag_high - flag_low) / flag_high * 100
            if flag_range_pct > self.max_flag_range_pct:
                continue

            # Breakout: last close must be above flag high
            if breakout_close <= flag_high:
                continue

            # Search for the pole ending just before the flag
            for pole_len in range(
                self.min_pole_days, self.max_pole_days + 1
            ):
                pole_end_abs = flag_start_abs - 1
                pole_start_abs = pole_end_abs - pole_len + 1

                if pole_start_abs < 0:
                    continue

                pole_slice = df.iloc[pole_start_abs : pole_end_abs + 1]
                pole_start_price = float(pole_slice["close"].iloc[0])
                pole_end_price = float(pole_slice["close"].iloc[-1])

                if pole_start_price <= 0:
                    continue

                pole_gain_pct = (
                    (pole_end_price - pole_start_price)
                    / pole_start_price * 100
                )
                if pole_gain_pct < self.min_pole_gain_pct:
                    continue

                pole_height = pole_end_price - pole_start_price

                # Flag retracement must not exceed max_flag_retrace_pct
                # of the pole height
                flag_retrace = pole_end_price - flag_low
                retrace_pct = (
                    flag_retrace / pole_height * 100
                    if pole_height > 0
                    else 999
                )
                if retrace_pct > self.max_flag_retrace_pct:
                    continue

                # Valid pattern found
                return {
                    "flag_high": flag_high,
                    "flag_low": flag_low,
                    "flag_range_pct": flag_range_pct,
                    "flag_days": flag_len,
                    "flag_start_idx": flag_start_abs - n,  # negative
                    "pole_height": pole_height,
                    "pole_gain_pct": pole_gain_pct,
                    "pole_days": pole_len,
                    "pole_start_price": pole_start_price,
                    "pole_start_idx": pole_start_abs - n,  # negative
                    "pole_end_idx": pole_end_abs - n,       # negative
                    "retrace_pct": retrace_pct,
                }

        return None

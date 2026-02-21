"""
Symmetrical Triangle Breakout Strategy

Purpose:
    Identifies symmetrical triangles by fitting regression lines through
    descending pivot highs (resistance) and ascending pivot lows (support).
    Signals BUY when price closes above the resistance trendline on the last
    bar, confirming an upside breakout.

Algorithm:
    1. Trend filter   : last close > 50 EMA.
    2. Pivot detection: find local highs (descending) and local lows (ascending)
                        within the last ``pattern_lookback`` candles.
    3. Regression     : fit a line through >= 2 pivot highs and >= 2 pivot lows.
    4. Convergence    : resistance slope < 0  (falling), support slope > 0 (rising).
    5. R² quality     : both regression fits must meet ``min_r_squared``.
    6. Breakout       : last close > projected resistance value at last bar.
    7. Volume         : breakout volume >= 1.3x 20-day average.
    8. SL             : support line value at last bar (or fixed %).
    9. Target         : fixed % from entry.

Metadata for ChartVisualizer:
    resistance_points  – list of [neg_idx, price] pairs (upper trendline)
    support_points     – list of [neg_idx, price] pairs (lower trendline)
    resistance_at_breakout – value of resistance line at last bar
    support_at_breakout    – value of support line at last bar

Dependencies:
    - numpy, pandas
    - src.strategies.base_strategy

Logging:
    Signals at INFO; rejections at DEBUG.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class SymmetricalTriangleStrategy(BaseStrategy):
    """
    Symmetrical Triangle breakout strategy for BTST setups.

    Fires when pivot-line convergence indicates coiling price action and
    the last bar breaks above the descending resistance trendline.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        params = self.strategy_config.get("params", {})

        self.pattern_lookback = params.get("pattern_lookback", 50)
        self.min_pivot_points = params.get("min_pivot_points", 2)
        self.pivot_order = params.get("pivot_order", 3)

        self.min_r_squared = params.get("min_r_squared", 0.70)
        self.resistance_max_slope = params.get(
            "resistance_max_slope", -0.05
        )
        self.support_min_slope = params.get("support_min_slope", 0.05)

        self.breakout_vol_multiplier = params.get(
            "breakout_vol_multiplier", 1.3
        )
        self.vol_avg_period = params.get("vol_avg_period", 20)

        self.require_uptrend = params.get("require_uptrend", True)
        self.trend_ema_period = params.get("trend_ema_period", 50)

        self.target_pct = params.get("target_pct", 8.0)
        self.stop_loss_pct = params.get("stop_loss_pct", 3.0)
        self.max_stop_loss_pct = params.get("max_stop_loss_pct", 5.0)
        self.use_support_line_sl = params.get(
            "use_support_line_sl", True
        )

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
        return dict(self._scan_stats)

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        self._scan_stats["total"] += 1

        if not self.apply_pre_filters(company_info):
            self._scan_stats["pre_filter_rejected"] += 1
            return None

        min_data = max(
            self.pattern_lookback + 10, self.trend_ema_period + 10
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
        pattern = self._find_triangle(df)
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
        target_price = round(
            entry_price * (1 + self.target_pct / 100), 2
        )

        support_sl = round(pattern["support_at_breakout"], 2)
        fixed_sl = round(
            entry_price * (1 - self.stop_loss_pct / 100), 2
        )

        if self.use_support_line_sl and support_sl > fixed_sl:
            stop_loss = support_sl
            sl_method = "support_line"
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
        confidence = 0.68
        avg_r2 = (
            pattern["resistance_r2"] + pattern["support_r2"]
        ) / 2
        if avg_r2 >= 0.85:
            confidence += 0.07
        elif avg_r2 >= 0.75:
            confidence += 0.04
        if vol_ratio >= 2.0:
            confidence += 0.06
        if len(pattern["resistance_points"]) >= 3:
            confidence += 0.04
        if rr_ratio >= 2.0:
            confidence += 0.05
        price_vs_ema = (last_close - last_ema) / last_ema * 100
        if price_vs_ema > 5:
            confidence += 0.04
        confidence = min(round(confidence, 4), 1.0)

        # ── 6. Build signal ───────────────────────────────────────────────
        indicator_details = {
            "symmetrical_triangle": {
                "passed": True,
                "resistance_pivots": len(pattern["resistance_points"]),
                "support_pivots": len(pattern["support_points"]),
                "resistance_r2": round(pattern["resistance_r2"], 3),
                "support_r2": round(pattern["support_r2"], 3),
                "resistance_slope": round(
                    pattern["resistance_slope"], 4
                ),
                "support_slope": round(pattern["support_slope"], 4),
            },
            "resistance_breakout": {
                "passed": True,
                "resistance_at_breakout": round(
                    pattern["resistance_at_breakout"], 2
                ),
                "close": round(last_close, 2),
                "break_pct": round(
                    (last_close - pattern["resistance_at_breakout"])
                    / pattern["resistance_at_breakout"] * 100,
                    2,
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
            indicators_met=3,
            total_indicators=3,
            indicator_details=indicator_details,
            metadata={
                "timeframe": "1D",
                "mode": "daily",
                # Trendline drawing data for ChartVisualizer
                "resistance_points": pattern["resistance_points"],
                "support_points": pattern["support_points"],
                "resistance_at_breakout": round(
                    pattern["resistance_at_breakout"], 2
                ),
                "support_at_breakout": round(
                    pattern["support_at_breakout"], 2
                ),
                "resistance_slope": round(
                    pattern["resistance_slope"], 4
                ),
                "support_slope": round(pattern["support_slope"], 4),
                # Risk metrics
                "vol_ratio": round(vol_ratio, 2),
                "sl_method": sl_method,
                "sl_distance_pct": round(sl_distance_pct, 2),
                "rr_ratio": rr_ratio,
                "target_pct": self.target_pct,
                "trend_ema": round(last_ema, 2),
            },
        )

        logger.info(
            f"SIGNAL: {self.name} — {symbol} "
            f"| Pivots: {len(pattern['resistance_points'])}R/"
            f"{len(pattern['support_points'])}S "
            f"| R²: {avg_r2:.2f} "
            f"| Entry: {entry_price} "
            f"| Vol: {vol_ratio:.2f}x "
            f"| R:R 1:{rr_ratio} "
            f"| Conf: {confidence:.0%}"
        )

        self._scan_stats["signals"] += 1
        return signal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_triangle(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a symmetrical triangle in the most recent pattern_lookback
        candles, ending at the second-to-last bar.

        Returns a dict with trendline parameters or None.
        """
        n = len(df)
        # Work with the last pattern_lookback candles (excluding the
        # breakout bar which is the last candle)
        window = df.iloc[max(0, n - self.pattern_lookback - 1) : n - 1]
        w_len = len(window)
        if w_len < (self.min_pivot_points * 2 + self.pivot_order * 2):
            return None

        highs = window["high"].values
        lows = window["low"].values
        order = self.pivot_order

        # ── Pivot highs ────────────────────────────────────────────────
        pivot_highs: List[Tuple[int, float]] = []
        for i in range(order, w_len - order):
            if all(
                highs[i] >= highs[i - j] for j in range(1, order + 1)
            ) and all(
                highs[i] >= highs[i + j] for j in range(1, order + 1)
            ):
                pivot_highs.append((i, highs[i]))

        # ── Pivot lows ─────────────────────────────────────────────────
        pivot_lows: List[Tuple[int, float]] = []
        for i in range(order, w_len - order):
            if all(
                lows[i] <= lows[i - j] for j in range(1, order + 1)
            ) and all(
                lows[i] <= lows[i + j] for j in range(1, order + 1)
            ):
                pivot_lows.append((i, lows[i]))

        if (
            len(pivot_highs) < self.min_pivot_points
            or len(pivot_lows) < self.min_pivot_points
        ):
            return None

        # ── Regression on pivots ───────────────────────────────────────
        res = self._fit_and_validate(
            pivot_highs, w_len - 1, slope_max=self.resistance_max_slope
        )
        sup = self._fit_and_validate(
            pivot_lows, w_len - 1, slope_min=self.support_min_slope
        )

        if res is None or sup is None:
            return None

        # ── R² quality gate ────────────────────────────────────────────
        if (
            res["r_squared"] < self.min_r_squared
            or sup["r_squared"] < self.min_r_squared
        ):
            return None

        # ── Breakout check ─────────────────────────────────────────────
        last_close = float(df["close"].iloc[-1])
        resistance_at_breakout = res["value_at_end"]
        support_at_breakout = sup["value_at_end"]

        if last_close <= resistance_at_breakout:
            return None

        # ── Convert window-relative indices to full-df negative offsets ──
        window_offset = max(0, n - self.pattern_lookback - 1)

        def to_neg_idx(win_idx: int) -> int:
            return (window_offset + win_idx) - n

        resistance_points = [
            [to_neg_idx(p[0]), round(float(p[1]), 2)]
            for p in pivot_highs
        ]
        support_points = [
            [to_neg_idx(p[0]), round(float(p[1]), 2)]
            for p in pivot_lows
        ]

        return {
            "resistance_points": resistance_points,
            "support_points": support_points,
            "resistance_slope": res["slope"],
            "support_slope": sup["slope"],
            "resistance_r2": res["r_squared"],
            "support_r2": sup["r_squared"],
            "resistance_at_breakout": resistance_at_breakout,
            "support_at_breakout": support_at_breakout,
        }

    def _fit_and_validate(
        self,
        points: List[Tuple[int, float]],
        end_idx: int,
        slope_max: Optional[float] = None,
        slope_min: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fit a linear regression line through pivot points and validate slope.

        Args:
            points: List of (x_index, price) tuples.
            end_idx: The x-index at the last consolidation bar.
            slope_max: If given, slope must be <= this (for resistance).
            slope_min: If given, slope must be >= this (for support).

        Returns:
            Dict with slope, intercept, r_squared, value_at_end, or None.
        """
        xs = np.array([float(p[0]) for p in points])
        ys = np.array([float(p[1]) for p in points])

        if len(xs) < 2:
            return None

        coeffs = np.polyfit(xs, ys, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        # Slope direction check
        if slope_max is not None and slope > slope_max:
            return None
        if slope_min is not None and slope < slope_min:
            return None

        # R² calculation
        y_pred = slope * xs + intercept
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        value_at_end = slope * end_idx + intercept

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "value_at_end": value_at_end,
        }

"""
Descending Channel Breakout Strategy

Purpose:
    Identifies stocks in a short-term descending channel (corrective pullback)
    within a primary uptrend.  Signals BUY when price breaks above the upper
    channel line, marking the end of the correction and the resumption of the
    primary uptrend.

Algorithm:
    1. Trend filter    : last close > 50 EMA (primary uptrend).
    2. Pivot detection : find >= 2 descending pivot highs and >= 2 descending
                         pivot lows in the last ``channel_lookback`` candles.
    3. Channel lines   : fit regression lines through pivot highs (upper) and
                         pivot lows (lower).
    4. Channel slope   : both lines must slope downward within configured bounds.
    5. R² quality      : both regression fits must meet ``min_r_squared``.
    6. Breakout        : last close > projected upper-line value at last bar.
    7. Volume          : breakout volume >= 1.3x 20-day average.
    8. SL              : lower channel value at breakout bar (or fixed %).
    9. Target          : fixed % from entry.

Metadata for ChartVisualizer:
    upper_line_points  – list of [neg_idx, price] pairs (upper channel)
    lower_line_points  – list of [neg_idx, price] pairs (lower channel)
    upper_at_breakout  – value of upper line at last bar
    lower_at_breakout  – value of lower line at last bar

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


class DescendingChannelStrategy(BaseStrategy):
    """
    Descending Channel pullback-breakout strategy for BTST setups.

    Fires when a stock correcting inside a channel within a primary uptrend
    breaks back above the upper channel resistance, signalling the pullback
    is over and the uptrend is resuming.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        params = self.strategy_config.get("params", {})

        self.channel_lookback = params.get("channel_lookback", 30)
        self.min_pivot_points = params.get("min_pivot_points", 2)
        self.pivot_order = params.get("pivot_order", 2)
        self.max_channel_slope = params.get("max_channel_slope", -0.05)
        self.min_channel_slope = params.get("min_channel_slope", -5.0)
        self.min_r_squared = params.get("min_r_squared", 0.65)

        self.breakout_vol_multiplier = params.get(
            "breakout_vol_multiplier", 1.3
        )
        self.vol_avg_period = params.get("vol_avg_period", 20)

        self.require_uptrend = params.get("require_uptrend", True)
        self.trend_ema_period = params.get("trend_ema_period", 50)

        self.target_pct = params.get("target_pct", 7.0)
        self.stop_loss_pct = params.get("stop_loss_pct", 3.0)
        self.max_stop_loss_pct = params.get("max_stop_loss_pct", 5.0)
        self.use_lower_channel_sl = params.get(
            "use_lower_channel_sl", True
        )

        # Validation rules (configurable via YAML params)
        self.min_rr = params.get("min_rr", 1.5)
        self.dma_wall_pct = params.get("dma_wall_pct", 2.0)

        # Risk-at-Risk position sizing
        _ps = self.risk_config.get("position_size", {})
        self.risk_capital = params.get("capital", 1_000_000.0)
        self.risk_pct = _ps.get("risk_per_trade_percent", 1.0)

        self._scan_stats: Dict[str, int] = {
            "total": 0,
            "pre_filter_rejected": 0,
            "insufficient_data": 0,
            "trend_rejected": 0,
            "overbought_rejected": 0,
            "no_pattern": 0,
            "volume_rejected": 0,
            "sl_too_wide": 0,
            "rr_too_low": 0,
            "dma_wall_blocked": 0,
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
            self.channel_lookback + 10, self.trend_ema_period + 10
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

        # ── 1b. Overbought filter (RSI > 75 → Neutral) ───────────────────
        from src.strategies.indicators.oscillators import rsi as calc_rsi
        rsi_series = calc_rsi(df["close"], 14)
        last_rsi = float(rsi_series.iloc[-1])
        if not pd.isna(last_rsi) and last_rsi > 75:
            self._scan_stats["overbought_rejected"] += 1
            logger.debug(
                f"{symbol}: Overbought — RSI {last_rsi:.1f} > 75, skipping"
            )
            return None

        # ── 2. Pattern detection ──────────────────────────────────────────
        pattern = self._find_channel(df)
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

        lower_sl = round(pattern["lower_at_breakout"], 2)
        fixed_sl = round(
            entry_price * (1 - self.stop_loss_pct / 100), 2
        )

        if self.use_lower_channel_sl and lower_sl > fixed_sl:
            stop_loss = lower_sl
            sl_method = "lower_channel"
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

        # ── 4b. Universal signal validation (Rules 1-3) ───────────────────
        passed, rule_reason = self.validate_signal_rules(
            entry_price, target_price, stop_loss, df,
            min_rr=self.min_rr,
            dma_wall_pct=self.dma_wall_pct,
        )
        if not passed:
            if "rr" in rule_reason:
                self._scan_stats["rr_too_low"] += 1
            elif "dma" in rule_reason:
                self._scan_stats["dma_wall_blocked"] += 1
            logger.debug(f"{symbol}: Signal rejected — {rule_reason}")
            return None

        # ── 5. Confidence ─────────────────────────────────────────────────
        confidence = 0.67
        avg_r2 = (
            pattern["upper_r2"] + pattern["lower_r2"]
        ) / 2
        if avg_r2 >= 0.80:
            confidence += 0.07
        elif avg_r2 >= 0.70:
            confidence += 0.04
        if vol_ratio >= 2.0:
            confidence += 0.06
        if len(pattern["upper_line_points"]) >= 3:
            confidence += 0.04
        if rr_ratio >= 2.0:
            confidence += 0.05
        price_vs_ema = (last_close - last_ema) / last_ema * 100
        if price_vs_ema > 3:
            confidence += 0.04
        confidence = min(round(confidence, 4), 1.0)

        # ── 5b. Risk-at-Risk position sizing ─────────────────────────────
        shares, risk_amount = self.calculate_position_size(
            entry_price, stop_loss,
            capital=self.risk_capital,
            risk_pct=self.risk_pct,
        )

        # ── 6. Build signal ───────────────────────────────────────────────
        indicator_details = {
            "descending_channel": {
                "passed": True,
                "upper_pivots": len(pattern["upper_line_points"]),
                "lower_pivots": len(pattern["lower_line_points"]),
                "upper_r2": round(pattern["upper_r2"], 3),
                "lower_r2": round(pattern["lower_r2"], 3),
                "channel_slope": round(pattern["upper_slope"], 4),
            },
            "upper_line_breakout": {
                "passed": True,
                "upper_at_breakout": round(
                    pattern["upper_at_breakout"], 2
                ),
                "close": round(last_close, 2),
                "break_pct": round(
                    (last_close - pattern["upper_at_breakout"])
                    / pattern["upper_at_breakout"] * 100,
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
                "upper_line_points": pattern["upper_line_points"],
                "lower_line_points": pattern["lower_line_points"],
                "upper_at_breakout": round(
                    pattern["upper_at_breakout"], 2
                ),
                "lower_at_breakout": round(
                    pattern["lower_at_breakout"], 2
                ),
                "upper_slope": round(pattern["upper_slope"], 4),
                "lower_slope": round(pattern["lower_slope"], 4),
                # Risk metrics
                "vol_ratio": round(vol_ratio, 2),
                "sl_method": sl_method,
                "sl_distance_pct": round(sl_distance_pct, 2),
                "rr_ratio": rr_ratio,
                "target_pct": self.target_pct,
                "trend_ema": round(last_ema, 2),
                # Risk-at-Risk position sizing
                "position_size_shares": shares,
                "risk_amount_inr": risk_amount,
                "capital": self.risk_capital,
            },
        )

        logger.info(
            f"SIGNAL: {self.name} — {symbol} "
            f"| Slope: {pattern['upper_slope']:.4f} "
            f"| R²: {avg_r2:.2f} "
            f"| Entry: {entry_price} "
            f"| Vol: {vol_ratio:.2f}x "
            f"| R:R 1:{rr_ratio} "
            f"| Qty: {shares} "
            f"| Risk: ₹{risk_amount:,.0f} "
            f"| Conf: {confidence:.0%}"
        )

        self._scan_stats["signals"] += 1
        return signal

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_channel(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a descending channel in the most recent channel_lookback
        candles (excluding the breakout bar).

        Returns a dict with trendline parameters or None.
        """
        n = len(df)
        # Exclude the last (breakout) candle; work on the window before it
        window = df.iloc[max(0, n - self.channel_lookback - 1) : n - 1]
        w_len = len(window)
        order = self.pivot_order
        min_pts = self.min_pivot_points

        if w_len < (min_pts * 2 + order * 2 + 2):
            return None

        highs = window["high"].values
        lows = window["low"].values

        # ── Pivot highs (for upper channel line) ──────────────────────
        pivot_highs: List[Tuple[int, float]] = []
        for i in range(order, w_len - order):
            if all(
                highs[i] >= highs[i - j] for j in range(1, order + 1)
            ) and all(
                highs[i] >= highs[i + j] for j in range(1, order + 1)
            ):
                pivot_highs.append((i, highs[i]))

        # ── Pivot lows (for lower channel line) ───────────────────────
        pivot_lows: List[Tuple[int, float]] = []
        for i in range(order, w_len - order):
            if all(
                lows[i] <= lows[i - j] for j in range(1, order + 1)
            ) and all(
                lows[i] <= lows[i + j] for j in range(1, order + 1)
            ):
                pivot_lows.append((i, lows[i]))

        if len(pivot_highs) < min_pts or len(pivot_lows) < min_pts:
            return None

        # ── Fit regression lines ───────────────────────────────────────
        upper = self._fit_line(pivot_highs, w_len - 1)
        lower = self._fit_line(pivot_lows, w_len - 1)

        if upper is None or lower is None:
            return None

        # ── Both lines must slope downward ─────────────────────────────
        for line in (upper, lower):
            if not (
                self.min_channel_slope
                <= line["slope"]
                <= self.max_channel_slope
            ):
                return None

        # ── R² gate ────────────────────────────────────────────────────
        if (
            upper["r_squared"] < self.min_r_squared
            or lower["r_squared"] < self.min_r_squared
        ):
            return None

        # ── Breakout: last close > upper line projection ───────────────
        last_close = float(df["close"].iloc[-1])
        upper_at_breakout = upper["value_at_end"]
        lower_at_breakout = lower["value_at_end"]

        if last_close <= upper_at_breakout:
            return None

        # ── Convert to negative offsets ────────────────────────────────
        window_offset = max(0, n - self.channel_lookback - 1)

        def to_neg(win_idx: int) -> int:
            return (window_offset + win_idx) - n

        upper_points = [
            [to_neg(p[0]), round(float(p[1]), 2)] for p in pivot_highs
        ]
        lower_points = [
            [to_neg(p[0]), round(float(p[1]), 2)] for p in pivot_lows
        ]

        return {
            "upper_line_points": upper_points,
            "lower_line_points": lower_points,
            "upper_slope": upper["slope"],
            "lower_slope": lower["slope"],
            "upper_r2": upper["r_squared"],
            "lower_r2": lower["r_squared"],
            "upper_at_breakout": upper_at_breakout,
            "lower_at_breakout": lower_at_breakout,
        }

    def _fit_line(
        self,
        points: List[Tuple[int, float]],
        end_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Fit OLS regression through (index, price) pivot points.

        Args:
            points: List of (x, y) pivot tuples.
            end_idx: x-index of the last consolidation bar.

        Returns:
            Dict with slope, r_squared, value_at_end, or None.
        """
        if len(points) < 2:
            return None

        xs = np.array([float(p[0]) for p in points])
        ys = np.array([float(p[1]) for p in points])

        coeffs = np.polyfit(xs, ys, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])

        y_pred = slope * xs + intercept
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "value_at_end": slope * end_idx + intercept,
        }

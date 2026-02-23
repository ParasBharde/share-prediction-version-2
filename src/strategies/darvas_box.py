"""
Darvas Box Breakout Strategy

Purpose:
    Identifies price consolidation boxes (a ceiling held for N+ days with
    a defined floor) inside a primary uptrend and signals BUY on a
    high-volume breakout above the ceiling.

Algorithm:
    1. Trend filter  : last close > 50 EMA (configurable).
    2. Box detection : scan backwards from the second-to-last candle for a
       pivot high (the ceiling) that was not exceeded by any subsequent
       candle up to (but not including) the breakout bar.  The floor is the
       lowest low over the same span.
    3. Breakout      : last close > ceiling.
    4. Volume        : breakout-day volume >= 1.5x 20-day average.
    5. SL            : box floor (or fixed %).
    6. Target        : fixed % from entry.

Metadata for ChartVisualizer:
    box_top          – ceiling price
    box_bottom       – floor price
    box_start_idx    – negative index of ceiling candle in full df
    box_end_idx      – always -2 (last consolidation bar)

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


class DarvasBoxStrategy(BaseStrategy):
    """
    Darvas Box breakout strategy optimized for BTST setups.

    Scans for consolidation boxes within an uptrend and fires on the
    first high-volume candle that closes above the ceiling.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        params = self.strategy_config.get("params", {})

        # Box detection
        self.max_lookback = params.get("max_lookback", 60)
        self.box_hold_days = params.get("box_hold_days", 3)
        self.max_box_depth_pct = params.get("max_box_depth_pct", 12.0)

        # Volume
        self.breakout_vol_multiplier = params.get(
            "breakout_vol_multiplier", 1.5
        )
        self.vol_avg_period = params.get("vol_avg_period", 20)

        # Trend filter
        self.require_uptrend = params.get("require_uptrend", True)
        self.trend_ema_period = params.get("trend_ema_period", 50)

        # Risk
        self.target_pct = params.get("target_pct", 7.0)
        self.stop_loss_pct = params.get("stop_loss_pct", 3.0)
        self.max_stop_loss_pct = params.get("max_stop_loss_pct", 6.0)
        self.use_box_bottom_sl = params.get("use_box_bottom_sl", True)

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
        """Return diagnostic counters for this scan run."""
        return dict(self._scan_stats)

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        """
        Scan a single stock for a Darvas Box breakout.

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

        min_data = max(self.max_lookback + 10, self.trend_ema_period + 10)
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

        # ── 2. Darvas Box detection ───────────────────────────────────────
        box = self._find_darvas_box(df)
        if box is None:
            self._scan_stats["no_pattern"] += 1
            return None

        # ── 3. Volume confirmation ────────────────────────────────────────
        vol_window = df["volume"].iloc[
            -(self.vol_avg_period + 1) : -1
        ]
        vol_avg = float(vol_window.mean()) if len(vol_window) > 0 else 0
        breakout_vol = float(df["volume"].iloc[-1])
        vol_ratio = breakout_vol / vol_avg if vol_avg > 0 else 0.0

        if vol_ratio < self.breakout_vol_multiplier:
            self._scan_stats["volume_rejected"] += 1
            logger.debug(
                f"{symbol}: Volume rejected — {vol_ratio:.2f}x "
                f"< {self.breakout_vol_multiplier}x"
            )
            return None

        # ── 4. Entry / Target / Stop-Loss ─────────────────────────────────
        entry_price = round(last_close, 2)
        target_price = round(
            entry_price * (1 + self.target_pct / 100), 2
        )

        box_bottom_sl = round(box["box_bottom"], 2)
        fixed_sl = round(
            entry_price * (1 - self.stop_loss_pct / 100), 2
        )

        if self.use_box_bottom_sl and box_bottom_sl > fixed_sl:
            stop_loss = box_bottom_sl
            sl_method = "box_bottom"
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
        confidence = 0.68
        if vol_ratio >= 2.0:
            confidence += 0.08
        if vol_ratio >= 3.0:
            confidence += 0.04
        if box["box_hold_days_actual"] >= 5:
            confidence += 0.05
        if rr_ratio >= 2.0:
            confidence += 0.05
        price_vs_ema_pct = (last_close - last_ema) / last_ema * 100
        if price_vs_ema_pct > 5:
            confidence += 0.05
        confidence = min(round(confidence, 4), 1.0)

        # ── 5b. Risk-at-Risk position sizing ─────────────────────────────
        shares, risk_amount = self.calculate_position_size(
            entry_price, stop_loss,
            capital=self.risk_capital,
            risk_pct=self.risk_pct,
        )

        # ── 6. Build signal ───────────────────────────────────────────────
        box_top = round(box["box_top"], 2)
        box_bottom = round(box["box_bottom"], 2)

        indicator_details = {
            "darvas_box": {
                "passed": True,
                "box_top": box_top,
                "box_bottom": box_bottom,
                "box_hold_days": box["box_hold_days_actual"],
                "box_depth_pct": round(
                    (box_top - box_bottom) / box_top * 100, 2
                ),
            },
            "volume_breakout": {
                "passed": True,
                "vol_ratio": round(vol_ratio, 2),
                "threshold": self.breakout_vol_multiplier,
            },
            "trend_filter": {
                "passed": True,
                "price": round(last_close, 2),
                "ema": round(last_ema, 2),
                "price_vs_ema_pct": round(price_vs_ema_pct, 2),
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
                # Box drawing coordinates for ChartVisualizer
                "box_top": box_top,
                "box_bottom": box_bottom,
                "box_start_idx": box["ceiling_idx"] - len(df),  # negative
                "box_end_idx": -2,
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
            f"| Box: {box_top}–{box_bottom} "
            f"| Hold: {box['box_hold_days_actual']}d "
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

    def _find_darvas_box(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Scan backwards for a valid Darvas Box.

        A valid box requires:
        - A ceiling candle whose high is not exceeded by ANY subsequent
          candle up to (but not including) the last bar.
        - At least ``box_hold_days`` candles between ceiling and last bar.
        - Last close > ceiling  (the breakout itself).
        - Box depth within ``max_box_depth_pct``.

        Args:
            df: Full OHLCV DataFrame.

        Returns:
            Dict with box_top, box_bottom, ceiling_idx,
            box_hold_days_actual, or None.
        """
        n = len(df)
        breakout_close = float(df["close"].iloc[-1])

        # Scan from second-to-last backwards
        for ceiling_pos in range(
            n - 2, max(n - self.max_lookback - 2, 1), -1
        ):
            ceiling = float(df["high"].iloc[ceiling_pos])

            # Must have at least box_hold_days candles between ceiling and
            # the breakout bar
            post_candles = df.iloc[ceiling_pos + 1 : n - 1]
            if len(post_candles) < self.box_hold_days:
                continue

            # No subsequent candle (before breakout) should exceed ceiling
            if float(post_candles["high"].max()) >= ceiling:
                continue

            # Breakout bar closes above ceiling
            if breakout_close <= ceiling:
                continue

            # Floor = lowest low from ceiling bar to last consolidation bar
            box_bottom = float(
                df["low"].iloc[ceiling_pos : n - 1].min()
            )

            # Validate box depth
            box_depth_pct = (ceiling - box_bottom) / ceiling * 100
            if box_depth_pct > self.max_box_depth_pct:
                continue

            return {
                "box_top": ceiling,
                "box_bottom": box_bottom,
                "ceiling_idx": ceiling_pos,
                "box_hold_days_actual": n - 2 - ceiling_pos,
            }

        return None

"""
Mother Candle Breakout Strategy V2 - Research-Based Pure Price Action

═══════════════════════════════════════════════════
CONCEPT (from established trading literature):
    The "Inside Bar" / "Mother Candle" pattern is a consolidation
    pattern where a large candle (Mother) contains subsequent smaller
    candles (Babies) entirely within its high-low range.
    When price finally breaks out, it signals strong directional momentum.

    Sources: Nial Fuller, PriceAction.com, Subashish Pani (Power of Stocks),
    Al Brooks (Price Action), multiple candlestick pattern references.

═══════════════════════════════════════════════════
ALGORITHM (Right-to-Left scan):
    Step 1: Start from last candle (potential breakout), scan backward
    Step 2: For each candidate Mother, check if it's a large candle
    Step 3: Validate ALL candles between Mother and last are babies
            (strictly contained within Mother's high-low range)
    Step 4: Confirm last candle CLOSES above Mother High (fresh breakout)
    Step 5: Volume confirmation (Mother + Breakout candle)
    Step 6: Generate signal with configurable target/SL

═══════════════════════════════════════════════════
RULES (research-based):
    - Mother must be a significant candle (range > multiplier x avg prior)
    - ALL baby candles must be inside Mother H/L (with tiny tolerance)
    - NO baby candle may have closed above Mother High (fresh breakout only)
    - Last candle must CLOSE above Mother High
    - Mother volume should be above average (institutional candle)
    - Breakout volume should be above average (real participation)
    - Volume should decline during babies (consolidation)
    - Target and StopLoss are configurable fixed percentages
    - Max SL cap to reject patterns where Mother is too large
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MotherCandleV2Strategy(BaseStrategy):
    """
    Research-based Mother Candle (Inside Bar) Breakout strategy.

    Scans right-to-left on daily chart to find a Mother Candle
    with 2+ baby candles strictly inside, confirmed by a fresh
    breakout on the last candle with volume confirmation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "1D")
        params = self.strategy_config.get("params", {})

        # Pattern detection params
        self.max_lookback = params.get("max_lookback", 20)
        self.min_babies = params.get("min_baby_candles", 2)
        self.mother_range_multiplier = params.get(
            "mother_range_multiplier", 1.5
        )
        self.baby_tolerance_pct = params.get(
            "baby_tolerance_pct", 0.1
        )

        # Volume params
        self.mother_vol_multiplier = params.get(
            "mother_vol_multiplier", 1.3
        )
        self.breakout_vol_multiplier = params.get(
            "breakout_vol_multiplier", 1.2
        )

        # Target & Stop Loss params (fixed percentage based)
        self.target_pct = params.get("target_pct", 5.0)
        self.stop_loss_pct = params.get("stop_loss_pct", 3.0)
        self.max_stop_loss_pct = params.get("max_stop_loss_pct", 5.0)
        self.use_mother_low_sl = params.get(
            "use_mother_low_sl", True
        )

        # Entry buffer - reject if breakout candle already moved too far
        self.max_entry_buffer_pct = params.get(
            "max_entry_buffer_pct", 2.0
        )

        self._scan_stats = {
            "total": 0,
            "pre_filter_rejected": 0,
            "insufficient_data": 0,
            "no_pattern": 0,
            "volume_rejected": 0,
            "sl_too_wide": 0,
            "signals": 0,
        }

    def get_scan_stats(self) -> Dict[str, int]:
        """Return scan statistics for diagnostics."""
        return dict(self._scan_stats)

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        self._scan_stats["total"] += 1

        # Pre-filters (market cap, volume, price range)
        if not self.apply_pre_filters(company_info):
            self._scan_stats["pre_filter_rejected"] += 1
            return None

        # Need enough data for lookback + averages
        min_data = self.max_lookback + 20
        if len(df) < min_data:
            self._scan_stats["insufficient_data"] += 1
            return None

        # ============================================================
        # STEP 1-4: Find Mother, Validate Babies, Check Breakout
        # ============================================================
        pattern = self._discover_mother_candle(df)
        if pattern is None:
            self._scan_stats["no_pattern"] += 1
            return None

        mother_high = pattern["mother_high"]
        mother_low = pattern["mother_low"]
        mother_idx = pattern["mother_position"]
        baby_count = pattern["baby_count"]
        breakout_close = pattern["breakout_close"]

        # ============================================================
        # STEP 5: Volume Confirmation
        # ============================================================
        # 5a. Mother candle volume > multiplier x avg of prior 10
        mother_abs_idx = len(df) + mother_idx
        if mother_abs_idx < 10:
            self._scan_stats["volume_rejected"] += 1
            return None

        mother_volume = float(df.iloc[mother_idx]["volume"])
        prev_10_avg_vol = float(
            df["volume"].iloc[
                mother_abs_idx - 10: mother_abs_idx
            ].mean()
        )
        mother_vol_ratio = (
            mother_volume / prev_10_avg_vol
            if prev_10_avg_vol > 0
            else 0
        )
        if mother_vol_ratio < self.mother_vol_multiplier:
            self._scan_stats["volume_rejected"] += 1
            return None

        # 5b. Breakout candle volume > multiplier x 20-day avg
        vol_r = volume_ratio(df["volume"], 20)
        breakout_vol_ratio = float(vol_r.iloc[-1])
        if breakout_vol_ratio < self.breakout_vol_multiplier:
            self._scan_stats["volume_rejected"] += 1
            return None

        # ============================================================
        # STEP 6: Entry, Target, Stop Loss Calculation
        # ============================================================
        entry_price = round(breakout_close, 2)

        # Check breakout isn't overextended beyond Mother High
        break_distance = (
            (breakout_close - mother_high) / mother_high * 100
        )
        if break_distance > self.max_entry_buffer_pct:
            self._scan_stats["no_pattern"] += 1
            return None

        # Target = entry + target_pct%
        target_price = round(
            entry_price * (1 + self.target_pct / 100), 2
        )

        # Stop Loss logic:
        # Option A: Fixed percentage SL
        fixed_sl = round(
            entry_price * (1 - self.stop_loss_pct / 100), 2
        )

        # Option B: Mother Low SL (traditional approach)
        mother_low_sl = round(mother_low, 2)

        # Use the TIGHTER (higher) stop loss - less risk
        if self.use_mother_low_sl and mother_low_sl > fixed_sl:
            stop_loss = mother_low_sl
            sl_method = "mother_low"
        else:
            stop_loss = fixed_sl
            sl_method = f"fixed_{self.stop_loss_pct}pct"

        # Max SL cap - reject if SL distance is too wide
        sl_distance_pct = (
            abs(entry_price - stop_loss) / entry_price * 100
        )
        if sl_distance_pct > self.max_stop_loss_pct:
            self._scan_stats["sl_too_wide"] += 1
            return None

        # Calculate actual R:R
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        if risk <= 0:
            self._scan_stats["sl_too_wide"] += 1
            return None
        rr_ratio = round(reward / risk, 2)

        # ============================================================
        # Build indicator details for alert
        # ============================================================
        indicator_details = {
            "mother_candle": {
                "passed": True,
                "mother_high": round(mother_high, 2),
                "mother_low": round(mother_low, 2),
                "mother_range": round(
                    mother_high - mother_low, 2
                ),
                "baby_count": baby_count,
                "days_consolidation": baby_count,
                "mother_position": (
                    f"{abs(mother_idx)} candles ago"
                ),
            },
            "fresh_breakout": {
                "passed": True,
                "breakout_close": round(breakout_close, 2),
                "mother_high": round(mother_high, 2),
                "break_amount": round(
                    breakout_close - mother_high, 2
                ),
                "break_pct": round(break_distance, 2),
            },
            "mother_volume": {
                "passed": True,
                "mother_vol_ratio": round(
                    mother_vol_ratio, 2
                ),
                "threshold": self.mother_vol_multiplier,
            },
            "breakout_volume": {
                "passed": True,
                "breakout_vol_ratio": round(
                    breakout_vol_ratio, 2
                ),
                "threshold": self.breakout_vol_multiplier,
            },
        }

        # Confidence calculation
        # Base: pattern + volumes = 0.70
        confidence = 0.70
        if breakout_vol_ratio >= 2.0:
            confidence += 0.10
        if baby_count >= 4:
            confidence += 0.05
        if baby_count >= 7:
            confidence += 0.05
        if mother_vol_ratio >= 2.0:
            confidence += 0.05
        if rr_ratio >= 2.0:
            confidence += 0.05
        confidence = min(confidence, 1.0)

        signal = TradingSignal(
            symbol=symbol,
            company_name=company_info.get("name", symbol),
            strategy_name=self.name,
            signal_type=SignalType.BUY,
            confidence=round(confidence, 4),
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            priority=AlertPriority.HIGH,
            indicators_met=4,
            total_indicators=4,
            indicator_details=indicator_details,
            metadata={
                "timeframe": self.timeframe,
                "mode": "daily",
                "baby_count": baby_count,
                "mother_high": round(mother_high, 2),
                "mother_low": round(mother_low, 2),
                "mother_vol_ratio": round(
                    mother_vol_ratio, 2
                ),
                "breakout_vol_ratio": round(
                    breakout_vol_ratio, 2
                ),
                "sl_distance_pct": round(
                    sl_distance_pct, 2
                ),
                "sl_method": sl_method,
                "rr_ratio": rr_ratio,
                "target_pct": self.target_pct,
                "stop_loss_pct": round(
                    sl_distance_pct, 2
                ),
            },
        )

        logger.info(
            f"SIGNAL: {self.name} - {symbol} "
            f"| Babies: {baby_count} "
            f"| Mother: {round(mother_high, 2)}"
            f"-{round(mother_low, 2)} "
            f"| Entry: {entry_price} "
            f"| Target: {target_price} (+{self.target_pct}%) "
            f"| SL: {stop_loss} (-{round(sl_distance_pct, 1)}%) "
            f"| R:R 1:{rr_ratio} "
            f"| Vol: {round(breakout_vol_ratio, 2)}x "
            f"| Conf: {confidence:.0%}"
        )

        self._scan_stats["signals"] += 1
        return signal

    def _discover_mother_candle(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Right-to-Left scan to find Mother Candle pattern.

        Starts from the last candle (potential breakout) and scans
        backward to find a large Mother Candle with all intermediate
        candles (babies) strictly contained within it.

        Returns:
            Dict with pattern details or None if no valid pattern.
        """
        last_candle = df.iloc[-1]
        last_close = float(last_candle["close"])
        last_high = float(last_candle["high"])

        # Scan backward: mother can be at -2 to -(max_lookback+1)
        # Position -1 is the last candle (breakout candidate)
        for mother_offset in range(2, self.max_lookback + 2):
            if mother_offset >= len(df):
                break

            mother_pos = -mother_offset
            mother = df.iloc[mother_pos]
            m_high = float(mother["high"])
            m_low = float(mother["low"])
            m_range = m_high - m_low

            if m_range <= 0 or m_low <= 0:
                continue

            # ──────────────────────────────────────────────
            # STEP 1: Mother Range Validation
            # Mother range must be > multiplier x avg range
            # of the 5 candles before it
            # ──────────────────────────────────────────────
            mother_abs_idx = len(df) + mother_pos
            if mother_abs_idx < 5:
                continue

            pre_mother_ranges = []
            for k in range(1, 6):
                idx = mother_abs_idx - k
                if idx < 0:
                    break
                c = df.iloc[idx]
                pre_mother_ranges.append(
                    float(c["high"]) - float(c["low"])
                )

            if not pre_mother_ranges:
                continue

            avg_pre_range = (
                sum(pre_mother_ranges) / len(pre_mother_ranges)
            )
            if avg_pre_range <= 0:
                continue

            if m_range < (
                self.mother_range_multiplier * avg_pre_range
            ):
                continue

            # ──────────────────────────────────────────────
            # STEP 2: Baby Candle Strict Containment
            # All candles between mother and last must be
            # inside Mother's H/L range (with small tolerance)
            # ──────────────────────────────────────────────
            baby_start = mother_pos + 1
            baby_count = mother_offset - 2

            if baby_count < self.min_babies:
                continue

            # Tolerance: allow tiny wick breach
            tolerance = m_range * (self.baby_tolerance_pct / 100)
            upper_limit = m_high + tolerance
            lower_limit = m_low - tolerance

            all_inside = True
            for j in range(baby_start, -1):
                baby = df.iloc[j]
                baby_high = float(baby["high"])
                baby_low = float(baby["low"])

                if baby_high > upper_limit or baby_low < lower_limit:
                    all_inside = False
                    break

            if not all_inside:
                continue

            # ──────────────────────────────────────────────
            # STEP 3: Fresh Breakout Validation
            # No baby candle may have closed above Mother High
            # Only the last candle should break out
            # ──────────────────────────────────────────────
            old_breakout = False
            for j in range(baby_start, -1):
                baby_close = float(df.iloc[j]["close"])
                if baby_close > m_high:
                    old_breakout = True
                    break

            if old_breakout:
                continue

            # Last candle must CLOSE above Mother High
            if last_close <= m_high:
                continue

            # ────────────────────────
            # VALID PATTERN FOUND
            # ────────────────────────
            return {
                "mother_high": m_high,
                "mother_low": m_low,
                "mother_range": m_range,
                "mother_position": mother_pos,
                "baby_count": baby_count,
                "breakout_close": last_close,
                "avg_pre_range": round(avg_pre_range, 2),
                "range_multiplier": round(
                    m_range / avg_pre_range, 2
                ),
            }

        return None

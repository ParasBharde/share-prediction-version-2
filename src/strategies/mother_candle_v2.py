"""
Mother Candle Breakout Strategy V2 - Pure Price Action Logic

═══════════════════════════════════════════════════
OBJECTIVE:
    Identify stocks where the LAST candle (current day's completed candle)
    is the FIRST to break above a Mother Candle that formed up to 15 days ago,
    with a minimum of 3 baby candles consolidating strictly inside.

═══════════════════════════════════════════════════
LOGIC FLOW:
    Step 1: Dynamic Mother Candle Discovery (look-back up to 15 days)
    Step 2: Validate Baby Candle Consolidation (strict containment)
    Step 3: Validate Fresh Breakout on Last Candle (FIRST break only)
    Step 4: Volume Confirmation (mother + breakout candle)
    Step 5: Momentum Filters (optional, configurable)
    Step 6: Final Output with full details

═══════════════════════════════════════════════════
HARD RULES:
    - Mother range must be > 1.5x avg range of 5 candles before it
    - Minimum 3 baby candles, maximum 14 (up to 15 day look-back)
    - ALL baby candles must be STRICTLY inside Mother (no wick breach)
    - ONLY the last candle can break above Mother High (fresh breakout)
    - If any prior candle closed above Mother High, REJECT (old breakout)
    - Mother volume > 1.5x avg volume of prior 10 candles
    - Breakout volume > 1.2x of 20-day avg volume
    - R:R must be >= 1.5:1 (SL at Mother Low)
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import adx, rsi
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MotherCandleV2Strategy(BaseStrategy):
    """
    Pure price-action Mother Candle Breakout strategy.
    Focuses on strict containment and fresh-breakout-only logic.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "1D")
        params = self.strategy_config.get("params", {})
        self.max_lookback = params.get("max_lookback", 15)
        self.min_babies = params.get("min_baby_candles", 3)
        self.mother_range_multiplier = params.get(
            "mother_range_multiplier", 1.5
        )
        self.momentum_filters_enabled = params.get(
            "momentum_filters_enabled", False
        )
        self._scan_stats = {
            "total": 0,
            "pre_filter_rejected": 0,
            "insufficient_data": 0,
            "no_pattern": 0,
            "volume_rejected": 0,
            "rr_rejected": 0,
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

        # Pre-filters
        if not self.apply_pre_filters(company_info):
            self._scan_stats["pre_filter_rejected"] += 1
            return None

        # Need at least max_lookback + 15 extra candles for averages
        min_data = self.max_lookback + 20
        if len(df) < min_data:
            self._scan_stats["insufficient_data"] += 1
            return None

        # ============================================================
        # STEP 1 + 2 + 3: Find Mother, Validate Babies, Check Breakout
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
        # STEP 4: Volume Confirmation
        # ============================================================
        # 4a. Mother candle volume > 1.5x avg volume of previous 10 candles
        mother_abs_idx = len(df) + mother_idx  # convert negative to absolute
        if mother_abs_idx < 10:
            self._scan_stats["volume_rejected"] += 1
            return None

        mother_volume = float(df.iloc[mother_idx]["volume"])
        prev_10_avg_vol = float(
            df["volume"].iloc[mother_abs_idx - 10 : mother_abs_idx].mean()
        )
        mother_vol_ratio = (
            mother_volume / prev_10_avg_vol if prev_10_avg_vol > 0 else 0
        )
        if mother_vol_ratio < 1.5:
            self._scan_stats["volume_rejected"] += 1
            return None

        # 4b. Breakout candle volume > 1.2x of 20-day avg volume
        vol_r = volume_ratio(df["volume"], 20)
        breakout_vol_ratio = float(vol_r.iloc[-1])
        if breakout_vol_ratio < 1.2:
            self._scan_stats["volume_rejected"] += 1
            return None

        # ============================================================
        # STEP 5: Momentum Filters (OPTIONAL)
        # ============================================================
        momentum_details = {}
        momentum_passed = True

        if self.momentum_filters_enabled:
            try:
                # RSI between 50 and 80
                rsi_values = rsi(df["close"], 14)
                current_rsi = float(rsi_values.iloc[-1])
                rsi_ok = 50 <= current_rsi <= 80
                momentum_details["rsi"] = {
                    "value": round(current_rsi, 2),
                    "range": "50-80",
                    "passed": rsi_ok,
                }
                if not rsi_ok:
                    momentum_passed = False
            except Exception:
                pass

            try:
                # ADX > 25
                adx_data = adx(
                    df["high"], df["low"], df["close"], 14
                )
                current_adx = float(adx_data["adx"].iloc[-1])
                adx_ok = current_adx >= 25
                momentum_details["adx"] = {
                    "value": round(current_adx, 2),
                    "threshold": 25,
                    "passed": adx_ok,
                }
                if not adx_ok:
                    momentum_passed = False
            except Exception:
                pass

            try:
                # Close > 20 EMA
                ema_20 = float(ema(df["close"], 20).iloc[-1])
                above_ema = breakout_close > ema_20
                momentum_details["above_ema20"] = {
                    "ema_20": round(ema_20, 2),
                    "close": round(breakout_close, 2),
                    "passed": above_ema,
                }
                if not above_ema:
                    momentum_passed = False
            except Exception:
                pass

            if not momentum_passed:
                self._scan_stats["no_pattern"] += 1
                return None

        # ============================================================
        # STEP 6: Final Validation & Signal Generation
        # ============================================================
        entry_price = breakout_close
        stop_loss = round(mother_low, 2)

        # Stop loss distance check (max 7%)
        sl_distance_pct = (
            abs(entry_price - stop_loss) / entry_price * 100
        )
        if sl_distance_pct > 7.0:
            self._scan_stats["rr_rejected"] += 1
            return None

        # Target = 2:1 risk-reward
        risk = entry_price - stop_loss
        target_price = round(entry_price + (risk * 2.0), 2)

        # R:R floor check - must be >= 1.5:1
        if risk <= 0:
            self._scan_stats["rr_rejected"] += 1
            return None
        rr_ratio = (target_price - entry_price) / risk
        if rr_ratio < 1.5:
            self._scan_stats["rr_rejected"] += 1
            return None

        # Build indicator details
        indicator_details = {
            "mother_candle": {
                "passed": True,
                "mother_high": round(mother_high, 2),
                "mother_low": round(mother_low, 2),
                "mother_range": round(mother_high - mother_low, 2),
                "baby_count": baby_count,
                "days_consolidation": baby_count,
                "mother_position": f"{abs(mother_idx)} candles ago",
            },
            "fresh_breakout": {
                "passed": True,
                "breakout_close": round(breakout_close, 2),
                "mother_high": round(mother_high, 2),
                "break_amount": round(breakout_close - mother_high, 2),
                "break_pct": round(
                    (breakout_close - mother_high) / mother_high * 100, 2
                ),
            },
            "mother_volume": {
                "passed": True,
                "mother_vol_ratio": round(mother_vol_ratio, 2),
                "threshold": 1.5,
            },
            "breakout_volume": {
                "passed": True,
                "breakout_vol_ratio": round(breakout_vol_ratio, 2),
                "threshold": 1.2,
            },
        }

        if momentum_details:
            indicator_details["momentum_filters"] = momentum_details

        # Confidence calculation
        # Base: pattern found + volumes confirmed = 0.70
        confidence = 0.70
        if breakout_vol_ratio >= 2.0:
            confidence += 0.10  # Strong volume breakout
        if baby_count >= 5:
            confidence += 0.05  # Longer consolidation = stronger
        if baby_count >= 8:
            confidence += 0.05  # Very long consolidation
        if momentum_passed and self.momentum_filters_enabled:
            confidence += 0.10
        confidence = min(confidence, 1.0)

        indicators_met = 4  # pattern + fresh breakout + 2 volumes
        total_indicators = 4
        if self.momentum_filters_enabled:
            total_indicators += len(momentum_details)
            indicators_met += sum(
                1 for d in momentum_details.values()
                if d.get("passed", False)
            )

        signal = TradingSignal(
            symbol=symbol,
            company_name=company_info.get("name", symbol),
            strategy_name=self.name,
            signal_type=SignalType.BUY,
            confidence=round(confidence, 4),
            entry_price=round(entry_price, 2),
            target_price=target_price,
            stop_loss=stop_loss,
            priority=AlertPriority.HIGH,
            indicators_met=indicators_met,
            total_indicators=total_indicators,
            indicator_details=indicator_details,
            metadata={
                "timeframe": self.timeframe,
                "mode": "daily",
                "baby_count": baby_count,
                "mother_high": round(mother_high, 2),
                "mother_low": round(mother_low, 2),
                "mother_vol_ratio": round(mother_vol_ratio, 2),
                "breakout_vol_ratio": round(breakout_vol_ratio, 2),
                "sl_distance_pct": round(sl_distance_pct, 2),
            },
        )

        logger.info(
            f"SIGNAL: {self.name} - {symbol} "
            f"| Babies: {baby_count} "
            f"| Mother: {round(mother_high,2)}-{round(mother_low,2)} "
            f"| Breakout: {round(breakout_close,2)} "
            f"| Vol: {round(breakout_vol_ratio,2)}x "
            f"| Conf: {confidence:.0%}"
        )

        self._scan_stats["signals"] += 1
        return signal

    def _discover_mother_candle(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        STEP 1-3: Dynamic Mother Candle Discovery with
        strict baby containment and fresh-breakout-only validation.

        Scans backward from candle before last (index -2) up to
        index -(max_lookback+1) to find an anchor Mother Candle.

        Returns:
            Dict with pattern details or None if no valid pattern.
        """
        last_candle = df.iloc[-1]
        last_close = float(last_candle["close"])

        # Calculate average range of recent candles for reference
        # (used to validate Mother candle significance)

        # Scan backward: the mother can be at position -2 to -(max_lookback+1)
        # (position -1 is the last candle which is the potential breakout)
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
            # Mother_Range must be > 1.5x average range
            # of the 5 candles immediately BEFORE the Mother
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

            avg_pre_range = sum(pre_mother_ranges) / len(pre_mother_ranges)
            if avg_pre_range <= 0:
                continue

            if m_range < self.mother_range_multiplier * avg_pre_range:
                continue  # Mother candle not significant enough

            # ──────────────────────────────────────────────
            # STEP 2: Baby Candle Strict Containment
            # All candles between mother and last must be
            # STRICTLY inside Mother's range (no wick breach)
            # ──────────────────────────────────────────────
            # Baby candles are from mother_pos+1 to -2 (inclusive)
            # (i.e., all candles BETWEEN mother and the last candle)
            baby_start = mother_pos + 1  # first baby
            baby_end = -1  # last baby (candle before the last)

            # Calculate baby count
            # e.g., mother at -5, babies at -4, -3, -2 = 3 babies
            baby_count = abs(baby_end) - abs(mother_pos) + 1
            # Simpler: number of candles between mother and last
            baby_count = mother_offset - 2  # offset=2 means 0 babies, offset=5 means 3

            if baby_count < self.min_babies:
                continue  # Not enough babies

            # Check strict containment for ALL baby candles
            # Babies are from baby_start to -2 (inclusive)
            # Index -1 is the breakout candle, NOT a baby
            all_inside = True
            for j in range(baby_start, -1):  # e.g., -5 to -2 inclusive
                baby = df.iloc[j]
                baby_high = float(baby["high"])
                baby_low = float(baby["low"])

                # STRICT: Baby must be completely inside Mother
                if baby_high > m_high or baby_low < m_low:
                    all_inside = False
                    break

            if not all_inside:
                continue

            # ──────────────────────────────────────────────
            # STEP 3: Fresh Breakout Validation
            # The LAST candle must be the ONLY one to close
            # above Mother High. If any baby closed above
            # Mother High before, this is an OLD breakout = REJECT.
            # ──────────────────────────────────────────────

            # Check that no baby candle closed above Mother High
            old_breakout = False
            for j in range(baby_start, -1):  # check babies only, not last
                baby_close = float(df.iloc[j]["close"])
                if baby_close > m_high:
                    old_breakout = True
                    break

            if old_breakout:
                continue  # Old breakout - not fresh

            # The last candle must close ABOVE Mother High
            if last_close <= m_high:
                continue  # No breakout yet

            # ────────────────────────
            # VALID PATTERN FOUND!
            # ────────────────────────
            return {
                "mother_high": m_high,
                "mother_low": m_low,
                "mother_range": m_range,
                "mother_position": mother_pos,
                "baby_count": baby_count,
                "breakout_close": last_close,
                "avg_pre_range": round(avg_pre_range, 2),
                "range_multiplier": round(m_range / avg_pre_range, 2),
            }

        return None

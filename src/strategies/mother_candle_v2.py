"""
Mother Candle Breakout Strategy V2 - IMPROVED VERSION
Added critical filters to prevent false signals like SICALLOG

NEW FILTERS ADDED:
1. Trend Filter: Only trade in uptrend (price > 50 EMA)
2. Minimum Breakout Strength: Breakout must be > min threshold
3. Price Structure: Ensure Mother is near recent highs, not in downtrend
4. Failed Breakout Filter: Check recent history for failed breakouts
5. ATR-based validation: Ensure sufficient volatility
"""

from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MotherCandleV2Strategy(BaseStrategy):
    """
    IMPROVED Mother Candle strategy with trend and context filters.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "1D")
        params = self.strategy_config.get("params", {})

        # Pattern detection params
        self.max_lookback = params.get("max_lookback", 20)
        self.min_babies = params.get("min_baby_candles", 2)
        self.mother_range_multiplier = params.get("mother_range_multiplier", 1.5)
        self.baby_tolerance_pct = params.get("baby_tolerance_pct", 0.1)
        # Maximum age of mother candle in bars — if mother candle is older than
        # this, its price levels are stale and should not be traded.
        # Default: 10 bars (2 trading weeks). Beyond this, the pattern decays.
        self.max_pattern_age_bars = params.get("max_pattern_age_bars", 10)

        # Volume params
        self.mother_vol_multiplier = params.get("mother_vol_multiplier", 1.3)
        self.breakout_vol_multiplier = params.get("breakout_vol_multiplier", 1.2)

        # Target & Stop Loss params
        self.target_pct = params.get("target_pct", 5.0)
        self.stop_loss_pct = params.get("stop_loss_pct", 3.0)
        self.max_stop_loss_pct = params.get("max_stop_loss_pct", 5.0)
        self.use_mother_low_sl = params.get("use_mother_low_sl", True)

        # ═══════════════════════════════════════════════════════════
        # NEW FILTERS TO PREVENT FALSE SIGNALS
        # ═══════════════════════════════════════════════════════════
        
        # 1. TREND FILTER
        self.require_uptrend = params.get("require_uptrend", True)
        self.trend_ema_period = params.get("trend_ema_period", 50)
        self.price_above_ema_pct = params.get("price_above_ema_pct", 0)  # Price must be >= EMA
        
        # 2. MINIMUM BREAKOUT STRENGTH
        self.min_breakout_pct = params.get("min_breakout_pct", 0.3)  # Breakout must be > 0.3% above Mother High
        self.max_entry_buffer_pct = params.get("max_entry_buffer_pct", 2.0)
        
        # 3. PRICE STRUCTURE FILTER
        # Mother should be near recent highs, not in middle of downtrend
        self.mother_position_filter = params.get("mother_position_filter", True)
        self.recent_high_lookback = params.get("recent_high_lookback", 30)
        self.mother_near_high_pct = params.get("mother_near_high_pct", 10)  # Mother High within 10% of recent high
        
        # 4. FAILED BREAKOUT CHECK
        # Don't trade if there were recent failed breakouts
        self.check_failed_breakouts = params.get("check_failed_breakouts", True)
        self.failed_breakout_lookback = params.get("failed_breakout_lookback", 10)
        
        # 5. ATR VALIDATION
        self.min_atr_pct = params.get("min_atr_pct", 1.0)  # Minimum ATR as % of price
        self.atr_period = params.get("atr_period", 14)

        # Validation rules (configurable via YAML params)
        self.min_rr = params.get("min_rr", 1.5)
        self.dma_wall_pct = params.get("dma_wall_pct", 2.0)

        # Risk-at-Risk position sizing
        _ps = self.risk_config.get("position_size", {})
        self.risk_capital = params.get("capital", 1_000_000.0)
        self.risk_pct = _ps.get("risk_per_trade_percent", 1.0)

        self._scan_stats = {
            "total": 0,
            "pre_filter_rejected": 0,
            "insufficient_data": 0,
            "no_pattern": 0,
            "volume_rejected": 0,
            "sl_too_wide": 0,
            "rr_too_low": 0,
            "dma_wall_blocked": 0,
            "trend_rejected": 0,
            "weak_breakout": 0,
            "bad_position": 0,
            "failed_breakout_history": 0,
            "low_volatility": 0,
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

        # Need enough data
        min_data = max(self.max_lookback + 20, self.trend_ema_period + 10, self.recent_high_lookback + 10)
        if len(df) < min_data:
            self._scan_stats["insufficient_data"] += 1
            return None

        # ════════════════════════════════════════════════════════════
        # NEW FILTER 1: TREND CHECK (price > EMA)
        # ════════════════════════════════════════════════════════════
        if self.require_uptrend:
            ema = df["close"].ewm(span=self.trend_ema_period, adjust=False).mean()
            last_close = float(df["close"].iloc[-1])
            last_ema = float(ema.iloc[-1])
            
            price_vs_ema_pct = ((last_close - last_ema) / last_ema) * 100
            
            if price_vs_ema_pct < self.price_above_ema_pct:
                self._scan_stats["trend_rejected"] += 1
                logger.debug(
                    f"{symbol}: Trend rejected - Price {last_close:.2f} is "
                    f"{price_vs_ema_pct:.2f}% vs EMA{self.trend_ema_period} {last_ema:.2f}"
                )
                return None

        # ════════════════════════════════════════════════════════════
        # NEW FILTER 5: ATR VALIDATION (sufficient volatility)
        # ════════════════════════════════════════════════════════════
        atr = self._calculate_atr(df, self.atr_period)
        last_close = float(df["close"].iloc[-1])
        atr_pct = (atr / last_close) * 100 if last_close > 0 else 0
        
        if atr_pct < self.min_atr_pct:
            self._scan_stats["low_volatility"] += 1
            logger.debug(
                f"{symbol}: Low volatility - ATR {atr_pct:.2f}% < {self.min_atr_pct}%"
            )
            return None

        # ════════════════════════════════════════════════════════════
        # FIND MOTHER CANDLE PATTERN (original logic)
        # ════════════════════════════════════════════════════════════
        pattern = self._discover_mother_candle(df)
        if pattern is None:
            self._scan_stats["no_pattern"] += 1
            return None

        mother_high = pattern["mother_high"]
        mother_low = pattern["mother_low"]
        mother_idx = pattern["mother_position"]
        baby_count = pattern["baby_count"]
        breakout_close = pattern["breakout_close"]

        # ════════════════════════════════════════════════════════════
        # PATTERN AGE CHECK: reject stale mother candle levels
        # mother_idx is a negative offset (e.g. -7 means 7 bars ago)
        # Total pattern age = abs(mother_idx) - 1 (exclude breakout bar)
        # ════════════════════════════════════════════════════════════
        pattern_age_bars = abs(mother_idx) - 1  # bars since mother candle
        if pattern_age_bars > self.max_pattern_age_bars:
            self._scan_stats["no_pattern"] += 1
            logger.debug(
                f"{symbol}: Pattern too old — mother candle was "
                f"{pattern_age_bars} bars ago "
                f"(max: {self.max_pattern_age_bars})"
            )
            return None

        # ════════════════════════════════════════════════════════════
        # NEW FILTER 2: MINIMUM BREAKOUT STRENGTH
        # ════════════════════════════════════════════════════════════
        breakout_strength_pct = ((breakout_close - mother_high) / mother_high) * 100
        
        if breakout_strength_pct < self.min_breakout_pct:
            self._scan_stats["weak_breakout"] += 1
            logger.debug(
                f"{symbol}: Weak breakout - {breakout_strength_pct:.2f}% < {self.min_breakout_pct}%"
            )
            return None
        
        # Also check max buffer (don't chase overextended breakouts)
        if breakout_strength_pct > self.max_entry_buffer_pct:
            self._scan_stats["weak_breakout"] += 1
            return None

        # ════════════════════════════════════════════════════════════
        # NEW FILTER 3: PRICE STRUCTURE - Mother near recent highs
        # ════════════════════════════════════════════════════════════
        if self.mother_position_filter:
            recent_high = float(df["high"].iloc[-self.recent_high_lookback:].max())
            mother_vs_high_pct = ((recent_high - mother_high) / recent_high) * 100
            
            if mother_vs_high_pct > self.mother_near_high_pct:
                self._scan_stats["bad_position"] += 1
                logger.debug(
                    f"{symbol}: Bad position - Mother High {mother_high:.2f} is "
                    f"{mother_vs_high_pct:.2f}% below recent high {recent_high:.2f}"
                )
                return None

        # ════════════════════════════════════════════════════════════
        # NEW FILTER 4: CHECK FOR RECENT FAILED BREAKOUTS
        # ════════════════════════════════════════════════════════════
        if self.check_failed_breakouts:
            # Look for candles in last N days that closed above a prior high
            # but then fell back below it
            failed_breakout = self._has_recent_failed_breakout(
                df, self.failed_breakout_lookback, mother_high
            )
            if failed_breakout:
                self._scan_stats["failed_breakout_history"] += 1
                logger.debug(
                    f"{symbol}: Recent failed breakout detected"
                )
                return None

        # ════════════════════════════════════════════════════════════
        # VOLUME CONFIRMATION (original logic)
        # ════════════════════════════════════════════════════════════
        mother_abs_idx = len(df) + mother_idx
        if mother_abs_idx < 10:
            self._scan_stats["volume_rejected"] += 1
            return None

        mother_volume = float(df.iloc[mother_idx]["volume"])
        prev_10_avg_vol = float(df["volume"].iloc[mother_abs_idx - 10: mother_abs_idx].mean())
        mother_vol_ratio = mother_volume / prev_10_avg_vol if prev_10_avg_vol > 0 else 0
        
        if mother_vol_ratio < self.mother_vol_multiplier:
            self._scan_stats["volume_rejected"] += 1
            return None

        vol_r = volume_ratio(df["volume"], 20)
        breakout_vol_ratio = float(vol_r.iloc[-1])
        
        if breakout_vol_ratio < self.breakout_vol_multiplier:
            self._scan_stats["volume_rejected"] += 1
            return None

        # ════════════════════════════════════════════════════════════
        # ENTRY, TARGET, STOP LOSS (original logic)
        # ════════════════════════════════════════════════════════════
        entry_price = round(breakout_close, 2)
        target_price = round(entry_price * (1 + self.target_pct / 100), 2)
        
        fixed_sl = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        mother_low_sl = round(mother_low, 2)
        
        if self.use_mother_low_sl and mother_low_sl > fixed_sl:
            stop_loss = mother_low_sl
            sl_method = "mother_low"
        else:
            stop_loss = fixed_sl
            sl_method = f"fixed_{self.stop_loss_pct}pct"
        
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        
        if sl_distance_pct > self.max_stop_loss_pct:
            self._scan_stats["sl_too_wide"] += 1
            return None
        
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        if risk <= 0:
            self._scan_stats["sl_too_wide"] += 1
            return None
        rr_ratio = round(reward / risk, 2)

        # ── Universal signal validation (Rules 1-3) ──────────────────────
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

        # ════════════════════════════════════════════════════════════
        # BUILD SIGNAL (enhanced with new filters)
        # ════════════════════════════════════════════════════════════
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
                "break_pct": round(breakout_strength_pct, 2),
            },
            "mother_volume": {
                "passed": True,
                "mother_vol_ratio": round(mother_vol_ratio, 2),
                "threshold": self.mother_vol_multiplier,
            },
            "breakout_volume": {
                "passed": True,
                "breakout_vol_ratio": round(breakout_vol_ratio, 2),
                "threshold": self.breakout_vol_multiplier,
            },
            "trend_filter": {
                "passed": True,
                "price": round(last_close, 2),
                "ema": round(last_ema, 2) if self.require_uptrend else "N/A",
                "price_vs_ema_pct": round(price_vs_ema_pct, 2) if self.require_uptrend else "N/A",
            },
            "volatility": {
                "passed": True,
                "atr_pct": round(atr_pct, 2),
                "min_required": self.min_atr_pct,
            },
        }

        # Enhanced confidence calculation
        confidence = 0.70
        
        # Volume bonuses
        if breakout_vol_ratio >= 2.0:
            confidence += 0.10
        if mother_vol_ratio >= 2.0:
            confidence += 0.05
        
        # Pattern quality bonuses
        if baby_count >= 4:
            confidence += 0.05
        if baby_count >= 7:
            confidence += 0.05
        
        # NEW: Trend strength bonus
        if self.require_uptrend and price_vs_ema_pct > 5:
            confidence += 0.05
        
        # NEW: Breakout strength bonus
        if breakout_strength_pct >= 1.0:
            confidence += 0.05
        
        # R:R bonus
        if rr_ratio >= 2.0:
            confidence += 0.05
        
        confidence = min(confidence, 1.0)

        # Risk-at-Risk position sizing
        shares, risk_amount = self.calculate_position_size(
            entry_price, stop_loss,
            capital=self.risk_capital,
            risk_pct=self.risk_pct,
        )

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
            indicators_met=6,  # Now 6 filters instead of 4
            total_indicators=6,
            indicator_details=indicator_details,
            metadata={
                "timeframe": self.timeframe,
                "mode": "daily",
                "baby_count": baby_count,
                "mother_high": round(mother_high, 2),
                "mother_low": round(mother_low, 2),
                "mother_vol_ratio": round(mother_vol_ratio, 2),
                "breakout_vol_ratio": round(breakout_vol_ratio, 2),
                "breakout_strength_pct": round(breakout_strength_pct, 2),
                "sl_distance_pct": round(sl_distance_pct, 2),
                "sl_method": sl_method,
                "rr_ratio": rr_ratio,
                "target_pct": self.target_pct,
                "stop_loss_pct": round(sl_distance_pct, 2),
                "atr_pct": round(atr_pct, 2),
                "trend_ema": round(last_ema, 2) if self.require_uptrend else None,
                # Chart visualizer indices (negative offsets from end of df)
                "mother_start_idx": mother_idx,  # e.g. -7 → mother candle
                "mother_end_idx": -2,             # last baby candle (day before breakout)
                # Risk-at-Risk position sizing
                "position_size_shares": shares,
                "risk_amount_inr": risk_amount,
                "capital": self.risk_capital,
            },
        )

        logger.info(
            f"SIGNAL: {self.name} - {symbol} "
            f"| Babies: {baby_count} "
            f"| Mother: {round(mother_high, 2)}-{round(mother_low, 2)} "
            f"| Entry: {entry_price} "
            f"| Breakout: +{round(breakout_strength_pct, 2)}% "
            f"| Target: {target_price} (+{self.target_pct}%) "
            f"| SL: {stop_loss} (-{round(sl_distance_pct, 1)}%) "
            f"| R:R 1:{rr_ratio} "
            f"| Qty: {shares} "
            f"| Risk: ₹{risk_amount:,.0f} "
            f"| Vol: {round(breakout_vol_ratio, 2)}x "
            f"| ATR: {round(atr_pct, 2)}% "
            f"| Conf: {confidence:.0%}"
        )

        self._scan_stats["signals"] += 1
        return signal

    def _discover_mother_candle(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Right-to-Left scan to find Mother Candle pattern.
        (Same as original - unchanged)
        """
        last_candle = df.iloc[-1]
        last_close = float(last_candle["close"])
        last_high = float(last_candle["high"])

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

            # Mother Range Validation
            mother_abs_idx = len(df) + mother_pos
            if mother_abs_idx < 5:
                continue

            pre_mother_ranges = []
            for k in range(1, 6):
                idx = mother_abs_idx - k
                if idx < 0:
                    break
                c = df.iloc[idx]
                pre_mother_ranges.append(float(c["high"]) - float(c["low"]))

            if not pre_mother_ranges:
                continue

            avg_pre_range = sum(pre_mother_ranges) / len(pre_mother_ranges)
            if avg_pre_range <= 0:
                continue

            if m_range < (self.mother_range_multiplier * avg_pre_range):
                continue

            # Baby Candle Strict Containment
            baby_start = mother_pos + 1
            baby_count = mother_offset - 2

            if baby_count < self.min_babies:
                continue

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

            # Fresh Breakout Validation
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

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if len(atr) > 0 else 0.0

    def _has_recent_failed_breakout(
        self, df: pd.DataFrame, lookback: int, threshold: float
    ) -> bool:
        """
        Check if there were recent failed breakouts above threshold.
        
        A failed breakout is when:
        - A candle closed above threshold
        - But then fell back below threshold in next 1-3 candles
        """
        if len(df) < lookback + 5:
            return False
        
        recent_candles = df.iloc[-(lookback + 1):-1]  # Exclude last candle
        
        for i in range(len(recent_candles) - 3):
            candle = recent_candles.iloc[i]
            close = float(candle["close"])
            
            # Did this candle close above threshold?
            if close > threshold:
                # Check next 1-3 candles - did they fall back below?
                for j in range(1, min(4, len(recent_candles) - i)):
                    next_candle = recent_candles.iloc[i + j]
                    next_close = float(next_candle["close"])
                    
                    if next_close < threshold:
                        # Failed breakout detected
                        return True
        
        return False
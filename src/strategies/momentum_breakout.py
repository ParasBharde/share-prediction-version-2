"""
Momentum Breakout Strategy (V2 - Strict)

Purpose:
    Identifies stocks breaking 52-week highs with volume confirmation.
    Targets stocks with strong upward momentum and institutional interest.

HARD REJECTS:
    - 52W high proximity is MANDATORY (within 2% of 52W high)
    - Volume must be >= 1.5x average (no breakout on thin volume)
    - EMA alignment (20>50>200) is MANDATORY (must be strong uptrend)
    - R:R must be >= 1.5:1

Dependencies:
    - base_strategy for interface
    - indicators for technical calculations
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import adx, rsi
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """Identifies momentum breakout opportunities."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "1D")
        self._scan_stats = {
            "total": 0,
            "pre_filter_rejected": 0,
            "insufficient_data": 0,
            "no_pattern": 0,
            "low_confidence": 0,
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

        if not self.apply_pre_filters(company_info):
            self._scan_stats["pre_filter_rejected"] += 1
            return None

        if len(df) < 200:
            self._scan_stats["insufficient_data"] += 1
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # ============================================================
        # MANDATORY 1: 52-Week High Proximity (within 2%)
        # ============================================================
        try:
            high_52w = float(df["high"].tail(252).max())
            proximity = entry_price / high_52w if high_52w > 0 else 0
            near_high = proximity >= 0.98

            indicator_details["52w_high"] = {
                "high_52w": round(high_52w, 2),
                "proximity": round(proximity, 4),
                "passed": near_high,
            }

            if not near_high:
                self._scan_stats["no_pattern"] += 1
                return None

            indicators_met += 1
            weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: 52W high calc error: {e}")
            return None

        # ============================================================
        # MANDATORY 2: Volume >= 1.5x (hard reject below average)
        # ============================================================
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol_ratio = float(vol_r.iloc[-1])

            # Hard reject: volume below average = no breakout
            if current_vol_ratio < 1.0:
                self._scan_stats["no_pattern"] += 1
                return None

            vol_surge = current_vol_ratio >= 1.5

            indicator_details["volume_surge"] = {
                "volume_ratio": round(current_vol_ratio, 2),
                "threshold": 1.5,
                "passed": vol_surge,
            }

            if vol_surge:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: Volume calc error: {e}")
            return None

        # ============================================================
        # MANDATORY 3: EMA Alignment (20 > 50 > 200 = strong uptrend)
        # ============================================================
        try:
            ema_20 = float(ema(df["close"], 20).iloc[-1])
            ema_50 = float(ema(df["close"], 50).iloc[-1])
            ema_200 = float(ema(df["close"], 200).iloc[-1])
            aligned = ema_20 > ema_50 > ema_200

            indicator_details["ema_alignment"] = {
                "ema_20": round(ema_20, 2),
                "ema_50": round(ema_50, 2),
                "ema_200": round(ema_200, 2),
                "passed": aligned,
            }

            if not aligned:
                self._scan_stats["no_pattern"] += 1
                return None

            indicators_met += 1
            weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: EMA calc error: {e}")
            return None

        # ============================================================
        # OPTIONAL 4: RSI Momentum Zone (55-75)
        # Tighter range: must show momentum but not overbought
        # ============================================================
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])
            rsi_ok = 55 <= current_rsi <= 75

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "range": "55-75",
                "passed": rsi_ok,
            }

            # Hard reject if overbought (RSI > 80)
            if current_rsi > 80:
                return None

            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI calc error: {e}")

        # ============================================================
        # OPTIONAL 5: ADX > 25 (trending market, not range-bound)
        # ============================================================
        try:
            adx_data = adx(df["high"], df["low"], df["close"], 14)
            current_adx = float(adx_data["adx"].iloc[-1])
            adx_ok = current_adx >= 25

            indicator_details["adx_trend"] = {
                "value": round(current_adx, 2),
                "threshold": 25,
                "passed": adx_ok,
            }

            if adx_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: ADX calc error: {e}")

        # ============================================================
        # Signal Generation
        # 3 mandatory already met + need at least 1 optional = 4 min
        # ============================================================
        min_conditions = self.signal_config.get(
            "min_conditions_met", 4
        )
        confidence_threshold = self.signal_config.get(
            "confidence_threshold", 0.70
        )

        if (
            indicators_met >= min_conditions
            and weighted_score >= confidence_threshold
        ):
            stop_loss = self.calculate_stop_loss(entry_price, df)
            target = self.calculate_target(
                entry_price, stop_loss, df
            )

            # R:R floor check - must be >= 1.5
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            if risk <= 0 or reward / risk < 1.5:
                logger.debug(
                    f"{symbol}: R:R too low "
                    f"({reward/risk:.1f} < 1.5)"
                    if risk > 0 else f"{symbol}: Zero risk"
                )
                self._scan_stats["low_confidence"] += 1
                return None

            signal = TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=SignalType.BUY,
                confidence=round(weighted_score, 4),
                entry_price=entry_price,
                target_price=round(target, 2),
                stop_loss=round(stop_loss, 2),
                priority=AlertPriority.HIGH,
                indicators_met=indicators_met,
                total_indicators=5,
                indicator_details=indicator_details,
                metadata={
                    "timeframe": self.timeframe,
                    "mode": "daily",
                    "volume_ratio": current_vol_ratio,
                    "rsi": indicator_details.get("rsi", {}).get(
                        "value", 0
                    ),
                    "high_52w": round(high_52w, 2),
                },
            )

            logger.info(
                f"SIGNAL: {self.name} - {symbol} "
                f"(confidence: {weighted_score:.2f}, "
                f"indicators: {indicators_met}/5)",
                extra=signal.to_dict(),
            )

            self._scan_stats["signals"] += 1
            return signal

        if indicators_met > 0:
            self._scan_stats["low_confidence"] += 1
        else:
            self._scan_stats["no_pattern"] += 1
        return None

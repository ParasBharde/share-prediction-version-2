"""
Volume Surge Strategy (V2 - Strict)

Purpose:
    Identifies unusual volume activity signaling institutional interest.
    Targets stocks with 3x+ volume and price confirmation.

HARD REJECTS:
    - Volume 3x average is MANDATORY (no signal without massive volume)
    - Price must be bullish (close > open) MANDATORY
    - Trend must be up (EMA 20 > EMA 50) MANDATORY
    - R:R must be >= 1.5:1
    - RSI > 80 = overbought, reject

Dependencies:
    - base_strategy for interface
    - indicators for technical calculations
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi
from src.strategies.indicators.volume_indicators import (
    volume_ratio,
    vwap,
)
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class VolumeSurgeStrategy(BaseStrategy):
    """Identifies institutional volume surge opportunities."""

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

        if len(df) < 50:
            self._scan_stats["insufficient_data"] += 1
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # ============================================================
        # MANDATORY 1: Volume Spike >= 3x average
        # ============================================================
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol_ratio = float(vol_r.iloc[-1])
            vol_spike = current_vol_ratio >= 3.0

            indicator_details["volume_spike"] = {
                "volume_ratio": round(current_vol_ratio, 2),
                "threshold": 3.0,
                "passed": vol_spike,
            }

            if not vol_spike:
                self._scan_stats["no_pattern"] += 1
                return None

            indicators_met += 1
            weighted_score += 0.30
        except Exception as e:
            logger.debug(f"{symbol}: Volume spike error: {e}")
            return None

        # ============================================================
        # MANDATORY 2: Price Confirmation (bullish candle, >= 1.5% gain)
        # ============================================================
        try:
            open_price = float(latest["open"])
            price_gain_pct = (
                (entry_price - open_price) / open_price * 100
                if open_price > 0
                else 0
            )
            price_confirmed = (
                entry_price > open_price and price_gain_pct >= 1.5
            )

            indicator_details["price_confirmation"] = {
                "open": round(open_price, 2),
                "close": entry_price,
                "gain_pct": round(price_gain_pct, 2),
                "passed": price_confirmed,
            }

            if not price_confirmed:
                self._scan_stats["no_pattern"] += 1
                return None

            indicators_met += 1
            weighted_score += 0.25
        except Exception as e:
            logger.debug(
                f"{symbol}: Price confirmation error: {e}"
            )
            return None

        # ============================================================
        # MANDATORY 3: Trend Up (EMA 20 > EMA 50)
        # ============================================================
        try:
            ema_20 = float(ema(df["close"], 20).iloc[-1])
            ema_50 = float(ema(df["close"], 50).iloc[-1])
            trend_up = ema_20 > ema_50

            indicator_details["trend"] = {
                "ema_20": round(ema_20, 2),
                "ema_50": round(ema_50, 2),
                "passed": trend_up,
            }

            if not trend_up:
                self._scan_stats["no_pattern"] += 1
                return None

            indicators_met += 1
            weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: Trend error: {e}")
            return None

        # ============================================================
        # OPTIONAL 4: RSI Momentum (50-75, reject if > 80)
        # ============================================================
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])
            rsi_ok = 50 <= current_rsi <= 75

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "range": "50-75",
                "passed": rsi_ok,
            }

            # Hard reject overbought
            if current_rsi > 80:
                return None

            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI error: {e}")

        # ============================================================
        # OPTIONAL 5: Close near day's high (strong close)
        # Within top 25% of day's range = buyers in control
        # ============================================================
        try:
            day_high = float(latest["high"])
            day_low = float(latest["low"])
            day_range = day_high - day_low
            if day_range > 0:
                close_position = (entry_price - day_low) / day_range
                strong_close = close_position >= 0.75
            else:
                strong_close = False

            indicator_details["close_strength"] = {
                "close_in_range": round(close_position, 2) if day_range > 0 else 0,
                "threshold": 0.75,
                "passed": strong_close,
            }

            if strong_close:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: Close strength error: {e}")

        # ============================================================
        # Signal Generation
        # 3 mandatory + need 1 optional = min 4
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

            # R:R floor check
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            if risk <= 0 or reward / risk < 1.5:
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
                    "volume_ratio": round(current_vol_ratio, 2),
                },
            )

            logger.info(
                f"SIGNAL: {self.name} - {symbol} "
                f"(confidence: {weighted_score:.2f})",
                extra=signal.to_dict(),
            )

            self._scan_stats["signals"] += 1
            return signal

        if indicators_met > 0:
            self._scan_stats["low_confidence"] += 1
        else:
            self._scan_stats["no_pattern"] += 1
        return None

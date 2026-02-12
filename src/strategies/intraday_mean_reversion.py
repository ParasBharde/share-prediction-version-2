"""
Intraday Mean Reversion Strategy

Adapted from daily Mean Reversion for 5m/15m/30m timeframes.
Uses session VWAP as the 'mean' instead of 200 EMA.
Uses Stochastic oversold instead of just RSI.
Targets reversion back to session VWAP level.

HARD REJECTS:
    - RSI must be < 40 (stock must be weak, not neutral/strong)
    - VWAP distance must be < 5% (prevents unrealistic targets)
    - Stochastic cross_up only counts if %K < 30 (must be near oversold)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema, sma
from src.strategies.indicators.oscillators import macd, rsi, stochastic
from src.strategies.indicators.volume_indicators import volume_ratio, session_vwap
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class IntradayMeanReversionStrategy(BaseStrategy):
    """Intraday oversold bounce back to VWAP."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "15m")

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        if not self.apply_pre_filters(company_info):
            return None

        if len(df) < 50:
            return None

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # HARD REJECT: RSI must be < 40 (stock must be weak)
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])
            if current_rsi > 40:
                return None
        except Exception:
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}

        # 1. RSI Oversold (RSI <= 30)
        is_oversold = current_rsi <= 30
        indicator_details["rsi_oversold"] = {
            "value": round(current_rsi, 2),
            "threshold": 30,
            "passed": is_oversold,
        }
        if is_oversold:
            indicators_met += 1
            weighted_score += 0.25

        # 2. Stochastic Oversold (%K < 20, or %K cross up %D while %K < 30)
        try:
            stoch = stochastic(df["high"], df["low"], df["close"], 14, 3)
            stoch_k = float(stoch["stoch_k"].iloc[-1])
            stoch_d = float(stoch["stoch_d"].iloc[-1])
            prev_k = float(stoch["stoch_k"].iloc[-2])
            prev_d = float(stoch["stoch_d"].iloc[-2])
            stoch_oversold = stoch_k < 20
            # Cross up only counts if %K is near oversold zone (<30)
            stoch_cross_up = prev_k <= prev_d and stoch_k > stoch_d and stoch_k < 30

            stoch_ok = stoch_oversold or stoch_cross_up

            indicator_details["stochastic"] = {
                "stoch_k": round(stoch_k, 2),
                "stoch_d": round(stoch_d, 2),
                "oversold": stoch_oversold,
                "cross_up": stoch_cross_up,
                "passed": stoch_ok,
            }
            if stoch_ok:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: Stochastic error: {e}")

        # 3. Bollinger Band Lower Touch
        try:
            bb_sma = sma(df["close"], 20)
            bb_std = df["close"].rolling(window=20).std()
            bb_lower = bb_sma - (2 * bb_std)
            at_lower_band = entry_price <= float(bb_lower.iloc[-1])

            indicator_details["bollinger_lower"] = {
                "bb_lower": round(float(bb_lower.iloc[-1]), 2),
                "price": entry_price,
                "passed": at_lower_band,
            }
            if at_lower_band:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: BB error: {e}")

        # 4. Below Session VWAP (price below VWAP = room to revert UP)
        try:
            vwap_values = session_vwap(df["high"], df["low"], df["close"], df["volume"])
            current_vwap = float(vwap_values.iloc[-1])
            below_vwap = entry_price < current_vwap
            distance_to_vwap = ((current_vwap - entry_price) / entry_price) * 100

            # HARD REJECT: If VWAP is > 5% away, target is unrealistic
            if distance_to_vwap > 5.0:
                logger.debug(
                    f"{symbol}: VWAP too far ({distance_to_vwap:.1f}%), "
                    f"unrealistic target"
                )
                return None

            indicator_details["vwap_support"] = {
                "vwap": round(current_vwap, 2),
                "price": entry_price,
                "distance_pct": round(distance_to_vwap, 2),
                "passed": below_vwap,
            }
            if below_vwap:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")
            return None

        # 5. MACD Histogram Improving (momentum turning up)
        try:
            macd_data = macd(df["close"], 12, 26, 9)
            hist = macd_data["histogram"]
            hist_improving = float(hist.iloc[-1]) > float(hist.iloc[-2])

            indicator_details["macd_improving"] = {
                "histogram": round(float(hist.iloc[-1]), 4),
                "prev_histogram": round(float(hist.iloc[-2]), 4),
                "passed": hist_improving,
            }
            if hist_improving:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: MACD error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get("confidence_threshold", 0.55)

        if indicators_met >= min_conditions and weighted_score >= confidence_threshold:
            stop_loss = self.calculate_stop_loss(entry_price, df)

            # Target is session VWAP level (mean reversion)
            target_vwap = indicator_details.get("vwap_support", {}).get("vwap", 0)
            if target_vwap > entry_price:
                target = target_vwap
            else:
                target = self.calculate_target(entry_price, stop_loss, df)

            # Final check: R:R must be > 1.0
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            if risk <= 0 or reward / risk < 1.0:
                return None

            return TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=SignalType.BUY,
                confidence=round(weighted_score, 4),
                entry_price=entry_price,
                target_price=round(target, 2),
                stop_loss=round(stop_loss, 2),
                priority=AlertPriority.MEDIUM,
                indicators_met=indicators_met,
                total_indicators=5,
                indicator_details=indicator_details,
                metadata={
                    "timeframe": self.timeframe,
                    "mode": "intraday",
                    "target_type": "vwap_reversion",
                },
            )

        return None

"""
Intraday Momentum Breakout Strategy

Adapted from daily Momentum Breakout for 5m/15m/30m timeframes.
Instead of 52-week high, uses TODAY's session high breakout.
Instead of EMA(200), uses session VWAP as structural level.

HARD REJECTS:
    - Session high proximity is MANDATORY (must be within 0.5% of today's high)
    - Minimum volume ratio of 1.0 (not below average)
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi
from src.strategies.indicators.volume_indicators import volume_ratio, session_vwap
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class IntradayMomentumStrategy(BaseStrategy):
    """Session high breakout with volume surge for intraday."""

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
            logger.debug(f"{symbol}: Insufficient intraday data ({len(df)} < 50)")
            return None

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # Get TODAY's data only for session high
        if hasattr(df.index, 'date'):
            today = df.index[-1].date()
            today_df = df[df.index.date == today]
        else:
            today_df = df.tail(26)  # ~1 day on 15m

        if len(today_df) < 3:
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}

        # 1. TODAY'S Session High Proximity (MANDATORY - within 0.5%)
        try:
            session_high = float(today_df["high"].max())
            proximity = entry_price / session_high if session_high > 0 else 0
            near_high = proximity >= 0.995  # Within 0.5% of today's high

            indicator_details["session_high"] = {
                "session_high": round(session_high, 2),
                "proximity": round(proximity, 4),
                "passed": near_high,
            }

            # MANDATORY: Must be near session high
            if not near_high:
                return None

            indicators_met += 1
            weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: Session high error: {e}")
            return None

        # 2. Volume Surge (MANDATORY - must be >= 1.5x average)
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol_ratio = float(vol_r.iloc[-1])

            # HARD REJECT: Volume must be above average with surge
            if current_vol_ratio < 1.2:
                return None

            vol_surge = current_vol_ratio >= 1.5

            indicator_details["volume"] = {
                "volume_ratio": round(current_vol_ratio, 2),
                "threshold": 1.5,
                "passed": vol_surge,
            }
            if vol_surge:
                indicators_met += 1
                weighted_score += 0.30
        except Exception as e:
            logger.debug(f"{symbol}: Volume error: {e}")
            return None

        # 3. RSI Momentum Zone (50-75)
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])
            rsi_ok = 50 <= current_rsi <= 75

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "range": "50-75",
                "passed": rsi_ok,
            }
            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI error: {e}")

        # 4. Above Session VWAP
        try:
            vwap_values = session_vwap(df["high"], df["low"], df["close"], df["volume"])
            current_vwap = float(vwap_values.iloc[-1])
            above_vwap = entry_price > current_vwap

            indicator_details["vwap"] = {
                "vwap": round(current_vwap, 2),
                "price": entry_price,
                "passed": above_vwap,
            }
            if above_vwap:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")

        # 5. EMA Alignment (9 > 21 for intraday)
        try:
            ema_9 = float(ema(df["close"], 9).iloc[-1])
            ema_21 = float(ema(df["close"], 21).iloc[-1])
            aligned = ema_9 > ema_21

            indicator_details["ema_alignment"] = {
                "ema_9": round(ema_9, 2),
                "ema_21": round(ema_21, 2),
                "passed": aligned,
            }
            if aligned:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: EMA error: {e}")

        # Need at least 3 indicators (session high is already 1, so need 2 more)
        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get("confidence_threshold", 0.55)

        if indicators_met >= min_conditions and weighted_score >= confidence_threshold:
            stop_loss = self.calculate_stop_loss(entry_price, df)
            target = self.calculate_target(entry_price, stop_loss, df)

            return TradingSignal(
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
                    "mode": "intraday",
                    "session_high": indicator_details.get("session_high", {}).get(
                        "session_high", 0
                    ),
                },
            )

        return None

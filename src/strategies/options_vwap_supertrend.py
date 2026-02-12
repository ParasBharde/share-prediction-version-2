"""
Options VWAP + Supertrend Strategy

Directional intraday strategy for index options (NIFTY/BANKNIFTY).
Uses VWAP for bias direction and Supertrend for entry timing.

Logic:
    - Price > VWAP AND Supertrend turns green → BUY CE
    - Price < VWAP AND Supertrend turns red → BUY PE
    - Exit: Supertrend reversal or 3:15 PM
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi, supertrend
from src.strategies.indicators.volume_indicators import vwap, volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class OptionsVWAPSupertrendStrategy(BaseStrategy):
    """VWAP + Supertrend directional strategy for options."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "5m")
        self.st_period = 10
        self.st_multiplier = 3.0

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        if len(df) < 30:
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}

        entry_price = float(df.iloc[-1]["close"])
        signal_type = None

        # 1. VWAP Direction (the bias)
        try:
            vwap_values = vwap(df["high"], df["low"], df["close"], df["volume"])
            current_vwap = float(vwap_values.iloc[-1])
            above_vwap = entry_price > current_vwap
            below_vwap = entry_price < current_vwap

            indicator_details["vwap"] = {
                "vwap": round(current_vwap, 2),
                "price": entry_price,
                "direction": "bullish" if above_vwap else "bearish",
                "passed": above_vwap or below_vwap,
            }

            if above_vwap or below_vwap:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")
            return None

        # 2. SUPERTREND Signal (the trigger)
        try:
            st_data = supertrend(
                df["high"], df["low"], df["close"],
                self.st_period, self.st_multiplier,
            )
            current_dir = int(st_data["direction"].iloc[-1])
            prev_dir = int(st_data["direction"].iloc[-2])

            # Fresh signal = direction just changed
            st_buy = current_dir == 1 and prev_dir == -1   # Just turned green
            st_sell = current_dir == -1 and prev_dir == 1   # Just turned red
            st_bullish = current_dir == 1                   # Currently green
            st_bearish = current_dir == -1                  # Currently red

            indicator_details["supertrend"] = {
                "direction": "bullish" if current_dir == 1 else "bearish",
                "fresh_signal": st_buy or st_sell,
                "value": round(float(st_data["supertrend"].iloc[-1]), 2)
                if pd.notna(st_data["supertrend"].iloc[-1]) else 0,
                "passed": st_buy or st_sell or st_bullish or st_bearish,
            }

            # Combine VWAP + Supertrend
            if above_vwap and (st_buy or st_bullish):
                signal_type = "BUY_CE"
                indicators_met += 1
                weighted_score += 0.30
                # Bonus for fresh signal
                if st_buy:
                    weighted_score += 0.05
            elif below_vwap and (st_sell or st_bearish):
                signal_type = "BUY_PE"
                indicators_met += 1
                weighted_score += 0.30
                if st_sell:
                    weighted_score += 0.05
            else:
                # VWAP and Supertrend disagree
                return None

        except Exception as e:
            logger.debug(f"{symbol}: Supertrend error: {e}")
            return None

        # 3. Volume Confirmation
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol = float(vol_r.iloc[-1])
            vol_ok = current_vol >= 1.3

            indicator_details["volume"] = {
                "volume_ratio": round(current_vol, 2),
                "passed": vol_ok,
            }
            if vol_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: Volume error: {e}")

        # 4. RSI Momentum Confirmation
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])

            if signal_type == "BUY_CE":
                rsi_ok = current_rsi > 50 and current_rsi < 80
            else:
                rsi_ok = current_rsi < 50 and current_rsi > 20

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "passed": rsi_ok,
            }
            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI error: {e}")

        # 5. EMA 9 direction
        try:
            ema_9 = ema(df["close"], 9)
            ema_rising = float(ema_9.iloc[-1]) > float(ema_9.iloc[-3])
            ema_falling = float(ema_9.iloc[-1]) < float(ema_9.iloc[-3])

            if signal_type == "BUY_CE":
                ema_ok = ema_rising
            else:
                ema_ok = ema_falling

            indicator_details["ema_9"] = {
                "value": round(float(ema_9.iloc[-1]), 2),
                "rising": ema_rising,
                "passed": ema_ok,
            }
            if ema_ok:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: EMA error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get("confidence_threshold", 0.55)

        if indicators_met >= min_conditions and weighted_score >= confidence_threshold:
            # SL at Supertrend level
            st_value = float(st_data["supertrend"].iloc[-1]) if pd.notna(st_data["supertrend"].iloc[-1]) else 0

            if signal_type == "BUY_CE":
                stop_loss = max(st_value, entry_price * 0.995) if st_value > 0 else entry_price * 0.995
                risk = entry_price - stop_loss
                target = round(entry_price + (risk * 2.0), 2)
                trade_signal = SignalType.BUY
            else:
                stop_loss = min(st_value, entry_price * 1.005) if st_value > 0 else entry_price * 1.005
                risk = stop_loss - entry_price
                target = round(entry_price - (risk * 2.0), 2)
                trade_signal = SignalType.SELL

            option_chain = company_info.get("option_chain", {})
            atm_strike = round(entry_price / 50) * 50  # Round to nearest 50

            return TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=trade_signal,
                confidence=round(weighted_score, 4),
                entry_price=entry_price,
                target_price=target,
                stop_loss=round(stop_loss, 2),
                priority=AlertPriority.HIGH,
                indicators_met=indicators_met,
                total_indicators=5,
                indicator_details=indicator_details,
                metadata={
                    "timeframe": self.timeframe,
                    "mode": "options",
                    "option_type": signal_type,
                    "atm_strike": atm_strike,
                    "supertrend_value": round(st_value, 2) if st_value else 0,
                    "vwap": indicator_details.get("vwap", {}).get("vwap", 0),
                    "exit_rule": "Supertrend reversal or 15:15 IST",
                },
            )

        return None

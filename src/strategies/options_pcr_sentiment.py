"""
Options PCR (Put-Call Ratio) Sentiment Strategy

Uses overall market PCR to determine sentiment direction.
Combines with price action for entry timing.

Logic:
    - PCR > 1.2 → Bullish sentiment → BUY CE
    - PCR < 0.8 → Bearish sentiment → BUY PE
    - PCR 0.8-1.2 → Neutral/sideways → No trade
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi
from src.strategies.indicators.volume_indicators import volume_ratio, vwap
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class OptionsPCRStrategy(BaseStrategy):
    """PCR-based sentiment strategy for options."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "5m")

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        option_chain = company_info.get("option_chain", {})
        if not option_chain:
            return None

        if len(df) < 20:
            return None

        pcr = option_chain.get("pcr", 1.0)
        underlying = option_chain.get("underlying_price", 0)

        if not underlying:
            return None

        entry_price = float(df.iloc[-1]["close"])
        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}
        signal_type = None

        # 1. PCR SENTIMENT (core signal)
        try:
            if pcr > 1.2:
                signal_type = "BUY_CE"
                pcr_ok = True
                sentiment = "bullish"
            elif pcr < 0.8:
                signal_type = "BUY_PE"
                pcr_ok = True
                sentiment = "bearish"
            else:
                indicator_details["pcr"] = {
                    "value": round(pcr, 3),
                    "sentiment": "neutral",
                    "passed": False,
                }
                return None  # No trade in neutral zone

            indicator_details["pcr"] = {
                "value": round(pcr, 3),
                "sentiment": sentiment,
                "passed": pcr_ok,
            }
            if pcr_ok:
                indicators_met += 1
                weighted_score += 0.30
        except Exception as e:
            logger.debug(f"{symbol}: PCR error: {e}")
            return None

        # 2. OI Support/Resistance Alignment
        try:
            support = option_chain.get("support", 0)
            resistance = option_chain.get("resistance", 0)

            if signal_type == "BUY_CE":
                oi_ok = entry_price > support  # Price above support
            else:
                oi_ok = entry_price < resistance  # Price below resistance

            indicator_details["oi_levels"] = {
                "support": support,
                "resistance": resistance,
                "price": entry_price,
                "passed": oi_ok,
            }
            if oi_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: OI levels error: {e}")

        # 3. VWAP Alignment
        try:
            vwap_values = vwap(df["high"], df["low"], df["close"], df["volume"])
            current_vwap = float(vwap_values.iloc[-1])

            if signal_type == "BUY_CE":
                vwap_ok = entry_price > current_vwap
            else:
                vwap_ok = entry_price < current_vwap

            indicator_details["vwap"] = {
                "vwap": round(current_vwap, 2),
                "price": entry_price,
                "passed": vwap_ok,
            }
            if vwap_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")

        # 4. EMA Trend
        try:
            ema_9 = float(ema(df["close"], 9).iloc[-1])
            ema_21 = float(ema(df["close"], 21).iloc[-1])

            if signal_type == "BUY_CE":
                trend_ok = ema_9 > ema_21
            else:
                trend_ok = ema_9 < ema_21

            indicator_details["trend"] = {
                "ema_9": round(ema_9, 2),
                "ema_21": round(ema_21, 2),
                "passed": trend_ok,
            }
            if trend_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: EMA error: {e}")

        # 5. RSI Confirmation
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])

            if signal_type == "BUY_CE":
                rsi_ok = 45 <= current_rsi <= 75
            else:
                rsi_ok = 25 <= current_rsi <= 55

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "passed": rsi_ok,
            }
            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get("confidence_threshold", 0.55)

        if indicators_met >= min_conditions and weighted_score >= confidence_threshold:
            atr = self._calculate_atr(df, 14)

            if signal_type == "BUY_CE":
                stop_loss = round(entry_price - (atr * 1.5), 2)
                risk = entry_price - stop_loss
                target = round(entry_price + (risk * 2.0), 2)
                trade_signal = SignalType.BUY
            else:
                stop_loss = round(entry_price + (atr * 1.5), 2)
                risk = stop_loss - entry_price
                target = round(entry_price - (risk * 2.0), 2)
                trade_signal = SignalType.SELL

            atm_strike = round(entry_price / 50) * 50

            return TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=trade_signal,
                confidence=round(weighted_score, 4),
                entry_price=entry_price,
                target_price=target,
                stop_loss=stop_loss,
                priority=AlertPriority.HIGH,
                indicators_met=indicators_met,
                total_indicators=5,
                indicator_details=indicator_details,
                metadata={
                    "timeframe": self.timeframe,
                    "mode": "options",
                    "option_type": signal_type,
                    "atm_strike": atm_strike,
                    "pcr": pcr,
                    "sentiment": indicator_details.get("pcr", {}).get("sentiment", "unknown"),
                },
            )

        return None

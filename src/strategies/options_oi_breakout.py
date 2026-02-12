"""
Options OI (Open Interest) Breakout Strategy

Detects when price breaks above max CALL OI strike (resistance) or
below max PUT OI strike (support). Smart money positioning analysis.

Logic:
    - Max CALL OI strike = Resistance (writers don't expect price above)
    - Max PUT OI strike = Support (writers don't expect price below)
    - When price breaks resistance → BUY CE (call option)
    - When price breaks support → BUY PE (put option)
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


class OptionsOIBreakoutStrategy(BaseStrategy):
    """Options strategy based on OI support/resistance breakout."""

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
        """
        Scan for OI breakout signals.

        company_info must contain option_chain data:
            - resistance: Max CALL OI strike
            - support: Max PUT OI strike
            - pcr: Put-Call Ratio
            - max_ce_oi: Max CALL OI value
            - max_pe_oi: Max PUT OI value
        """
        option_chain = company_info.get("option_chain", {})
        if not option_chain:
            return None

        if len(df) < 20:
            return None

        resistance = option_chain.get("resistance", 0)
        support = option_chain.get("support", 0)
        pcr = option_chain.get("pcr", 1.0)
        underlying = option_chain.get("underlying_price", 0)

        if not resistance or not support or not underlying:
            return None

        entry_price = float(df.iloc[-1]["close"])
        prev_close = float(df.iloc[-2]["close"])

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}
        signal_type = None

        # 1. OI BREAKOUT CHECK (the core pattern)
        broke_resistance = entry_price > resistance and prev_close <= resistance
        broke_support = entry_price < support and prev_close >= support

        if broke_resistance:
            signal_type = "BUY_CE"
            indicators_met += 1
            weighted_score += 0.35
            indicator_details["oi_breakout"] = {
                "type": "resistance_break",
                "resistance": resistance,
                "price": entry_price,
                "passed": True,
            }
        elif broke_support:
            signal_type = "BUY_PE"
            indicators_met += 1
            weighted_score += 0.35
            indicator_details["oi_breakout"] = {
                "type": "support_break",
                "support": support,
                "price": entry_price,
                "passed": True,
            }
        else:
            # No breakout, check if price is near levels
            dist_to_resistance = abs(entry_price - resistance) / entry_price * 100
            dist_to_support = abs(entry_price - support) / entry_price * 100

            indicator_details["oi_levels"] = {
                "resistance": resistance,
                "support": support,
                "price": entry_price,
                "dist_to_resistance_pct": round(dist_to_resistance, 2),
                "dist_to_support_pct": round(dist_to_support, 2),
                "passed": False,
            }
            return None

        # 2. PCR CONFIRMATION
        try:
            if signal_type == "BUY_CE":
                pcr_ok = pcr > 1.0  # High PUT OI = bullish
            else:
                pcr_ok = pcr < 0.8  # High CALL OI = bearish

            indicator_details["pcr"] = {
                "value": round(pcr, 3),
                "interpretation": "bullish" if pcr > 1.0 else "bearish" if pcr < 0.8 else "neutral",
                "passed": pcr_ok,
            }
            if pcr_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: PCR error: {e}")

        # 3. Volume confirmation
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol = float(vol_r.iloc[-1])
            vol_ok = current_vol >= 1.5

            indicator_details["volume"] = {
                "volume_ratio": round(current_vol, 2),
                "threshold": 1.5,
                "passed": vol_ok,
            }
            if vol_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: Volume error: {e}")

        # 4. EMA trend alignment
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
                "direction": "bullish" if ema_9 > ema_21 else "bearish",
                "passed": trend_ok,
            }
            if trend_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: EMA error: {e}")

        # 5. RSI not extreme
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
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: RSI error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get("confidence_threshold", 0.55)

        if indicators_met >= min_conditions and weighted_score >= confidence_threshold:
            # Calculate SL and target for the underlying
            if signal_type == "BUY_CE":
                stop_loss = round(resistance * 0.995, 2)  # 0.5% below resistance
                risk = entry_price - stop_loss
                target = round(entry_price + (risk * 2.0), 2)
                trade_signal_type = SignalType.BUY
            else:
                stop_loss = round(support * 1.005, 2)  # 0.5% above support
                risk = stop_loss - entry_price
                target = round(entry_price - (risk * 2.0), 2)
                trade_signal_type = SignalType.SELL

            # Find the strike to trade
            atm_strike = self._find_atm_strike(option_chain, entry_price)

            return TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=trade_signal_type,
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
                    "resistance": resistance,
                    "support": support,
                    "pcr": pcr,
                    "max_ce_oi": option_chain.get("max_ce_oi", 0),
                    "max_pe_oi": option_chain.get("max_pe_oi", 0),
                },
            )

        return None

    def _find_atm_strike(self, option_chain: Dict, price: float) -> float:
        """Find the at-the-money strike closest to current price."""
        strikes = option_chain.get("strikes", [])
        if not strikes:
            return round(price / 50) * 50  # Round to nearest 50

        closest = min(strikes, key=lambda s: abs(s["strike"] - price))
        return closest["strike"]

"""
Intraday Volume Surge Strategy

Adapted from daily Volume Surge for 5m/15m/30m timeframes.
Drops delivery % check (not available intraday).
Uses session VWAP reset, adds MFI for money flow confirmation.
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi
from src.strategies.indicators.volume_indicators import (
    mfi,
    volume_ratio,
    session_vwap,
)
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class IntradayVolumeSurgeStrategy(BaseStrategy):
    """Detects intraday volume spikes with price confirmation."""

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

        if len(df) < 30:
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details = {}

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # 1. Volume Spike (3x average on intraday)
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol_ratio = float(vol_r.iloc[-1])
            vol_spike = current_vol_ratio >= 3.0

            indicator_details["volume_spike"] = {
                "volume_ratio": round(current_vol_ratio, 2),
                "threshold": 3.0,
                "passed": vol_spike,
            }
            if vol_spike:
                indicators_met += 1
                weighted_score += 0.35
        except Exception as e:
            logger.debug(f"{symbol}: Volume spike error: {e}")

        # 2. Price Confirmation (bullish candle, >= 0.5% gain on intraday)
        try:
            open_price = float(latest["open"])
            price_gain_pct = (
                (entry_price - open_price) / open_price * 100
                if open_price > 0
                else 0
            )
            price_confirmed = entry_price > open_price and price_gain_pct >= 0.5

            indicator_details["price_confirmation"] = {
                "open": round(open_price, 2),
                "close": entry_price,
                "gain_pct": round(price_gain_pct, 2),
                "passed": price_confirmed,
            }
            if price_confirmed:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: Price confirmation error: {e}")

        # 3. MFI > 60 (Money Flow Index replaces Delivery % for intraday)
        try:
            mfi_values = mfi(df["high"], df["low"], df["close"], df["volume"], 14)
            current_mfi = float(mfi_values.iloc[-1])
            mfi_ok = current_mfi >= 60

            indicator_details["mfi"] = {
                "value": round(current_mfi, 2),
                "threshold": 60,
                "passed": mfi_ok,
            }
            if mfi_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: MFI error: {e}")

        # 4. Above VWAP
        try:
            vwap_values = session_vwap(df["high"], df["low"], df["close"], df["volume"])
            current_vwap = float(vwap_values.iloc[-1])
            above_vwap = entry_price >= current_vwap

            indicator_details["vwap"] = {
                "vwap": round(current_vwap, 2),
                "price": entry_price,
                "passed": above_vwap,
            }
            if above_vwap:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")

        # 5. Trend Direction (EMA 9 > EMA 21)
        try:
            ema_9 = float(ema(df["close"], 9).iloc[-1])
            ema_21 = float(ema(df["close"], 21).iloc[-1])
            trend_up = ema_9 > ema_21

            indicator_details["trend"] = {
                "ema_9": round(ema_9, 2),
                "ema_21": round(ema_21, 2),
                "passed": trend_up,
            }
            if trend_up:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: Trend error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get("confidence_threshold", 0.60)

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
                    "volume_ratio": indicator_details.get("volume_spike", {}).get(
                        "volume_ratio", 0
                    ),
                },
            )

        return None

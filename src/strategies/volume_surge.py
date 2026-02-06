"""
Volume Surge Strategy

Purpose:
    Identifies unusual volume activity signaling institutional interest.
    Targets stocks with 3x+ volume and price confirmation.

Dependencies:
    - base_strategy for interface
    - indicators for technical calculations

Logging:
    - Each stock scan at DEBUG
    - Signal generation at INFO

Fallbacks:
    If individual indicator fails, it is skipped.
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.volume_indicators import (
    volume_ratio,
    vwap,
)
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class VolumeSurgeStrategy(BaseStrategy):
    """Identifies institutional volume surge opportunities."""

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        """
        Scan for volume surge signals.

        Args:
            symbol: Stock symbol.
            df: OHLCV DataFrame.
            company_info: Company metadata.

        Returns:
            TradingSignal if volume surge detected.
        """
        if not self.apply_pre_filters(company_info):
            return None

        if len(df) < 50:
            return None

        indicators_met = 0
        total_indicators = len(self.indicators_config)
        weighted_score = 0.0
        indicator_details = {}

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # 1. Volume Spike (3x average)
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

        # 2. Price Confirmation (bullish candle, >=2% gain)
        try:
            open_price = float(latest["open"])
            price_gain_pct = (
                (entry_price - open_price) / open_price * 100
                if open_price > 0
                else 0
            )
            price_confirmed = (
                entry_price > open_price and price_gain_pct >= 2
            )

            indicator_details["price_confirmation"] = {
                "open": open_price,
                "close": entry_price,
                "gain_pct": round(price_gain_pct, 2),
                "passed": price_confirmed,
            }

            if price_confirmed:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(
                f"{symbol}: Price confirmation error: {e}"
            )

        # 3. High Delivery Percentage (>=60%)
        try:
            delivery_pct = latest.get("delivery_percent", 0) or 0
            delivery_pct = float(delivery_pct)
            high_delivery = delivery_pct >= 60

            indicator_details["delivery_percent"] = {
                "value": round(delivery_pct, 2),
                "threshold": 60,
                "passed": high_delivery,
            }

            if high_delivery:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: Delivery % error: {e}")

        # 4. Above VWAP
        try:
            vwap_values = vwap(
                df["high"], df["low"], df["close"], df["volume"]
            )
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

        # 5. Trend Direction (EMA 20 > EMA 50)
        try:
            ema_20 = float(ema(df["close"], 20).iloc[-1])
            ema_50 = float(ema(df["close"], 50).iloc[-1])
            trend_up = ema_20 > ema_50

            indicator_details["trend"] = {
                "ema_20": round(ema_20, 2),
                "ema_50": round(ema_50, 2),
                "passed": trend_up,
            }

            if trend_up:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: Trend error: {e}")

        # Check signal criteria
        min_conditions = self.signal_config.get(
            "min_conditions_met", 3
        )
        confidence_threshold = self.signal_config.get(
            "confidence_threshold", 0.65
        )

        if (
            indicators_met >= min_conditions
            and weighted_score >= confidence_threshold
        ):
            stop_loss = self.calculate_stop_loss(entry_price, df)
            target = self.calculate_target(
                entry_price, stop_loss, df
            )

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
                total_indicators=total_indicators,
                indicator_details=indicator_details,
                metadata={
                    "volume_surge": current_vol_ratio
                    if "current_vol_ratio" in dir()
                    else 0,
                },
            )

            logger.info(
                f"SIGNAL: {self.name} - {symbol} "
                f"(confidence: {weighted_score:.2f})",
                extra=signal.to_dict(),
            )

            return signal

        return None

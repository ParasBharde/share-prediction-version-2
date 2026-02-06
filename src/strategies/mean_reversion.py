"""
Mean Reversion Strategy

Purpose:
    Identifies oversold quality stocks likely to revert to the mean.
    Targets stocks at Bollinger Band lows with volume confirmation.

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

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema, sma
from src.strategies.indicators.oscillators import macd, rsi
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Identifies mean reversion opportunities in oversold stocks."""

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        """
        Scan a stock for mean reversion signals.

        Args:
            symbol: Stock symbol.
            df: OHLCV DataFrame.
            company_info: Company metadata.

        Returns:
            TradingSignal if mean reversion setup detected.
        """
        if not self.apply_pre_filters(company_info):
            return None

        if len(df) < 200:
            return None

        indicators_met = 0
        total_indicators = len(self.indicators_config)
        weighted_score = 0.0
        indicator_details = {}

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # 1. RSI Oversold (RSI <= 30)
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])
            is_oversold = current_rsi <= 30

            indicator_details["rsi_oversold"] = {
                "value": round(current_rsi, 2),
                "threshold": 30,
                "passed": is_oversold,
            }

            if is_oversold:
                indicators_met += 1
                weighted_score += 0.30
        except Exception as e:
            logger.debug(f"{symbol}: RSI error: {e}")

        # 2. Bollinger Band Lower Touch
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
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: BB error: {e}")

        # 3. Volume Confirmation (1.5x average)
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol_ratio = float(vol_r.iloc[-1])
            vol_confirmed = current_vol_ratio >= 1.5

            indicator_details["volume_confirmation"] = {
                "volume_ratio": round(current_vol_ratio, 2),
                "threshold": 1.5,
                "passed": vol_confirmed,
            }

            if vol_confirmed:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: Volume error: {e}")

        # 4. 200 DMA Support (within 5% of 200 EMA)
        try:
            ema_200 = float(ema(df["close"], 200).iloc[-1])
            above_support = entry_price >= ema_200 * 0.95

            indicator_details["200dma_support"] = {
                "ema_200": round(ema_200, 2),
                "threshold": round(ema_200 * 0.95, 2),
                "passed": above_support,
            }

            if above_support:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: 200DMA error: {e}")

        # 5. MACD Divergence (histogram improving)
        try:
            macd_data = macd(df["close"], 12, 26, 9)
            hist = macd_data["histogram"]
            hist_improving = (
                float(hist.iloc[-1]) > float(hist.iloc[-2])
            )

            indicator_details["macd_divergence"] = {
                "histogram": round(float(hist.iloc[-1]), 4),
                "prev_histogram": round(
                    float(hist.iloc[-2]), 4
                ),
                "passed": hist_improving,
            }

            if hist_improving:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: MACD error: {e}")

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
                priority=AlertPriority.MEDIUM,
                indicators_met=indicators_met,
                total_indicators=total_indicators,
                indicator_details=indicator_details,
            )

            logger.info(
                f"SIGNAL: {self.name} - {symbol} "
                f"(confidence: {weighted_score:.2f})",
                extra=signal.to_dict(),
            )

            return signal

        return None

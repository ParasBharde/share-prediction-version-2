"""
Momentum Breakout Strategy

Purpose:
    Identifies stocks breaking 52-week highs with volume confirmation.
    Targets stocks with strong upward momentum and institutional interest.

Dependencies:
    - base_strategy for interface
    - indicators for technical calculations

Logging:
    - Each stock scan at DEBUG
    - Signal generation at INFO

Fallbacks:
    If individual indicator fails, that indicator is skipped.
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """Identifies momentum breakout opportunities."""

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        """
        Scan a stock for momentum breakout signals.

        Args:
            symbol: Stock symbol.
            df: OHLCV DataFrame (min 200 rows recommended).
            company_info: Company metadata.

        Returns:
            TradingSignal if breakout detected, None otherwise.
        """
        # Apply pre-filters
        if not self.apply_pre_filters(company_info):
            return None

        # Need sufficient data
        if len(df) < 200:
            logger.debug(
                f"{symbol}: Insufficient data "
                f"({len(df)} < 200 rows)"
            )
            return None

        indicators_met = 0
        total_indicators = len(self.indicators_config)
        weighted_score = 0.0
        indicator_details = {}

        latest = df.iloc[-1]
        entry_price = float(latest["close"])

        # 1. 52-Week High Proximity
        try:
            high_52w = float(df["high"].tail(252).max())
            proximity = entry_price / high_52w if high_52w > 0 else 0
            near_high = proximity >= 0.98

            indicator_details["52w_high"] = {
                "high_52w": high_52w,
                "proximity": round(proximity, 4),
                "passed": near_high,
            }

            if near_high:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: 52W high calc error: {e}")

        # 2. Volume Surge
        try:
            vol_ratio = volume_ratio(df["volume"], 20)
            current_vol_ratio = float(vol_ratio.iloc[-1])
            vol_surge = current_vol_ratio >= 2.0

            indicator_details["volume_surge"] = {
                "volume_ratio": round(current_vol_ratio, 2),
                "threshold": 2.0,
                "passed": vol_surge,
            }

            if vol_surge:
                indicators_met += 1
                weighted_score += 0.30
        except Exception as e:
            logger.debug(f"{symbol}: Volume calc error: {e}")

        # 3. RSI Range (50-70)
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])
            rsi_ok = 50 <= current_rsi <= 70

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "range": "50-70",
                "passed": rsi_ok,
            }

            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI calc error: {e}")

        # 4. EMA Alignment (20 > 50 > 200)
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

            if aligned:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: EMA calc error: {e}")

        # 5. Relative Strength vs Market
        try:
            stock_return = (
                (entry_price - float(df["close"].iloc[-21]))
                / float(df["close"].iloc[-21])
                * 100
            )
            # Use stock return vs 0 as simplified check
            rs_ok = stock_return > 0

            indicator_details["relative_strength"] = {
                "stock_return_20d": round(stock_return, 2),
                "passed": rs_ok,
            }

            if rs_ok:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(
                f"{symbol}: Relative strength calc error: {e}"
            )

        # Check signal generation criteria
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
                    "volume_surge": indicator_details.get(
                        "volume_surge", {}
                    ).get("volume_ratio", 0),
                    "rsi": indicator_details.get("rsi", {}).get(
                        "value", 0
                    ),
                    "high_52w": indicator_details.get(
                        "52w_high", {}
                    ).get("high_52w", 0),
                },
            )

            logger.info(
                f"SIGNAL: {self.name} - {symbol} "
                f"(confidence: {weighted_score:.2f}, "
                f"indicators: {indicators_met}/{total_indicators})",
                extra=signal.to_dict(),
            )

            return signal

        return None

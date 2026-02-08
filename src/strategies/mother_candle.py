"""
Mother Candle (Inside Bar) Strategy

Purpose:
    Detects Mother Candle patterns where a large candle (Mother)
    is followed by 3-6 smaller candles (Babies) that fit entirely
    within the Mother's range. Generates alerts on breakout when
    any candle from the 3rd to 6th breaks the Mother's high or low.

Pattern:
    - Mother Candle: A large candle that sets the range
      (High = resistance, Low = support).
    - Baby Candles: 3-6 subsequent candles where each has:
      High <= Mother High AND Low >= Mother Low
    - Breakout Trigger:
      BUY  -> when a candle closes above Mother's High
      SELL -> when a candle closes below Mother's Low
    - Stop Loss: Opposite end of Mother candle
    - Works best in trending markets on higher timeframes

Dependencies:
    - base_strategy for interface
    - indicators for trend confirmation

Logging:
    - Pattern detection at DEBUG
    - Signal generation at INFO

Fallbacks:
    If individual indicator fails, that check is skipped.
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi, adx
from src.strategies.indicators.volume_indicators import volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)


class MotherCandleStrategy(BaseStrategy):
    """Detects Mother Candle (Inside Bar) breakout patterns."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Scan stats for diagnostics
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
        """
        Scan a stock for Mother Candle breakout signals.

        Looks for a large candle followed by 3-6 smaller candles
        contained within it, then checks for breakout.

        Args:
            symbol: Stock symbol.
            df: OHLCV DataFrame (min 50 rows recommended).
            company_info: Company metadata.

        Returns:
            TradingSignal if breakout detected, None otherwise.
        """
        self._scan_stats["total"] += 1

        # Apply pre-filters
        if not self.apply_pre_filters(company_info):
            self._scan_stats["pre_filter_rejected"] += 1
            return None

        if len(df) < 50:
            self._scan_stats["insufficient_data"] += 1
            return None

        # Strategy parameters from config
        params = self.strategy_config.get("params", {})
        min_babies = params.get("min_baby_candles", 3)
        max_babies = params.get("max_baby_candles", 6)
        mother_body_min_pct = params.get(
            "mother_body_min_percent", 1.0
        )
        trend_confirmation = params.get(
            "trend_confirmation", True
        )

        indicators_met = 0
        total_indicators = len(self.indicators_config) or 5
        weighted_score = 0.0
        indicator_details = {}

        # Detect Mother Candle pattern
        pattern = self._find_mother_candle_pattern(
            df, min_babies, max_babies, mother_body_min_pct
        )

        if pattern is None:
            self._scan_stats["no_pattern"] += 1
            return None

        mother_idx = pattern["mother_idx"]
        baby_count = pattern["baby_count"]
        breakout_type = pattern["breakout_type"]
        mother_high = pattern["mother_high"]
        mother_low = pattern["mother_low"]
        breakout_price = pattern["breakout_price"]

        indicator_details["mother_candle"] = {
            "mother_high": round(mother_high, 2),
            "mother_low": round(mother_low, 2),
            "mother_range_pct": round(
                (mother_high - mother_low) / mother_low * 100, 2
            ),
            "baby_count": baby_count,
            "breakout_type": breakout_type,
            "passed": True,
        }
        indicators_met += 1
        weighted_score += 0.35

        latest = df.iloc[-1]
        entry_price = float(breakout_price)

        # Confirmation indicators

        # 1. Volume confirmation on breakout candle
        try:
            vol_ratio_series = volume_ratio(df["volume"], 20)
            breakout_vol = float(vol_ratio_series.iloc[-1])
            vol_ok = breakout_vol >= 1.5

            indicator_details["volume_confirmation"] = {
                "volume_ratio": round(breakout_vol, 2),
                "threshold": 1.5,
                "passed": vol_ok,
            }

            if vol_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: Volume calc error: {e}")

        # 2. Trend alignment (EMA 20 > EMA 50 for BUY)
        if trend_confirmation:
            try:
                ema_20 = float(ema(df["close"], 20).iloc[-1])
                ema_50 = float(ema(df["close"], 50).iloc[-1])

                if breakout_type == "BUY":
                    trend_ok = ema_20 > ema_50
                else:
                    trend_ok = ema_20 < ema_50

                indicator_details["trend_alignment"] = {
                    "ema_20": round(ema_20, 2),
                    "ema_50": round(ema_50, 2),
                    "direction": breakout_type,
                    "passed": trend_ok,
                }

                if trend_ok:
                    indicators_met += 1
                    weighted_score += 0.20
            except Exception as e:
                logger.debug(
                    f"{symbol}: Trend confirmation error: {e}"
                )

        # 3. RSI not overbought/oversold
        try:
            rsi_values = rsi(df["close"], 14)
            current_rsi = float(rsi_values.iloc[-1])

            if breakout_type == "BUY":
                rsi_ok = 40 <= current_rsi <= 75
            else:
                rsi_ok = 25 <= current_rsi <= 60

            indicator_details["rsi"] = {
                "value": round(current_rsi, 2),
                "passed": rsi_ok,
            }

            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
        except Exception as e:
            logger.debug(f"{symbol}: RSI calc error: {e}")

        # 4. ADX trend strength (higher ADX = stronger trend)
        try:
            adx_df = adx(df["high"], df["low"], df["close"], 14)
            current_adx = float(adx_df["adx"].iloc[-1])
            adx_ok = current_adx >= 20

            indicator_details["adx"] = {
                "value": round(current_adx, 2),
                "threshold": 20,
                "passed": adx_ok,
            }

            if adx_ok:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: ADX calc error: {e}")

        # Check signal generation criteria
        min_conditions = self.signal_config.get(
            "min_conditions_met", 3
        )
        confidence_threshold = self.signal_config.get(
            "confidence_threshold", 0.50
        )

        if (
            indicators_met >= min_conditions
            and weighted_score >= confidence_threshold
        ):
            # Stop loss at opposite end of mother candle
            if breakout_type == "BUY":
                signal_type = SignalType.BUY
                stop_loss = round(mother_low, 2)
                # Target: risk-reward based
                risk = entry_price - stop_loss
                ratio = self.risk_config.get("target", {}).get(
                    "ratio", 2.0
                )
                target = round(entry_price + (risk * ratio), 2)
            else:
                signal_type = SignalType.SELL
                stop_loss = round(mother_high, 2)
                risk = stop_loss - entry_price
                ratio = self.risk_config.get("target", {}).get(
                    "ratio", 2.0
                )
                target = round(entry_price - (risk * ratio), 2)

            signal = TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=signal_type,
                confidence=round(weighted_score, 4),
                entry_price=entry_price,
                target_price=target,
                stop_loss=stop_loss,
                priority=AlertPriority.HIGH,
                indicators_met=indicators_met,
                total_indicators=total_indicators,
                indicator_details=indicator_details,
                metadata={
                    "pattern": "mother_candle",
                    "baby_count": baby_count,
                    "mother_range": round(
                        mother_high - mother_low, 2
                    ),
                    "breakout_direction": breakout_type,
                    "sector": company_info.get(
                        "sector", "Unknown"
                    ),
                },
            )

            logger.info(
                f"SIGNAL: {self.name} - {symbol} "
                f"({breakout_type}, babies={baby_count}, "
                f"confidence: {weighted_score:.2f}, "
                f"indicators: {indicators_met}/{total_indicators})",
                extra=signal.to_dict(),
            )

            self._scan_stats["signals"] += 1
            return signal

        # Pattern found but indicators didn't meet threshold
        self._scan_stats["low_confidence"] += 1
        logger.info(
            f"{symbol}: Mother Candle pattern found "
            f"(babies={baby_count}, {breakout_type}) but "
            f"only {indicators_met}/{min_conditions} indicators met "
            f"(score={weighted_score:.2f}/{confidence_threshold})"
        )
        return None

    def _find_mother_candle_pattern(
        self,
        df: pd.DataFrame,
        min_babies: int,
        max_babies: int,
        mother_body_min_pct: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Search recent candles for a Mother Candle pattern.

        Looks backwards from the latest candle to find a pattern
        where a large candle is followed by 3-6 smaller candles
        within its range, with the latest candle breaking out.

        Args:
            df: OHLCV DataFrame.
            min_babies: Minimum baby candles required (default 3).
            max_babies: Maximum baby candles to consider (default 6).
            mother_body_min_pct: Minimum body size of mother candle
                as percentage of price.

        Returns:
            Pattern dict or None if not found.
        """
        if len(df) < min_babies + 2:
            return None

        # The latest candle is the potential breakout candle
        latest = df.iloc[-1]

        # Search for mother candle starting from max_babies+1 bars back
        # to min_babies+1 bars back
        search_start = min(len(df) - 1, max_babies + 1)
        search_end = min_babies + 1

        for lookback in range(search_start, search_end - 1, -1):
            if lookback >= len(df):
                continue

            mother_idx = len(df) - 1 - lookback
            mother = df.iloc[mother_idx]
            mother_high = float(mother["high"])
            mother_low = float(mother["low"])
            mother_open = float(mother["open"])
            mother_close = float(mother["close"])

            # Mother candle must be significant
            if mother_low <= 0:
                continue

            mother_range_pct = (
                (mother_high - mother_low) / mother_low * 100
            )
            if mother_range_pct < mother_body_min_pct:
                continue

            # Check if subsequent candles are inside the mother
            baby_count = 0
            all_inside = True

            for j in range(mother_idx + 1, len(df) - 1):
                candle = df.iloc[j]
                c_high = float(candle["high"])
                c_low = float(candle["low"])

                if c_high <= mother_high and c_low >= mother_low:
                    baby_count += 1
                else:
                    all_inside = False
                    break

            if not all_inside or baby_count < min_babies:
                continue

            if baby_count > max_babies:
                continue

            # Check if latest candle breaks out
            latest_close = float(latest["close"])
            latest_high = float(latest["high"])
            latest_low = float(latest["low"])

            breakout_type = None
            breakout_price = None

            if latest_close > mother_high:
                breakout_type = "BUY"
                breakout_price = latest_close
            elif latest_close < mother_low:
                breakout_type = "SELL"
                breakout_price = latest_close

            if breakout_type is None:
                continue

            logger.debug(
                f"Mother Candle pattern found: "
                f"mother_idx={mother_idx}, "
                f"babies={baby_count}, "
                f"breakout={breakout_type}, "
                f"range={mother_low:.2f}-{mother_high:.2f}"
            )

            return {
                "mother_idx": mother_idx,
                "baby_count": baby_count,
                "breakout_type": breakout_type,
                "mother_high": mother_high,
                "mother_low": mother_low,
                "breakout_price": breakout_price,
            }

        return None

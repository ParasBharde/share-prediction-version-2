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
    """
    Final Logic: Triggers ONLY if the last closing candle breaks the range.
    Includes Entry Buffer, Mandatory Volume, and RSI Overbought filters.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "1D")
        self.lookback_bars = self.data_config.get("lookback_bars", 200)

    def scan(
        self, symbol: str, df: pd.DataFrame, company_info: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        # Basic Pre-filters
        if not self.apply_pre_filters(company_info) or len(df) < self.lookback_bars:
            return None

        params = self.strategy_config.get("params", {})
        min_babies = params.get("min_baby_candles", 2)
        max_babies = params.get("max_baby_candles", 6)
        min_pct = params.get("mother_body_min_percent", 0.8)
        max_buffer = params.get("max_entry_buffer_pct", 2.5)

        # 1. CORE PATTERN: Must break on the CURRENT (last) candle
        pattern = self._find_last_candle_breakout(df, min_babies, max_babies, min_pct)
        if pattern is None:
            return None

        breakout_type = pattern["breakout_type"]
        mother_high = pattern["mother_high"]
        mother_low = pattern["mother_low"]
        entry_price = float(df.iloc[-1]["close"])

        # IMPROVEMENT: Max Entry Buffer Check (Don't chase the price)
        breakout_level = mother_high if breakout_type == "BUY" else mother_low
        distance_pct = abs(entry_price - breakout_level) / breakout_level * 100
        if distance_pct > max_buffer:
            logger.debug(f"{symbol}: Breakout overextended ({distance_pct:.2f}%)")
            return None

        indicators_met = 1
        weighted_score = 0.35
        indicator_details = {
            "mother_candle": {"passed": True, "baby_count": pattern["baby_count"]}
        }

        # 2. MANDATORY VOLUME: Reject if Volume < 1.3
        try:
            vol_r = float(volume_ratio(df["volume"], 20).iloc[-1])
            if params.get("mandatory_volume", True) and vol_r < 1.3:
                return None
            vol_ok = vol_r >= 1.5
            if vol_ok:
                indicators_met += 1
                weighted_score += 0.20
            indicator_details["volume"] = {"value": round(vol_r, 2), "passed": vol_ok}
        except Exception:
            pass

        # 3. TREND CONFIRMATION
        try:
            e20, e50 = float(ema(df["close"], 20).iloc[-1]), float(
                ema(df["close"], 50).iloc[-1]
            )
            trend_ok = (e20 > e50) if breakout_type == "BUY" else (e20 < e50)
            if trend_ok:
                indicators_met += 1
                weighted_score += 0.20
            indicator_details["trend"] = {"passed": trend_ok}
        except Exception:
            pass

        # 4. RSI FILTER (Reject Overbought > 70 for BUY)
        try:
            curr_rsi = float(rsi(df["close"], 14).iloc[-1])
            if breakout_type == "BUY" and curr_rsi > 70:
                return None
            rsi_ok = (
                (40 <= curr_rsi <= 70)
                if breakout_type == "BUY"
                else (30 <= curr_rsi <= 60)
            )
            if rsi_ok:
                indicators_met += 1
                weighted_score += 0.15
            indicator_details["rsi"] = {"value": round(curr_rsi, 2), "passed": rsi_ok}
        except Exception:
            pass

        # 5. ADX STRENGTH
        try:
            curr_adx = float(
                adx(df["high"], df["low"], df["close"], 14)["adx"].iloc[-1]
            )
            adx_ok = curr_adx >= 20
            if adx_ok:
                indicators_met += 1
                weighted_score += 0.10
            indicator_details["adx"] = {"value": round(curr_adx, 2), "passed": adx_ok}
        except Exception:
            pass

        # Final Approval
        min_met = self.signal_config.get("min_conditions_met", 4)
        if indicators_met >= min_met:
            sl = round(mother_low if breakout_type == "BUY" else mother_high, 2)
            # Automatic Stop Loss Distance Filter
            if (abs(entry_price - sl) / entry_price * 100) > 7.0:
                return None

            return TradingSignal(
                symbol=symbol,
                company_name=company_info.get("name", symbol),
                strategy_name=self.name,
                signal_type=(
                    SignalType.BUY if breakout_type == "BUY" else SignalType.SELL
                ),
                confidence=round(weighted_score, 4),
                entry_price=entry_price,
                target_price=round(entry_price + (abs(entry_price - sl) * 2.0), 2),
                stop_loss=sl,
                priority=AlertPriority.HIGH,
                indicators_met=indicators_met,
                total_indicators=5,
                indicator_details=indicator_details,
                metadata={
                    "timeframe": self.timeframe,
                    "baby_count": pattern["baby_count"],
                },
            )
        return None

    def _find_last_candle_breakout(self, df, min_babies, max_babies, min_pct):
        last_close = float(df.iloc[-1]["close"])
        for b in range(min_babies, max_babies + 1):
            mother_idx = -(b + 2)
            if abs(mother_idx) > len(df):
                break
            mother = df.iloc[mother_idx]
            m_h, m_l = float(mother["high"]), float(mother["low"])
            if m_l <= 0 or ((m_h - m_l) / m_l * 100) < min_pct:
                continue

            # Ensure all candles between mother and last are inside
            if all(
                float(df.iloc[j]["high"]) <= m_h * 1.003
                and float(df.iloc[j]["low"]) >= m_l * 0.997
                for j in range(mother_idx + 1, -1)
            ):
                if last_close > m_h:
                    return {
                        "mother_high": m_h,
                        "mother_low": m_l,
                        "baby_count": b,
                        "breakout_type": "BUY",
                    }
                if last_close < m_l:
                    return {
                        "mother_high": m_h,
                        "mother_low": m_l,
                        "baby_count": b,
                        "breakout_type": "SELL",
                    }
        return None

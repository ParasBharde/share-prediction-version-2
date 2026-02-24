"""
Options PCR (Put-Call Ratio) Sentiment Strategy — v2

Uses overall market PCR to determine sentiment direction.
Combines with VWAP alignment, EMA trend, and MACD confirmation.

Improvements over v1:
    - PCR thresholds now configurable from YAML
      (bullish_pcr, bearish_pcr — defaults 1.2 / 0.8)
    - MACD momentum added as 5th indicator (replaces RSI)
    - Per-scan statistics via get_scan_stats()
    - Rich metadata: VWAP level, OI levels, PCR, sentiment stored
      for chart visualization overlay

Logic:
    PCR > bullish_pcr  → Bullish sentiment → BUY CE (if VWAP / EMA aligned)
    PCR < bearish_pcr  → Bearish sentiment → BUY PE (if VWAP / EMA aligned)
    PCR neutral        → No trade
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi
from src.strategies.indicators.volume_indicators import volume_ratio, vwap
from src.utils.constants import AlertPriority, SignalType
from src.strategies.options_oi_breakout import _LOT_SIZE

logger = get_logger(__name__)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line


class OptionsPCRStrategy(BaseStrategy):
    """PCR-based sentiment strategy for options — v2."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "5m")
        params = config.get("signal_generation", {})
        # Configurable PCR thresholds
        self.bullish_pcr = params.get("bullish_pcr_threshold", 1.2)
        self.bearish_pcr = params.get("bearish_pcr_threshold", 0.8)
        # IV filter: skip if ATM IV is above this % (options too expensive)
        self.max_iv_to_buy = params.get("max_iv_to_buy", 40.0)

        self._scan_stats = {
            "total": 0,
            "no_option_chain": 0,
            "insufficient_data": 0,
            "pcr_neutral": 0,
            "expiry_day_skip": 0,
            "iv_too_high": 0,
            "volume_rejected": 0,
            "macd_rejected": 0,
            "low_confidence": 0,
            "signals": 0,
        }

    def get_scan_stats(self) -> Dict[str, int]:
        return dict(self._scan_stats)

    def scan(
        self,
        symbol: str,
        df: pd.DataFrame,
        company_info: Dict[str, Any],
    ) -> Optional[TradingSignal]:
        self._scan_stats["total"] += 1

        option_chain = company_info.get("option_chain", {})
        if not option_chain:
            self._scan_stats["no_option_chain"] += 1
            return None

        if len(df) < 30:
            self._scan_stats["insufficient_data"] += 1
            return None

        pcr = option_chain.get("pcr", 1.0)
        underlying = option_chain.get("underlying_price", 0)
        is_expiry_day = option_chain.get("is_expiry_day", False)

        if not underlying:
            self._scan_stats["no_option_chain"] += 1
            return None

        # ── Expiry-day guard: PCR is unreliable on expiry ────────────────────
        # On expiry day, OI unwinds aggressively. PCR values spike without
        # reflecting real sentiment. Skip this strategy on expiry day.
        if is_expiry_day:
            self._scan_stats["expiry_day_skip"] += 1
            logger.debug(
                f"{symbol}: PCR strategy skipped — today is expiry day "
                "(PCR unreliable on expiry)"
            )
            return None

        # ── IV filter ─────────────────────────────────────────────────────────
        atm_ce_iv = option_chain.get("atm_ce_iv", 0) or 0
        atm_pe_iv = option_chain.get("atm_pe_iv", 0) or 0
        if atm_ce_iv > self.max_iv_to_buy or atm_pe_iv > self.max_iv_to_buy:
            self._scan_stats["iv_too_high"] += 1
            logger.debug(
                f"{symbol}: IV filter — "
                f"CE IV={atm_ce_iv:.1f}%, PE IV={atm_pe_iv:.1f}% "
                f"(max={self.max_iv_to_buy}%)"
            )
            return None

        atm_ce_ltp = option_chain.get("atm_ce_ltp", 0) or 0
        atm_pe_ltp = option_chain.get("atm_pe_ltp", 0) or 0
        atm_strike = option_chain.get("atm_strike", 0) or 0

        entry_price = float(df.iloc[-1]["close"])
        indicators_met = 0
        weighted_score = 0.0
        indicator_details: Dict[str, Any] = {}
        signal_type = None

        # ── 1. PCR Sentiment (core signal) ────────────────────────────────
        if pcr > self.bullish_pcr:
            signal_type = "BUY_CE"
            sentiment = "bullish"
        elif pcr < self.bearish_pcr:
            signal_type = "BUY_PE"
            sentiment = "bearish"
        else:
            indicator_details["pcr"] = {
                "value": round(pcr, 3),
                "sentiment": "neutral",
                "bullish_threshold": self.bullish_pcr,
                "bearish_threshold": self.bearish_pcr,
                "passed": False,
            }
            self._scan_stats["pcr_neutral"] += 1
            return None

        indicator_details["pcr"] = {
            "value": round(pcr, 3),
            "sentiment": sentiment,
            "bullish_threshold": self.bullish_pcr,
            "bearish_threshold": self.bearish_pcr,
            "passed": True,
        }
        indicators_met += 1
        weighted_score += 0.30

        # ── 2. OI Support / Resistance Alignment ──────────────────────────
        support = option_chain.get("support", 0)
        resistance = option_chain.get("resistance", 0)
        try:
            if signal_type == "BUY_CE":
                oi_ok = entry_price > support
            else:
                oi_ok = entry_price < resistance

            indicator_details["oi_levels"] = {
                "support": support,
                "resistance": resistance,
                "price": round(entry_price, 2),
                "passed": oi_ok,
            }
            if oi_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: OI levels error: {e}")

        # ── 3. VWAP Alignment ─────────────────────────────────────────────
        vwap_value = 0.0
        try:
            vwap_series = vwap(df["high"], df["low"], df["close"], df["volume"])
            vwap_value = round(float(vwap_series.iloc[-1]), 2)
            vwap_ok = (
                entry_price > vwap_value
                if signal_type == "BUY_CE"
                else entry_price < vwap_value
            )
            indicator_details["vwap"] = {
                "vwap": vwap_value,
                "price": round(entry_price, 2),
                "passed": vwap_ok,
            }
            if vwap_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")

        # ── 4. EMA Trend Alignment ─────────────────────────────────────────
        try:
            ema_9 = float(ema(df["close"], 9).iloc[-1])
            ema_21 = float(ema(df["close"], 21).iloc[-1])
            trend_ok = (
                ema_9 > ema_21
                if signal_type == "BUY_CE"
                else ema_9 < ema_21
            )
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

        # ── 5. MACD Momentum Confirmation ─────────────────────────────────
        try:
            _, _, hist = _macd(df["close"])
            hist_now = float(hist.iloc[-1])
            hist_prev = float(hist.iloc[-2])
            macd_ok = (
                hist_now > 0 and hist_now > hist_prev
                if signal_type == "BUY_CE"
                else hist_now < 0 and hist_now < hist_prev
            )
            indicator_details["macd"] = {
                "histogram": round(hist_now, 4),
                "prev_histogram": round(hist_prev, 4),
                "direction": "bullish" if hist_now > 0 else "bearish",
                "passed": macd_ok,
            }
            if macd_ok:
                indicators_met += 1
                weighted_score += 0.15
            else:
                self._scan_stats["macd_rejected"] += 1
        except Exception as e:
            logger.debug(f"{symbol}: MACD error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get(
            "confidence_threshold", 0.55
        )

        if indicators_met < min_conditions or weighted_score < confidence_threshold:
            self._scan_stats["low_confidence"] += 1
            return None

        # ── Risk management ────────────────────────────────────────────────
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

        # Use ATM strike from option chain data if available
        # (more accurate than rounding spot price to lot-size step)
        if not atm_strike:
            step = _LOT_SIZE.get(symbol.upper(), 50)
            atm_strike = round(entry_price / step) * step

        self._scan_stats["signals"] += 1

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
                # ATM option premiums — shown in Telegram alert
                "atm_ce_ltp": round(atm_ce_ltp, 2),
                "atm_pe_ltp": round(atm_pe_ltp, 2),
                "atm_ce_iv": round(atm_ce_iv, 1),
                "atm_pe_iv": round(atm_pe_iv, 1),
                "is_expiry_day": False,   # already skipped above if True
                # Levels for chart overlay
                "oi_resistance": resistance,
                "oi_support": support,
                "vwap_value": vwap_value,
                "pcr": pcr,
                "sentiment": sentiment,
                "bullish_pcr": self.bullish_pcr,
                "bearish_pcr": self.bearish_pcr,
            },
        )

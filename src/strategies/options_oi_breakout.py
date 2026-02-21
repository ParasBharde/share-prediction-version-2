"""
Options OI (Open Interest) Breakout Strategy — v2

Detects when price breaks and holds above max CALL OI strike (resistance)
or breaks below max PUT OI strike (support).  Uses multi-candle confirmation
and MACD momentum to reduce false breakouts.

Improvements over v1:
    - Multi-candle confirmation: price must have been below resistance for
      at least 2 candles before the breakout bar (not just prev_close check)
    - MACD momentum indicator added (replaces noisy single-candle signal)
    - Candle strength filter: breakout candle must close in top/bottom 40%
    - ATM strike rounded to correct lot size (NIFTY=50, BANKNIFTY=100)
    - Rich metadata for chart visualization (levels, VWAP, PCR)
    - Per-scan statistics via get_scan_stats()

Logic:
    BUY CE  → price breaks above max CALL OI strike (resistance) with volume
    BUY PE  → price breaks below max PUT OI strike  (support)   with volume
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

# Strike rounding by index name
_LOT_SIZE = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "MIDCPNIFTY": 25,
    "FINNIFTY": 50,
    "SENSEX": 100,
}


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) as pandas Series."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


class OptionsOIBreakoutStrategy(BaseStrategy):
    """Enhanced options strategy based on OI support/resistance breakout."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        params = config.get("signal_generation", {})
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "5m")
        # How many candles below resistance before valid breakout
        self.confirm_bars = params.get("confirm_bars_below", 2)
        # Candle strength: breakout close must be in top X% of candle range
        self.min_candle_strength = params.get("min_candle_strength", 0.55)

        self._scan_stats = {
            "total": 0,
            "no_option_chain": 0,
            "insufficient_data": 0,
            "no_breakout": 0,
            "candle_weak": 0,
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
        """
        Scan for OI breakout signals.

        company_info must contain an ``option_chain`` dict with keys:
            resistance, support, pcr, underlying_price,
            max_ce_oi, max_pe_oi, strikes (optional list)
        """
        self._scan_stats["total"] += 1

        option_chain = company_info.get("option_chain", {})
        if not option_chain:
            self._scan_stats["no_option_chain"] += 1
            return None

        if len(df) < 30:
            self._scan_stats["insufficient_data"] += 1
            return None

        resistance = option_chain.get("resistance", 0)
        support = option_chain.get("support", 0)
        pcr = option_chain.get("pcr", 1.0)
        underlying = option_chain.get("underlying_price", 0)

        if not resistance or not support or not underlying:
            self._scan_stats["no_option_chain"] += 1
            return None

        close = df["close"]
        entry_price = float(close.iloc[-1])

        # ── Multi-candle breakout confirmation ─────────────────────────────
        # Check that price was BELOW resistance for the last N candles
        # (or ABOVE support for bearish), then just broke through.
        recent_closes = close.iloc[-(self.confirm_bars + 1): -1]
        broke_resistance = (
            entry_price > resistance
            and (recent_closes <= resistance).all()
        )
        broke_support = (
            entry_price < support
            and (recent_closes >= support).all()
        )

        if not broke_resistance and not broke_support:
            self._scan_stats["no_breakout"] += 1
            return None

        signal_type = "BUY_CE" if broke_resistance else "BUY_PE"

        # ── Candle strength filter ─────────────────────────────────────────
        last = df.iloc[-1]
        candle_range = float(last["high"]) - float(last["low"])
        if candle_range > 0:
            if signal_type == "BUY_CE":
                close_strength = (
                    float(last["close"]) - float(last["low"])
                ) / candle_range
            else:
                close_strength = (
                    float(last["high"]) - float(last["close"])
                ) / candle_range
            if close_strength < self.min_candle_strength:
                self._scan_stats["candle_weak"] += 1
                return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details: Dict[str, Any] = {}

        # ── 1. OI Breakout (core signal) ───────────────────────────────────
        indicators_met += 1
        weighted_score += 0.35
        if signal_type == "BUY_CE":
            indicator_details["oi_breakout"] = {
                "type": "resistance_break",
                "resistance": resistance,
                "confirmed_bars": self.confirm_bars,
                "price": round(entry_price, 2),
                "passed": True,
            }
        else:
            indicator_details["oi_breakout"] = {
                "type": "support_break",
                "support": support,
                "confirmed_bars": self.confirm_bars,
                "price": round(entry_price, 2),
                "passed": True,
            }

        # ── 2. PCR confirmation ────────────────────────────────────────────
        try:
            pcr_ok = pcr > 1.0 if signal_type == "BUY_CE" else pcr < 0.8
            indicator_details["pcr"] = {
                "value": round(pcr, 3),
                "interpretation": (
                    "bullish" if pcr > 1.0
                    else "bearish" if pcr < 0.8
                    else "neutral"
                ),
                "passed": pcr_ok,
            }
            if pcr_ok:
                indicators_met += 1
                weighted_score += 0.20
        except Exception as e:
            logger.debug(f"{symbol}: PCR error: {e}")

        # ── 3. Volume confirmation ─────────────────────────────────────────
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
                weighted_score += 0.15
            else:
                self._scan_stats["volume_rejected"] += 1
        except Exception as e:
            logger.debug(f"{symbol}: Volume error: {e}")

        # ── 4. MACD momentum ───────────────────────────────────────────────
        try:
            _, _, hist = _macd(df["close"])
            hist_now = float(hist.iloc[-1])
            hist_prev = float(hist.iloc[-2])
            # Rising histogram = accelerating momentum
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
                weighted_score += 0.20
            else:
                self._scan_stats["macd_rejected"] += 1
        except Exception as e:
            logger.debug(f"{symbol}: MACD error: {e}")

        # ── 5. EMA trend alignment ─────────────────────────────────────────
        try:
            ema_9 = float(ema(df["close"], 9).iloc[-1])
            ema_21 = float(ema(df["close"], 21).iloc[-1])
            trend_ok = ema_9 > ema_21 if signal_type == "BUY_CE" else ema_9 < ema_21
            indicator_details["trend"] = {
                "ema_9": round(ema_9, 2),
                "ema_21": round(ema_21, 2),
                "direction": "bullish" if ema_9 > ema_21 else "bearish",
                "passed": trend_ok,
            }
            if trend_ok:
                indicators_met += 1
                weighted_score += 0.10
        except Exception as e:
            logger.debug(f"{symbol}: EMA error: {e}")

        min_conditions = self.signal_config.get("min_conditions_met", 3)
        confidence_threshold = self.signal_config.get(
            "confidence_threshold", 0.55
        )

        if indicators_met < min_conditions or weighted_score < confidence_threshold:
            self._scan_stats["low_confidence"] += 1
            return None

        # ── Risk management ────────────────────────────────────────────────
        if signal_type == "BUY_CE":
            # SL: 0.5% below resistance (the OI wall)
            stop_loss = round(resistance * 0.995, 2)
            risk = entry_price - stop_loss
            target = round(entry_price + (risk * 2.0), 2)
            trade_signal_type = SignalType.BUY
        else:
            stop_loss = round(support * 1.005, 2)
            risk = stop_loss - entry_price
            target = round(entry_price - (risk * 2.0), 2)
            trade_signal_type = SignalType.SELL

        atm_strike = self._find_atm_strike(option_chain, symbol, entry_price)

        # ── VWAP for chart visualization ───────────────────────────────────
        vwap_value = 0.0
        try:
            vwap_values = vwap(df["high"], df["low"], df["close"], df["volume"])
            vwap_value = round(float(vwap_values.iloc[-1]), 2)
        except Exception:
            pass

        self._scan_stats["signals"] += 1

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
                # Levels stored for chart overlay
                "oi_resistance": resistance,
                "oi_support": support,
                "vwap_value": vwap_value,
                "pcr": pcr,
                "max_ce_oi": option_chain.get("max_ce_oi", 0),
                "max_pe_oi": option_chain.get("max_pe_oi", 0),
                "breakout_type": (
                    "resistance_break" if broke_resistance else "support_break"
                ),
                "confirm_bars": self.confirm_bars,
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _find_atm_strike(
        self, option_chain: Dict, symbol: str, price: float
    ) -> float:
        """Round price to the nearest ATM strike for the given index."""
        step = _LOT_SIZE.get(symbol.upper(), 50)
        # First try the OI data's own strikes list
        strikes = option_chain.get("strikes", [])
        if strikes:
            available = [
                s["strike"] if isinstance(s, dict) else float(s)
                for s in strikes
            ]
            return min(available, key=lambda s: abs(s - price))
        return round(price / step) * step

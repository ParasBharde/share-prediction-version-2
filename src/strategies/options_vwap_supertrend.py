"""
Options VWAP + Supertrend Strategy — v2

Directional intraday strategy for index options (NIFTY/BANKNIFTY).
VWAP sets the overall bias; Supertrend provides the entry trigger.

Improvements over v1:
    - MACD momentum confirmation added (replaces noisy RSI check)
    - Time filter: skips first 15 min (9:30–9:45) and last 30 min (14:45+)
      of the session to avoid choppy open and low-liquidity close
    - Supertrend history stored in metadata for chart visualization
    - VWAP series stored in metadata for chart overlay
    - Per-scan statistics via get_scan_stats()
    - Fresh-signal bonus for supertrend flip (was already there, kept)

Logic:
    Price > VWAP  AND Supertrend turns green   → BUY CE
    Price < VWAP  AND Supertrend turns red     → BUY PE
    Exit: Supertrend reversal OR 14:45 IST cut-off
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.indicators.moving_averages import ema
from src.strategies.indicators.oscillators import rsi, supertrend
from src.strategies.indicators.volume_indicators import vwap, volume_ratio
from src.utils.constants import AlertPriority, SignalType

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# Don't enter trades in the first/last N minutes of the session
_AVOID_FIRST_MINUTES = 15   # 9:30–9:45 = choppy open
_AVOID_LAST_MINUTES = 30    # 14:45+ = liquidity dries up


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line


def _is_valid_intraday_time(bypass: bool = False) -> bool:
    """Return False during the first/last avoid windows."""
    if bypass:
        return True
    now = datetime.now(IST)
    total_min = now.hour * 60 + now.minute
    # Market open: 9:30 = 570 min
    open_min = 9 * 60 + 30
    # Avoid first 15 min
    if total_min < open_min + _AVOID_FIRST_MINUTES:
        return False
    # Avoid last 30 min (14:45 = 885 min)
    close_min = 14 * 60 + 45
    if total_min >= close_min:
        return False
    return True


class OptionsVWAPSupertrendStrategy(BaseStrategy):
    """VWAP + Supertrend directional strategy for options — v2."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_config = config.get("data", {})
        self.timeframe = self.data_config.get("timeframe", "5m")
        params = config.get("signal_generation", {})
        self.st_period = params.get("supertrend_period", 10)
        self.st_multiplier = params.get("supertrend_multiplier", 3.0)
        self.bypass_time = params.get("bypass_time_filter", False)

        self._scan_stats = {
            "total": 0,
            "insufficient_data": 0,
            "time_blocked": 0,
            "vwap_st_conflict": 0,
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

        if len(df) < 30:
            self._scan_stats["insufficient_data"] += 1
            return None

        # Time filter (intraday guard)
        if not _is_valid_intraday_time(self.bypass_time):
            self._scan_stats["time_blocked"] += 1
            return None

        indicators_met = 0
        weighted_score = 0.0
        indicator_details: Dict[str, Any] = {}
        signal_type = None
        above_vwap = below_vwap = False

        entry_price = float(df.iloc[-1]["close"])

        # ── 1. VWAP Direction (bias) ───────────────────────────────────────
        try:
            vwap_series = vwap(
                df["high"], df["low"], df["close"], df["volume"]
            )
            current_vwap = float(vwap_series.iloc[-1])
            above_vwap = entry_price > current_vwap
            below_vwap = entry_price < current_vwap

            indicator_details["vwap"] = {
                "vwap": round(current_vwap, 2),
                "price": round(entry_price, 2),
                "direction": "bullish" if above_vwap else "bearish",
                "passed": above_vwap or below_vwap,
            }
            if above_vwap or below_vwap:
                indicators_met += 1
                weighted_score += 0.25
        except Exception as e:
            logger.debug(f"{symbol}: VWAP error: {e}")
            return None

        # ── 2. Supertrend Signal (entry trigger) ───────────────────────────
        st_data = None
        st_value = 0.0
        try:
            st_data = supertrend(
                df["high"], df["low"], df["close"],
                self.st_period, self.st_multiplier,
            )
            current_dir = int(st_data["direction"].iloc[-1])
            prev_dir = int(st_data["direction"].iloc[-2])

            st_buy = current_dir == 1 and prev_dir == -1  # Just flipped green
            st_sell = current_dir == -1 and prev_dir == 1  # Just flipped red
            st_bullish = current_dir == 1
            st_bearish = current_dir == -1

            st_value_raw = st_data["supertrend"].iloc[-1]
            st_value = (
                round(float(st_value_raw), 2)
                if pd.notna(st_value_raw)
                else 0.0
            )

            indicator_details["supertrend"] = {
                "direction": "bullish" if current_dir == 1 else "bearish",
                "fresh_signal": st_buy or st_sell,
                "value": st_value,
                "passed": st_buy or st_sell or st_bullish or st_bearish,
            }

            if above_vwap and (st_buy or st_bullish):
                signal_type = "BUY_CE"
                indicators_met += 1
                weighted_score += 0.30
                if st_buy:
                    weighted_score += 0.05   # Fresh-signal bonus
            elif below_vwap and (st_sell or st_bearish):
                signal_type = "BUY_PE"
                indicators_met += 1
                weighted_score += 0.30
                if st_sell:
                    weighted_score += 0.05
            else:
                self._scan_stats["vwap_st_conflict"] += 1
                return None

        except Exception as e:
            logger.debug(f"{symbol}: Supertrend error: {e}")
            return None

        # ── 3. Volume Confirmation ─────────────────────────────────────────
        try:
            vol_r = volume_ratio(df["volume"], 20)
            current_vol = float(vol_r.iloc[-1])
            vol_ok = current_vol >= 1.3
            indicator_details["volume"] = {
                "volume_ratio": round(current_vol, 2),
                "threshold": 1.3,
                "passed": vol_ok,
            }
            if vol_ok:
                indicators_met += 1
                weighted_score += 0.15
            else:
                self._scan_stats["volume_rejected"] += 1
        except Exception as e:
            logger.debug(f"{symbol}: Volume error: {e}")

        # ── 4. MACD Momentum Confirmation ──────────────────────────────────
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

        # ── 5. EMA 9 Direction ─────────────────────────────────────────────
        try:
            ema_9 = ema(df["close"], 9)
            ema_rising = float(ema_9.iloc[-1]) > float(ema_9.iloc[-3])
            ema_falling = float(ema_9.iloc[-1]) < float(ema_9.iloc[-3])
            ema_ok = ema_rising if signal_type == "BUY_CE" else ema_falling
            indicator_details["ema_9"] = {
                "value": round(float(ema_9.iloc[-1]), 2),
                "rising": ema_rising,
                "passed": ema_ok,
            }
            if ema_ok:
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
            stop_loss = (
                max(st_value, entry_price * 0.995)
                if st_value > 0 else entry_price * 0.995
            )
            stop_loss = round(stop_loss, 2)
            risk = entry_price - stop_loss
            target = round(entry_price + (risk * 2.0), 2)
            trade_signal = SignalType.BUY
        else:
            stop_loss = (
                min(st_value, entry_price * 1.005)
                if st_value > 0 else entry_price * 1.005
            )
            stop_loss = round(stop_loss, 2)
            risk = stop_loss - entry_price
            target = round(entry_price - (risk * 2.0), 2)
            trade_signal = SignalType.SELL

        # Round ATM strike to index lot size
        from src.strategies.options_oi_breakout import _LOT_SIZE
        step = _LOT_SIZE.get(symbol.upper(), 50)
        atm_strike = round(entry_price / step) * step

        # Collect last N Supertrend + VWAP values for chart overlay
        n_chart = min(30, len(df))
        try:
            vwap_series = vwap(
                df["high"], df["low"], df["close"], df["volume"]
            )
            vwap_chart: List[float] = [
                round(float(v), 2)
                for v in vwap_series.iloc[-n_chart:]
            ]
        except Exception:
            vwap_chart = []

        st_line_chart: List[float] = []
        st_dir_chart: List[int] = []
        if st_data is not None:
            try:
                st_line_chart = [
                    round(float(v), 2) if pd.notna(v) else 0.0
                    for v in st_data["supertrend"].iloc[-n_chart:]
                ]
                st_dir_chart = [
                    int(d)
                    for d in st_data["direction"].iloc[-n_chart:]
                ]
            except Exception:
                pass

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
                "supertrend_value": st_value,
                "supertrend_direction": (
                    "bullish"
                    if signal_type == "BUY_CE"
                    else "bearish"
                ),
                # Series for chart drawing (last n_chart bars)
                "vwap_series": vwap_chart,
                "supertrend_series": st_line_chart,
                "supertrend_dir_series": st_dir_chart,
                "n_chart_bars": n_chart,
                "vwap": indicator_details.get("vwap", {}).get("vwap", 0),
                "exit_rule": "Supertrend reversal or 14:45 IST",
            },
        )

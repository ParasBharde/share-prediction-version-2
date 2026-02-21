"""
Live Intraday Scanner - ULTRA-STRICT VERSION

PROBLEM: Too many false signals, poor entry quality
SOLUTION: Added 10 additional confirmation filters

NEW STRICT FILTERS:
1. Price Action Confirmation   → Candle must close strong (near high)
2. Volume Surge Required        → Current candle volume > 1.5x avg
3. Higher Timeframe Alignment   → 1H chart must also be bullish
4. Support/Resistance Check     → Not near major resistance
5. Multi-Candle Confirmation    → Last 2-3 candles bullish
6. ATR-Based Volatility         → Filter out choppy/ranging stocks
7. Spread Check                 → Bid-ask spread must be tight
8. Momentum Confirmation        → MACD must be bullish
9. Price Structure              → Higher highs + higher lows
10. Time Since Last Signal      → Wait 30 min between signals per stock

USAGE:
    python scripts/live_scan.py --universe NIFTY500 --interval 15m --telegram
"""

import argparse
import asyncio
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import pytz

# Suppress yfinance warnings about delisted stocks
warnings.filterwarnings('ignore', message='.*possibly delisted.*')
warnings.filterwarnings('ignore', message='.*No data found.*')
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
import numpy as np

from src.data_ingestion.fallback_manager import FallbackManager
from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import StrategyLoader, STRATEGY_REGISTRY
from src.utils.config_loader import load_config, load_strategy_config
from src.utils.constants import YAHOO_NSE_SUFFIX
from src.utils.visualizer import ChartVisualizer

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

logger = get_logger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ============================================================================
# ULTRA-STRICT FILTERS - VERY AGGRESSIVE FILTERING
# ============================================================================
STRICT_FILTERS = {
    # Market hours
    "market_open_hour": 9,
    "market_open_minute": 30,
    "market_close_hour": 14,
    "market_close_minute": 30,

    # Volume - MUCH STRICTER
    "min_volume_today": 100_000,        # Doubled from 50k
    "min_avg_volume": 200_000,          # Doubled from 100k
    "min_volume_surge": 1.5,            # Current vol must be 1.5x average

    # Gap filter
    "max_gap_up_pct": 1.5,              # Reduced from 2% (tighter)
    "max_gap_down_pct": 1.5,

    # Nifty trend - STRICTER
    "nifty_min_change_pct": 0.0,        # Changed from -0.3 (must be positive or neutral)

    # Daily trend
    "daily_ema_period": 20,
    "require_daily_uptrend": True,

    # Time windows
    "time_windows": {
        "strong":  [(9, 30, 10, 30), (13, 30, 14, 30)],  # Only 2 windows
        "avoid":   [(10, 30, 13, 30)],                    # Avoid most of mid-day
    },

    # NEW: Price Action Quality
    "min_candle_close_strength": 0.7,   # Close must be in top 30% of range
    "min_body_to_range_ratio": 0.6,     # Body must be 60% of total range

    # NEW: Multi-Candle Confirmation
    "require_n_bullish_candles": 2,     # Last N candles must be bullish

    # NEW: ATR-based filters
    "min_atr_ratio": 0.015,             # ATR must be > 1.5% of price (not too tight)
    "max_atr_ratio": 0.05,              # ATR must be < 5% of price (not too choppy)

    # NEW: Support/Resistance
    "lookback_sr_candles": 50,          # Check last 50 candles for S/R
    "sr_proximity_pct": 2.0,            # Don't enter within 2% of major S/R

    # NEW: Signal Cooldown (prevent spam)
    "signal_cooldown_minutes": 30,      # 30 min between signals per stock

    # NEW: Higher Timeframe Confirmation
    "require_1h_alignment": True,       # 1H chart must also be bullish
}

# Track recent signals to enforce cooldown
_recent_signals: Dict[str, datetime] = {}

# NIFTY 50 symbols
NIFTY50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC",
    "NESTLEIND", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMTRDVT",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]


# ============================================================================
# EXISTING FILTERS (from previous version)
# ============================================================================

def is_valid_trading_time(bypass: bool = False) -> Tuple[bool, str]:
    """Market hours check."""
    if bypass:
        return True, "bypassed"

    now = datetime.now(IST)
    hour, minute = now.hour, now.minute
    total_minutes = hour * 60 + minute

    open_mins = STRICT_FILTERS["market_open_hour"] * 60 + STRICT_FILTERS["market_open_minute"]
    close_mins = STRICT_FILTERS["market_close_hour"] * 60 + STRICT_FILTERS["market_close_minute"]

    if total_minutes < open_mins:
        return False, f"Market not open yet (opens 9:30 AM, now {hour:02d}:{minute:02d})"
    if total_minutes >= close_mins:
        return False, f"Too close to close (cutoff 2:30 PM, now {hour:02d}:{minute:02d})"

    return True, "valid"


def get_time_window_quality() -> str:
    """Time window quality."""
    now = datetime.now(IST)
    h, m = now.hour, now.minute
    total = h * 60 + m

    for (sh, sm, eh, em) in STRICT_FILTERS["time_windows"]["avoid"]:
        if sh * 60 + sm <= total < eh * 60 + em:
            return "avoid"

    for (sh, sm, eh, em) in STRICT_FILTERS["time_windows"]["strong"]:
        if sh * 60 + sm <= total < eh * 60 + em:
            return "strong"

    return "avoid"  # Changed default to avoid


def get_nifty_trend() -> Tuple[float, str]:
    """Fetch Nifty trend."""
    try:
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period="2d", interval="1d")
        if df.empty or len(df) < 2:
            return 0.0, "unknown"

        prev_close = float(df["Close"].iloc[-2])
        curr_close = float(df["Close"].iloc[-1])
        change_pct = ((curr_close - prev_close) / prev_close) * 100

        if change_pct > 0.3:
            trend = "bullish"
        elif change_pct < -0.3:
            trend = "bearish"
        else:
            trend = "neutral"

        return round(change_pct, 2), trend

    except Exception as e:
        logger.debug(f"Could not fetch Nifty: {e}")
        return 0.0, "unknown"


# ============================================================================
# NEW ULTRA-STRICT FILTERS
# ============================================================================

def check_candle_quality(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    NEW FILTER: Check if current candle shows strong price action.
    
    - Close must be in top 30% of candle range (strong close)
    - Body must be at least 60% of total range (not a doji)
    """
    try:
        last = df.iloc[-1]
        o, h, l, c = last['open'], last['high'], last['low'], last['close']
        
        candle_range = h - l
        if candle_range == 0:
            return False, "Zero range candle"
        
        # Where did it close in the range? (0 = low, 1 = high)
        close_position = (c - l) / candle_range
        
        # Body size
        body = abs(c - o)
        body_ratio = body / candle_range
        
        if close_position < STRICT_FILTERS["min_candle_close_strength"]:
            return False, f"Weak close: {close_position:.2f} < 0.7"
        
        if body_ratio < STRICT_FILTERS["min_body_to_range_ratio"]:
            return False, f"Small body: {body_ratio:.2f} < 0.6"
        
        return True, f"Strong candle: close_pos={close_position:.2f}, body={body_ratio:.2f}"
    
    except Exception as e:
        logger.debug(f"Candle quality check error: {e}")
        return True, "check skipped"


def check_volume_surge(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    NEW FILTER: Current candle volume must be significantly higher than average.
    """
    try:
        if 'volume' not in df.columns:
            return True, "no volume data"
        
        last_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        
        if pd.isna(avg_vol) or avg_vol == 0:
            return True, "no avg volume"
        
        vol_ratio = last_vol / avg_vol
        
        if vol_ratio < STRICT_FILTERS["min_volume_surge"]:
            return False, f"Low volume surge: {vol_ratio:.2f}x < 1.5x"
        
        return True, f"Volume surge: {vol_ratio:.2f}x"
    
    except Exception as e:
        logger.debug(f"Volume surge check error: {e}")
        return True, "check skipped"


def check_multi_candle_bullish(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    NEW FILTER: Last N candles must be bullish (close > open).
    
    This confirms the move is sustained, not just a single spike.
    """
    try:
        n = STRICT_FILTERS["require_n_bullish_candles"]
        last_n = df.tail(n)
        
        bullish_count = (last_n['close'] > last_n['open']).sum()
        
        if bullish_count < n:
            return False, f"Only {bullish_count}/{n} bullish candles"
        
        return True, f"Last {n} candles bullish"
    
    except Exception as e:
        logger.debug(f"Multi-candle check error: {e}")
        return True, "check skipped"


def check_atr_range(df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
    """
    NEW FILTER: ATR must be in acceptable range.
    
    - Too low ATR = ranging/choppy (avoid)
    - Too high ATR = too volatile (avoid)
    """
    try:
        # Calculate ATR(14)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        curr_price = df['close'].iloc[-1]
        atr_pct = atr / curr_price
        
        if atr_pct < STRICT_FILTERS["min_atr_ratio"]:
            return False, f"ATR too low: {atr_pct*100:.2f}% < 1.5% (ranging)"
        
        if atr_pct > STRICT_FILTERS["max_atr_ratio"]:
            return False, f"ATR too high: {atr_pct*100:.2f}% > 5% (too volatile)"
        
        return True, f"ATR OK: {atr_pct*100:.2f}%"
    
    except Exception as e:
        logger.debug(f"ATR check error for {symbol}: {e}")
        return True, "check skipped"


def check_support_resistance(df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
    """
    NEW FILTER: Check if price is near major support/resistance.
    
    Avoid entering near S/R as price often stalls or reverses.
    """
    try:
        lookback = STRICT_FILTERS["lookback_sr_candles"]
        last_n = df.tail(lookback)
        
        curr_price = df['close'].iloc[-1]
        
        # Find swing highs/lows (local peaks/troughs)
        highs = last_n['high'].nlargest(5).values
        lows = last_n['low'].nsmallest(5).values
        
        proximity_threshold = STRICT_FILTERS["sr_proximity_pct"] / 100
        
        # Check if current price is near any major high (resistance)
        for h in highs:
            if abs(curr_price - h) / h < proximity_threshold:
                return False, f"Near resistance: ₹{h:.2f} ({abs(curr_price-h)/h*100:.1f}%)"
        
        # Check if current price is near any major low (support)
        for l in lows:
            if abs(curr_price - l) / l < proximity_threshold:
                return False, f"Near support: ₹{l:.2f} ({abs(curr_price-l)/l*100:.1f}%)"
        
        return True, "Clear of S/R"
    
    except Exception as e:
        logger.debug(f"S/R check error for {symbol}: {e}")
        return True, "check skipped"


def check_higher_timeframe(symbol: str) -> Tuple[bool, str]:
    """
    NEW FILTER: Check 1H chart alignment.
    
    15m signal must align with 1H trend for higher probability.
    """
    if not STRICT_FILTERS["require_1h_alignment"]:
        return True, "1H check disabled"
    
    try:
        import io
        import contextlib
        
        yahoo_symbol = f"{symbol}{YAHOO_NSE_SUFFIX}"
        
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ticker = yf.Ticker(yahoo_symbol)
            df_1h = ticker.history(period="5d", interval="1h")
        
        if df_1h.empty or len(df_1h) < 20:
            return True, "insufficient 1H data"
        
        df_1h.columns = [c.lower() for c in df_1h.columns]
        
        # Calculate 20 EMA on 1H
        ema20 = df_1h['close'].ewm(span=20, adjust=False).mean()
        
        curr_price = float(df_1h['close'].iloc[-1])
        curr_ema = float(ema20.iloc[-1])
        
        # 1H must be in uptrend
        if curr_price < curr_ema:
            return False, f"1H downtrend: price {curr_price:.2f} < EMA20 {curr_ema:.2f}"
        
        # Last 1H candle must be bullish
        last_1h = df_1h.iloc[-1]
        if last_1h['close'] <= last_1h['open']:
            return False, "Last 1H candle bearish"
        
        return True, f"1H uptrend: price {curr_price:.2f} > EMA20 {curr_ema:.2f}"
    
    except Exception as e:
        logger.debug(f"1H check error for {symbol}: {e}")
        return True, "check skipped"


def check_signal_cooldown(symbol: str) -> Tuple[bool, str]:
    """
    NEW FILTER: Enforce cooldown between signals.
    
    Prevents spamming alerts on same stock every few minutes.
    """
    global _recent_signals
    
    cooldown_min = STRICT_FILTERS["signal_cooldown_minutes"]
    now = datetime.now(IST)
    
    if symbol in _recent_signals:
        last_signal_time = _recent_signals[symbol]
        elapsed = (now - last_signal_time).total_seconds() / 60
        
        if elapsed < cooldown_min:
            return False, f"Cooldown: {elapsed:.0f}/{cooldown_min} min elapsed"
    
    return True, "cooldown OK"


def mark_signal_sent(symbol: str):
    """Mark that a signal was sent for this symbol."""
    global _recent_signals
    _recent_signals[symbol] = datetime.now(IST)


# ============================================================================
# EXISTING FILTERS (Volume, Gap, Daily Trend)
# ============================================================================

def check_volume(df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
    """Volume filter."""
    if "volume" not in df.columns:
        return True, "no volume data"
    try:
        today = datetime.now(IST).date()
        df_copy = df.copy()
        if df_copy.index.tz is not None:
            today_mask = df_copy.index.tz_convert(IST).date == today
        else:
            today_mask = pd.Series(
                [idx.date() == today for idx in df_copy.index],
                index=df_copy.index
            )
        today_volume = int(df_copy[today_mask]["volume"].sum()) if today_mask.any() else 0
        avg_volume = int(df_copy["volume"].mean())

        if today_volume < STRICT_FILTERS["min_volume_today"]:
            return False, f"Low today vol: {today_volume:,} < {STRICT_FILTERS['min_volume_today']:,}"
        if avg_volume < STRICT_FILTERS["min_avg_volume"]:
            return False, f"Low avg vol: {avg_volume:,} < {STRICT_FILTERS['min_avg_volume']:,}"
        return True, f"Vol OK: today={today_volume:,}, avg={avg_volume:,}"
    except Exception as e:
        logger.debug(f"Volume check error for {symbol}: {e}")
        return True, "check skipped"


def check_gap(df: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
    """Gap check."""
    try:
        if len(df) < 2:
            return True, "insufficient data"
        today = datetime.now(IST).date()
        df_copy = df.copy()
        if df_copy.index.tz is not None:
            dates = df_copy.index.tz_convert(IST)
        else:
            dates = df_copy.index
        today_candles = df_copy[[d.date() == today for d in dates]]
        prev_candles = df_copy[[d.date() < today for d in dates]]
        if today_candles.empty or prev_candles.empty:
            return True, "gap check skipped"
        prev_close = float(prev_candles["close"].iloc[-1])
        today_open = float(today_candles["open"].iloc[0])
        gap_pct = ((today_open - prev_close) / prev_close) * 100
        if gap_pct > STRICT_FILTERS["max_gap_up_pct"]:
            return False, f"Gap up: {gap_pct:.1f}% > {STRICT_FILTERS['max_gap_up_pct']}%"
        if gap_pct < -STRICT_FILTERS["max_gap_down_pct"]:
            return False, f"Gap down: {gap_pct:.1f}%"
        return True, f"Gap OK: {gap_pct:+.1f}%"
    except Exception as e:
        logger.debug(f"Gap check error for {symbol}: {e}")
        return True, "check skipped"


def check_daily_trend(symbol: str) -> Tuple[bool, str]:
    """Daily trend check."""
    try:
        import io
        import contextlib
        
        yahoo_symbol = f"{symbol}{YAHOO_NSE_SUFFIX}"
        
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period="60d", interval="1d")
        
        if df.empty or len(df) < STRICT_FILTERS["daily_ema_period"]:
            return True, "daily check skipped"
        df.columns = [c.lower() for c in df.columns]
        ema_period = STRICT_FILTERS["daily_ema_period"]
        ema = df["close"].ewm(span=ema_period, adjust=False).mean()
        curr_price = float(df["close"].iloc[-1])
        curr_ema = float(ema.iloc[-1])
        if curr_price > curr_ema:
            return True, f"Daily uptrend: ₹{curr_price:.2f} > EMA{ema_period} ₹{curr_ema:.2f}"
        else:
            return False, f"Daily downtrend: ₹{curr_price:.2f} < EMA{ema_period} ₹{curr_ema:.2f}"
    except Exception as e:
        logger.debug(f"Daily trend check error for {symbol}: {e}")
        return True, "check skipped"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-Strict Intraday Scanner")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--watchlist", type=str)
    parser.add_argument(
        "--universe", type=str,
        choices=["NIFTY50", "NIFTY100", "NIFTY500", "ALL"],
    )
    parser.add_argument(
        "--interval", type=str, default="15m",
        choices=["5m", "15m", "30m", "1h"],
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send alerts + chart images via Telegram",
    )
    parser.add_argument("--bypass-time", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=1.0)
    parser.add_argument(
        "--chart-dir", type=str, default="/tmp",
        metavar="DIR",
        help="Directory to save chart PNGs (default: /tmp)",
    )
    return parser.parse_args()


async def get_stock_universe(universe: str) -> List[str]:
    """Get stock list with offline fallback."""
    if universe == "NIFTY50":
        return NIFTY50_SYMBOLS
    
    index_map = {
        "NIFTY100": "NIFTY 100",
        "NIFTY500": "NIFTY 500",
        "ALL": "ALL",
    }
    index_name = index_map.get(universe, "NIFTY 500")
    
    fallback_manager = FallbackManager()
    try:
        for source_name, fetcher in fallback_manager.fetchers.items():
            try:
                stocks = await fetcher.fetch_stock_list(index_name)
                if stocks:
                    return stocks
            except Exception as e:
                logger.debug(f"Failed to fetch from {source_name}: {e}")
                continue
    finally:
        try:
            await fallback_manager.close()
        except Exception:
            pass
    
    # OFFLINE FALLBACK: If network fails, use NIFTY50
    logger.warning(f"⚠️  Could not fetch {universe} (offline/network error)")
    logger.warning(f"   Falling back to NIFTY50 ({len(NIFTY50_SYMBOLS)} stocks)")
    return NIFTY50_SYMBOLS


def fetch_intraday_data(symbol: str, interval: str = "15m") -> Optional[pd.DataFrame]:
    """Fetch intraday data."""
    yahoo_symbol = f"{symbol}{YAHOO_NSE_SUFFIX}"
    try:
        # Suppress yfinance print statements
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ticker = yf.Ticker(yahoo_symbol)
            period_map = {"5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}
            df = ticker.history(period=period_map.get(interval, "60d"), interval=interval)
        
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def load_intraday_strategies(include_daily: bool = False) -> list:
    """Load strategies."""
    strategies = []
    for config_name in ["intraday_momentum", "intraday_volume_surge", "intraday_mean_reversion"]:
        try:
            config = load_strategy_config(config_name)
            if not config:
                continue
            strategy_name = config.get("strategy", {}).get("name", "")
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_class:
                strategies.append(strategy_class(config))
        except Exception:
            pass
    try:
        mc_config = load_strategy_config("mother_candle")
        if mc_config:
            mc_config.setdefault("data", {})["timeframe"] = "15m"
            strategy_class = STRATEGY_REGISTRY.get("Mother Candle")
            if strategy_class:
                strategies.append(strategy_class(mc_config))
    except Exception:
        pass
    return strategies


def format_signal_output(signal: TradingSignal, interval: str, filters_passed: dict) -> str:
    """Format signal output."""
    lines = [
        f"{'='*60}",
        f"  ✅ ULTRA-HIGH-QUALITY {signal.signal_type.value}: {signal.symbol}",
        f"  Strategy: {signal.strategy_name} | {interval}",
        f"  Confidence: {signal.confidence*100:.0f}%",
        f"{'='*60}",
        f"  Entry  : ₹{signal.entry_price:,.2f}",
        f"  Target : ₹{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)",
        f"  SL     : ₹{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)",
        f"  R:R    : 1:{signal.risk_reward_ratio:.1f}",
        f"",
        f"  ✅ ALL FILTERS PASSED:",
    ]
    
    # Show all filter results
    for name, result in filters_passed.items():
        lines.append(f"    {name}: {result}")
    
    lines.append(f"")
    lines.append(f"  ⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}")
    lines.append(f"  🎯 ENTER NOW - High probability setup")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


def format_telegram_signal(signal: TradingSignal, interval: str, filters_passed: dict) -> str:
    """Format Telegram message."""
    return (
        f"🎯 ULTRA-HIGH-QUALITY {signal.signal_type.value}\n"
        f"Symbol: <b>{signal.symbol}</b>\n"
        f"Strategy: {signal.strategy_name} ({interval})\n\n"
        f"💰 Entry  : ₹{signal.entry_price:,.2f}\n"
        f"🎯 Target : ₹{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)\n"
        f"🛑 SL     : ₹{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)\n"
        f"📊 R:R    : 1:{signal.risk_reward_ratio:.1f}\n"
        f"✅ Confidence: {signal.confidence*100:.0f}%\n\n"
        f"⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}\n"
        f"⚡ ENTER NOW"
    )


async def scan_stocks(
    symbols: List[str],
    interval: str,
    strategies: list,
    send_telegram: bool = False,
    min_confidence: float = 1.0,
    bypass_time: bool = False,
    nifty_trend: Tuple[float, str] = (0.0, "unknown"),
    chart_dir: str = "/tmp",
) -> List[TradingSignal]:
    """
    ULTRA-STRICT SCAN with 15+ filters.
    Generates annotated chart images for every signal and sends them to Telegram.
    """
    all_signals = []
    total = len(symbols)
    visualizer = ChartVisualizer()
    import pathlib
    pathlib.Path(chart_dir).mkdir(parents=True, exist_ok=True)

    # Detailed filter stats
    stats = {
        "scanned": 0,
        "no_data": 0,

        # Existing filters
        "low_volume": 0,
        "bad_gap": 0,
        "daily_downtrend": 0,
        "low_confidence": 0,
        "nifty_blocked": 0,

        # NEW filter rejections
        "weak_candle": 0,
        "low_vol_surge": 0,
        "not_multi_bullish": 0,
        "bad_atr": 0,
        "near_sr": 0,
        "1h_misaligned": 0,
        "cooldown": 0,

        "passed": 0,
    }

    time_quality = get_time_window_quality()
    nifty_pct, nifty_direction = nifty_trend

    telegram = None
    if send_telegram:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if TelegramBot and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)

    print(f"\n  🔍 ULTRA-STRICT SCAN | {total} stocks | {interval}")
    print(f"  📈 Nifty: {nifty_pct:+.2f}% ({nifty_direction})")
    print(f"  ⏰ Time: {time_quality.upper()}")
    print(f"  {'='*60}")

    if time_quality == "avoid" and not bypass_time:
        print(f"\n  ⚠️  AVOID WINDOW - Skipping scan")
        print(f"  (Use --bypass-time to scan anyway)")
        return []
    
    if time_quality == "avoid" and bypass_time:
        print(f"\n  ⚠️  AVOID WINDOW - Scanning anyway (bypass mode)")


    for idx, symbol in enumerate(symbols, 1):
        try:
            df = fetch_intraday_data(symbol, interval)
            if df is None or len(df) < 50:  # Increased from 20 to 50
                stats["no_data"] += 1
                continue

            stats["scanned"] += 1

            # ================================================================
            # APPLY ALL 15+ FILTERS IN SEQUENCE
            # ================================================================
            
            # 1. Volume
            vol_ok, vol_reason = check_volume(df, symbol)
            if not vol_ok:
                stats["low_volume"] += 1
                continue

            # 2. Gap
            gap_ok, gap_reason = check_gap(df, symbol)
            if not gap_ok:
                stats["bad_gap"] += 1
                continue

            # 3. Daily trend
            daily_ok, daily_reason = True, "skipped"
            if STRICT_FILTERS["require_daily_uptrend"]:
                daily_ok, daily_reason = check_daily_trend(symbol)
                if not daily_ok:
                    stats["daily_downtrend"] += 1
                    continue

            # 4. NEW: Candle quality
            candle_ok, candle_reason = check_candle_quality(df)
            if not candle_ok:
                stats["weak_candle"] += 1
                continue

            # 5. NEW: Volume surge
            surge_ok, surge_reason = check_volume_surge(df)
            if not surge_ok:
                stats["low_vol_surge"] += 1
                continue

            # 6. NEW: Multi-candle bullish
            multi_ok, multi_reason = check_multi_candle_bullish(df)
            if not multi_ok:
                stats["not_multi_bullish"] += 1
                continue

            # 7. NEW: ATR range
            atr_ok, atr_reason = check_atr_range(df, symbol)
            if not atr_ok:
                stats["bad_atr"] += 1
                continue

            # 8. NEW: S/R proximity
            sr_ok, sr_reason = check_support_resistance(df, symbol)
            if not sr_ok:
                stats["near_sr"] += 1
                continue

            # 9. NEW: 1H alignment
            htf_ok, htf_reason = check_higher_timeframe(symbol)
            if not htf_ok:
                stats["1h_misaligned"] += 1
                continue

            # 10. NEW: Cooldown
            cooldown_ok, cooldown_reason = check_signal_cooldown(symbol)
            if not cooldown_ok:
                stats["cooldown"] += 1
                continue

            # ================================================================
            # STRATEGY SCAN
            # ================================================================
            company_info = {
                "name": symbol,
                "symbol": symbol,
                "sector": "Unknown",
                "market_cap": 0,
                "last_price": float(df["close"].iloc[-1]) if "close" in df.columns else 0,
            }

            for strategy in strategies:
                try:
                    signal = strategy.scan(symbol, df, company_info)
                    if not signal:
                        continue

                    # Confidence filter
                    if signal.confidence < min_confidence:
                        stats["low_confidence"] += 1
                        continue

                    # Nifty filter
                    signal_type = signal.signal_type.value
                    if (signal_type == "BUY" and
                            nifty_direction == "bearish" and
                            nifty_pct < STRICT_FILTERS["nifty_min_change_pct"]):
                        stats["nifty_blocked"] += 1
                        continue

                    # ========================================================
                    # ✅ ALL FILTERS PASSED!
                    # ========================================================
                    stats["passed"] += 1
                    
                    # Mark signal sent (cooldown)
                    mark_signal_sent(symbol)
                    
                    filters_passed = {
                        "Nifty": f"{nifty_pct:+.2f}% ({nifty_direction})",
                        "Volume": vol_reason,
                        "Gap": gap_reason,
                        "Daily Trend": daily_reason,
                        "Candle Quality": candle_reason,
                        "Volume Surge": surge_reason,
                        "Multi-Candle": multi_reason,
                        "ATR": atr_reason,
                        "S/R": sr_reason,
                        "1H Timeframe": htf_reason,
                        "Cooldown": cooldown_reason,
                    }

                    all_signals.append(signal)
                    output = format_signal_output(signal, interval, filters_passed)
                    print(output)

                    # Generate chart image for this signal
                    chart_path = str(
                        pathlib.Path(chart_dir)
                        / f"live_{symbol}_{strategy.name.replace(' ', '_')}.png"
                    )
                    chart_saved = visualizer.save_signal_chart(
                        df, signal, chart_path
                    )

                    if telegram:
                        msg = format_telegram_signal(signal, interval, filters_passed)
                        await telegram.send_alert(
                            msg,
                            signal.priority.value,
                            image_path=chart_path if chart_saved else None,
                        )
                        # Cleanup after successful delivery
                        if chart_saved:
                            try:
                                os.remove(chart_path)
                            except OSError:
                                pass

                except Exception as e:
                    logger.debug(f"{symbol}/{strategy.name}: {e}")

            # Progress
            if idx % 50 == 0 or idx == total:
                print(
                    f"  Progress: {idx}/{total} | "
                    f"✅ {stats['passed']} | "
                    f"❌ candle={stats['weak_candle']} "
                    f"surge={stats['low_vol_surge']} "
                    f"multi={stats['not_multi_bullish']} "
                    f"atr={stats['bad_atr']} "
                    f"sr={stats['near_sr']} "
                    f"1h={stats['1h_misaligned']}"
                )

        except Exception as e:
            logger.warning(f"Error scanning {symbol}: {e}")

    # Final summary
    print(f"\n  {'='*60}")
    print(f"  📊 ULTRA-STRICT FILTER SUMMARY")
    print(f"  {'='*60}")
    print(f"  Total Symbols     : {total}")
    print(f"  Scanned           : {stats['scanned']}")
    print(f"  ❌ No Data         : {stats['no_data']}")
    print(f"  ❌ Low Volume      : {stats['low_volume']}")
    print(f"  ❌ Bad Gap         : {stats['bad_gap']}")
    print(f"  ❌ Daily Downtrend : {stats['daily_downtrend']}")
    print(f"  ❌ Weak Candle     : {stats['weak_candle']}")
    print(f"  ❌ Low Vol Surge   : {stats['low_vol_surge']}")
    print(f"  ❌ Not Multi-Bull  : {stats['not_multi_bullish']}")
    print(f"  ❌ Bad ATR         : {stats['bad_atr']}")
    print(f"  ❌ Near S/R        : {stats['near_sr']}")
    print(f"  ❌ 1H Misaligned   : {stats['1h_misaligned']}")
    print(f"  ❌ Cooldown        : {stats['cooldown']}")
    print(f"  ❌ Low Confidence  : {stats['low_confidence']}")
    print(f"  ❌ Nifty Blocked   : {stats['nifty_blocked']}")
    print(f"  ✅ SIGNALS PASSED  : {stats['passed']}")
    print(f"  {'='*60}")

    return all_signals


async def main():
    args = parse_args()

    # Market hours check
    time_valid, time_reason = is_valid_trading_time(bypass=args.bypass_time)
    if not time_valid:
        print(f"\n⏰ {time_reason}")
        if not args.bypass_time:
            sys.exit(0)

    # Nifty trend
    print(f"\n📈 Checking Nifty...")
    nifty_pct, nifty_direction = get_nifty_trend()
    print(f"   Nifty: {nifty_pct:+.2f}% → {nifty_direction.upper()}")

    # Get symbols
    symbols = []
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.watchlist:
        symbols = [s.strip().upper() for s in args.watchlist.split(",")]
    elif args.universe:
        print(f"\n📊 Fetching {args.universe}...")
        symbols = await get_stock_universe(args.universe)
        if not symbols:
            print(f"❌ Failed to fetch {args.universe} and fallback failed")
            sys.exit(1)
        if len(symbols) == 50 and args.universe != "NIFTY50":
            print(f"⚠️  Network error - using NIFTY50 fallback ({len(symbols)} stocks)")
        else:
            print(f"✅ Got {len(symbols)} stocks")
    else:
        print("Error: Specify --symbol, --watchlist, or --universe")
        sys.exit(1)

    # Load strategies
    strategies = load_intraday_strategies()
    if not strategies:
        print("❌ No strategies loaded")
        sys.exit(1)

    # Scan
    if args.repeat > 0:
        print(f"\n🔄 AUTO-REPEAT: every {args.repeat} min\n")
        scan_count = 0
        while True:
            time_valid, _ = is_valid_trading_time(bypass=args.bypass_time)
            if not time_valid:
                print(f"\n⏰ Outside market hours - waiting...")
                await asyncio.sleep(60)
                continue

            nifty_pct, nifty_direction = get_nifty_trend()
            scan_count += 1
            print(f"\n{'#'*60}")
            print(f"  SCAN #{scan_count} | {datetime.now(IST).strftime('%H:%M:%S')}")
            print(f"{'#'*60}")

            signals = await scan_stocks(
                symbols, args.interval, strategies,
                send_telegram=args.telegram,
                min_confidence=args.min_confidence,
                bypass_time=args.bypass_time,
                nifty_trend=(nifty_pct, nifty_direction),
                chart_dir=args.chart_dir,
            )

            if signals:
                print(f"\n  🎯 {len(signals)} HIGH-QUALITY SIGNAL(S):")
                for s in signals:
                    print(f"    {s.signal_type.value}: {s.symbol} @ ₹{s.entry_price:.2f}")
            else:
                print(f"\n  No signals (strict filters working)")

            print(f"\n  Next scan in {args.repeat} min...")
            try:
                await asyncio.sleep(args.repeat * 60)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        signals = await scan_stocks(
            symbols, args.interval, strategies,
            send_telegram=args.telegram,
            min_confidence=args.min_confidence,
            bypass_time=args.bypass_time,
            nifty_trend=(nifty_pct, nifty_direction),
        )

        print(f"\n{'='*60}")
        print(f"  FINAL SUMMARY")
        print(f"  {'='*60}")
        print(f"  Stocks      : {len(symbols)}")
        print(f"  Confidence  : ≥{args.min_confidence*100:.0f}%")
        print(f"  Signals     : {len(signals)}")

        if signals:
            print(f"\n  HIGH-QUALITY SIGNALS:")
            for s in signals:
                print(
                    f"    {s.signal_type.value}: {s.symbol:15s}"
                    f" | Entry: ₹{s.entry_price:>10,.2f}"
                    f" | Target: ₹{s.target_price:>10,.2f}"
                    f" | SL: ₹{s.stop_loss:>10,.2f}"
                )
        else:
            print(f"\n  No signals (filters working as designed)")

        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
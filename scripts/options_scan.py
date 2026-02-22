"""
Options Trading Scanner

Purpose:
    Scans NIFTY/BANKNIFTY for options trading signals.
    Uses OI analysis, VWAP+Supertrend, and PCR sentiment strategies.
    Fetches option chain data from NSE and 5m candles from Yahoo.
    Generates annotated chart images and sends them to Telegram.

Usage:
    # Scan NIFTY options (single run)
    python scripts/options_scan.py --symbol NIFTY

    # Scan BANKNIFTY options
    python scripts/options_scan.py --symbol BANKNIFTY

    # Scan both indices
    python scripts/options_scan.py --symbol NIFTY --symbol BANKNIFTY

    # Auto-repeat every 5 minutes
    python scripts/options_scan.py --symbol NIFTY --repeat 5

    # With Telegram alerts + chart images
    python scripts/options_scan.py --symbol NIFTY --telegram

    # Dry run (print but don't send Telegram)
    python scripts/options_scan.py --symbol NIFTY --dry-run

    # Custom chart output dir (default /tmp)
    python scripts/options_scan.py --symbol NIFTY --telegram --chart-dir ./charts/options

Environment:
    TELEGRAM_BOT_TOKEN   — bot token
    TELEGRAM_CHAT_ID     — target chat / channel
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.alerts.alert_formatter import AlertFormatter
from src.data_ingestion.option_chain_fetcher import OptionChainFetcher
from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import STRATEGY_REGISTRY
from src.utils.config_loader import load_strategy_config
from src.utils.visualizer import ChartVisualizer

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

IST = pytz.timezone("Asia/Kolkata")
logger = get_logger(__name__)


def _compute_atr14(df: "pd.DataFrame") -> float:
    """Compute ATR-14 (Wilder) from an OHLCV DataFrame."""
    try:
        if len(df) < 15 or not {"high", "low", "close"}.issubset(df.columns):
            return 0.0
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = df["close"].astype(float).shift(1)
        tr = (
            (high - low)
            .combine(abs(high - prev_close), max)
            .combine(abs(low - prev_close), max)
        )
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return 0.0


# ============================================================================
# OPTIONS TRADING CONFIGURATION
# ============================================================================

OPTIONS_CONFIG = {
    # Risk management
    "max_premium_per_lot": 5000,        # Max ₹5000 premium per lot
    "min_premium_per_lot": 100,         # Min ₹100 premium (avoid illiquid)
    "max_delta_long": 0.7,              # Max delta 0.7 for long positions
    "min_delta_long": 0.3,              # Min delta 0.3 (avoid deep OTM)
    
    # Liquidity filters
    "min_oi": 100,                      # Minimum Open Interest
    "min_volume": 50,                   # Minimum volume today
    "max_bid_ask_spread_pct": 2.0,      # Max 2% spread
    
    # IV filters
    "min_iv_percentile": 30,            # Min IV percentile (for buying)
    "max_iv_percentile": 70,            # Max IV percentile (for selling)
    
    # Time decay
    "min_dte": 0,                       # Minimum days to expiry (0 = 0DTE allowed)
    "max_dte": 7,                       # Maximum days to expiry
    
    # Strike selection
    "strikes_around_spot": 10,          # Scan 10 strikes above/below spot
}

# Lot sizes
LOT_SIZES = {
    "NIFTY": 25,        # As of 2024 (verify current)
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Options Trading Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol", type=str, action="append", default=[],
        help="Index symbol (NIFTY, BANKNIFTY). Can specify multiple."
    )
    parser.add_argument(
        "--interval", type=str, default="5m",
        choices=["5m", "15m"],
        help="Candle interval (default: 5m)"
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Auto-repeat every N minutes (0=once)"
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send alerts + chart images via Telegram"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print signals but do NOT send Telegram"
    )
    parser.add_argument(
        "--chart-dir", type=str, default="/tmp",
        metavar="DIR",
        help="Directory to save chart PNG files (default: /tmp)"
    )
    return parser.parse_args()

def get_spot_price(index: str) -> float:
    """
    Get current spot price of index.
    Uses option chain data as it contains underlying value.
    """
    try:
        # Fetch option chain - it contains underlying (spot) price
        chain = nse_optionchain_scrapper(index)
        
        if not chain:
            return 0.0
        
        # Extract underlying value (spot price)
        records = chain.get("records", {})
        underlying_value = records.get("underlyingValue")
        
        if underlying_value:
            return float(underlying_value)
        
        # Alternative: Try strikePrices and find ATM
        # (spot is usually near ATM strike)
        data = records.get("data", [])
        if data and len(data) > 0:
            # Get middle strike as approximation
            strikes = [d.get("strikePrice", 0) for d in data]
            if strikes:
                return float(sorted(strikes)[len(strikes)//2])
        
        return 0.0
        
    except Exception as e:
        print(f"  ❌ Error fetching spot for {index}: {e}")
        # Last resort: Use fallback values (update these daily)
        fallback = {
            "NIFTY": 23500,
            "BANKNIFTY": 48000,
            "FINNIFTY": 22000,
        }
        print(f"  ⚠️  Using fallback spot: ₹{fallback.get(index, 0):,.2f}")
        return fallback.get(index, 0)


def get_expiry_dates(index: str) -> List[str]:
    """Get available expiry dates for index."""
    try:
        # Get option chain
        if index == "NIFTY":
            data = nse_optionchain_scrapper("NIFTY")
        elif index == "BANKNIFTY":
            data = nse_optionchain_scrapper("BANKNIFTY")
        elif index == "FINNIFTY":
            data = nse_optionchain_scrapper("FINNIFTY")
        else:
            return []
        
        # Extract expiry dates
        expiries = data.get("records", {}).get("expiryDates", [])
        return expiries[:4]  # Return next 4 expiries
    
    except Exception as e:
        print(f"  ❌ Error fetching expiries for {index}: {e}")
        return []


def get_options_chain(index: str, expiry: str) -> pd.DataFrame:
    """
    Fetch options chain data for given index and expiry.
    
    Returns DataFrame with columns:
    - strike
    - CE_ltp, CE_bid, CE_ask, CE_oi, CE_volume, CE_iv, CE_delta
    - PE_ltp, PE_bid, PE_ask, PE_oi, PE_volume, PE_iv, PE_delta
    """
    try:
        # Fetch option chain
        print(f"    Fetching option chain for {index}...", end=" ")
        data = nse_optionchain_scrapper(index)
        
        if not data:
            print(f"No data returned")
            return pd.DataFrame()
        
        records = data.get("records", {}).get("data", [])
        
        if not records:
            print(f"No records in data")
            return pd.DataFrame()
        
        print(f"{len(records)} strikes fetched")
        
        # Parse into structured format
        rows = []
        for rec in records:
            # Check if this record is for the requested expiry
            rec_expiry = rec.get("expiryDate", "")
            if rec_expiry != expiry:
                continue
            
            strike = rec.get("strikePrice", 0)
            
            # Call data
            ce = rec.get("CE", {})
            ce_ltp = ce.get("lastPrice", 0)
            ce_bid = ce.get("bidprice", 0)
            ce_ask = ce.get("askPrice", 0)
            ce_oi = ce.get("openInterest", 0)
            ce_vol = ce.get("totalTradedVolume", 0)
            ce_iv = ce.get("impliedVolatility", 0)
            ce_delta = ce.get("delta", 0)
            
            # Put data
            pe = rec.get("PE", {})
            pe_ltp = pe.get("lastPrice", 0)
            pe_bid = pe.get("bidprice", 0)
            pe_ask = pe.get("askPrice", 0)
            pe_oi = pe.get("openInterest", 0)
            pe_vol = pe.get("totalTradedVolume", 0)
            pe_iv = pe.get("impliedVolatility", 0)
            pe_delta = pe.get("delta", 0)
            
            rows.append({
                "strike": strike,
                "CE_ltp": ce_ltp,
                "CE_bid": ce_bid,
                "CE_ask": ce_ask,
                "CE_oi": ce_oi,
                "CE_volume": ce_vol,
                "CE_iv": ce_iv,
                "CE_delta": ce_delta,
                "PE_ltp": pe_ltp,
                "PE_bid": pe_bid,
                "PE_ask": pe_ask,
                "PE_oi": pe_oi,
                "PE_volume": pe_vol,
                "PE_iv": pe_iv,
                "PE_delta": pe_delta,
            })
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            print(f"    ⚠️  No data for expiry {expiry}")
        else:
            print(f"    ✅ {len(df)} strikes for expiry {expiry}")
        
        return df
    
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================================
# OPTIONS ANALYSIS & FILTERING
# ============================================================================

def calculate_moneyness(spot: float, strike: float, option_type: str) -> str:
    """
    Calculate if option is ITM/ATM/OTM.
    
    Returns: "ITM", "ATM", "OTM"
    """
    diff_pct = abs((strike - spot) / spot) * 100
    
    if diff_pct < 1:
        return "ATM"
    
    if option_type == "CE":
        if strike < spot:
            return "ITM"
        else:
            return "OTM"
    else:  # PE
        if strike > spot:
            return "ITM"
        else:
            return "OTM"


def calculate_bid_ask_spread(bid: float, ask: float) -> float:
    """Calculate bid-ask spread percentage."""
    if bid == 0 or ask == 0:
        return 100.0  # Invalid
    mid = (bid + ask) / 2
    if mid == 0:
        return 100.0
    return ((ask - bid) / mid) * 100


def estimate_delta(spot: float, strike: float, option_type: str, days_to_expiry: int) -> float:
    """
    Rough delta estimation (simplified Black-Scholes approximation).
    
    NOTE: This is a rough estimate. Real delta requires proper pricing model.
    """
    moneyness = (spot - strike) / spot
    
    # Time decay factor
    time_factor = max(0.1, days_to_expiry / 7)
    
    if option_type == "CE":
        # Call delta: 0 (deep OTM) → 1 (deep ITM)
        if strike <= spot - (spot * 0.05):  # Deep ITM
            return min(0.9, 0.7 + time_factor * 0.2)
        elif strike >= spot + (spot * 0.05):  # Deep OTM
            return max(0.1, 0.3 - time_factor * 0.2)
        else:  # ATM
            return 0.5
    else:  # PE
        # Put delta: -1 (deep ITM) → 0 (deep OTM)
        if strike >= spot + (spot * 0.05):  # Deep ITM
            return min(-0.1, -0.7 - time_factor * 0.2)
        elif strike <= spot - (spot * 0.05):  # Deep OTM
            return max(-0.9, -0.3 + time_factor * 0.2)
        else:  # ATM
            return -0.5


def filter_liquid_options(df: pd.DataFrame, option_type: str = "CE") -> pd.DataFrame:
    """
    Filter for liquid options based on OI, volume, spread.
    """
    prefix = option_type  # "CE" or "PE"
    
    # Filter by OI
    df = df[df[f"{prefix}_oi"] >= OPTIONS_CONFIG["min_oi"]].copy()
    
    # Filter by volume
    df = df[df[f"{prefix}_volume"] >= OPTIONS_CONFIG["min_volume"]].copy()
    
    # Filter by bid-ask spread
    df["spread"] = df.apply(
        lambda row: calculate_bid_ask_spread(
            row[f"{prefix}_bid"], row[f"{prefix}_ask"]
        ), axis=1
    )
    df = df[df["spread"] <= OPTIONS_CONFIG["max_bid_ask_spread_pct"]].copy()
    
    # Filter by premium range
    df = df[
        (df[f"{prefix}_ltp"] >= OPTIONS_CONFIG["min_premium_per_lot"]) &
        (df[f"{prefix}_ltp"] <= OPTIONS_CONFIG["max_premium_per_lot"])
    ].copy()
    
    return df


def calculate_days_to_expiry(expiry_str: str) -> int:
    """Calculate days to expiry from date string."""
    try:
        # Parse expiry date (format: "19-Feb-2026")
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y")
        today = datetime.now()
        days = (expiry_date - today).days
        return max(0, days)
    except Exception:
        return 0


# ============================================================================
# TRADING STRATEGIES
# ============================================================================

def strategy_trend_following(
    index: str,
    spot: float,
    df: pd.DataFrame,
    expiry: str,
    option_type: str = "CE"
) -> List[Dict]:
    """
    STRATEGY 1: Trend Following
    
    BUY ITM/ATM options when strong trend detected.
    - Look for high OI buildup at ITM strikes
    - Good IV (not too high)
    - Tight spread
    """
    signals = []
    dte = calculate_days_to_expiry(expiry)
    
    # Filter liquid options
    df_filtered = filter_liquid_options(df, option_type)
    
    if df_filtered.empty:
        return signals
    
    prefix = option_type
    
    for _, row in df_filtered.iterrows():
        strike = row["strike"]
        ltp = row[f"{prefix}_ltp"]
        oi = row[f"{prefix}_oi"]
        volume = row[f"{prefix}_volume"]
        iv = row[f"{prefix}_iv"]
        
        moneyness = calculate_moneyness(spot, strike, option_type)
        
        # Only ITM/ATM for trend following
        if moneyness not in ["ITM", "ATM"]:
            continue
        
        # Estimate delta
        delta = estimate_delta(spot, strike, option_type, dte)
        
        # Check delta range
        if option_type == "CE":
            if not (OPTIONS_CONFIG["min_delta_long"] <= delta <= OPTIONS_CONFIG["max_delta_long"]):
                continue
        else:
            if not (-OPTIONS_CONFIG["max_delta_long"] <= delta <= -OPTIONS_CONFIG["min_delta_long"]):
                continue
        
        # OI should be high (liquidity + interest)
        if oi < 500:
            continue
        
        # IV should not be too high (expensive)
        if iv > 50:
            continue
        
        # Calculate risk-reward
        # For calls: SL = premium, Target = 2x premium
        sl = ltp
        target = ltp * 2
        risk_reward = 2.0
        
        signals.append({
            "index": index,
            "strategy": "Trend Following",
            "type": option_type,
            "strike": strike,
            "moneyness": moneyness,
            "entry": ltp,
            "sl": 0,  # Premium loss (full)
            "target": target,
            "rr": risk_reward,
            "oi": oi,
            "volume": volume,
            "iv": iv,
            "delta": delta,
            "expiry": expiry,
            "dte": dte,
        })
    
    return signals


def strategy_breakout(
    index: str,
    spot: float,
    df: pd.DataFrame,
    expiry: str,
    option_type: str = "CE"
) -> List[Dict]:
    """
    STRATEGY 2: Breakout Trading
    
    BUY ATM options on strong directional moves.
    - High volume spike
    - ATM strikes
    - Tight stop loss
    """
    signals = []
    dte = calculate_days_to_expiry(expiry)
    
    df_filtered = filter_liquid_options(df, option_type)
    
    if df_filtered.empty:
        return signals
    
    prefix = option_type
    
    # Find ATM strike
    atm_strike = min(df_filtered["strike"], key=lambda x: abs(x - spot))
    
    row = df_filtered[df_filtered["strike"] == atm_strike]
    if row.empty:
        return signals
    
    row = row.iloc[0]
    
    ltp = row[f"{prefix}_ltp"]
    oi = row[f"{prefix}_oi"]
    volume = row[f"{prefix}_volume"]
    iv = row[f"{prefix}_iv"]
    
    # Volume should be high (breakout confirmation)
    if volume < 200:
        return signals
    
    delta = estimate_delta(spot, atm_strike, option_type, dte)
    
    signals.append({
        "index": index,
        "strategy": "Breakout",
        "type": option_type,
        "strike": atm_strike,
        "moneyness": "ATM",
        "entry": ltp,
        "sl": 0,
        "target": ltp * 1.5,
        "rr": 1.5,
        "oi": oi,
        "volume": volume,
        "iv": iv,
        "delta": delta,
        "expiry": expiry,
        "dte": dte,
    })
    
    return signals


# ============================================================================
# MAIN SCANNER
# ============================================================================

def scan_index_options(
    index: str,
    option_types: List[str] = ["CE", "PE"],
    expiry_filter: str = "weekly"
) -> List[Dict]:
    """
    Scan all options for given index.
    
    Args:
        index: "NIFTY", "BANKNIFTY", "FINNIFTY"
        option_types: ["CE", "PE"] or ["CE"] or ["PE"]
        expiry_filter: "0DTE", "weekly", "monthly", "all"
    
    Returns:
        List of trading signals
    """
    all_signals = []
    
    print(f"\n{'='*60}")
    print(f"  📊 SCANNING {index} OPTIONS")
    print(f"{'='*60}")
    
    # Get spot price
    spot = get_spot_price(index)
    if spot == 0:
        print(f"  ❌ Could not fetch spot price")
        return all_signals
    
    print(f"  Spot Price: ₹{spot:,.2f}")
    
    # Get expiry dates
    expiries = get_expiry_dates(index)
    if not expiries:
        print(f"  ❌ Could not fetch expiries")
        return all_signals
    
    print(f"  Expiries: {', '.join(expiries[:3])}")
    
    # Filter expiries based on preference
    if expiry_filter == "0DTE":
        # Only today's expiry
        expiries = [expiries[0]]
    elif expiry_filter == "weekly":
        # Next 2 weeklies
        expiries = expiries[:2]
    elif expiry_filter == "monthly":
        # Only monthly (last in list usually)
        expiries = [expiries[-1]]
    
    # Scan each expiry
    for expiry in expiries:
        dte = calculate_days_to_expiry(expiry)
        
        # Filter by DTE config
        if dte < OPTIONS_CONFIG["min_dte"] or dte > OPTIONS_CONFIG["max_dte"]:
            continue
        
        print(f"\n  Scanning expiry: {expiry} ({dte} DTE)")
        
        # Fetch options chain
        df = get_options_chain(index, expiry)
        
        if df.empty:
            print(f"    No data")
            continue
        
        print(f"    Strikes available: {len(df)}")
        
        # Apply strategies
        for opt_type in option_types:
            # Strategy 1: Trend Following
            signals = strategy_trend_following(index, spot, df, expiry, opt_type)
            all_signals.extend(signals)
            
            # Strategy 2: Breakout
            signals = strategy_breakout(index, spot, df, expiry, opt_type)
            all_signals.extend(signals)
    
    return all_signals


def format_signal_output(signal: Dict) -> str:
    """Format option signal for console output."""
    lines = [
        f"{'='*60}",
        f"  🎯 {signal['strategy'].upper()} - {signal['type']}",
        f"  {signal['index']} {signal['strike']} {signal['type']} ({signal['moneyness']})",
        f"  Expiry: {signal['expiry']} ({signal['dte']} DTE)",
        f"{'='*60}",
        f"  Entry    : ₹{signal['entry']:.2f}",
        f"  Target   : ₹{signal['target']:.2f} ({(signal['target']/signal['entry']-1)*100:.0f}%)",
        f"  R:R      : 1:{signal['rr']:.1f}",
        f"  Delta    : {signal['delta']:.2f}",
        f"",
        f"  📊 Greeks & Liquidity:",
        f"    IV       : {signal['iv']:.1f}%",
        f"    OI       : {signal['oi']:,}",
        f"    Volume   : {signal['volume']:,}",
        f"",
        f"  💰 Position Size:",
        f"    Lot Size : {LOT_SIZES.get(signal['index'], 0)} qty",
        f"    Premium  : ₹{signal['entry'] * LOT_SIZES.get(signal['index'], 0):,.0f} per lot",
        f"",
        f"  ⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


def format_telegram_signal(signal: Dict) -> str:
    """Format option signal for Telegram."""
    lot_size = LOT_SIZES.get(signal['index'], 0)
    premium_per_lot = signal['entry'] * lot_size
    
    return (
        f"🎯 <b>{signal['strategy'].upper()}</b>\n"
        f"<b>{signal['index']} {signal['strike']} {signal['type']}</b> ({signal['moneyness']})\n\n"
        f"💰 Entry: ₹{signal['entry']:.2f}\n"
        f"🎯 Target: ₹{signal['target']:.2f} (+{(signal['target']/signal['entry']-1)*100:.0f}%)\n"
        f"📊 R:R: 1:{signal['rr']:.1f}\n\n"
        f"📈 Delta: {signal['delta']:.2f}\n"
        f"📉 IV: {signal['iv']:.1f}%\n"
        f"📊 OI: {signal['oi']:,} | Vol: {signal['volume']:,}\n\n"
        f"💵 Premium: ₹{premium_per_lot:,.0f}/lot ({lot_size} qty)\n"
        f"📅 Expiry: {signal['expiry']} ({signal['dte']} DTE)\n\n"
        f"⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}"
    )


async def scan_options(
    symbols: List[str],
    interval: str,
    strategies: list,
    send_telegram: bool = False,
    dry_run: bool = False,
    chart_dir: str = "/tmp",
) -> List[TradingSignal]:
    """Scan indices for options trading signals with chart images."""
    all_signals = []
    oc_fetcher = OptionChainFetcher()
    alert_formatter = AlertFormatter()
    visualizer = ChartVisualizer()
    Path(chart_dir).mkdir(parents=True, exist_ok=True)

    # Telegram setup
    telegram = None
    if send_telegram and not dry_run:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if TelegramBot and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)
            logger.info("Telegram delivery enabled for options scan")
        else:
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "signals will be logged only"
            )
    elif dry_run:
        logger.info("Dry-run mode — Telegram delivery disabled")

    print(f"\nOptions Scanner — {len(symbols)} index/indices, {len(strategies)} strategies")
    print(f"Interval : {interval}  |  Chart dir: {chart_dir}")
    print(f"{'='*55}")

    for symbol in symbols:
        symbol = symbol.upper()
        print(f"\n  Scanning {symbol}...")

        # 1. Fetch option chain
        print("  Fetching option chain...")
        option_chain = await oc_fetcher.fetch_option_chain(symbol)

        if option_chain:
            print(format_option_chain_summary(option_chain))
        else:
            print(f"  WARNING: Could not fetch option chain for {symbol}")
            print("  (NSE may block requests. OI-based strategies will be skipped.)")
            option_chain = {}

        # 2. Fetch intraday price data
        print(f"  Fetching {interval} candle data...")
        df = fetch_index_data(symbol, interval)

        if df is None or len(df) < 20:
            print(f"  ERROR: Insufficient price data for {symbol}")
            continue

        print(
            f"  Got {len(df)} candles, "
            f"latest close: ₹{float(df['close'].iloc[-1]):,.2f}"
        )

        # 3. Build company_info with option chain
        company_info = {
            "name": symbol,
            "symbol": symbol,
            "market_cap": 0,
            "last_price": float(df["close"].iloc[-1]),
            "option_chain": option_chain,
        }

        # 4. Run strategies
        for strategy in strategies:
            try:
                signal = strategy.scan(symbol, df, company_info)
                if not signal:
                    continue

                all_signals.append(signal)

                # Enrich metadata with ATR14 if strategy didn't provide it
                if not signal.metadata.get("atr_pct"):
                    atr14 = _compute_atr14(df)
                    if atr14 > 0:
                        signal.metadata["atr"] = round(atr14, 4)
                        signal.metadata["atr_pct"] = (
                            round(atr14 / signal.entry_price * 100, 2)
                            if signal.entry_price > 0 else 0.0
                        )

                # ── Console output ─────────────────────────────────────
                output = format_options_signal(signal)
                print(output)

                # ── Generate chart image ───────────────────────────────
                chart_path = str(
                    Path(chart_dir)
                    / f"options_{symbol}_{signal.strategy_name.replace(' ', '_')}.png"
                )
                chart_saved = visualizer.save_signal_chart(df, signal, chart_path)

                # ── Format Telegram message ────────────────────────────
                signal_dict = signal.__dict__.copy() if hasattr(signal, "__dict__") else {}
                # Prefer to_dict() if available
                if hasattr(signal, "to_dict"):
                    signal_dict = signal.to_dict()
                # Ensure indicator_details top-level for formatter
                if "indicator_details" not in signal_dict:
                    signal_dict["indicator_details"] = signal.indicator_details
                signal_dict["individual_signals"] = []  # no aggregation for options

                tg_message = alert_formatter.format_options_signal(signal_dict)

                # ── Send via Telegram ──────────────────────────────────
                if telegram and not dry_run:
                    await telegram.send_alert(
                        tg_message,
                        signal.priority.value,
                        image_path=chart_path if chart_saved else None,
                    )
                elif dry_run:
                    logger.info(
                        f"[DRY-RUN] Options signal: {signal.symbol} "
                        f"{signal.metadata.get('option_type', '')} "
                        f"{signal.metadata.get('atm_strike', '')} "
                        f"conf={signal.confidence:.0%}"
                    )

                # Cleanup chart after delivery
                if chart_saved and telegram and not dry_run:
                    try:
                        os.remove(chart_path)
                    except OSError:
                        pass

            except Exception as e:
                logger.debug(f"{symbol}/{strategy.name}: {e}")

    await oc_fetcher.close()
    return all_signals


async def main():
    args = parse_args()

    symbols = args.symbol if args.symbol else ["NIFTY"]

    strategies = load_options_strategies()
    if not strategies:
        print("Error: No options strategies loaded")
        sys.exit(1)

    print(f"\nStrategies: {', '.join(s.name for s in strategies)}")

    if args.repeat > 0:
        print(f"\nAUTO-REPEAT: Scanning every {args.repeat} minutes")
        print("Press Ctrl+C to stop\n")

        scan_count = 0
        
        while True:
            scan_count += 1
            print(f"\n{'#'*55}")
            print(
                f"  OPTIONS SCAN #{scan_count} — "
                f"{datetime.now().strftime('%H:%M:%S IST')}"
            )
            print(f"{'#'*55}")

            signals = await scan_options(
                symbols, args.interval, strategies,
                send_telegram=args.telegram,
                dry_run=args.dry_run,
                chart_dir=args.chart_dir,
            )

            if signals:
                print(f"\n  TOTAL SIGNALS: {len(signals)}")
            else:
                print(f"\n  No signals. Next scan in {args.repeat} min...")

            try:
                time.sleep(args.repeat * 60)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        signals = await scan_options(
            symbols, args.interval, strategies,
            send_telegram=args.telegram,
            dry_run=args.dry_run,
            chart_dir=args.chart_dir,
        )

        print(f"\n{'='*55}")
        print(f"  OPTIONS SCAN SUMMARY")
        print(f"{'='*55}")
        print(f"  Indices Scanned : {len(symbols)}")
        print(f"  Strategies      : {len(strategies)}")
        print(f"  Signals Found   : {len(signals)}")

        if signals:
            for s in signals:
                option_type = s.metadata.get("option_type", "")
                atm = s.metadata.get("atm_strike", 0)
                print(
                    f"    {option_type}: {s.symbol} {atm} "
                    f"| {s.strategy_name} "
                    f"| Spot: Rs.{s.entry_price:,.2f} "
                    f"| Conf: {s.confidence*100:.0f}%"
                )
        else:
            print(f"  No options signals found. Market may be range-bound.")

        print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(main())
"""
NSE OPTIONS SCANNER - OFFLINE MODE

WHAT THIS DOES:
- Works WITHOUT NSE API (uses Yahoo Finance for spot prices)
- Manual expiry/strike configuration
- Greeks calculation using Black-Scholes
- Strategy signals based on spot movement + technical indicators

USAGE:
    # Basic scan
    python scripts/options_scan_offline.py --index BANKNIFTY
    
    # With specific strikes
    python scripts/options_scan_offline.py --index BANKNIFTY --strikes 48000 48100 48200
    
    # Auto-update every 5 min
    python scripts/options_scan_offline.py --index BANKNIFTY --repeat 5

NOTE: This is for ANALYSIS & LEARNING. For live trading, use broker's platform.
"""

import argparse
import asyncio
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
import math

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

IST = pytz.timezone("Asia/Kolkata")

# ============================================================================
# OPTIONS CONFIGURATION
# ============================================================================

# Yahoo Finance symbols for indices
YAHOO_SYMBOLS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^NIFIN",
}

# Lot sizes (verify these - they change periodically)
LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
}

# Next expiries (UPDATE THESE EVERY WEEK!)
# Format: "DD-MMM-YYYY"
NEXT_EXPIRIES = {
    "BANKNIFTY": ["19-Feb-2026", "26-Feb-2026"],  # Wednesday expiries
    "NIFTY": ["20-Feb-2026", "27-Feb-2026"],      # Thursday expiries
    "FINNIFTY": ["18-Feb-2026", "25-Feb-2026"],   # Tuesday expiries
}

# Risk-free rate (India)
RISK_FREE_RATE = 0.065  # 6.5% (approx RBI repo rate)

# ============================================================================
# BLACK-SCHOLES MODEL
# ============================================================================

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option price using Black-Scholes model.
    
    S: Spot price
    K: Strike price
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Volatility (IV)
    option_type: "call" or "put"
    """
    if T <= 0:
        # At expiry
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(price, 0)


def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option Greeks.
    
    Returns: dict with delta, gamma, theta, vega
    """
    if T <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% IV change
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)) / 365
    
    return {
        "delta": round(delta, 3),
        "gamma": round(gamma, 5),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
    }


# ============================================================================
# SPOT PRICE & VOLATILITY
# ============================================================================

def get_spot_price(index: str) -> float:
    """Get current spot price from Yahoo Finance."""
    try:
        symbol = YAHOO_SYMBOLS.get(index)
        if not symbol:
            return 0.0
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        
        if data.empty:
            return 0.0
        
        spot = float(data["Close"].iloc[-1])
        return spot
    
    except Exception as e:
        print(f"  ❌ Error fetching spot for {index}: {e}")
        return 0.0


def calculate_historical_volatility(index: str, days: int = 30) -> float:
    """
    Calculate historical volatility (annualized).
    Used as proxy for IV.
    """
    try:
        symbol = YAHOO_SYMBOLS.get(index)
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days*2}d", interval="1d")
        
        if len(data) < days:
            return 0.20  # Default 20% IV
        
        # Calculate returns
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        
        # Annualized volatility
        vol = data["returns"].std() * np.sqrt(252)
        
        return vol
    
    except Exception:
        return 0.20  # Default


def get_trend_direction(index: str) -> Tuple[str, float]:
    """
    Determine trend direction and strength.
    
    Returns: ("bullish"/"bearish"/"neutral", strength 0-1)
    """
    try:
        symbol = YAHOO_SYMBOLS.get(index)
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5d", interval="15m")
        
        if len(data) < 20:
            return "neutral", 0.5
        
        # Calculate EMAs
        ema20 = data["Close"].ewm(span=20).mean()
        ema50 = data["Close"].ewm(span=50).mean() if len(data) >= 50 else ema20
        
        curr_price = data["Close"].iloc[-1]
        ema20_val = ema20.iloc[-1]
        ema50_val = ema50.iloc[-1]
        
        # Determine trend
        if curr_price > ema20_val > ema50_val:
            # Strong uptrend
            strength = min(1.0, (curr_price - ema50_val) / ema50_val * 100)
            return "bullish", strength
        elif curr_price < ema20_val < ema50_val:
            # Strong downtrend
            strength = min(1.0, (ema50_val - curr_price) / ema50_val * 100)
            return "bearish", strength
        else:
            return "neutral", 0.3
    
    except Exception:
        return "neutral", 0.5


# ============================================================================
# STRIKE GENERATION
# ============================================================================

def generate_strikes(spot: float, index: str, count: int = 10) -> List[float]:
    """
    Generate strike prices around spot.
    
    Uses appropriate intervals for each index.
    """
    # Strike intervals
    intervals = {
        "NIFTY": 50,
        "BANKNIFTY": 100,
        "FINNIFTY": 50,
    }
    
    interval = intervals.get(index, 100)
    
    # Round spot to nearest interval
    atm_strike = round(spot / interval) * interval
    
    # Generate strikes
    strikes = []
    for i in range(-count, count + 1):
        strikes.append(atm_strike + (i * interval))
    
    return sorted(strikes)


# ============================================================================
# OPTION CHAIN SIMULATION
# ============================================================================

def simulate_option_chain(
    index: str,
    spot: float,
    strikes: List[float],
    expiry: str,
    iv: float
) -> pd.DataFrame:
    """
    Simulate option chain using Black-Scholes.
    
    Returns DataFrame with calculated premiums and greeks.
    """
    # Calculate days to expiry
    try:
        expiry_date = datetime.strptime(expiry, "%d-%b-%Y")
        dte = (expiry_date - datetime.now()).days
        T = max(dte / 365, 0.001)  # Years
    except Exception:
        T = 7 / 365  # Default 1 week
        dte = 7
    
    rows = []
    
    for strike in strikes:
        # Calculate Call
        ce_price = black_scholes_price(spot, strike, T, RISK_FREE_RATE, iv, "call")
        ce_greeks = calculate_greeks(spot, strike, T, RISK_FREE_RATE, iv, "call")
        
        # Calculate Put
        pe_price = black_scholes_price(spot, strike, T, RISK_FREE_RATE, iv, "put")
        pe_greeks = calculate_greeks(spot, strike, T, RISK_FREE_RATE, iv, "put")
        
        # Moneyness
        diff_pct = abs((strike - spot) / spot) * 100
        if diff_pct < 1:
            ce_money = pe_money = "ATM"
        elif strike < spot:
            ce_money = "ITM"
            pe_money = "OTM"
        else:
            ce_money = "OTM"
            pe_money = "ITM"
        
        rows.append({
            "strike": strike,
            "CE_price": round(ce_price, 2),
            "CE_delta": ce_greeks["delta"],
            "CE_theta": ce_greeks["theta"],
            "CE_vega": ce_greeks["vega"],
            "CE_money": ce_money,
            "PE_price": round(pe_price, 2),
            "PE_delta": pe_greeks["delta"],
            "PE_theta": pe_greeks["theta"],
            "PE_vega": pe_greeks["vega"],
            "PE_money": pe_money,
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# STRATEGY: DIRECTIONAL TRADING
# ============================================================================

def strategy_directional(
    index: str,
    spot: float,
    df: pd.DataFrame,
    expiry: str,
    trend: str,
    trend_strength: float
) -> List[Dict]:
    """
    STRATEGY: Buy options in trend direction.
    
    - Bullish trend → Buy ATM/ITM Calls
    - Bearish trend → Buy ATM/ITM Puts
    """
    signals = []
    
    if trend == "neutral" or trend_strength < 0.3:
        return signals
    
    # Filter for good strikes
    if trend == "bullish":
        # Look for ITM/ATM calls
        candidates = df[df["CE_money"].isin(["ATM", "ITM"])].copy()
        option_type = "CE"
    else:
        # Look for ITM/ATM puts
        candidates = df[df["PE_money"].isin(["ATM", "ITM"])].copy()
        option_type = "PE"
    
    if candidates.empty:
        return signals
    
    # Pick best strike (highest delta, reasonable premium)
    for _, row in candidates.iterrows():
        price = row[f"{option_type}_price"]
        delta = abs(row[f"{option_type}_delta"])
        theta = abs(row[f"{option_type}_theta"])
        
        # Filters
        if price < 50 or price > 5000:
            continue
        
        if delta < 0.4 or delta > 0.8:
            continue
        
        # Calculate R:R
        sl = price * 0.4  # 40% stop loss
        target = price * 2.0  # 100% profit target
        
        signals.append({
            "index": index,
            "strategy": "Directional",
            "type": option_type,
            "strike": row["strike"],
            "moneyness": row[f"{option_type}_money"],
            "entry": price,
            "sl": price - sl,
            "target": target,
            "delta": row[f"{option_type}_delta"],
            "theta": row[f"{option_type}_theta"],
            "expiry": expiry,
            "trend": trend,
            "strength": trend_strength,
        })
    
    # Return top 3
    return signals[:3]


# ============================================================================
# MAIN SCANNER
# ============================================================================

def scan_index(
    index: str,
    strikes: Optional[List[float]] = None,
    expiry: Optional[str] = None
) -> List[Dict]:
    """Main scanning function."""
    
    print(f"\n{'='*60}")
    print(f"  📊 SCANNING {index} (OFFLINE MODE)")
    print(f"{'='*60}")
    
    # Get spot
    spot = get_spot_price(index)
    if spot == 0:
        print(f"  ❌ Could not fetch spot price")
        return []
    
    print(f"  Spot Price: ₹{spot:,.2f}")
    
    # Get volatility
    iv = calculate_historical_volatility(index)
    print(f"  Historical Vol: {iv*100:.1f}%")
    
    # Get trend
    trend, strength = get_trend_direction(index)
    print(f"  Trend: {trend.upper()} (strength: {strength:.2f})")
    
    # Use provided expiry or next available
    if not expiry:
        expiries = NEXT_EXPIRIES.get(index, [])
        if not expiries:
            print(f"  ❌ No expiry configured. Update NEXT_EXPIRIES in script.")
            return []
        expiry = expiries[0]
    
    print(f"  Expiry: {expiry}")
    
    # Generate strikes if not provided
    if not strikes:
        strikes = generate_strikes(spot, index, count=10)
    
    print(f"  Strikes: {len(strikes)} generated")
    
    # Simulate option chain
    df = simulate_option_chain(index, spot, strikes, expiry, iv)
    
    if df.empty:
        print(f"  ❌ Could not generate option chain")
        return []
    
    # Apply strategies
    signals = strategy_directional(index, spot, df, expiry, trend, strength)
    
    return signals


def format_signal_output(signal: Dict) -> str:
    """Format signal for console."""
    lot_size = LOT_SIZES.get(signal["index"], 0)
    premium = signal["entry"] * lot_size
    
    lines = [
        f"{'='*60}",
        f"  🎯 {signal['strategy'].upper()} - {signal['type']}",
        f"  {signal['index']} {signal['strike']} {signal['type']} ({signal['moneyness']})",
        f"  Expiry: {signal['expiry']}",
        f"{'='*60}",
        f"  Entry    : ₹{signal['entry']:.2f}",
        f"  Target   : ₹{signal['target']:.2f} (+{(signal['target']/signal['entry']-1)*100:.0f}%)",
        f"  Stop Loss: ₹{signal['sl']:.2f} (-40%)",
        f"",
        f"  Greeks:",
        f"    Delta: {signal['delta']:.3f}",
        f"    Theta: {signal['theta']:.2f}/day",
        f"",
        f"  Position:",
        f"    Lot Size: {lot_size} qty",
        f"    Premium : ₹{premium:,.0f}/lot",
        f"",
        f"  Trend: {signal['trend'].upper()} (strength {signal['strength']:.2f})",
        f"",
        f"  ⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Options Scanner (Offline)")
    parser.add_argument(
        "--index",
        type=str,
        default="BANKNIFTY",
        choices=["NIFTY", "BANKNIFTY", "FINNIFTY"],
        help="Index to scan"
    )
    parser.add_argument(
        "--strikes",
        nargs="+",
        type=float,
        help="Custom strikes (space-separated)"
    )
    parser.add_argument(
        "--expiry",
        type=str,
        help="Expiry date (DD-MMM-YYYY)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Auto-repeat every N minutes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n🚀 NSE OPTIONS SCANNER (OFFLINE MODE)")
    print(f"{'='*60}")
    print(f"Index: {args.index}")
    print(f"Mode : Simulation using Black-Scholes")
    print(f"{'='*60}")
    print(f"\n⚠️  NOTE: This uses simulated prices for analysis.")
    print(f"    For live trading, use your broker's platform.")
    print(f"{'='*60}")
    
    def run_scan():
        signals = scan_index(args.index, args.strikes, args.expiry)
        
        if signals:
            print(f"\n{'='*60}")
            print(f"  ✅ FOUND {len(signals)} SIGNALS")
            print(f"{'='*60}")
            
            for signal in signals:
                print(format_signal_output(signal))
        else:
            print(f"\n  No signals (neutral trend or no suitable strikes)")
        
        return len(signals)
    
    if args.repeat > 0:
        print(f"\n🔄 AUTO-REPEAT: every {args.repeat} min\n")
        scan_count = 0
        
        while True:
            scan_count += 1
            print(f"\n{'#'*60}")
            print(f"  SCAN #{scan_count} | {datetime.now(IST).strftime('%H:%M:%S IST')}")
            print(f"{'#'*60}")
            
            count = run_scan()
            
            print(f"\n  Next scan in {args.repeat} min...")
            try:
                time.sleep(args.repeat * 60)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        run_scan()


if __name__ == "__main__":
    main()
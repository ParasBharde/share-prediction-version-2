import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import concurrent.futures
import time
import os
import io
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8343278396:AAGmgn3v_jS-LQNykx6zq609KStj8Fqap5Y")
TELEGRAM_CHAT_ID =  os.environ.get("TELEGRAM_CHAT_ID", "-1003704255112")

# --- NEW FEATURE: MARKET FILTER ---
# Set to True to ONLY scan when Nifty 50 is in an Uptrend.
# Set to False to scan regardless of market conditions.
ENABLE_NIFTY_FILTER = True 

# NSE Ticker Suffix for Yahoo Finance
SUFFIX = ".NS"

# Technical Parameters
MIN_PRICE = 50
MAX_PRICE = 5000
MIN_AVG_VOLUME = 500000
EMA_PERIOD = 50
RSI_PERIOD = 14
RSI_MIN = 50
RSI_MAX = 70

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def send_telegram_alert(message):
    """Sends a formatted message to Telegram."""
    if "YOUR_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        print("[-] Telegram Token not set. Skipping alert.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"[!] Telegram Error: {e}")

def check_market_status():
    """
    Checks if the overall market (Nifty 50) is Bullish or Bearish.
    Returns: True if Bullish, False if Bearish.
    """
    print("[-] Checking Market Sentiment (Nifty 50)...")
    try:
        # Download Nifty 50 Data (^NSEI)
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period="6mo", interval="1d")
        
        if df.empty:
            print("[!] Could not fetch Nifty data. Proceeding with caution.")
            return True # Assume Bullish if data fails (Fallback)

        # Calculate 50 EMA
        df['ema50'] = ta.ema(df['Close'], length=50)
        
        curr_close = df['Close'].iloc[-1]
        curr_ema = df['ema50'].iloc[-1]
        
        if curr_close > curr_ema:
            print(f"✅ MARKET IS BULLISH (Nifty {int(curr_close)} > 50 EMA {int(curr_ema)})")
            return True
        else:
            print(f"❌ MARKET IS BEARISH (Nifty {int(curr_close)} < 50 EMA {int(curr_ema)})")
            return False

    except Exception as e:
        print(f"[!] Market Check Error: {e}. Proceeding...")
        return True

def get_nifty50_hardcoded():
    """Fallback list of Nifty 50 symbols if dynamic fetch fails."""
    symbols = [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
        "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
        "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
        "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
        "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
        "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
        "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
    ]
    return [f"{s}{SUFFIX}" for s in symbols]

def get_nifty500_symbols():
    """Dynamically fetches Nifty 500 symbols from NSE Archives."""
    print("[-] Fetching dynamic stock list from NSE Archives...")
    try:
        url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            symbols = df['Symbol'].tolist()
            return [f"{sym}{SUFFIX}" for sym in symbols]
        else:
            print(f"[!] HTTP Error {response.status_code}. Switching to fallback.")
            return get_nifty50_hardcoded()
            
    except Exception as e:
        print(f"[!] Error fetching list: {e}. Using fallback list (Nifty 50).")
        return get_nifty50_hardcoded()

# ==========================================
# CORE ANALYSIS LOGIC
# ==========================================

def analyze_stock(symbol):
    """Downloads data using the Thread-Safe Ticker Object."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")
        
        if df.empty or len(df) < 60: return None

        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns: return None

        # Indicators
        df['ema50'] = ta.ema(df['close'], length=EMA_PERIOD)
        df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
        df['avg_vol'] = ta.sma(df['volume'], length=20)
        df['range'] = df['high'] - df['low']
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Filter A: Liquidity
        if not (MIN_PRICE <= curr['close'] <= MAX_PRICE): return None
        if pd.isna(curr['avg_vol']) or curr['avg_vol'] < MIN_AVG_VOLUME: return None

        # Filter B: Trend (Uptrend)
        if curr['close'] <= curr['ema50']: return None

        # Filter C: Setup (Volatility)
        is_nr7 = False
        is_inside_bar = False
        
        last_7_ranges = df['range'].tail(7)
        if len(last_7_ranges) == 7 and curr['range'] == last_7_ranges.min():
            is_nr7 = True
            
        if curr['high'] < prev['high'] and curr['low'] > prev['low']:
            is_inside_bar = True
            
        if not (is_nr7 or is_inside_bar): return None

        # Filter D: Momentum
        if not (RSI_MIN <= curr['rsi'] <= RSI_MAX): return None

        # Outputs
        pattern_name = []
        if is_nr7: pattern_name.append("NR7")
        if is_inside_bar: pattern_name.append("Inside Bar")
        pattern_str = " + ".join(pattern_name)

        buffer = curr['close'] * 0.001
        entry_price = round(curr['high'] + buffer, 2)
        stop_loss = round(curr['low'], 2)
        risk = entry_price - stop_loss
        
        if risk <= 0: return None
        
        target1 = round(entry_price + risk, 2)
        target2 = round(entry_price + (risk * 2), 2)
        
        vol_rel = round(curr['volume'] / curr['avg_vol'], 2) if curr['avg_vol'] > 0 else 0

        return {
            "symbol": symbol.replace(".NS", ""),
            "pattern": pattern_str,
            "entry": entry_price,
            "sl": stop_loss,
            "t1": target1,
            "t2": target2,
            "curr_vol": int(curr['volume']),
            "rel_vol": vol_rel
        }

    except Exception as e:
        return None

def worker(symbol):
    return analyze_stock(symbol)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"\n🚀 STARTING NSE BREAKOUT SCANNER :: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("===============================================================")
    
    # 1. Market Filter Check (The New Feature)
    proceed_with_scan = True
    
    if ENABLE_NIFTY_FILTER:
        is_bullish = check_market_status()
        if not is_bullish:
            print("⚠️ ALERT: Market is Bearish. 'ENABLE_NIFTY_FILTER' is ON.")
            print("🛑 STOPPING SCAN to protect capital. See you tomorrow!")
            proceed_with_scan = False
        else:
            print("✅ Market is favorable. Proceeding with scan...")
    
    if proceed_with_scan:
        # 2. Get Symbols
        symbols = get_nifty500_symbols()
        print(f"[-] Loaded {len(symbols)} stocks to scan.")
        
        # 3. Parallel Processing
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(worker, sym): sym for sym in symbols}
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                data = future.result()
                if data:
                    results.append(data)
                    print(f"[+] FOUND: {data['symbol']} ({data['pattern']})")
                
                completed_count += 1
                if completed_count % 50 == 0:
                    print(f"    ... Scanned {completed_count}/{len(symbols)} stocks")

        # 4. Report & Alert
        print("===============================================================")
        print(f"[-] Scan Complete. Found {len(results)} potential breakouts.")
        print(f"[-] Time taken: {round(time.time() - start_time, 2)} seconds")
        
        for setup in results:
            msg = (
                f"🚀 <b>BULLISH BREAKOUT ALERT</b> 🚀\n"
                f"Symbol: <b>{setup['symbol']}</b>\n"
                f"Pattern: {setup['pattern']}\n\n"
                f"✅ BUY ABOVE: {setup['entry']}\n"
                f"🛑 STOP LOSS: {setup['sl']}\n"
                f"🎯 TARGET 1: {setup['t1']}\n"
                f"🎯 TARGET 2: {setup['t2']}\n\n"
                f"📊 Vol: {setup['curr_vol']} | Rel Vol: {setup['rel_vol']}"
            )
            
            send_telegram_alert(msg)
            print(f" -> Alert sent for {setup['symbol']}")
            time.sleep(0.5)
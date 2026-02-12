"""
Mother Candle V2 - Dedicated Scanner

Purpose:
    Runs the Mother Candle V2 (pure price-action breakout) strategy
    on all stocks. Uses the same data pipeline as daily_scan.py but
    runs ONLY the Mother Candle V2 strategy.

Usage:
    # Scan all stocks (uses system config universe: NIFTY500/ALL)
    python scripts/mother_candle_scan.py

    # Force run on non-trading day
    python scripts/mother_candle_scan.py --force

    # Scan specific stock(s)
    python scripts/mother_candle_scan.py --symbol RELIANCE
    python scripts/mother_candle_scan.py --watchlist "RELIANCE,TCS,INFY,HDFCBANK"

    # Scan NIFTY 50 only
    python scripts/mother_candle_scan.py --universe NIFTY50

    # Enable momentum filters (RSI/ADX/EMA)
    python scripts/mother_candle_scan.py --momentum

    # With Telegram alerts
    python scripts/mother_candle_scan.py --telegram
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.mother_candle_v2 import MotherCandleV2Strategy
from src.strategies.strategy_loader import STRATEGY_REGISTRY
from src.utils.config_loader import load_strategy_config
from src.utils.constants import YAHOO_NSE_SUFFIX

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

try:
    from src.alerts.alert_formatter import AlertFormatter
except Exception:
    AlertFormatter = None

try:
    from src.data_ingestion.fallback_manager import FallbackManager
    from src.data_ingestion.data_validator import DataValidator
except Exception:
    FallbackManager = None
    DataValidator = None

try:
    from src.utils.time_helpers import is_trading_day, now_ist
except Exception:
    is_trading_day = None
    now_ist = None

logger = get_logger(__name__)

# NIFTY 50 stocks for quick scanning
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mother Candle V2 - Breakout Scanner"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force run on non-trading days"
    )
    parser.add_argument(
        "--symbol", type=str,
        help="Single stock symbol (e.g., RELIANCE)"
    )
    parser.add_argument(
        "--watchlist", type=str,
        help="Comma-separated symbols"
    )
    parser.add_argument(
        "--universe", type=str,
        choices=["NIFTY50", "NIFTY500", "ALL"],
        help="Predefined stock universe (default: system config)"
    )
    parser.add_argument(
        "--momentum", action="store_true",
        help="Enable momentum filters (RSI/ADX/EMA)"
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send alerts via Telegram"
    )
    return parser.parse_args()


def fetch_daily_data_yahoo(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch 1 year daily OHLCV from Yahoo Finance."""
    yahoo_symbol = f"{symbol}{YAHOO_NSE_SUFFIX}"
    try:
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period="1y", interval="1d")
        if df.empty or len(df) < 50:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.debug(f"Yahoo fetch failed for {symbol}: {e}")
        return None


def format_mc_signal(signal: TradingSignal) -> str:
    """Format Mother Candle V2 signal for console output."""
    meta = signal.metadata
    details = signal.indicator_details
    mc = details.get("mother_candle", {})
    fb = details.get("fresh_breakout", {})

    lines = [
        f"\n{'='*60}",
        f"  MOTHER CANDLE BREAKOUT: {signal.symbol}",
        f"{'='*60}",
        f"  Strategy     : {signal.strategy_name}",
        f"  Mother High  : Rs.{mc.get('mother_high', 0):,.2f}",
        f"  Mother Low   : Rs.{mc.get('mother_low', 0):,.2f}",
        f"  Mother Range : Rs.{mc.get('mother_range', 0):,.2f}",
        f"  Baby Count   : {mc.get('baby_count', 0)} candles inside",
        f"  Mother At    : {mc.get('mother_position', 'N/A')}",
        f"",
        f"  Breakout Close : Rs.{fb.get('breakout_close', 0):,.2f}",
        f"  Break Amount   : Rs.{fb.get('break_amount', 0):,.2f} ({fb.get('break_pct', 0):+.2f}%)",
        f"",
        f"  Entry Price  : Rs.{signal.entry_price:,.2f}",
        f"  Target       : Rs.{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)",
        f"  Stop Loss    : Rs.{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)",
        f"  R:R Ratio    : 1:{signal.risk_reward_ratio:.1f}",
        f"  Confidence   : {signal.confidence*100:.0f}%",
        f"",
        f"  Mother Vol   : {meta.get('mother_vol_ratio', 0):.2f}x (need 1.5x)",
        f"  Breakout Vol : {meta.get('breakout_vol_ratio', 0):.2f}x (need 1.2x)",
        f"  SL Distance  : {meta.get('sl_distance_pct', 0):.1f}%",
    ]

    # Momentum details if present
    mom = details.get("momentum_filters", {})
    if mom:
        lines.append(f"")
        lines.append(f"  Momentum Filters:")
        for name, detail in mom.items():
            icon = "+" if detail.get("passed", False) else "-"
            lines.append(f"    {icon} {name}: {detail}")

    lines.append(f"")
    lines.append(f"  Time: {datetime.now().strftime('%H:%M:%S IST')}")
    lines.append(f"  Action: BUY on next day open / limit at breakout level")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def format_telegram_mc_signal(signal: TradingSignal) -> str:
    """Format signal for Telegram."""
    meta = signal.metadata
    mc = signal.indicator_details.get("mother_candle", {})

    return (
        f"MOTHER CANDLE BREAKOUT: {signal.symbol}\n\n"
        f"Mother: Rs.{mc.get('mother_high',0):,.2f} - Rs.{mc.get('mother_low',0):,.2f}\n"
        f"Baby Candles: {mc.get('baby_count', 0)} days consolidation\n\n"
        f"Entry: Rs.{signal.entry_price:,.2f}\n"
        f"Target: Rs.{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)\n"
        f"SL: Rs.{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)\n"
        f"R:R: 1:{signal.risk_reward_ratio:.1f}\n\n"
        f"Vol: Mother {meta.get('mother_vol_ratio',0):.1f}x | Breakout {meta.get('breakout_vol_ratio',0):.1f}x\n"
        f"Confidence: {signal.confidence*100:.0f}%\n\n"
        f"Action: BUY on next open"
    )


async def scan_with_fallback_manager(
    strategy: MotherCandleV2Strategy,
    universe: str,
    force: bool,
    send_telegram: bool,
) -> List[TradingSignal]:
    """Scan using the full data pipeline (FallbackManager)."""
    from src.utils.config_loader import load_config

    config = load_config("system")
    fallback_manager = FallbackManager()
    data_validator = DataValidator()

    # Get stock list
    index_map = {
        "NIFTY50": "NIFTY 50",
        "NIFTY500": "NIFTY 500",
        "ALL": "ALL",
    }

    all_signals = []

    try:
        if universe == "ALL":
            indices = ["NIFTY 500", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250"]
            stocks = set()
            for idx_name in indices:
                for source_name, fetcher in fallback_manager.fetchers.items():
                    try:
                        s = await fetcher.fetch_stock_list(idx_name)
                        if s:
                            stocks.update(s)
                            break
                    except Exception:
                        continue
            stock_list = sorted(list(stocks))
        else:
            idx_name = index_map.get(universe, "NIFTY 500")
            stock_list = []
            for source_name, fetcher in fallback_manager.fetchers.items():
                try:
                    stock_list = await fetcher.fetch_stock_list(idx_name)
                    if stock_list:
                        break
                except Exception:
                    continue

        if not stock_list:
            print("ERROR: Could not fetch stock list")
            return []

        # Telegram
        telegram = None
        if send_telegram:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if TelegramBot and bot_token and chat_id:
                telegram = TelegramBot(bot_token, chat_id)

        total = len(stock_list)
        print(f"\nMother Candle V2 Scanner")
        print(f"{'='*60}")
        print(f"  Stocks to scan : {total}")
        print(f"  Strategy       : {strategy.name} v{strategy.version}")
        print(f"  Momentum       : {'ON' if strategy.momentum_filters_enabled else 'OFF'}")
        print(f"{'='*60}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        chunk_size = 50

        for i in range(0, total, chunk_size):
            chunk = stock_list[i: i + chunk_size]
            chunk_num = i // chunk_size + 1

            for symbol in chunk:
                try:
                    data = await fallback_manager.fetch_stock_data(
                        symbol, start_date, end_date
                    )
                    if not data or not data.get("records"):
                        continue

                    clean_records = data_validator.clean_records(
                        data["records"], symbol
                    )
                    if len(clean_records) < 50:
                        continue

                    df = pd.DataFrame(clean_records)
                    if "date" in df.columns:
                        df.set_index("date", inplace=True)

                    last_price = float(df["close"].iloc[-1]) if "close" in df.columns else 0
                    company_info = {
                        "name": symbol,
                        "symbol": symbol,
                        "sector": "Unknown",
                        "market_cap": 0,
                        "last_price": last_price,
                    }

                    signal = strategy.scan(symbol, df, company_info)
                    if signal:
                        all_signals.append(signal)
                        output = format_mc_signal(signal)
                        print(output)

                        if telegram:
                            msg = format_telegram_mc_signal(signal)
                            await telegram.send_alert(msg, signal.priority.value)

                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")

            signals_so_far = len(all_signals)
            scanned = min(i + chunk_size, total)
            print(f"  Progress: {scanned}/{total} stocks | {signals_so_far} signals found")

    finally:
        await fallback_manager.close()

    return all_signals


async def scan_with_yahoo(
    strategy: MotherCandleV2Strategy,
    symbols: List[str],
    send_telegram: bool,
) -> List[TradingSignal]:
    """Scan specific symbols using Yahoo Finance directly."""
    all_signals = []

    telegram = None
    if send_telegram:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if TelegramBot and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)

    total = len(symbols)
    print(f"\nMother Candle V2 Scanner (Yahoo)")
    print(f"{'='*60}")
    print(f"  Stocks to scan : {total}")
    print(f"  Strategy       : {strategy.name} v{strategy.version}")
    print(f"  Momentum       : {'ON' if strategy.momentum_filters_enabled else 'OFF'}")
    print(f"{'='*60}")

    for idx, symbol in enumerate(symbols, 1):
        try:
            df = fetch_daily_data_yahoo(symbol)
            if df is None:
                continue

            company_info = {
                "name": symbol,
                "symbol": symbol,
                "sector": "Unknown",
                "market_cap": 0,
                "last_price": float(df["close"].iloc[-1]),
            }

            signal = strategy.scan(symbol, df, company_info)
            if signal:
                all_signals.append(signal)
                output = format_mc_signal(signal)
                print(output)

                if telegram:
                    msg = format_telegram_mc_signal(signal)
                    await telegram.send_alert(msg, signal.priority.value)

        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")

        if idx % 10 == 0 or idx == total:
            print(f"  Progress: {idx}/{total} stocks | {len(all_signals)} signals found")

    return all_signals


async def main():
    args = parse_args()

    # Trading day check
    if not args.force and not args.symbol and not args.watchlist:
        if is_trading_day and now_ist:
            scan_date = now_ist()
            if not is_trading_day(scan_date.date()):
                print("Not a trading day. Use --force to override.")
                sys.exit(0)

    # Load strategy config
    config = load_strategy_config("mother_candle_v2")
    if not config:
        print("ERROR: Could not load mother_candle_v2.yaml config")
        sys.exit(1)

    # Override momentum filter if --momentum flag
    if args.momentum:
        config.setdefault("strategy", {}).setdefault("params", {})[
            "momentum_filters_enabled"
        ] = True

    strategy = MotherCandleV2Strategy(config)

    start_time = time.time()

    # Determine scan mode
    if args.symbol:
        # Single symbol via Yahoo
        symbols = [args.symbol.upper()]
        all_signals = await scan_with_yahoo(strategy, symbols, args.telegram)

    elif args.watchlist:
        # Multiple symbols via Yahoo
        symbols = [s.strip().upper() for s in args.watchlist.split(",")]
        all_signals = await scan_with_yahoo(strategy, symbols, args.telegram)

    elif args.universe == "NIFTY50":
        # NIFTY 50 via Yahoo
        all_signals = await scan_with_yahoo(strategy, NIFTY50_SYMBOLS, args.telegram)

    elif FallbackManager and DataValidator:
        # Full universe via FallbackManager (data pipeline)
        universe = args.universe or "NIFTY500"
        all_signals = await scan_with_fallback_manager(
            strategy, universe, args.force, args.telegram
        )
    else:
        # Fallback to NIFTY50 via Yahoo
        print("FallbackManager not available, scanning NIFTY50 via Yahoo")
        all_signals = await scan_with_yahoo(strategy, NIFTY50_SYMBOLS, args.telegram)

    # Summary
    duration = time.time() - start_time
    stats = strategy.get_scan_stats()

    print(f"\n{'='*60}")
    print(f"  MOTHER CANDLE V2 - SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"  Stocks Scanned    : {stats['total']}")
    print(f"  Pre-filter Reject : {stats['pre_filter_rejected']}")
    print(f"  Insufficient Data : {stats['insufficient_data']}")
    print(f"  No Pattern Found  : {stats['no_pattern']}")
    print(f"  Volume Rejected   : {stats['volume_rejected']}")
    print(f"  R:R Rejected      : {stats['rr_rejected']}")
    print(f"  SIGNALS FOUND     : {stats['signals']}")
    print(f"  Duration          : {duration:.1f}s")

    if all_signals:
        print(f"\n  SIGNALS:")
        for s in all_signals:
            mc = s.indicator_details.get("mother_candle", {})
            print(
                f"    {s.symbol:15s} "
                f"| Babies: {mc.get('baby_count', 0):2d} "
                f"| Entry: Rs.{s.entry_price:>10,.2f} "
                f"| Target: Rs.{s.target_price:>10,.2f} "
                f"| SL: Rs.{s.stop_loss:>10,.2f} "
                f"| R:R 1:{s.risk_reward_ratio:.1f} "
                f"| Vol: {s.metadata.get('breakout_vol_ratio', 0):.1f}x"
            )
    else:
        print(f"\n  No Mother Candle breakouts found today.")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

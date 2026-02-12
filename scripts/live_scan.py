"""
Live / Intraday Stock Scanner

Purpose:
    Scans stocks during market hours using intraday data (5m/15m).
    Runs ALL intraday strategies (+ Mother Candle on intraday candles).
    Supports single stock, watchlist, or full universe scanning.

Usage:
    # Check single stock
    python scripts/live_scan.py --symbol RELIANCE

    # Check multiple stocks
    python scripts/live_scan.py --watchlist "RELIANCE,TCS,INFY,TATASTEEL"

    # Scan NIFTY 50 stocks
    python scripts/live_scan.py --universe NIFTY50

    # Scan with different interval
    python scripts/live_scan.py --symbol RELIANCE --interval 5m

    # Auto-repeat every N minutes
    python scripts/live_scan.py --universe NIFTY50 --repeat 15

    # Run all strategies (daily + intraday) on intraday data
    python scripts/live_scan.py --symbol RELIANCE --all-strategies
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import StrategyLoader, STRATEGY_REGISTRY
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

logger = get_logger(__name__)

# Intraday strategy names
INTRADAY_STRATEGIES = [
    "Mother Candle",  # Works on intraday too
    "Intraday Momentum",
    "Intraday Volume Surge",
    "Intraday Mean Reversion",
]

# NIFTY 50 stocks for quick scanning
# Note: TATAMOTORS may not work on Yahoo (delisted/renamed) - using TATAMOTOR
# Note: M&M uses M%26M on Yahoo Finance
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
        description="Live Intraday Stock Scanner"
    )
    parser.add_argument(
        "--symbol", type=str, help="Single stock symbol (e.g., RELIANCE)"
    )
    parser.add_argument(
        "--watchlist", type=str,
        help="Comma-separated symbols (e.g., RELIANCE,TCS,INFY)"
    )
    parser.add_argument(
        "--universe", type=str, choices=["NIFTY50"],
        help="Predefined stock universe"
    )
    parser.add_argument(
        "--interval", type=str, default="15m",
        choices=["5m", "15m", "30m", "1h"],
        help="Candle interval (default: 15m)"
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Auto-repeat scan every N minutes (0=once)"
    )
    parser.add_argument(
        "--all-strategies", action="store_true",
        help="Include daily strategies too (run on intraday data)"
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send alerts via Telegram"
    )
    return parser.parse_args()


def fetch_intraday_data(symbol: str, interval: str = "15m") -> Optional[pd.DataFrame]:
    """Fetch intraday OHLCV data from Yahoo Finance."""
    yahoo_symbol = f"{symbol}{YAHOO_NSE_SUFFIX}"

    try:
        ticker = yf.Ticker(yahoo_symbol)
        period_map = {
            "5m": "60d",
            "15m": "60d",
            "30m": "60d",
            "1h": "730d",
        }
        df = ticker.history(
            period=period_map.get(interval, "60d"),
            interval=interval,
        )

        if df.empty:
            return None

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Reset index to make datetime a column, then set back
        if df.index.name and df.index.name.lower() in ("date", "datetime"):
            pass  # already has proper index

        return df

    except Exception as e:
        logger.warning(f"Failed to fetch intraday data for {symbol}: {e}")
        return None


def load_intraday_strategies(include_daily: bool = False) -> list:
    """Load intraday strategies (and optionally daily ones too)."""
    strategies = []

    # Load intraday strategy configs and instantiate
    intraday_configs = [
        "intraday_momentum",
        "intraday_volume_surge",
        "intraday_mean_reversion",
    ]

    for config_name in intraday_configs:
        try:
            config = load_strategy_config(config_name)
            if not config:
                continue
            strategy_name = config.get("strategy", {}).get("name", "")
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_class:
                strategy = strategy_class(config)
                strategies.append(strategy)
                logger.info(f"Loaded intraday strategy: {strategy_name}")
        except Exception as e:
            logger.warning(f"Failed to load {config_name}: {e}")

    # Mother Candle works on intraday - always load it
    try:
        mc_config = load_strategy_config("mother_candle")
        if mc_config:
            # Override timeframe for intraday
            mc_config.setdefault("data", {})["timeframe"] = "15m"
            strategy_class = STRATEGY_REGISTRY.get("Mother Candle")
            if strategy_class:
                strategy = strategy_class(mc_config)
                strategies.append(strategy)
                logger.info("Loaded Mother Candle (intraday mode)")
    except Exception as e:
        logger.warning(f"Failed to load Mother Candle: {e}")

    if include_daily:
        # Also load daily strategies (they can still detect patterns on intraday data)
        for config_name in ["momentum_breakout", "volume_surge", "mean_reversion"]:
            try:
                config = load_strategy_config(config_name)
                if not config or not config.get("strategy", {}).get("enabled", False):
                    continue
                strategy_name = config.get("strategy", {}).get("name", "")
                strategy_class = STRATEGY_REGISTRY.get(strategy_name)
                if strategy_class:
                    strategy = strategy_class(config)
                    strategies.append(strategy)
                    logger.info(f"Loaded daily strategy: {strategy_name} (running on intraday data)")
            except Exception as e:
                logger.warning(f"Failed to load {config_name}: {e}")

    return strategies


def format_signal_output(signal: TradingSignal, interval: str) -> str:
    """Format a signal for console and Telegram output."""
    signal_type = signal.signal_type.value
    r_r = signal.risk_reward_ratio

    lines = [
        f"{'='*50}",
        f"  {signal_type} SIGNAL: {signal.symbol}",
        f"  Strategy: {signal.strategy_name}",
        f"  Timeframe: {interval}",
        f"{'='*50}",
        f"  Entry Price : Rs.{signal.entry_price:,.2f}",
        f"  Target      : Rs.{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)",
        f"  Stop Loss   : Rs.{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)",
        f"  R:R Ratio   : 1:{r_r:.1f}",
        f"  Confidence  : {signal.confidence*100:.0f}%",
        f"  Indicators  : {signal.indicators_met}/{signal.total_indicators}",
        f"",
    ]

    # Add indicator details
    for name, detail in signal.indicator_details.items():
        passed = "PASS" if detail.get("passed", False) else "FAIL"
        lines.append(f"    [{passed}] {name}: {detail}")

    lines.append(f"  Time: {datetime.now().strftime('%H:%M:%S IST')}")
    lines.append(f"  Action: ENTER NOW during market hours")
    lines.append(f"{'='*50}")

    return "\n".join(lines)


def format_telegram_signal(signal: TradingSignal, interval: str) -> str:
    """Format signal for Telegram message."""
    signal_type = signal.signal_type.value
    indicators = []
    for name, detail in signal.indicator_details.items():
        icon = "+" if detail.get("passed", False) else "-"
        indicators.append(f"  {icon} {name}")

    return (
        f"INTRADAY {signal_type}: {signal.symbol}\n"
        f"Strategy: {signal.strategy_name} ({interval})\n\n"
        f"Entry: Rs.{signal.entry_price:,.2f}\n"
        f"Target: Rs.{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)\n"
        f"SL: Rs.{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)\n"
        f"R:R: 1:{signal.risk_reward_ratio:.1f}\n"
        f"Confidence: {signal.confidence*100:.0f}%\n\n"
        f"Indicators ({signal.indicators_met}/{signal.total_indicators}):\n"
        + "\n".join(indicators) + "\n\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')} IST\n"
        f"Action: ENTER NOW"
    )


async def scan_stocks(
    symbols: List[str],
    interval: str,
    strategies: list,
    send_telegram: bool = False,
) -> List[TradingSignal]:
    """Scan a list of stocks with all intraday strategies."""
    all_signals = []
    total = len(symbols)

    # Initialize Telegram if needed
    telegram = None
    if send_telegram:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if TelegramBot and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)

    print(f"\nScanning {total} stocks with {len(strategies)} strategies on {interval} candles...")
    print(f"Strategies: {', '.join(s.name for s in strategies)}")
    print(f"{'='*60}")

    for idx, symbol in enumerate(symbols, 1):
        try:
            df = fetch_intraday_data(symbol, interval)
            if df is None or len(df) < 20:
                continue

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
                    if signal:
                        all_signals.append(signal)
                        output = format_signal_output(signal, interval)
                        print(output)

                        # Send Telegram alert
                        if telegram:
                            msg = format_telegram_signal(signal, interval)
                            await telegram.send_alert(msg, signal.priority.value)
                except Exception as e:
                    logger.debug(f"{symbol}/{strategy.name}: {e}")

            # Progress indicator
            if idx % 10 == 0 or idx == total:
                signals_so_far = len(all_signals)
                print(f"  Progress: {idx}/{total} stocks scanned, {signals_so_far} signals found")

        except Exception as e:
            logger.warning(f"Error scanning {symbol}: {e}")

    return all_signals


async def main():
    args = parse_args()

    # Determine stock list
    symbols = []
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.watchlist:
        symbols = [s.strip().upper() for s in args.watchlist.split(",")]
    elif args.universe == "NIFTY50":
        symbols = NIFTY50_SYMBOLS
    else:
        print("Error: Specify --symbol, --watchlist, or --universe")
        sys.exit(1)

    # Load strategies
    strategies = load_intraday_strategies(include_daily=args.all_strategies)
    if not strategies:
        print("Error: No strategies could be loaded")
        sys.exit(1)

    # Single run or repeat mode
    if args.repeat > 0:
        print(f"\nAUTO-REPEAT MODE: Scanning every {args.repeat} minutes")
        print(f"Press Ctrl+C to stop\n")

        scan_count = 0
        while True:
            scan_count += 1
            print(f"\n{'#'*60}")
            print(f"  SCAN #{scan_count} at {datetime.now().strftime('%H:%M:%S IST')}")
            print(f"{'#'*60}")

            signals = await scan_stocks(
                symbols, args.interval, strategies, args.telegram
            )

            if signals:
                print(f"\n  TOTAL SIGNALS: {len(signals)}")
                for s in signals:
                    print(f"    {s.signal_type.value}: {s.symbol} ({s.strategy_name}) @ Rs.{s.entry_price:.2f}")
            else:
                print(f"\n  No signals found this scan.")

            print(f"\n  Next scan in {args.repeat} minutes...")
            try:
                await asyncio.sleep(args.repeat * 60)
            except KeyboardInterrupt:
                print("\nScan stopped by user.")
                break
    else:
        # Single run
        signals = await scan_stocks(
            symbols, args.interval, strategies, args.telegram
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"  SCAN SUMMARY")
        print(f"{'='*60}")
        print(f"  Stocks Scanned  : {len(symbols)}")
        print(f"  Interval        : {args.interval}")
        print(f"  Strategies Used : {len(strategies)}")
        print(f"  Signals Found   : {len(signals)}")

        if signals:
            print(f"\n  SIGNALS:")
            for s in signals:
                print(
                    f"    {s.signal_type.value}: {s.symbol:15s} "
                    f"| {s.strategy_name:25s} "
                    f"| Entry: Rs.{s.entry_price:>10,.2f} "
                    f"| Target: Rs.{s.target_price:>10,.2f} "
                    f"| SL: Rs.{s.stop_loss:>10,.2f} "
                    f"| R:R 1:{s.risk_reward_ratio:.1f}"
                )
        else:
            print(f"\n  No signals found. Try again in next candle.")

        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

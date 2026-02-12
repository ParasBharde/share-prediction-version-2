"""
Options Trading Scanner

Purpose:
    Scans NIFTY/BANKNIFTY for options trading signals.
    Uses OI analysis, VWAP+Supertrend, and PCR sentiment strategies.
    Fetches option chain data from NSE and 5m candles from Yahoo.

Usage:
    # Scan NIFTY options
    python scripts/options_scan.py --symbol NIFTY

    # Scan BANKNIFTY options
    python scripts/options_scan.py --symbol BANKNIFTY

    # Scan both indices
    python scripts/options_scan.py --symbol NIFTY --symbol BANKNIFTY

    # Auto-repeat every 5 minutes
    python scripts/options_scan.py --symbol NIFTY --repeat 5

    # With Telegram alerts
    python scripts/options_scan.py --symbol NIFTY --telegram
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from src.data_ingestion.option_chain_fetcher import OptionChainFetcher
from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import STRATEGY_REGISTRY
from src.utils.config_loader import load_strategy_config

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

logger = get_logger(__name__)

# Index to Yahoo symbol mapping
INDEX_YAHOO_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Options Trading Scanner")
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
        help="Send alerts via Telegram"
    )
    return parser.parse_args()


def fetch_index_data(symbol: str, interval: str = "5m") -> Optional[pd.DataFrame]:
    """Fetch intraday OHLCV data for an index from Yahoo Finance."""
    yahoo_symbol = INDEX_YAHOO_MAP.get(symbol.upper(), f"^NSEI")

    try:
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period="60d", interval=interval)

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch index data for {symbol}: {e}")
        return None


def load_options_strategies() -> list:
    """Load all options strategies."""
    strategies = []

    options_configs = [
        "options_oi_breakout",
        "options_vwap_supertrend",
        "options_pcr_sentiment",
    ]

    for config_name in options_configs:
        try:
            config = load_strategy_config(config_name)
            if not config:
                continue
            strategy_name = config.get("strategy", {}).get("name", "")
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_class:
                strategy = strategy_class(config)
                strategies.append(strategy)
                logger.info(f"Loaded options strategy: {strategy_name}")
        except Exception as e:
            logger.warning(f"Failed to load {config_name}: {e}")

    return strategies


def format_options_signal(signal: TradingSignal) -> str:
    """Format an options signal for console output."""
    meta = signal.metadata
    option_type = meta.get("option_type", "UNKNOWN")
    atm_strike = meta.get("atm_strike", 0)

    # Determine CE/PE recommendation
    if "CE" in option_type:
        option_name = f"{signal.symbol} {atm_strike} CE"
        action = "BUY CALL (CE)"
    elif "PE" in option_type:
        option_name = f"{signal.symbol} {atm_strike} PE"
        action = "BUY PUT (PE)"
    else:
        option_name = signal.symbol
        action = signal.signal_type.value

    lines = [
        f"\n{'='*55}",
        f"  OPTIONS SIGNAL: {action}",
        f"  Index: {signal.symbol} | Strategy: {signal.strategy_name}",
        f"{'='*55}",
        f"  Option     : {option_name}",
        f"  Spot Price : Rs.{signal.entry_price:,.2f}",
        f"  Target     : Rs.{signal.target_price:,.2f} ({signal.target_percent:+.1f}%)",
        f"  Stop Loss  : Rs.{signal.stop_loss:,.2f} ({signal.stop_loss_percent:.1f}%)",
        f"  R:R Ratio  : 1:{signal.risk_reward_ratio:.1f}",
        f"  Confidence : {signal.confidence*100:.0f}%",
        f"  Indicators : {signal.indicators_met}/{signal.total_indicators}",
    ]

    # Add key metadata
    if meta.get("pcr"):
        lines.append(f"  PCR        : {meta['pcr']:.3f}")
    if meta.get("resistance"):
        lines.append(f"  Resistance : {meta['resistance']} (Max CALL OI)")
    if meta.get("support"):
        lines.append(f"  Support    : {meta['support']} (Max PUT OI)")
    if meta.get("supertrend_value"):
        lines.append(f"  Supertrend : {meta['supertrend_value']}")
    if meta.get("exit_rule"):
        lines.append(f"  Exit Rule  : {meta['exit_rule']}")

    # Indicator details
    lines.append(f"\n  Indicators:")
    for name, detail in signal.indicator_details.items():
        icon = "+" if detail.get("passed", False) else "-"
        lines.append(f"    {icon} {name}: {detail}")

    lines.append(f"\n  Time: {datetime.now().strftime('%H:%M:%S IST')}")
    lines.append(f"  Action: ENTER NOW")
    lines.append(f"{'='*55}")

    return "\n".join(lines)


def format_option_chain_summary(chain: Dict[str, Any]) -> str:
    """Format option chain summary for display."""
    return (
        f"\n  Option Chain: {chain['symbol']}\n"
        f"  Spot: Rs.{chain['underlying_price']:,.2f}\n"
        f"  Expiry: {chain['current_expiry']}\n"
        f"  PCR: {chain['pcr']:.3f} "
        f"({'Bullish' if chain['pcr'] > 1.2 else 'Bearish' if chain['pcr'] < 0.8 else 'Neutral'})\n"
        f"  Max CALL OI: {chain['max_ce_oi_strike']} ({chain['max_ce_oi']:,}) = Resistance\n"
        f"  Max PUT OI:  {chain['max_pe_oi_strike']} ({chain['max_pe_oi']:,}) = Support\n"
        f"  Range: {chain['support']} - {chain['resistance']}\n"
    )


async def scan_options(
    symbols: List[str],
    interval: str,
    strategies: list,
    send_telegram: bool = False,
) -> List[TradingSignal]:
    """Scan indices for options trading signals."""
    all_signals = []
    oc_fetcher = OptionChainFetcher()

    # Telegram
    telegram = None
    if send_telegram:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if TelegramBot and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)

    print(f"\nOptions Scanner - {len(symbols)} indices, {len(strategies)} strategies")
    print(f"Interval: {interval}")
    print(f"{'='*55}")

    for symbol in symbols:
        symbol = symbol.upper()
        print(f"\n  Scanning {symbol}...")

        # 1. Fetch option chain
        print(f"  Fetching option chain...")
        option_chain = await oc_fetcher.fetch_option_chain(symbol)

        if option_chain:
            print(format_option_chain_summary(option_chain))
        else:
            print(f"  WARNING: Could not fetch option chain for {symbol}")
            print(f"  (NSE may block requests. Strategies needing OI data will be skipped.)")
            option_chain = {}

        # 2. Fetch intraday price data
        print(f"  Fetching {interval} candle data...")
        df = fetch_index_data(symbol, interval)

        if df is None or len(df) < 20:
            print(f"  ERROR: Insufficient price data for {symbol}")
            continue

        print(f"  Got {len(df)} candles, latest close: Rs.{float(df['close'].iloc[-1]):,.2f}")

        # 3. Run strategies
        company_info = {
            "name": symbol,
            "symbol": symbol,
            "market_cap": 0,
            "last_price": float(df["close"].iloc[-1]),
            "option_chain": option_chain,
        }

        for strategy in strategies:
            try:
                signal = strategy.scan(symbol, df, company_info)
                if signal:
                    all_signals.append(signal)
                    output = format_options_signal(signal)
                    print(output)

                    if telegram:
                        await telegram.send_alert(output, signal.priority.value)
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
        print(f"Press Ctrl+C to stop\n")

        scan_count = 0
        while True:
            scan_count += 1
            print(f"\n{'#'*55}")
            print(f"  OPTIONS SCAN #{scan_count} - {datetime.now().strftime('%H:%M:%S IST')}")
            print(f"{'#'*55}")

            signals = await scan_options(
                symbols, args.interval, strategies, args.telegram
            )

            if signals:
                print(f"\n  TOTAL SIGNALS: {len(signals)}")
            else:
                print(f"\n  No signals. Next scan in {args.repeat} min...")

            try:
                await asyncio.sleep(args.repeat * 60)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        signals = await scan_options(
            symbols, args.interval, strategies, args.telegram
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

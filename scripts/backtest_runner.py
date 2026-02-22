"""
Backtest Runner Script

Purpose:
    Runs historical backtests for trading strategies.
    Fetches OHLCV data from Yahoo Finance, runs walk-forward
    simulation via the backtesting engine, prints a formatted
    results table, and saves a per-strategy trade log CSV.

Usage:
    # Single strategy, NIFTY50 universe (default)
    python scripts/backtest_runner.py --strategy "Momentum Breakout"

    # All strategies, quick SAMPLE universe (10 stocks)
    python scripts/backtest_runner.py --all --universe SAMPLE

    # Custom symbols and date range
    python scripts/backtest_runner.py --strategy "Flag Pattern" \\
        --symbols RELIANCE,INFY,TCS,HDFCBANK \\
        --start 2023-01-01 --end 2024-12-31

    # All strategies, full NIFTY50, save results to my_results/
    python scripts/backtest_runner.py --all --universe NIFTY50 \\
        --output-dir my_results
"""

import argparse
import asyncio
import csv
import io
import contextlib
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", message=".*possibly delisted.*")
warnings.filterwarnings("ignore", message=".*No data found.*")
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from src.engine.backtester import run_backtest, BacktestResult
from src.monitoring.logger import get_logger
from src.strategies.strategy_loader import StrategyLoader
from src.utils.constants import YAHOO_NSE_SUFFIX

logger = get_logger(__name__)

# ── Stock Universes ────────────────────────────────────────────────────────────
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

# Smaller universe for quick smoke-tests
SAMPLE_SYMBOLS = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "BHARTIARTL", "SBIN", "WIPRO", "AXISBANK", "LT",
]

UNIVERSE_MAP = {
    "NIFTY50": NIFTY50_SYMBOLS,
    "SAMPLE":  SAMPLE_SYMBOLS,
}


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance.

    Args:
        symbol: NSE symbol (e.g. 'RELIANCE').
        start_date: Inclusive start date.
        end_date: Inclusive end date.

    Returns:
        DataFrame with lowercase columns [open, high, low, close, volume],
        or empty DataFrame on failure.
    """
    yahoo_symbol = f"{symbol}{YAHOO_NSE_SUFFIX}"
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                yahoo_symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

        if df.empty:
            return pd.DataFrame()

        # yfinance may return MultiIndex columns when downloading a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()

        return df[list(needed)].dropna()

    except Exception as e:
        logger.debug(f"Could not fetch {symbol}: {e}")
        return pd.DataFrame()


# ── Aggregation & Display ──────────────────────────────────────────────────────

def aggregate_metrics(results_list: List[BacktestResult]) -> Dict:
    """
    Aggregate per-symbol backtest results into a single summary dict.

    Uses average per-trade return as the headline return figure so the
    number is comparable across runs with different stock universes.
    """
    import math
    import numpy as np

    valid = [r for r in results_list if r is not None]
    all_trades = [t for r in valid for t in r.trades]
    all_daily_returns = [ret for r in valid for ret in r.daily_returns]

    if not all_trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_return_per_trade_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
        }

    winners = [t for t in all_trades if t.is_winner]
    losers  = [t for t in all_trades if not t.is_winner]

    win_rate = len(winners) / len(all_trades) * 100

    gross_wins   = sum(t.pnl for t in winners) if winners else 0.0
    gross_losses = abs(sum(t.pnl for t in losers)) if losers else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

    avg_return = (
        sum(t.return_pct for t in all_trades) / len(all_trades)
        if all_trades else 0.0
    )

    # Sharpe from combined daily returns
    sharpe = 0.0
    if len(all_daily_returns) > 1:
        arr = np.array(all_daily_returns)
        std = np.std(arr, ddof=1)
        if std > 0:
            daily_rf = 0.065 / 252
            sharpe = ((np.mean(arr) - daily_rf) / std) * math.sqrt(252)

    max_dd = max(
        (r.metrics.max_drawdown_pct for r in valid),
        default=0.0,
    )

    return {
        "total_trades": len(all_trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": win_rate,
        "avg_return_per_trade_pct": avg_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "profit_factor": profit_factor,
        "total_pnl": round(sum(t.pnl for t in all_trades), 2),
        "total_commission": round(sum(r.metrics.total_commission for r in valid), 2),
        "total_slippage": round(sum(r.metrics.total_slippage for r in valid), 2),
    }


def print_results_table(
    strategy_name: str,
    metrics_by_symbol: Dict[str, Dict],
    overall: Dict,
) -> None:
    """Print a formatted per-symbol and overall results table."""
    print(f"\n{'='*84}")
    print(f"  BACKTEST RESULTS: {strategy_name}")
    print(f"{'='*84}")
    print(
        f"  {'Symbol':<16} {'Trades':>7} {'Win%':>7} "
        f"{'AvgRet%':>9} {'Sharpe':>7} {'MaxDD%':>7} {'PF':>6} {'P&L':>14}"
    )
    print(f"  {'-'*80}")

    # Sort by avg return per trade, skip symbols with 0 trades
    sorted_rows = sorted(
        ((s, m) for s, m in metrics_by_symbol.items() if m["total_trades"] > 0),
        key=lambda x: -x[1]["avg_win_pct"],
    )

    for sym, m in sorted_rows:
        avg_ret = (
            (m["avg_win_pct"] * m["winning_trades"] + m["avg_loss_pct"] * m["losing_trades"])
            / m["total_trades"]
            if m["total_trades"] > 0 else 0.0
        )
        pnl = m["final_portfolio_value"] - m["initial_capital"]
        print(
            f"  {sym:<16} {m['total_trades']:>7} {m['win_rate']:>6.1f}% "
            f"{avg_ret:>+8.2f}% {m['sharpe_ratio']:>7.2f} "
            f"{m['max_drawdown_pct']:>6.1f}% {m['profit_factor']:>6.2f} "
            f"₹{pnl:>12,.0f}"
        )

    if not sorted_rows:
        print(f"  (no trades generated)")

    print(f"  {'-'*80}")
    if overall["total_trades"] > 0:
        print(
            f"  {'OVERALL':<16} {overall['total_trades']:>7} "
            f"{overall['win_rate']:>6.1f}% "
            f"{overall['avg_return_per_trade_pct']:>+8.2f}% "
            f"{overall['sharpe_ratio']:>7.2f} "
            f"{overall['max_drawdown_pct']:>6.1f}% "
            f"{overall['profit_factor']:>6.2f} "
            f"₹{overall['total_pnl']:>12,.0f}"
        )
    print(f"{'='*84}\n")


def save_trades_csv(
    strategy_name: str,
    all_trade_dicts: List[Dict],
    output_dir: Path,
) -> Optional[Path]:
    """
    Save every trade from the backtest to a timestamped CSV file.

    Returns the file path on success, None if there are no trades.
    """
    if not all_trade_dicts:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = strategy_name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"backtest_{safe_name}_{timestamp}.csv"

    fieldnames = [
        "symbol", "strategy_name",
        "entry_date", "entry_price",
        "exit_date", "exit_price",
        "quantity", "side",
        "signal_stop_loss", "signal_target_price",
        "commission", "slippage_cost",
        "pnl", "return_pct", "exit_reason",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_trade_dicts)

    return filepath


# ── Core Backtest Loop ─────────────────────────────────────────────────────────

def run_strategy_backtest(
    strategy,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    output_dir: Path,
) -> Dict:
    """
    Run a backtest for one strategy across all symbols.

    Fetches OHLCV data for each symbol, calls the backtesting engine,
    prints a results table, and saves a trade-log CSV.

    Returns the aggregated overall metrics dict.
    """
    print(f"\n  ─── {strategy.name} {'─'*max(0, 56-len(strategy.name))}")
    print(f"  Period  : {start_date.date()} → {end_date.date()}")
    print(f"  Symbols : {len(symbols)}  |  Capital: ₹{initial_capital:,.0f}")

    metrics_by_symbol: Dict[str, Dict] = {}
    all_trade_dicts: List[Dict] = []
    results_list: List[BacktestResult] = []
    fetched = 0
    skipped = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] Fetching {symbol:<16}", end="\r")
        df = fetch_ohlcv(symbol, start_date, end_date)
        if df.empty:
            skipped += 1
            continue

        fetched += 1
        result = run_backtest(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            df=df,
        )

        results_list.append(result)
        metrics_by_symbol[symbol] = result.metrics.to_dict()
        all_trade_dicts.extend([t.to_dict() for t in result.trades])

    print(f"  Done: {fetched} fetched, {skipped} skipped             ")

    overall = aggregate_metrics(results_list)
    print_results_table(strategy.name, metrics_by_symbol, overall)

    csv_path = save_trades_csv(strategy.name, all_trade_dicts, output_dir)
    if csv_path:
        print(f"  Trade log → {csv_path}")
    else:
        print(f"  (no trades — nothing to save)")

    return overall


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def run(
    strategy_name: Optional[str],
    run_all_strategies: bool,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    output_dir: Path,
) -> None:
    """Load strategies, iterate, collect and print final summary."""
    loader = StrategyLoader()
    all_strategies = loader.load_all()

    if not all_strategies:
        print(
            "❌ No strategies loaded. "
            "Check that config/strategies/*.yaml files exist and are enabled."
        )
        return

    if run_all_strategies:
        targets = all_strategies
    elif strategy_name:
        target = loader.get_strategy(strategy_name)
        if target is None:
            # Try case-insensitive partial match
            for s in all_strategies:
                if strategy_name.lower() in s.name.lower():
                    target = s
                    break
        if target is None:
            print(f"❌ Strategy not found: '{strategy_name}'")
            print(f"   Available: {[s.name for s in all_strategies]}")
            return
        targets = [target]
    else:
        print("Specify --strategy NAME or --all")
        return

    print(f"\n{'='*60}")
    print(f"  BACKTEST RUNNER")
    print(f"  Strategies : {len(targets)}")
    print(f"  Symbols    : {len(symbols)}")
    print(f"  Period     : {start_date.date()} → {end_date.date()}")
    print(f"  Capital    : ₹{initial_capital:,.0f} per symbol")
    print(f"  Output     : {output_dir}/")
    print(f"{'='*60}")

    summary_rows = []
    for strategy in targets:
        overall = run_strategy_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            output_dir=output_dir,
        )
        summary_rows.append({"strategy": strategy.name, **overall})

    # Multi-strategy leaderboard
    if len(targets) > 1:
        print(f"\n{'='*84}")
        print(f"  MULTI-STRATEGY LEADERBOARD  (sorted by avg return per trade)")
        print(f"{'='*84}")
        print(
            f"  {'Strategy':<30} {'Trades':>7} {'Win%':>7} "
            f"{'AvgRet%':>9} {'Sharpe':>7} {'PF':>6} {'P&L':>14}"
        )
        print(f"  {'-'*80}")
        for row in sorted(
            summary_rows,
            key=lambda x: -x["avg_return_per_trade_pct"],
        ):
            print(
                f"  {row['strategy']:<30} {row['total_trades']:>7} "
                f"{row['win_rate']:>6.1f}% "
                f"{row['avg_return_per_trade_pct']:>+8.2f}% "
                f"{row['sharpe_ratio']:>7.2f} "
                f"{row['profit_factor']:>6.2f} "
                f"₹{row['total_pnl']:>12,.0f}"
            )
        print(f"{'='*84}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run strategy backtests against historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Strategy
    parser.add_argument(
        "--strategy", type=str,
        metavar="NAME",
        help='Strategy name, e.g. "Momentum Breakout" or "Flag Pattern"',
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Backtest all enabled strategies",
    )

    # Symbol universe
    parser.add_argument(
        "--universe", type=str, default="NIFTY50",
        choices=list(UNIVERSE_MAP.keys()),
        help="Built-in stock universe (default: NIFTY50 = 50 stocks)",
    )
    parser.add_argument(
        "--symbols", type=str, metavar="SYM1,SYM2,...",
        help="Comma-separated NSE symbols (overrides --universe)",
    )

    # Date range
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        metavar="YYYY-MM-DD", help="Backtest start date (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default="2024-12-31",
        metavar="YYYY-MM-DD", help="Backtest end date (default: 2024-12-31)",
    )

    # Capital & output
    parser.add_argument(
        "--capital", type=float, default=1_000_000.0,
        help="Initial capital per symbol in INR (default: 10,00,000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="backtest_results",
        metavar="DIR",
        help="Directory for CSV trade logs (default: backtest_results/)",
    )

    args = parser.parse_args()

    if not args.strategy and not args.run_all:
        parser.print_help()
        sys.exit(1)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else UNIVERSE_MAP.get(args.universe, NIFTY50_SYMBOLS)
    )

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date   = datetime.strptime(args.end,   "%Y-%m-%d")
    except ValueError as e:
        print(f"❌ Invalid date: {e}")
        sys.exit(1)

    if start_date >= end_date:
        print("❌ --start must be earlier than --end")
        sys.exit(1)

    run(
        strategy_name=args.strategy,
        run_all_strategies=args.run_all,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

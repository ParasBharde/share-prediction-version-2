"""
Daily Stock Scanning Script - FINAL VERSION FOR 2700+ STOCKS

Purpose:
    Main entry point for the daily stock scan.
    Orchestrates data fetching, strategy execution,
    signal generation, chart visualization, and alert delivery.

BTST Timing:
    Scan starts at 15:15 IST (configured in config/system.yaml).
    With 2700+ stocks processed in chunks of 50, signals are ready
    by ~15:25 IST — a 5-minute window to place BTST orders before
    the 15:30 market close.

CRITICAL UPDATE:
    - Added multi-index fetching support
    - When universe="ALL", fetches from 3 indices and deduplicates
    - Gets 900+ unique stocks from archives (NIFTY 500 + MIDCAP + SMALLCAP)

Usage:
    python scripts/daily_scan.py                     # Full scan (all strategies)
    python scripts/daily_scan.py --force             # Skip trading day check
    python scripts/daily_scan.py --btst-only         # Run 4 BTST strategies only
    python scripts/daily_scan.py --mother-v2-only    # Mother Candle V2 only
    python scripts/daily_scan.py --strategies "Darvas Box" "Flag Pattern"
    python scripts/daily_scan.py --test-symbol ICICIBANK
    FORCE_RUN=true python scripts/daily_scan.py

BTST Strategy Names (use with --strategies or --btst-only):
    "Darvas Box"          "Flag Pattern"
    "Symmetrical Triangle"  "Descending Channel"

Dependencies:
    - All src modules

Logging:
    - Scan progress at INFO
    - Results at INFO
    - Failures at ERROR
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.alert_deduplicator import AlertDeduplicator
from src.alerts.alert_formatter import AlertFormatter
from src.utils.visualizer import ChartVisualizer

# Warn immediately (stdout) if chart dependencies are missing or mis-versioned
# so the user doesn't have to dig through log files to understand why images are absent.
try:
    import plotly as _plotly
    import kaleido  # noqa: F401
    _pver = tuple(int(x) for x in _plotly.__version__.split(".")[:2])
    if _pver >= (6, 0):
        print(
            f"\n[WARNING] Chart images DISABLED — plotly {_plotly.__version__} is not compatible "
            "with kaleido 0.2.x.\n"
            "  Fix: pip install \"plotly>=5.17.0,<6.0.0\"\n",
            flush=True,
        )
    del _plotly, _pver
except ImportError as _e:
    print(
        f"\n[WARNING] Chart images DISABLED — missing dependency: {_e}\n"
        "  Fix: pip install \"plotly>=5.17.0,<6.0.0\" \"kaleido>=0.2.1,<1.0.0\"\n",
        flush=True,
    )

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

try:
    from src.paper_trading.paper_trading_engine import PaperTradingEngine
except Exception:
    PaperTradingEngine = None
from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.fallback_manager import FallbackManager
from src.engine.ranking_engine import rank_signals
from src.engine.risk_filter import filter_signals, PortfolioState
from src.engine.signal_aggregator import aggregate_signals
from src.engine.strategy_executor import (
    execute_all,
    _process_single_stock,
    ExecutionResult,
)
from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    job_duration_histogram,
    job_success_counter,
)
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import StrategyLoader
from src.storage.redis_handler import RedisHandler
from src.utils.config_loader import load_config
from src.utils.time_helpers import (
    format_timestamp,
    is_trading_day,
    now_ist,
)

logger = get_logger(__name__)


def _compute_atr14(df: "pd.DataFrame") -> float:
    """
    Compute ATR-14 (Wilder) from an OHLCV DataFrame.

    Returns the ATR as an absolute price value, or 0.0 if the
    DataFrame has fewer than 15 rows or is missing required columns.
    """
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
        # Wilder smoothing: SMA for first value, then EWM
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return 0.0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AlgoTrade Scanner - Daily Stock Scan"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even on non-trading days",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        help="Run specific strategies (space-separated names)",
    )
    parser.add_argument(
        "--mother-v2-only",
        action="store_true",
        help=(
            "Run only Mother Candle V2 strategy. "
            "Useful to replace standalone mother_candle_scan.py runs."
        ),
    )
    parser.add_argument(
        "--btst-only",
        action="store_true",
        help=(
            "Run only the 4 BTST strategies: "
            "Darvas Box, Flag Pattern, Symmetrical Triangle, "
            "Descending Channel. Implies --force."
        ),
    )
    parser.add_argument(
        "--test-symbol",
        type=str,
        help=(
            "Test a single stock symbol (e.g. ICICIBANK). "
            "Fetches data and runs Mother Candle V2 with "
            "detailed output. Implies --force."
        ),
    )
    return parser.parse_args()


async def run_daily_scan(
    force_run: bool = False,
    strategy_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute the full daily stock scanning pipeline.

    Args:
        force_run: If True, skip trading day check.
        strategy_filter: Optional list of strategy names to run.

    Returns:
        Dictionary with scan results summary.
    """
    start_time = time.time()
    scan_date = now_ist()

    logger.info(
        f"Starting daily scan for {scan_date.date()}",
        extra={"scan_date": str(scan_date.date())},
    )

    # Check if trading day (can be bypassed)
    force_env = os.environ.get("FORCE_RUN", "").lower() in (
        "true", "1", "yes",
    )
    should_force = force_run or force_env

    if not should_force and not is_trading_day(scan_date.date()):
        logger.info("Not a trading day, skipping scan (use --force to override)")
        return {"status": "skipped", "reason": "not_trading_day"}

    if should_force and not is_trading_day(scan_date.date()):
        logger.info("Not a trading day, but running anyway (force mode)")

    results = {
        "scan_date": str(scan_date.date()),
        "stocks_scanned": 0,
        "signals_generated": 0,
        "alerts_sent": 0,
        "errors": 0,
    }

    fallback_manager = None

    try:
        # 1. Load configuration
        config = load_config("system")

        # 2. Load DAILY strategies only (not intraday/options)
        strategy_loader = StrategyLoader()
        strategies = strategy_loader.load_by_mode("daily")

        if not strategies:
            logger.warning("No strategies loaded, aborting scan")
            return {**results, "status": "error", "reason": "no_strategies"}

        # 3. Filter strategies if specified
        strategies_config = config.get("scanning", {}).get("strategies", {})
        config_mode = strategies_config.get("mode", "all")
        config_selected = strategies_config.get("selected", [])

        # CLI filter takes priority, then config
        if strategy_filter:
            strategies = [
                s for s in strategies
                if s.name in strategy_filter
            ]
            logger.info(
                f"CLI filter: running {len(strategies)} strategies: "
                f"{[s.name for s in strategies]}"
            )
        elif config_mode == "selected" and config_selected:
            strategies = [
                s for s in strategies
                if s.name in config_selected
            ]
            logger.info(
                f"Config filter: running {len(strategies)} strategies: "
                f"{[s.name for s in strategies]}"
            )

        if not strategies:
            logger.warning("No strategies after filtering, aborting scan")
            return {**results, "status": "error", "reason": "no_strategies_after_filter"}

        logger.info(f"Loaded {len(strategies)} strategies")

        # 4. Initialize components
        fallback_manager = FallbackManager()
        data_validator = DataValidator()
        alert_formatter = AlertFormatter()
        visualizer = ChartVisualizer()

        # Initialize Redis-dependent components (graceful if Redis is down)
        redis_handler = RedisHandler()
        deduplicator = AlertDeduplicator(redis_handler)

        # Initialize Telegram bot from environment
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        telegram = None
        if TelegramBot is not None and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)
        else:
            if TelegramBot is None:
                logger.warning(
                    "python-telegram-bot not installed, "
                    "alerts will be logged only"
                )
            else:
                logger.warning(
                    "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, "
                    "alerts will be logged only"
                )

        # Initialize Paper Trading Engine
        paper_engine = None
        if PaperTradingEngine is not None:
            try:
                paper_engine = PaperTradingEngine()
                logger.info(
                    "Paper trading engine initialized"
                )
            except Exception as e:
                logger.warning(
                    f"Paper trading engine init failed: {e}"
                )
        else:
            logger.warning(
                "Paper trading engine not available"
            )

        # 5. Market context check (Nifty trend + India VIX regime)
        #    Prevents BUY signals on confirmed bearish market days.
        from src.utils.market_context import get_market_context
        market_ctx = await get_market_context(fallback_manager)
        results["market_regime"] = market_ctx["regime"]

        logger.info(
            f"Market regime: {market_ctx['regime']} | "
            f"{market_ctx['reason']}"
        )

        if not market_ctx["allow_buys"] and telegram:
            regime_msg = (
                f"⚠️ MARKET REGIME: *{market_ctx['regime']}*\n"
                f"{market_ctx['reason']}\n\n"
                f"Nifty: {market_ctx['nifty_close']} "
                f"({market_ctx['nifty_vs_ema']:+.1f}% vs 20D EMA)\n"
                f"India VIX: {market_ctx['vix']} "
                f"({market_ctx['vix_regime']})\n\n"
                f"_No BUY signals will be sent today._"
            )
            try:
                await telegram.send_alert(regime_msg, priority="HIGH")
            except Exception:
                pass

        # 6. Get stock list - THIS IS THE CRITICAL PART FOR 2700+ STOCKS
        stock_list = await _get_stock_universe(
            fallback_manager, config
        )

        if not stock_list:
            logger.error("Failed to get stock list")
            return {**results, "status": "error", "reason": "no_stocks"}

        logger.info(f"📊 Scanning {len(stock_list)} stocks")

        # 7. Fetch data and run strategies on all stocks
        all_signals: List[TradingSignal] = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        chunk_size = config.get("scanning", {}).get(
            "chunk_size", 50
        )

        import pandas as pd

        # Track scan stats for visibility
        scan_stats = {
            "no_data": 0,
            "insufficient_records": 0,
            "strategy_scanned": 0,
            "signals_found": 0,
        }

        # Maps symbol -> temp chart image path (generated while df is in scope)
        chart_paths: dict = {}

        for i in range(0, len(stock_list), chunk_size):
            chunk = stock_list[i: i + chunk_size]
            chunk_num = i // chunk_size + 1
            chunk_end = min(i + chunk_size, len(stock_list))
            logger.info(
                f"Processing chunk {chunk_num}: "
                f"stocks {i + 1}-{chunk_end}"
            )

            chunk_signals = 0
            for symbol in chunk:
                try:
                    # Fetch data
                    data = await fallback_manager.fetch_stock_data(
                        symbol, start_date, end_date
                    )

                    if not data or not data.get("records"):
                        scan_stats["no_data"] += 1
                        continue

                    # Validate data
                    clean_records = data_validator.clean_records(
                        data["records"], symbol
                    )

                    if len(clean_records) < 50:
                        scan_stats["insufficient_records"] += 1
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(clean_records)
                    if "date" in df.columns:
                        df.set_index("date", inplace=True)

                    # Run all strategies on this stock
                    last_price = float(df["close"].iloc[-1]) if "close" in df.columns else 0
                    company_info = {
                        "name": symbol,
                        "symbol": symbol,
                        "sector": "Unknown",
                        "market_cap": 0,
                        "last_price": last_price,
                    }

                    scan_stats["strategy_scanned"] += 1
                    signals = _process_single_stock(
                        symbol, df, company_info, strategies
                    )

                    if signals:
                        chunk_signals += len(signals)
                        scan_stats["signals_found"] += len(signals)

                        # Compute ATR14 once per symbol while df is in scope.
                        # BTST strategies (Flag, Darvas, …) don't add ATR to
                        # their metadata, so get_trading_time_info() would
                        # default to atr_pct=0.0 and show "ATR 0.0%".
                        atr14 = _compute_atr14(df)

                        for sig in signals:
                            logger.info(
                                f"SIGNAL: {sig.strategy_name} -> "
                                f"{sig.symbol} ({sig.signal_type.value}) "
                                f"confidence={sig.confidence:.2f} "
                                f"entry={sig.entry_price:.2f} "
                                f"target={sig.target_price:.2f} "
                                f"SL={sig.stop_loss:.2f}"
                            )
                            # Enrich metadata with ATR if strategy didn't provide it
                            if atr14 > 0 and not sig.metadata.get("atr_pct"):
                                sig.metadata["atr"] = round(atr14, 4)
                                sig.metadata["atr_pct"] = round(
                                    atr14 / sig.entry_price * 100, 2
                                ) if sig.entry_price > 0 else 0.0

                            # Generate chart while df is still in scope.
                            # Use asyncio.to_thread so save_signal_chart
                            # (and its internal ThreadPoolExecutor) runs
                            # in a background thread, keeping the event loop
                            # free to process kaleido's subprocess I/O on
                            # Windows (ProactorEventLoop).
                            if symbol not in chart_paths:
                                temp_path = os.path.join(
                                    tempfile.gettempdir(),
                                    f"chart_{symbol}.png",
                                )
                                chart_ok = await asyncio.to_thread(
                                    visualizer.save_signal_chart,
                                    df, sig, temp_path
                                )
                                if chart_ok:
                                    chart_paths[symbol] = temp_path
                                    logger.info(
                                        f"Chart queued for {symbol}: {temp_path}"
                                    )
                                else:
                                    logger.warning(
                                        f"Chart generation FAILED for {symbol} "
                                        f"— alert will be text-only"
                                    )
                    all_signals.extend(signals)
                    results["stocks_scanned"] += 1

                except (asyncio.CancelledError, KeyboardInterrupt):
                    logger.info(
                        "Scan interrupted, returning partial results"
                    )
                    results["status"] = "interrupted"
                    break

                except Exception as e:
                    results["errors"] += 1
                    logger.error(
                        f"Error processing {symbol}: {e}",
                        exc_info=True,
                    )
            else:
                # Inner loop completed normally
                logger.info(
                    f"Chunk {chunk_num} done: "
                    f"{chunk_signals} signals found"
                )
                continue
            # Inner loop was broken (interrupted), break outer too
            break

        # Log scan summary
        logger.info(
            f"Scan stats: {scan_stats['strategy_scanned']} stocks "
            f"analyzed by strategies, "
            f"{scan_stats['no_data']} had no data, "
            f"{scan_stats['insufficient_records']} had <50 records, "
            f"{scan_stats['signals_found']} total signals generated"
        )
        for s in strategies:
            # Log per-strategy stats if available
            if hasattr(s, "get_scan_stats"):
                stats = s.get_scan_stats()
                logger.info(
                    f"Strategy '{s.name}' stats: "
                    f"total={stats.get('total', 0)}, "
                    f"no_pattern={stats.get('no_pattern', 0)}, "
                    f"low_confidence={stats.get('low_confidence', 0)}, "
                    f"pre_filter_rejected={stats.get('pre_filter_rejected', 0)}, "
                    f"signals={stats.get('signals', 0)}"
                )
            else:
                logger.info(
                    f"Strategy '{s.name}': scanned "
                    f"{scan_stats['strategy_scanned']} stocks"
                )

        # 7. Aggregate and rank signals
        if all_signals:
            aggregated = aggregate_signals(all_signals)

            # Enrich aggregated signals with sector data so that the
            # ranking engine's sector diversification cap works correctly.
            # We only fetch sector for the few signal symbols (not all
            # 2700+ stocks), so this is cheap (~10-50 yfinance calls).
            if aggregated:
                signal_symbols = [s.symbol for s in aggregated]
                sector_map = await _fetch_sector_map(signal_symbols)
                for agg_sig in aggregated:
                    sect = sector_map.get(agg_sig.symbol, "Unknown")
                    if sect and sect != "Unknown":
                        for ind_sig in agg_sig.individual_signals:
                            ind_sig.setdefault("metadata", {})["sector"] = sect

            ranked = rank_signals(aggregated)
            filtered = filter_signals(ranked)

            # Apply minimum confidence threshold — only alert on high-quality setups
            min_conf = config.get("scanning", {}).get(
                "min_signal_confidence", 0.65
            )
            before_conf = len(filtered)
            filtered = [
                s for s in filtered if s.weighted_confidence >= min_conf
            ]
            if len(filtered) < before_conf:
                logger.info(
                    f"Confidence filter ({min_conf:.0%}): "
                    f"{before_conf} → {len(filtered)} signals"
                )

            # Apply signal type filter from config
            signal_type_filter = config.get("scanning", {}).get(
                "signal_type_filter", "BOTH"
            ).upper()
            if signal_type_filter != "BOTH":
                before_count = len(filtered)
                filtered = [
                    s for s in filtered
                    if s.signal_type.value == signal_type_filter
                ]
                logger.info(
                    f"Signal type filter '{signal_type_filter}': "
                    f"{before_count} -> {len(filtered)} signals"
                )

            # Apply market context filter
            # In BEARISH regime, suppress BUY/STRONG_BUY signals.
            # (Options BUY_PE are stored as SELL signal_type — those pass through)
            if not market_ctx.get("allow_buys", True):
                before_count = len(filtered)
                filtered = [
                    s for s in filtered
                    if s.signal_type.value
                    not in ("BUY", "STRONG_BUY")
                ]
                suppressed = before_count - len(filtered)
                if suppressed:
                    logger.warning(
                        f"Market regime {market_ctx['regime']}: "
                        f"suppressed {suppressed} BUY signals"
                    )

            results["signals_generated"] = len(filtered)

            # 8. Send alerts and place paper trades
            paper_trade_signals = []
            for signal in filtered:
                try:
                    # Check deduplication — key is symbol + direction so
                    # the same BUY signal doesn't fire 3 days in a row,
                    # but a SELL can still go through after a prior BUY.
                    signal_direction = signal.signal_type.value
                    if deduplicator.is_duplicate(
                        signal.symbol, signal_direction
                    ):
                        logger.debug(
                            f"Skipping duplicate alert: "
                            f"{signal.symbol} ({signal_direction})"
                        )
                        continue

                    # Format alert - convert AggregatedSignal to dict
                    signal_dict = signal.to_dict()
                    # Use RAW confidence from the best individual signal
                    raw_conf = 0.0
                    best_met = "N/A"
                    best_total = "N/A"
                    for ind_sig in signal.individual_signals:
                        ic = ind_sig.get("confidence", 0)
                        if ic > raw_conf:
                            raw_conf = ic
                            best_met = ind_sig.get(
                                "indicators_met", "N/A"
                            )
                            best_total = ind_sig.get(
                                "total_indicators", "N/A"
                            )
                    signal_dict["confidence"] = round(
                        raw_conf * 100, 1
                    )
                    signal_dict["indicators_met"] = best_met
                    signal_dict["total_indicators"] = best_total
                    signal_dict["individual_signals"] = (
                        signal.individual_signals
                    )

                    # Add trading time info
                    if paper_engine:
                        trading_time = (
                            paper_engine.get_trading_time_info(
                                signal_dict
                            )
                        )
                        signal_dict["trading_time"] = trading_time

                    # Collect for paper trading
                    paper_trade_signals.append(signal_dict)

                    message = alert_formatter.format_buy_signal(
                        signal_dict
                    )

                    # Attach chart image when available
                    chart_path = chart_paths.get(signal.symbol)
                    logger.info(
                        f"Sending alert for {signal.symbol}: "
                        f"chart_path={chart_path!r} "
                        f"file_exists={bool(chart_path and os.path.isfile(chart_path))}"
                    )

                    # Send via Telegram (or log if not configured)
                    if telegram:
                        sent = await telegram.send_alert(
                            message,
                            signal.priority.value,
                            image_path=chart_path,
                        )
                    else:
                        logger.info(
                            f"ALERT (no Telegram): {message}"
                        )
                        sent = True

                    if sent:
                        deduplicator.mark_sent(
                            signal.symbol, signal_direction
                        )
                        results["alerts_sent"] += 1

                        # Clean up temp chart file after successful delivery
                        if chart_path and os.path.isfile(chart_path):
                            try:
                                os.remove(chart_path)
                            except OSError:
                                pass

                except Exception as e:
                    logger.error(
                        f"Alert delivery failed for "
                        f"{signal.symbol}: {e}",
                        exc_info=True,
                    )

            # 8b. Place paper trades for all qualifying signals
            if paper_engine and paper_trade_signals:
                try:
                    trade_results = paper_engine.process_signals(
                        paper_trade_signals
                    )
                    placed = [
                        t for t in trade_results
                        if t["status"] == "PLACED"
                    ]
                    results["paper_trades_placed"] = len(placed)
                    logger.info(
                        f"Paper trades: {len(placed)} placed "
                        f"out of {len(paper_trade_signals)} signals"
                    )

                    # Send paper trading summary if trades were placed
                    if placed and telegram:
                        pt_summary = (
                            alert_formatter.format_paper_trade_summary(
                                paper_engine.get_portfolio_summary(),
                                paper_engine.get_session_trades_summary(),
                            )
                        )
                        await telegram.send_alert(
                            pt_summary, "MEDIUM"
                        )
                    elif placed:
                        logger.info(
                            f"PAPER TRADES: {len(placed)} orders "
                            f"placed successfully"
                        )
                except Exception as e:
                    logger.error(
                        f"Paper trading failed: {e}",
                        exc_info=True,
                    )

            # 8c. Update existing positions with live prices
            try:
                from src.storage.postgres_handler import PostgresHandler
                from src.data_ingestion.yahoo_fetcher import YahooFetcher

                postgres = PostgresHandler()
                open_positions = postgres.get_open_positions()
                if open_positions:
                    yahoo = YahooFetcher()
                    sl_hits = []
                    target_hits = []
                    for pos in open_positions:
                        sym = pos["symbol"]
                        try:
                            quote = await yahoo.fetch_quote(sym)
                            if not quote or not quote.get("close"):
                                continue
                            live = round(float(quote["close"]), 2)
                            entry = pos["avg_entry_price"]
                            qty = pos["quantity"]
                            side = pos.get("side", "LONG")
                            is_long = side in (
                                "LONG", "BUY", "STRONG_BUY",
                            )

                            # P&L depends on position direction
                            if is_long:
                                upnl = round(
                                    (live - entry) * qty, 2
                                )
                            else:
                                upnl = round(
                                    (entry - live) * qty, 2
                                )
                            postgres.update_position_price(
                                pos["id"], live, upnl
                            )
                            sl = pos.get("stop_loss", 0)
                            tgt = pos.get("target_price", 0)

                            if is_long:
                                # LONG: SL hit when price drops
                                # below SL
                                if sl > 0 and live <= sl:
                                    rpnl = round(
                                        (sl - entry) * qty, 2
                                    )
                                    postgres.close_position(
                                        pos["id"], live, rpnl,
                                        "SL_HIT",
                                    )
                                    sl_hits.append(
                                        {"symbol": sym, "pnl": rpnl}
                                    )
                                # LONG: Target hit when price rises
                                # above target
                                elif tgt > 0 and live >= tgt:
                                    rpnl = round(
                                        (live - entry) * qty, 2
                                    )
                                    postgres.close_position(
                                        pos["id"], live, rpnl,
                                        "TARGET_HIT",
                                    )
                                    target_hits.append(
                                        {"symbol": sym, "pnl": rpnl}
                                    )
                            else:
                                # SHORT: SL hit when price rises
                                # above SL
                                if sl > 0 and live >= sl:
                                    rpnl = round(
                                        (entry - sl) * qty, 2
                                    )
                                    postgres.close_position(
                                        pos["id"], live, rpnl,
                                        "SL_HIT",
                                    )
                                    sl_hits.append(
                                        {"symbol": sym, "pnl": rpnl}
                                    )
                                # SHORT: Target hit when price drops
                                # below target
                                elif tgt > 0 and live <= tgt:
                                    rpnl = round(
                                        (entry - live) * qty, 2
                                    )
                                    postgres.close_position(
                                        pos["id"], live, rpnl,
                                        "TARGET_HIT",
                                    )
                                    target_hits.append(
                                        {"symbol": sym, "pnl": rpnl}
                                    )
                        except Exception:
                            pass

                    if (sl_hits or target_hits) and telegram:
                        hits_msg = []
                        for h in sl_hits:
                            hits_msg.append(
                                f"SL HIT: {h['symbol']} "
                                f"P&L: \u20b9{h['pnl']:+,.2f}"
                            )
                        for h in target_hits:
                            hits_msg.append(
                                f"TARGET HIT: {h['symbol']} "
                                f"P&L: \u20b9{h['pnl']:+,.2f}"
                            )
                        await telegram.send_alert(
                            "POSITION EXITS\n\n"
                            + "\n".join(hits_msg),
                            "HIGH",
                        )

                    logger.info(
                        f"Updated {len(open_positions)} positions, "
                        f"{len(sl_hits)} SL hits, "
                        f"{len(target_hits)} target hits"
                    )
            except Exception as e:
                logger.warning(
                    f"Position update failed: {e}"
                )

        # 9. Send daily summary
        duration = time.time() - start_time
        results["duration_seconds"] = round(duration, 2)
        if results.get("status") != "interrupted":
            results["status"] = "success"

        # Get paper trading portfolio state for summary
        pt_state = {}
        if paper_engine:
            try:
                pt_state = paper_engine.get_portfolio_summary()
            except Exception:
                pass

        try:
            summary_message = alert_formatter.format_daily_summary(
                {
                    "date": format_timestamp(),
                    "stocks_scanned": results["stocks_scanned"],
                    "signals_count": results["signals_generated"],
                    "alerts_sent": results["alerts_sent"],
                    "scan_duration": results["duration_seconds"],
                    "active_positions": pt_state.get(
                        "open_positions", 0
                    ),
                    "total_pnl_pct": pt_state.get(
                        "total_return_pct", 0
                    ),
                    "paper_trades_placed": results.get(
                        "paper_trades_placed", 0
                    ),
                    "top_signals": [
                        {
                            "symbol": s.symbol,
                            "confidence": round(
                                s.confidence * 100, 1
                            ),
                        }
                        for s in all_signals[:5]
                    ],
                }
            )
            if telegram:
                await telegram.send_alert(summary_message, "LOW")
            else:
                logger.info(f"SUMMARY (no Telegram): {summary_message}")
        except Exception as e:
            logger.error(
                f"Failed to send daily summary: {e}"
            )

        logger.info(
            f"Daily scan completed: {results}",
            extra=results,
        )

        return results

    except Exception as e:
        logger.error(
            "Daily scan failed with unhandled error",
            exc_info=True,
        )
        results["status"] = "error"
        results["error"] = str(e)
        return results

    finally:
        # Always clean up sessions
        if fallback_manager:
            try:
                await fallback_manager.close()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")


async def _get_stock_universe(
    fallback_manager: FallbackManager,
    config: Dict,
) -> List[str]:
    """
    Get the list of stocks to scan based on config.
    
    CRITICAL FUNCTION FOR 2700+ STOCKS:
    When universe="ALL", this fetches from multiple indices and deduplicates.
    
    Args:
        fallback_manager: Data source manager.
        config: System config.

    Returns:
        List of stock symbols.
    """
    universe = config.get("scanning", {}).get(
        "universe", "NIFTY500"
    )

    # Standard index mapping
    index_map = {
        "NIFTY50": "NIFTY 50",
        "NIFTY100": "NIFTY 100",
        "NIFTY500": "NIFTY 500",
        "ALL": "ALL",
    }

    # ========================================================================
    # MULTI-INDEX FETCHING FOR "ALL" UNIVERSE - THIS IS THE KEY PART!
    # ========================================================================
    if universe.upper() == "ALL":
        logger.info("=" * 70)
        logger.info("🎯 FETCHING COMPREHENSIVE STOCK UNIVERSE (ALL MODE)")
        logger.info("=" * 70)
        
        # Indices to combine for comprehensive coverage
        indices_to_fetch = [
            "NIFTY 500",           # 500 large-cap stocks
            "NIFTY MIDCAP 150",    # 150 mid-cap stocks
            "NIFTY SMALLCAP 250",  # 250 small-cap stocks
        ]
        
        all_stocks = set()  # Use set for automatic deduplication
        
        for index in indices_to_fetch:
            logger.info(f"\n📊 Fetching index: {index}")
            
            # Try each data source
            for source_name, fetcher in fallback_manager.fetchers.items():
                try:
                    logger.info(f"  Attempting {source_name}...")
                    stocks = await fetcher.fetch_stock_list(index)
                    
                    if stocks and len(stocks) > 0:
                        before_count = len(all_stocks)
                        all_stocks.update(stocks)
                        new_count = len(all_stocks) - before_count
                        
                        logger.info(
                            f"  ✅ SUCCESS: Got {len(stocks)} stocks from {source_name}, "
                            f"added {new_count} new unique stocks"
                        )
                        logger.info(f"  📈 Running total: {len(all_stocks)} unique stocks")
                        break  # Success - move to next index
                    else:
                        logger.warning(f"  ⚠️  {source_name} returned empty list")
                        
                except Exception as e:
                    logger.warning(f"  ❌ {source_name} failed: {e}")
                    continue
            else:
                # All sources failed for this index
                logger.error(f"  ❌ ALL SOURCES FAILED for {index}")
        
        if all_stocks:
            unique_stocks = sorted(list(all_stocks))  # Convert set to sorted list
            logger.info("\n" + "=" * 70)
            logger.info(f"✅ FINAL RESULT: {len(unique_stocks)} UNIQUE STOCKS")
            logger.info(f"   From {len(indices_to_fetch)} indices")
            logger.info("=" * 70 + "\n")
            return unique_stocks
        else:
            logger.error("\n❌ CRITICAL ERROR: Failed to get ANY stocks from multi-index fetch")
            return []
    
    # ========================================================================
    # SINGLE INDEX FETCHING (for NIFTY50, NIFTY100, NIFTY500)
    # ========================================================================
    index_name = index_map.get(universe, "NIFTY 500")
    logger.info(f"📊 Fetching single index: {universe} ({index_name})")

    for source_name, fetcher in fallback_manager.fetchers.items():
        try:
            stocks = await fetcher.fetch_stock_list(index_name)
            if stocks:
                logger.info(
                    f"✅ Got {len(stocks)} stocks from {source_name} for {universe}"
                )
                return stocks
        except Exception as e:
            logger.warning(
                f"Failed to get stock list from {source_name}: {e}"
            )

    logger.error(f"❌ Failed to get stock list for {universe}")
    return []


async def _fetch_sector_map(symbols: List[str]) -> Dict[str, str]:
    """
    Fetch Yahoo Finance sector for each symbol (NSE suffix appended).

    Called only for stocks that produced signals, typically ≤50 symbols,
    so the individual yfinance calls are fast.  Failures fall back to
    "Unknown" silently — sector enrichment is best-effort.

    Args:
        symbols: List of NSE ticker symbols (e.g. ["RELIANCE", "TCS"]).

    Returns:
        Dict mapping symbol → sector string.
    """
    import yfinance as yf

    sector_map: Dict[str, str] = {}
    for sym in symbols:
        try:
            info = yf.Ticker(f"{sym}.NS").info
            sector = info.get("sector") or "Unknown"
            sector_map[sym] = sector
            if sector != "Unknown":
                logger.debug(f"Sector for {sym}: {sector}")
        except Exception as exc:
            logger.debug(f"Sector fetch failed for {sym}: {exc}")
            sector_map[sym] = "Unknown"
    return sector_map


async def test_single_symbol(symbol: str) -> None:
    """
    Test a single stock against Mother Candle V2 strategy.

    Fetches data, runs the strategy, and prints detailed output.
    Useful for verifying signals on specific stocks.

    Usage:
        python scripts/daily_scan.py --test-symbol ICICIBANK
    """
    print(f"\n{'='*60}")
    print(f"  MOTHER CANDLE V2 - SINGLE STOCK TEST")
    print(f"  Symbol: {symbol}")
    print(f"{'='*60}\n")

    from src.data_ingestion.fallback_manager import FallbackManager
    import pandas as pd

    fallback = FallbackManager()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        print(f"Fetching data for {symbol}...")
        data = await fallback.fetch_stock_data(
            symbol, start_date, end_date
        )

        if not data or not data.get("records"):
            print(f"ERROR: No data found for {symbol}")
            return

        from src.data_ingestion.data_validator import (
            DataValidator,
        )
        validator = DataValidator()
        records = validator.clean_records(
            data["records"], symbol
        )

        if len(records) < 50:
            print(
                f"ERROR: Only {len(records)} records "
                f"(need 50+)"
            )
            return

        df = pd.DataFrame(records)
        if "date" in df.columns:
            df.set_index("date", inplace=True)

        print(f"Data: {len(df)} candles loaded")
        last = df.iloc[-1]
        print(
            f"Last candle: O={last['open']:.2f} "
            f"H={last['high']:.2f} "
            f"L={last['low']:.2f} "
            f"C={last['close']:.2f} "
            f"V={int(last['volume']):,}"
        )

        # Load Mother Candle V2 strategy
        loader = StrategyLoader()
        strategies = loader.load_by_mode("daily")
        mc_v2 = None
        for s in strategies:
            if s.name == "Mother Candle V2":
                mc_v2 = s
                break

        if mc_v2 is None:
            print("ERROR: Mother Candle V2 strategy not loaded")
            return

        print(f"\nStrategy params:")
        params = mc_v2.strategy_config.get("params", {})
        for k, v in params.items():
            print(f"  {k}: {v}")

        company_info = {
            "name": symbol,
            "symbol": symbol,
            "sector": "Unknown",
            "market_cap": 0,
            "last_price": float(last["close"]),
        }

        print(f"\nRunning scan...")
        signal = mc_v2.scan(symbol, df, company_info)

        stats = mc_v2.get_scan_stats()
        print(f"\nScan stats: {stats}")

        if signal:
            print(f"\n{'='*60}")
            print(f"  SIGNAL FOUND!")
            print(f"{'='*60}")
            print(f"  Symbol:     {signal.symbol}")
            print(f"  Strategy:   {signal.strategy_name}")
            print(f"  Type:       {signal.signal_type.value}")
            print(f"  Entry:      Rs.{signal.entry_price:.2f}")
            print(f"  Target:     Rs.{signal.target_price:.2f} "
                  f"(+{mc_v2.target_pct}%)")
            print(f"  Stop Loss:  Rs.{signal.stop_loss:.2f} "
                  f"(-{signal.metadata.get('sl_distance_pct', 0)}%)")
            print(f"  SL Method:  {signal.metadata.get('sl_method', 'N/A')}")
            print(f"  R:R Ratio:  1:{signal.metadata.get('rr_ratio', 0)}")
            print(f"  Confidence: {signal.confidence:.0%}")
            print(f"  Indicators: {signal.indicators_met}/{signal.total_indicators}")
            print(f"\n  Pattern Details:")
            details = signal.indicator_details
            mc = details.get("mother_candle", {})
            print(f"    Mother High:      Rs.{mc.get('mother_high', 0)}")
            print(f"    Mother Low:       Rs.{mc.get('mother_low', 0)}")
            print(f"    Mother Range:     Rs.{mc.get('mother_range', 0)}")
            print(f"    Baby Count:       {mc.get('baby_count', 0)}")
            print(f"    Mother Position:  {mc.get('mother_position', 'N/A')}")
            fb = details.get("fresh_breakout", {})
            print(f"    Breakout Close:   Rs.{fb.get('breakout_close', 0)}")
            print(f"    Break Amount:     Rs.{fb.get('break_amount', 0)}")
            print(f"    Break %:          {fb.get('break_pct', 0)}%")
            mv = details.get("mother_volume", {})
            print(f"    Mother Vol Ratio: {mv.get('mother_vol_ratio', 0)}x")
            bv = details.get("breakout_volume", {})
            print(f"    Breakout Vol:     {bv.get('breakout_vol_ratio', 0)}x")
        else:
            print(f"\n  NO SIGNAL - Pattern not found for {symbol}")
            print(f"  Rejection reason from stats: {stats}")

        print(f"\n{'='*60}\n")

    finally:
        try:
            await fallback.close()
        except Exception:
            pass


if __name__ == "__main__":
    args = parse_args()
    strategy_filter = args.strategies

    # Single stock test mode
    if args.test_symbol:
        asyncio.run(test_single_symbol(args.test_symbol))
        sys.exit(0)

    if args.mother_v2_only:
        strategy_filter = ["Mother Candle V2"]
        logger.info(
            "mother-v2-only mode enabled: "
            "running only 'Mother Candle V2' strategy"
        )

    if args.btst_only:
        strategy_filter = [
            "Darvas Box",
            "Flag Pattern",
            "Symmetrical Triangle",
            "Descending Channel",
        ]
        logger.info(
            "btst-only mode enabled: running 4 BTST strategies "
            "(Darvas Box, Flag Pattern, Symmetrical Triangle, Descending Channel)"
        )

    asyncio.run(
        run_daily_scan(
            force_run=args.force,
            strategy_filter=strategy_filter,
        )
    )

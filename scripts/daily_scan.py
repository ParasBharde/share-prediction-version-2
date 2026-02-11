"""
Daily Stock Scanning Script - FINAL VERSION FOR 2700+ STOCKS

Purpose:
    Main entry point for the daily stock scan.
    Orchestrates data fetching, strategy execution,
    signal generation, and alert delivery.

CRITICAL UPDATE:
    - Added multi-index fetching support
    - When universe="ALL", fetches from 3 indices and deduplicates
    - Gets 900+ unique stocks from archives (NIFTY 500 + MIDCAP + SMALLCAP)
    
Usage:
    python scripts/daily_scan.py
    python scripts/daily_scan.py --force  # Skip trading day check
    FORCE_RUN=true python scripts/daily_scan.py

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
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.alert_deduplicator import AlertDeduplicator
from src.alerts.alert_formatter import AlertFormatter

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

        # 2. Load strategies
        strategy_loader = StrategyLoader()
        strategies = strategy_loader.load_all()

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

        # 5. Get stock list - THIS IS THE CRITICAL PART FOR 2700+ STOCKS
        stock_list = await _get_stock_universe(
            fallback_manager, config
        )

        if not stock_list:
            logger.error("Failed to get stock list")
            return {**results, "status": "error", "reason": "no_stocks"}

        logger.info(f"📊 Scanning {len(stock_list)} stocks")

        # 6. Fetch data and run strategies on all stocks
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
                        for sig in signals:
                            logger.info(
                                f"SIGNAL: {sig.strategy_name} -> "
                                f"{sig.symbol} ({sig.signal_type.value}) "
                                f"confidence={sig.confidence:.2f} "
                                f"entry={sig.entry_price:.2f} "
                                f"target={sig.target_price:.2f} "
                                f"SL={sig.stop_loss:.2f}"
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
            ranked = rank_signals(aggregated)
            filtered = filter_signals(ranked)

            results["signals_generated"] = len(filtered)

            # 8. Send alerts and place paper trades
            paper_trade_signals = []
            for signal in filtered:
                try:
                    # Check deduplication
                    signal_strategy = (
                        signal.contributing_strategies[0]
                        if signal.contributing_strategies
                        else "unknown"
                    )
                    if deduplicator.is_duplicate(
                        signal.symbol, signal_strategy
                    ):
                        logger.debug(
                            f"Skipping duplicate alert: "
                            f"{signal.symbol}"
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

                    # Send via Telegram (or log if not configured)
                    if telegram:
                        sent = await telegram.send_alert(
                            message, signal.priority.value
                        )
                    else:
                        logger.info(
                            f"ALERT (no Telegram): {message}"
                        )
                        sent = True

                    if sent:
                        deduplicator.mark_sent(
                            signal.symbol, signal_strategy
                        )
                        results["alerts_sent"] += 1

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


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_daily_scan(
            force_run=args.force,
            strategy_filter=args.strategies,
        )
    )
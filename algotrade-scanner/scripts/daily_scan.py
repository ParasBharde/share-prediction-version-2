"""
Daily Stock Scanning Script

Purpose:
    Main entry point for the daily stock scan.
    Orchestrates data fetching, strategy execution,
    signal generation, and alert delivery.

Usage:
    python scripts/daily_scan.py
    # Or called by the scheduler automatically

Dependencies:
    - All src modules

Logging:
    - Scan progress at INFO
    - Results at INFO
    - Failures at ERROR
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.alert_deduplicator import AlertDeduplicator
from src.alerts.alert_formatter import AlertFormatter
from src.alerts.telegram_bot import TelegramBot
from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.fallback_manager import FallbackManager
from src.engine.ranking_engine import RankingEngine
from src.engine.risk_filter import RiskFilter
from src.engine.signal_aggregator import SignalAggregator
from src.engine.strategy_executor import StrategyExecutor
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


async def run_daily_scan() -> Dict[str, Any]:
    """
    Execute the full daily stock scanning pipeline.

    Returns:
        Dictionary with scan results summary.
    """
    start_time = time.time()
    scan_date = now_ist()

    logger.info(
        f"Starting daily scan for {scan_date.date()}",
        extra={"scan_date": str(scan_date.date())},
    )

    # Check if trading day
    if not is_trading_day(scan_date.date()):
        logger.info("Not a trading day, skipping scan")
        return {"status": "skipped", "reason": "not_trading_day"}

    results = {
        "scan_date": str(scan_date.date()),
        "stocks_scanned": 0,
        "signals_generated": 0,
        "alerts_sent": 0,
        "errors": 0,
    }

    try:
        # 1. Load configuration
        config = load_config("system")

        # 2. Load strategies
        strategy_loader = StrategyLoader()
        strategies = strategy_loader.load_all()

        if not strategies:
            logger.warning("No strategies loaded, aborting scan")
            return {**results, "status": "error", "reason": "no_strategies"}

        logger.info(f"Loaded {len(strategies)} strategies")

        # 3. Initialize components
        fallback_manager = FallbackManager()
        data_validator = DataValidator()
        strategy_executor = StrategyExecutor()
        signal_aggregator = SignalAggregator()
        ranking_engine = RankingEngine()
        risk_filter = RiskFilter()
        alert_formatter = AlertFormatter()
        deduplicator = AlertDeduplicator()
        telegram = TelegramBot()

        # 4. Get stock list
        stock_list = await _get_stock_universe(
            fallback_manager, config
        )

        if not stock_list:
            logger.error("Failed to get stock list")
            return {**results, "status": "error", "reason": "no_stocks"}

        logger.info(f"Scanning {len(stock_list)} stocks")

        # 5. Fetch data and run strategies
        all_signals: List[TradingSignal] = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        chunk_size = config.get("scanning", {}).get(
            "chunk_size", 50
        )

        for i in range(0, len(stock_list), chunk_size):
            chunk = stock_list[i: i + chunk_size]
            logger.info(
                f"Processing chunk {i // chunk_size + 1}: "
                f"stocks {i + 1}-{min(i + chunk_size, len(stock_list))}"
            )

            for symbol in chunk:
                try:
                    # Fetch data
                    data = await fallback_manager.fetch_stock_data(
                        symbol, start_date, end_date
                    )

                    if not data or not data.get("records"):
                        continue

                    # Validate data
                    clean_records = data_validator.clean_records(
                        data["records"], symbol
                    )

                    if len(clean_records) < 50:
                        continue

                    # Convert to DataFrame
                    import pandas as pd

                    df = pd.DataFrame(clean_records)
                    if "date" in df.columns:
                        df.set_index("date", inplace=True)

                    # Run all strategies
                    company_info = {
                        "name": symbol,
                        "symbol": symbol,
                        "sector": "Unknown",
                        "market_cap": 0,
                    }

                    signals = strategy_executor.execute_strategies(
                        symbol, df, company_info, strategies
                    )

                    all_signals.extend(signals)
                    results["stocks_scanned"] += 1

                except Exception as e:
                    results["errors"] += 1
                    logger.error(
                        f"Error processing {symbol}: {e}",
                        exc_info=True,
                    )

        # 6. Aggregate and rank signals
        if all_signals:
            aggregated = signal_aggregator.aggregate(all_signals)
            ranked = ranking_engine.rank_signals(aggregated)
            filtered = risk_filter.filter_signals(ranked)

            results["signals_generated"] = len(filtered)

            # 7. Send alerts
            for signal in filtered:
                try:
                    # Check deduplication
                    if deduplicator.is_duplicate(
                        signal.symbol, signal.strategy_name
                    ):
                        logger.debug(
                            f"Skipping duplicate alert: "
                            f"{signal.symbol}"
                        )
                        continue

                    # Format alert
                    message = alert_formatter.format_buy_signal(
                        signal
                    )

                    # Send via Telegram
                    sent = await telegram.send_alert(
                        message, signal.priority.value
                    )

                    if sent:
                        deduplicator.mark_sent(
                            signal.symbol, signal.strategy_name
                        )
                        results["alerts_sent"] += 1

                except Exception as e:
                    logger.error(
                        f"Alert delivery failed for "
                        f"{signal.symbol}: {e}",
                        exc_info=True,
                    )

        # 8. Send daily summary
        duration = time.time() - start_time
        results["duration_seconds"] = round(duration, 2)
        results["status"] = "success"

        try:
            summary_message = alert_formatter.format_daily_summary(
                {
                    "date": format_timestamp(),
                    "stocks_scanned": results["stocks_scanned"],
                    "signals_count": results["signals_generated"],
                    "alerts_sent": results["alerts_sent"],
                    "scan_duration": results["duration_seconds"],
                    "active_positions": 0,
                    "total_pnl_pct": 0,
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
            await telegram.send_alert(summary_message, "LOW")
        except Exception as e:
            logger.error(
                f"Failed to send daily summary: {e}"
            )

        logger.info(
            f"Daily scan completed: {results}",
            extra=results,
        )

        # Cleanup
        await fallback_manager.close()

        return results

    except Exception as e:
        logger.error(
            "Daily scan failed with unhandled error",
            exc_info=True,
        )
        results["status"] = "error"
        results["error"] = str(e)
        return results


async def _get_stock_universe(
    fallback_manager: FallbackManager,
    config: Dict,
) -> List[str]:
    """
    Get the list of stocks to scan based on config.

    Args:
        fallback_manager: Data source manager.
        config: System config.

    Returns:
        List of stock symbols.
    """
    universe = config.get("scanning", {}).get(
        "universe", "NIFTY500"
    )

    index_map = {
        "NIFTY50": "NIFTY 50",
        "NIFTY100": "NIFTY 100",
        "NIFTY500": "NIFTY 500",
    }

    index_name = index_map.get(universe, "NIFTY 500")

    # Try to get from primary source
    for source_name, fetcher in fallback_manager.fetchers.items():
        try:
            stocks = await fetcher.fetch_stock_list(index_name)
            if stocks:
                logger.info(
                    f"Got {len(stocks)} stocks from "
                    f"{source_name} for {universe}"
                )
                return stocks
        except Exception as e:
            logger.warning(
                f"Failed to get stock list from "
                f"{source_name}: {e}"
            )

    logger.error(
        f"Failed to get stock list for {universe}"
    )
    return []


if __name__ == "__main__":
    asyncio.run(run_daily_scan())

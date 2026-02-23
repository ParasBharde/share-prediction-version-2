"""
Parallel Strategy Executor

Purpose:
    Executes trading strategies across stocks in parallel.
    Uses ThreadPoolExecutor so strategy objects and DataFrames are
    shared in-process — no pickling required.  pandas/numpy release
    the GIL for most operations so threading gives real concurrency
    while avoiding the serialisation overhead and crash-on-pickle
    failures that ProcessPoolExecutor caused.

Dependencies:
    - concurrent.futures for parallel execution
    - strategies.base_strategy for TradingSignal
    - data sources for stock data

Logging:
    - Batch start/completion at INFO
    - Per-stock execution at DEBUG
    - Timeouts and failures at WARNING/ERROR

Fallbacks:
    Timed-out or crashed stocks are skipped with warnings.
    Partial results are returned even if some chunks fail.
"""

import os
import time
import traceback
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    strategy_execution_time,
    strategy_scan_counter,
    signal_generated_counter,
    job_success_counter,
    job_failure_counter,
)
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.utils.config_loader import load_config, get_nested

logger = get_logger(__name__)

# Defaults
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
DEFAULT_TIMEOUT_PER_STOCK = 30  # seconds
DEFAULT_CHUNK_SIZE = 50  # stocks per batch


@dataclass
class ExecutionResult:
    """Result container for a full execution run."""

    signals: List[TradingSignal] = field(default_factory=list)
    processed: int = 0
    failed: int = 0
    timed_out: int = 0
    skipped: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        """Total stocks attempted."""
        return self.processed + self.failed + self.timed_out + self.skipped

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "signals_count": len(self.signals),
            "processed": self.processed,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "skipped": self.skipped,
            "total": self.total,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


def execute_all(
    stocks: List[Dict[str, Any]],
    strategies: List[BaseStrategy],
    max_workers: Optional[int] = None,
) -> ExecutionResult:
    """
    Execute all strategies across all stocks in parallel.

    Splits the stock list into chunks of configurable size and
    processes each chunk using a process pool. Each stock has
    a per-stock timeout to prevent hangs.

    Args:
        stocks: List of stock dictionaries, each containing at
            minimum 'symbol', 'df' (pd.DataFrame), and
            'company_info' (Dict).
        strategies: List of initialized BaseStrategy instances
            to run against each stock.
        max_workers: Maximum number of parallel worker processes.
            Defaults to CPU cores minus one.

    Returns:
        ExecutionResult with all generated signals and statistics.
    """
    config = load_config("system")
    engine_config = config.get("scanning", {})

    workers = max_workers or get_nested(
        engine_config, "max_workers", DEFAULT_MAX_WORKERS
    )
    chunk_size = get_nested(
        engine_config, "chunk_size", DEFAULT_CHUNK_SIZE
    )
    timeout = get_nested(
        engine_config, "timeout_per_stock", DEFAULT_TIMEOUT_PER_STOCK
    )

    result = ExecutionResult()
    start_time = time.time()

    logger.info(
        f"Starting parallel execution: {len(stocks)} stocks, "
        f"{len(strategies)} strategies, {workers} workers, "
        f"chunk_size={chunk_size}",
        extra={
            "stock_count": len(stocks),
            "strategy_count": len(strategies),
            "workers": workers,
            "chunk_size": chunk_size,
        },
    )

    if not stocks or not strategies:
        logger.warning("No stocks or strategies provided, skipping execution")
        return result

    # Split into chunks
    chunks = [
        stocks[i : i + chunk_size]
        for i in range(0, len(stocks), chunk_size)
    ]
    logger.info(f"Split into {len(chunks)} chunks of up to {chunk_size} stocks")

    for chunk_idx, chunk in enumerate(chunks):
        logger.debug(
            f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
            f"({len(chunk)} stocks)"
        )
        try:
            chunk_signals, chunk_stats = _process_chunk(
                chunk, strategies, workers, timeout
            )
            result.signals.extend(chunk_signals)
            result.processed += chunk_stats["processed"]
            result.failed += chunk_stats["failed"]
            result.timed_out += chunk_stats["timed_out"]
        except Exception as e:
            logger.error(
                f"Chunk {chunk_idx + 1} crashed entirely: {e}",
                exc_info=True,
                extra={"chunk_index": chunk_idx},
            )
            result.failed += len(chunk)
            job_failure_counter.labels(job_name="strategy_execution").inc()

    result.elapsed_seconds = time.time() - start_time

    logger.info(
        f"Execution complete: {len(result.signals)} signals from "
        f"{result.processed} stocks in {result.elapsed_seconds:.1f}s "
        f"(failed={result.failed}, timed_out={result.timed_out})",
        extra=result.to_dict(),
    )

    job_success_counter.labels(job_name="strategy_execution").inc()
    return result


def _process_chunk(
    chunk: List[Dict[str, Any]],
    strategies: List[BaseStrategy],
    max_workers: int,
    timeout: int,
) -> Tuple[List[TradingSignal], Dict[str, int]]:
    """
    Process a chunk of stocks using a thread pool.

    Submits each stock to the pool and collects results,
    handling timeouts and individual stock failures gracefully.

    Args:
        chunk: List of stock dictionaries for this batch.
        strategies: List of strategy instances to execute.
        max_workers: Number of parallel workers for the pool.
        timeout: Per-stock timeout in seconds.

    Returns:
        Tuple of (list of signals, stats dict with processed/failed/timed_out).
    """
    signals: List[TradingSignal] = []
    stats = {"processed": 0, "failed": 0, "timed_out": 0}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {}

            for stock in chunk:
                symbol = stock.get("symbol", "UNKNOWN")
                df = stock.get("df")
                company_info = stock.get("company_info", {})

                if df is None or df.empty:
                    logger.debug(f"{symbol}: No data available, skipping")
                    stats["failed"] += 1
                    continue

                future = executor.submit(
                    _process_single_stock,
                    symbol,
                    df,
                    company_info,
                    strategies,
                )
                future_to_symbol[future] = symbol

            for future in as_completed(
                future_to_symbol, timeout=timeout * len(future_to_symbol) + 60
            ):
                symbol = future_to_symbol[future]
                try:
                    stock_signals = future.result(timeout=timeout)
                    if stock_signals:
                        signals.extend(stock_signals)
                    stats["processed"] += 1
                except FuturesTimeoutError:
                    logger.warning(
                        f"{symbol}: Timed out after {timeout}s",
                        extra={"symbol": symbol, "timeout": timeout},
                    )
                    stats["timed_out"] += 1
                    future.cancel()
                except Exception as e:
                    logger.error(
                        f"{symbol}: Strategy execution failed: {e}",
                        exc_info=True,
                        extra={"symbol": symbol, "error": str(e)},
                    )
                    stats["failed"] += 1

    except Exception as e:
        logger.error(
            f"Process pool error: {e}",
            exc_info=True,
        )
        stats["failed"] += len(chunk) - stats["processed"]

    return signals, stats


def _process_single_stock(
    symbol: str,
    df: pd.DataFrame,
    company_info: Dict[str, Any],
    strategies: List[BaseStrategy],
) -> List[TradingSignal]:
    """
    Run all strategies against a single stock.

    Iterates through each strategy, calling its scan method.
    Individual strategy failures are caught so other strategies
    can still execute for the same stock.

    Args:
        symbol: Stock ticker symbol.
        df: OHLCV DataFrame for the stock (sorted ascending by date).
        company_info: Company metadata dictionary containing fields
            such as sector, market_cap, and name.
        strategies: List of strategy instances to run.

    Returns:
        List of TradingSignal objects generated for this stock.
    """
    signals: List[TradingSignal] = []

    for strategy in strategies:
        if not strategy.enabled:
            continue

        logger.debug(
            f"{symbol}: Running strategy '{strategy.name}'..."
        )

        strategy_start = time.time()
        try:
            signal = strategy.scan(symbol, df, company_info)

            elapsed = time.time() - strategy_start
            strategy_execution_time.labels(
                strategy_name=strategy.name
            ).observe(elapsed)
            strategy_scan_counter.labels(
                strategy_name=strategy.name
            ).inc()

            if signal is not None:
                signals.append(signal)
                signal_generated_counter.labels(
                    strategy_name=strategy.name,
                    signal_type=signal.signal_type.value,
                ).inc()
                logger.info(
                    f"{symbol}: SIGNAL from '{strategy.name}' "
                    f"-> {signal.signal_type.value} "
                    f"confidence={signal.confidence:.2f} "
                    f"entry={signal.entry_price:.2f} "
                    f"target={signal.target_price:.2f} "
                    f"SL={signal.stop_loss:.2f}",
                    extra={
                        "symbol": symbol,
                        "strategy": strategy.name,
                        "confidence": signal.confidence,
                    },
                )

        except Exception as e:
            elapsed = time.time() - strategy_start
            logger.warning(
                f"{symbol}: {strategy.name} raised an error: {e}",
                extra={
                    "symbol": symbol,
                    "strategy": strategy.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "elapsed": round(elapsed, 3),
                },
            )

    return signals

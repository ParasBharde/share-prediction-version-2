"""
Backtest Runner Script

Purpose:
    Runs historical backtests for trading strategies.
    Generates performance reports.

Usage:
    python scripts/backtest_runner.py --strategy momentum_breakout
    python scripts/backtest_runner.py --all
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.backtester import Backtester
from src.monitoring.logger import get_logger
from src.strategies.strategy_loader import StrategyLoader

logger = get_logger(__name__)


async def run_backtest(
    strategy_name: str = None,
    run_all: bool = False,
):
    """
    Run backtests for strategies.

    Args:
        strategy_name: Specific strategy to backtest.
        run_all: Run backtests for all strategies.
    """
    loader = StrategyLoader()
    strategies = loader.load_all()

    if not strategies:
        logger.error("No strategies loaded")
        return

    backtester = Backtester()

    if run_all:
        targets = strategies
    elif strategy_name:
        target = loader.get_strategy(strategy_name)
        if not target:
            logger.error(f"Strategy not found: {strategy_name}")
            return
        targets = [target]
    else:
        logger.error("Specify --strategy or --all")
        return

    for strategy in targets:
        logger.info(f"Backtesting: {strategy.name}")

        bt_config = strategy.config.get("backtesting", {})
        start_date = datetime.strptime(
            bt_config.get("start_date", "2023-01-01"),
            "%Y-%m-%d",
        )
        end_date = datetime.strptime(
            bt_config.get("end_date", "2025-01-31"),
            "%Y-%m-%d",
        )
        initial_capital = bt_config.get(
            "initial_capital", 100000
        )

        results = await backtester.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )

        if results:
            logger.info(
                f"Backtest results for {strategy.name}:"
            )
            for key, value in results.items():
                logger.info(f"  {key}: {value}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run strategy backtests"
    )
    parser.add_argument(
        "--strategy", type=str, help="Strategy name to backtest"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backtest all strategies",
    )

    args = parser.parse_args()
    asyncio.run(
        run_backtest(
            strategy_name=args.strategy, run_all=args.all
        )
    )


if __name__ == "__main__":
    main()

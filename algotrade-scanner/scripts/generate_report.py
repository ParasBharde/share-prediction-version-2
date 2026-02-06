"""
Report Generation Script

Purpose:
    Generates performance reports for the trading system.
    Outputs HTML and text reports.

Usage:
    python scripts/generate_report.py --period weekly
    python scripts/generate_report.py --period monthly --output report.html
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.logger import get_logger
from src.storage.postgres_handler import PostgresHandler

logger = get_logger(__name__)


def generate_report(
    period: str = "weekly", output_path: str = None
):
    """
    Generate a performance report.

    Args:
        period: Report period (daily, weekly, monthly).
        output_path: Optional output file path.
    """
    postgres = PostgresHandler()

    # Determine date range
    end_date = datetime.now()
    if period == "daily":
        start_date = end_date - timedelta(days=1)
    elif period == "weekly":
        start_date = end_date - timedelta(weeks=1)
    elif period == "monthly":
        start_date = end_date - timedelta(days=30)
    else:
        logger.error(f"Invalid period: {period}")
        return

    # Get signals
    signals = postgres.get_recent_signals(
        hours=int((end_date - start_date).total_seconds() / 3600)
    )

    # Get positions
    positions = postgres.get_open_positions()

    # Build report
    report_lines = [
        f"{'=' * 60}",
        f"  AlgoTrade Scanner - {period.title()} Report",
        f"  Period: {start_date.date()} to {end_date.date()}",
        f"{'=' * 60}",
        "",
        f"Signals Generated: {len(signals)}",
        f"Active Positions: {len(positions)}",
        "",
    ]

    if signals:
        report_lines.append("Top Signals:")
        report_lines.append("-" * 40)
        for s in signals[:10]:
            report_lines.append(
                f"  {s['symbol']:>10s} | "
                f"{s['strategy_name']:>20s} | "
                f"Conf: {s['confidence']:.1%}"
            )

    if positions:
        report_lines.append("")
        report_lines.append("Open Positions:")
        report_lines.append("-" * 40)
        for p in positions:
            pnl = (
                p.get("unrealized_pnl", 0) or 0
            )
            report_lines.append(
                f"  {p['symbol']:>10s} | "
                f"Entry: {p['avg_entry_price']:.2f} | "
                f"P&L: {pnl:.2f}"
            )

    report_text = "\n".join(report_lines)
    print(report_text)

    if output_path:
        Path(output_path).write_text(report_text)
        logger.info(f"Report saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate performance report"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="weekly",
        choices=["daily", "weekly", "monthly"],
    )
    parser.add_argument(
        "--output", type=str, help="Output file path"
    )

    args = parser.parse_args()
    generate_report(args.period, args.output)


if __name__ == "__main__":
    main()

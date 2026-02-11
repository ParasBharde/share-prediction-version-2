"""
View Paper Trades - CLI Tool

Usage:
    python scripts/view_trades.py                  # Show open positions
    python scripts/view_trades.py --orders          # Show order history
    python scripts/view_trades.py --all             # Show all positions (open + closed)
    python scripts/view_trades.py --performance     # Show performance history
    python scripts/view_trades.py --summary         # Show portfolio summary
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_currency(value):
    """Format number as Indian Rupee."""
    if value is None:
        return "N/A"
    return f"Rs {value:,.2f}"


def format_date(dt):
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    try:
        return dt.strftime("%d %b %Y %H:%M")
    except Exception:
        return str(dt)


def show_open_positions(postgres):
    """Display all open positions."""
    positions = postgres.get_open_positions()

    if not positions:
        print("\nNo open positions found.")
        print(
            "Run the scanner first: "
            "python scripts/daily_scan.py --force"
        )
        return

    print(f"\n{'='*70}")
    print(f"  OPEN POSITIONS ({len(positions)})")
    print(f"{'='*70}")

    total_invested = 0
    for i, p in enumerate(positions, 1):
        entry = p.get("avg_entry_price", 0)
        qty = p.get("quantity", 0)
        sl = p.get("stop_loss", 0)
        target = p.get("target_price", 0)
        invested = entry * qty
        total_invested += invested

        sl_pct = (
            ((sl - entry) / entry) * 100 if entry > 0 else 0
        )
        tgt_pct = (
            ((target - entry) / entry) * 100
            if entry > 0
            else 0
        )

        print(f"\n  {i}. {p['symbol']}")
        print(f"     Side: {p.get('side', 'LONG')}")
        print(f"     Quantity: {qty}")
        print(f"     Entry Price: {format_currency(entry)}")
        print(f"     Stop Loss: {format_currency(sl)} ({sl_pct:+.1f}%)")
        print(
            f"     Target: {format_currency(target)} ({tgt_pct:+.1f}%)"
        )
        print(f"     Invested: {format_currency(invested)}")
        print(
            f"     Strategy: {p.get('strategy_name', 'N/A')}"
        )
        print(f"     Opened: {format_date(p.get('opened_at'))}")

    print(f"\n{'─'*70}")
    print(f"  Total Invested: {format_currency(total_invested)}")
    print(f"{'='*70}\n")


def show_all_positions(postgres):
    """Display all positions (open + closed)."""
    positions = postgres.get_all_positions(limit=50)

    if not positions:
        print("\nNo positions found in database.")
        return

    print(f"\n{'='*70}")
    print(f"  ALL POSITIONS ({len(positions)})")
    print(f"{'='*70}")

    for i, p in enumerate(positions, 1):
        status = p.get("status", "UNKNOWN")
        status_icon = "[OPEN]" if status == "OPEN" else "[CLOSED]"
        entry = p.get("avg_entry_price", 0)
        qty = p.get("quantity", 0)

        print(f"\n  {i}. {p['symbol']} {status_icon}")
        print(f"     Qty: {qty} | Entry: {format_currency(entry)}")
        print(
            f"     SL: {format_currency(p.get('stop_loss', 0))} | "
            f"Target: {format_currency(p.get('target_price', 0))}"
        )
        print(
            f"     Strategy: {p.get('strategy_name', 'N/A')}"
        )
        if status == "CLOSED":
            print(
                f"     Realized P&L: "
                f"{format_currency(p.get('realized_pnl', 0))}"
            )
            print(
                f"     Closed: {format_date(p.get('closed_at'))}"
            )
        else:
            print(
                f"     Opened: {format_date(p.get('opened_at'))}"
            )

    print(f"\n{'='*70}\n")


def show_orders(postgres):
    """Display order history."""
    orders = postgres.get_trade_history(limit=30)

    if not orders:
        print("\nNo orders found in database.")
        return

    print(f"\n{'='*70}")
    print(f"  ORDER HISTORY ({len(orders)} recent)")
    print(f"{'='*70}")

    for i, o in enumerate(orders, 1):
        print(
            f"\n  {i}. {o['side']} {o['symbol']} | "
            f"Qty: {o['quantity']}"
        )
        print(
            f"     Price: {format_currency(o.get('price', 0))} -> "
            f"Executed: {format_currency(o.get('executed_price', 0))}"
        )
        slip = o.get("slippage", 0) or 0
        comm = o.get("commission", 0) or 0
        print(
            f"     Slippage: {format_currency(slip)} | "
            f"Commission: {format_currency(comm)}"
        )
        print(
            f"     Strategy: {o.get('strategy_name', 'N/A')} | "
            f"Status: {o.get('status', 'N/A')}"
        )
        print(f"     Date: {format_date(o.get('created_at'))}")

    print(f"\n{'='*70}\n")


def show_performance(postgres):
    """Display performance history."""
    metrics = postgres.get_performance_history(limit=15)

    if not metrics:
        print("\nNo performance data found.")
        return

    print(f"\n{'='*70}")
    print(f"  PERFORMANCE HISTORY ({len(metrics)} days)")
    print(f"{'='*70}")
    print(
        f"  {'Date':<14} {'Portfolio':>14} {'Cash':>14} "
        f"{'P&L':>10} {'Return':>8} {'Pos':>4}"
    )
    print(f"  {'─'*64}")

    for m in reversed(metrics):
        date_str = format_date(m.get("date"))[:11]
        pv = m.get("portfolio_value", 0)
        cash = m.get("cash_balance", 0)
        pnl = m.get("total_pnl", 0)
        ret = m.get("total_return_pct", 0)
        pos = m.get("active_positions", 0)

        pnl_str = f"{pnl:+,.0f}"
        ret_str = f"{ret:+.2f}%"

        print(
            f"  {date_str:<14} {pv:>14,.0f} {cash:>14,.0f} "
            f"{pnl_str:>10} {ret_str:>8} {pos:>4}"
        )

    print(f"\n{'='*70}\n")


def show_summary(postgres):
    """Display a combined portfolio summary."""
    positions = postgres.get_open_positions()
    orders = postgres.get_trade_history(limit=100)
    metrics = postgres.get_performance_history(limit=1)

    total_invested = sum(
        p.get("avg_entry_price", 0) * p.get("quantity", 0)
        for p in positions
    )

    latest = metrics[0] if metrics else {}

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING - PORTFOLIO SUMMARY")
    print(f"{'='*70}")
    print(
        f"  Portfolio Value:  "
        f"{format_currency(latest.get('portfolio_value', 0))}"
    )
    print(
        f"  Cash Balance:     "
        f"{format_currency(latest.get('cash_balance', 0))}"
    )
    print(f"  Invested Value:   {format_currency(total_invested)}")
    print(
        f"  Total P&L:        "
        f"{format_currency(latest.get('total_pnl', 0))}"
    )
    print(
        f"  Total Return:     "
        f"{latest.get('total_return_pct', 0):+.4f}%"
    )
    print(f"  Open Positions:   {len(positions)}")
    print(f"  Total Orders:     {len(orders)}")
    print(f"{'='*70}")

    if positions:
        print(f"\n  Open Positions:")
        for p in positions:
            entry = p.get("avg_entry_price", 0)
            qty = p.get("quantity", 0)
            val = entry * qty
            print(
                f"    - {p['symbol']:<15} "
                f"Qty: {qty:<6} "
                f"Entry: {entry:>10.2f}  "
                f"Value: {format_currency(val)}"
            )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View paper trading positions and history"
    )
    parser.add_argument(
        "--orders",
        action="store_true",
        help="Show order history",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all positions (open + closed)",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Show performance history",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show portfolio summary",
    )
    args = parser.parse_args()

    try:
        from src.storage.postgres_handler import PostgresHandler

        postgres = PostgresHandler()
    except Exception as e:
        print(f"\nError connecting to database: {e}")
        print(
            "Make sure PostgreSQL is running and "
            "DATABASE_URL is set in .env"
        )
        sys.exit(1)

    if args.summary:
        show_summary(postgres)
    elif args.orders:
        show_orders(postgres)
    elif args.all:
        show_all_positions(postgres)
    elif args.performance:
        show_performance(postgres)
    else:
        # Default: show open positions + summary
        show_summary(postgres)
        show_open_positions(postgres)


if __name__ == "__main__":
    main()

"""
Portfolio Tracker - Live P&L Dashboard

Fetches live market prices for all open paper trading positions,
updates unrealized P&L, detects SL/Target hits, and displays
a real-time portfolio dashboard.

Usage:
    python scripts/portfolio_tracker.py              # Full dashboard
    python scripts/portfolio_tracker.py --update     # Update prices + check SL/Target
    python scripts/portfolio_tracker.py --closed     # Show closed trades with results
    python scripts/portfolio_tracker.py --report     # Full performance report
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def fmt_rs(value):
    """Format as Indian Rupee."""
    if value is None:
        return "N/A"
    return f"\u20b9{value:,.2f}"


def fmt_pnl(value):
    """Format P&L with sign and color hint."""
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.2f}"


def fmt_pct(value):
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


async def fetch_live_prices(symbols):
    """Fetch current prices for a list of symbols.

    Uses Yahoo Finance (most reliable for quick quotes).
    Falls back to returning None for failed symbols.
    """
    prices = {}

    try:
        from src.data_ingestion.yahoo_fetcher import YahooFetcher

        yahoo = YahooFetcher()

        for symbol in symbols:
            try:
                quote = await yahoo.fetch_quote(symbol)
                if quote and quote.get("close"):
                    prices[symbol] = round(float(quote["close"]), 2)
                    print(f"  Fetched {symbol}: {fmt_rs(prices[symbol])}")
                else:
                    print(f"  Warning: No price for {symbol}")
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")

    except ImportError:
        print("Error: Yahoo fetcher not available.")
        print("Install yfinance: pip install yfinance")

    return prices


async def update_positions(postgres):
    """Fetch live prices and update all open positions."""
    positions = postgres.get_open_positions()

    if not positions:
        print("\nNo open positions to update.")
        return

    symbols = [p["symbol"] for p in positions]
    print(f"\nFetching live prices for {len(symbols)} positions...")
    prices = await fetch_live_prices(symbols)

    if not prices:
        print("Could not fetch any prices. Market may be closed.")
        return

    updated = 0
    sl_hits = []
    target_hits = []

    for p in positions:
        symbol = p["symbol"]
        if symbol not in prices:
            continue

        live_price = prices[symbol]
        entry_price = p["avg_entry_price"]
        qty = p["quantity"]
        sl = p.get("stop_loss", 0)
        target = p.get("target_price", 0)

        # Calculate unrealized P&L
        if p.get("side", "LONG") == "LONG":
            unrealized_pnl = round((live_price - entry_price) * qty, 2)
        else:
            unrealized_pnl = round((entry_price - live_price) * qty, 2)

        pnl_pct = round(
            ((live_price - entry_price) / entry_price) * 100, 2
        ) if entry_price > 0 else 0

        # Update in database
        postgres.update_position_price(
            p["id"], live_price, unrealized_pnl
        )
        updated += 1

        # Check SL hit
        if sl > 0 and live_price <= sl:
            realized_pnl = round((sl - entry_price) * qty, 2)
            postgres.close_position(
                p["id"], live_price, realized_pnl, "SL_HIT"
            )
            sl_hits.append({
                "symbol": symbol,
                "entry": entry_price,
                "sl": sl,
                "exit": live_price,
                "pnl": realized_pnl,
            })

        # Check Target hit
        elif target > 0 and live_price >= target:
            realized_pnl = round((live_price - entry_price) * qty, 2)
            postgres.close_position(
                p["id"], live_price, realized_pnl, "TARGET_HIT"
            )
            target_hits.append({
                "symbol": symbol,
                "entry": entry_price,
                "target": target,
                "exit": live_price,
                "pnl": realized_pnl,
            })

    print(f"\nUpdated {updated}/{len(positions)} positions.")

    if sl_hits:
        print(f"\n  STOP LOSS HIT ({len(sl_hits)}):")
        for h in sl_hits:
            print(
                f"    {h['symbol']}: Entry {fmt_rs(h['entry'])} "
                f"-> SL {fmt_rs(h['sl'])} | "
                f"P&L: {fmt_rs(h['pnl'])}"
            )

    if target_hits:
        print(f"\n  TARGET HIT ({len(target_hits)}):")
        for h in target_hits:
            print(
                f"    {h['symbol']}: Entry {fmt_rs(h['entry'])} "
                f"-> Target {fmt_rs(h['exit'])} | "
                f"P&L: {fmt_rs(h['pnl'])}"
            )

    return {"sl_hits": sl_hits, "target_hits": target_hits}


def show_dashboard(postgres):
    """Display the main portfolio dashboard."""
    positions = postgres.get_open_positions()
    closed = postgres.get_all_positions(status="CLOSED", limit=50)

    # Calculate totals
    total_invested = 0
    total_current = 0
    total_unrealized = 0

    for p in positions:
        entry = p.get("avg_entry_price", 0)
        qty = p.get("quantity", 0)
        current = p.get("current_price", entry)
        unrealized = p.get("unrealized_pnl", 0) or 0
        total_invested += entry * qty
        total_current += current * qty
        total_unrealized += unrealized

    total_realized = sum(
        (c.get("realized_pnl", 0) or 0) for c in closed
    )
    winning = [c for c in closed if (c.get("realized_pnl", 0) or 0) > 0]
    losing = [c for c in closed if (c.get("realized_pnl", 0) or 0) <= 0]
    win_rate = (
        (len(winning) / len(closed)) * 100 if closed else 0
    )

    overall_pnl = total_unrealized + total_realized
    overall_pct = (
        (overall_pnl / total_invested) * 100
        if total_invested > 0
        else 0
    )

    # Header
    print(f"\n{'='*72}")
    print(f"  PAPER TRADING PORTFOLIO DASHBOARD")
    print(f"  {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    print(f"{'='*72}")

    # Portfolio Summary
    print(f"\n  Portfolio Summary")
    print(f"  {'─'*40}")
    print(f"  Total Invested:     {fmt_rs(total_invested)}")
    print(f"  Current Value:      {fmt_rs(total_current)}")
    print(f"  Unrealized P&L:     {fmt_pnl(total_unrealized)} ({fmt_pct((total_unrealized / total_invested * 100) if total_invested > 0 else 0)})")
    print(f"  Realized P&L:       {fmt_pnl(total_realized)}")
    print(f"  Overall P&L:        {fmt_pnl(overall_pnl)} ({fmt_pct(overall_pct)})")
    print(f"  Win Rate:           {win_rate:.1f}% ({len(winning)}W / {len(losing)}L / {len(closed)} total)")

    # Open Positions Table
    if positions:
        print(f"\n  Open Positions ({len(positions)})")
        print(f"  {'─'*68}")
        print(
            f"  {'Symbol':<14} {'Qty':>5} {'Entry':>10} "
            f"{'Current':>10} {'P&L':>12} {'P&L%':>8} {'SL':>9} {'Target':>9}"
        )
        print(f"  {'─'*68}")

        for p in positions:
            symbol = p["symbol"]
            qty = p.get("quantity", 0)
            entry = p.get("avg_entry_price", 0)
            current = p.get("current_price", entry)
            unrealized = p.get("unrealized_pnl", 0) or 0
            sl = p.get("stop_loss", 0)
            target = p.get("target_price", 0)
            pnl_pct = (
                ((current - entry) / entry) * 100
                if entry > 0
                else 0
            )

            # SL/Target distance
            sl_dist = ((sl - entry) / entry * 100) if entry > 0 and sl > 0 else 0
            tgt_dist = ((target - entry) / entry * 100) if entry > 0 and target > 0 else 0

            pnl_str = fmt_pnl(unrealized)
            pct_str = fmt_pct(pnl_pct)

            print(
                f"  {symbol:<14} {qty:>5} {entry:>10.2f} "
                f"{current:>10.2f} {pnl_str:>12} {pct_str:>8} "
                f"{sl:>9.2f} {target:>9.2f}"
            )

        print(f"  {'─'*68}")
        print(
            f"  {'TOTAL':<14} {'':>5} {total_invested:>10.0f} "
            f"{total_current:>10.0f} {fmt_pnl(total_unrealized):>12}"
        )
    else:
        print(f"\n  No open positions.")

    print(f"\n{'='*72}")

    # Hint for user
    if positions:
        any_stale = any(
            p.get("current_price") == p.get("avg_entry_price")
            for p in positions
        )
        if any_stale:
            print(
                f"\n  Tip: Prices may be stale. Run with --update "
                f"to fetch live market prices."
            )
    print()


def show_closed_trades(postgres):
    """Display closed trades with results."""
    closed = postgres.get_all_positions(status="CLOSED", limit=50)

    if not closed:
        print("\nNo closed trades yet.")
        print(
            "Positions close when SL or Target is hit "
            "during --update."
        )
        return

    total_pnl = 0
    print(f"\n{'='*72}")
    print(f"  CLOSED TRADES ({len(closed)})")
    print(f"{'='*72}")
    print(
        f"  {'#':>3} {'Symbol':<14} {'Entry':>10} {'Exit':>10} "
        f"{'P&L':>12} {'Reason':<12} {'Date'}"
    )
    print(f"  {'─'*72}")

    for i, c in enumerate(closed, 1):
        entry = c.get("avg_entry_price", 0)
        exit_p = c.get("current_price", 0)
        pnl = c.get("realized_pnl", 0) or 0
        reason = c.get("exit_reason", "N/A") or "N/A"
        closed_at = c.get("closed_at")
        date_str = (
            closed_at.strftime("%d %b %Y")
            if closed_at else "N/A"
        )
        total_pnl += pnl

        print(
            f"  {i:>3} {c['symbol']:<14} {entry:>10.2f} "
            f"{exit_p:>10.2f} {fmt_pnl(pnl):>12} "
            f"{reason:<12} {date_str}"
        )

    print(f"  {'─'*72}")
    print(f"  {'TOTAL':>18} {'':>10} {'':>10} {fmt_pnl(total_pnl):>12}")

    winning = [c for c in closed if (c.get("realized_pnl", 0) or 0) > 0]
    print(
        f"\n  Win Rate: {len(winning)}/{len(closed)} "
        f"({len(winning) / len(closed) * 100:.1f}%)"
    )
    print(f"{'='*72}\n")


def show_report(postgres):
    """Full performance report."""
    show_dashboard(postgres)
    show_closed_trades(postgres)

    metrics = postgres.get_performance_history(limit=15)
    if metrics:
        print(f"  Performance History (last {len(metrics)} sessions)")
        print(f"  {'─'*60}")
        print(
            f"  {'Date':<14} {'Value':>14} {'Positions':>10} "
            f"{'Return':>10}"
        )
        print(f"  {'─'*60}")
        for m in reversed(metrics):
            dt = m.get("date")
            date_str = dt.strftime("%d %b %Y") if dt else "N/A"
            pv = m.get("portfolio_value", 0)
            pos = m.get("active_positions", 0)
            ret = m.get("total_return_pct", 0)
            print(
                f"  {date_str:<14} {fmt_rs(pv):>14} "
                f"{pos:>10} {fmt_pct(ret):>10}"
            )
        print(f"  {'─'*60}\n")


async def send_telegram_update(postgres, hits):
    """Send portfolio P&L update to Telegram."""
    try:
        from src.alerts.alert_formatter import AlertFormatter
        from src.alerts.telegram_bot import TelegramBot

        telegram = TelegramBot()
        formatter = AlertFormatter()
    except Exception as e:
        print(f"  Telegram not configured: {e}")
        return

    positions = postgres.get_open_positions()
    closed = postgres.get_all_positions(status="CLOSED", limit=50)
    sl_hits = hits.get("sl_hits", []) if hits else []
    target_hits = hits.get("target_hits", []) if hits else []

    message = formatter.format_portfolio_update(
        positions, closed, sl_hits, target_hits
    )

    sent = await telegram.send_alert(message, "MEDIUM")
    if sent:
        print("  Portfolio update sent to Telegram!")
    else:
        print("  Failed to send Telegram update.")


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Portfolio Tracker"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch live prices, update P&L, check SL/Target",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send portfolio update to Telegram after --update",
    )
    parser.add_argument(
        "--closed",
        action="store_true",
        help="Show closed trades",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Full performance report",
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

    if args.update:
        hits = asyncio.run(update_positions(postgres))
        if args.telegram:
            asyncio.run(send_telegram_update(postgres, hits))
        print()
        show_dashboard(postgres)
    elif args.closed:
        show_closed_trades(postgres)
    elif args.report:
        show_report(postgres)
    else:
        show_dashboard(postgres)


if __name__ == "__main__":
    main()

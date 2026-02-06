"""
P&L Calculator

Purpose:
    Computes profit and loss at both individual position and
    portfolio levels. Handles realized P&L from closed trades,
    unrealized P&L from open positions, and daily P&L changes.

Dependencies:
    - src.monitoring.logger for structured logging
    - src.monitoring.metrics for Prometheus instrumentation
    - src.utils.constants for enums (OrderSide, PositionStatus)

Logging:
    - P&L calculations at DEBUG
    - Portfolio-level summaries at INFO
    - Errors at ERROR
"""

from typing import Any, Dict, List

from src.monitoring.logger import get_logger
from src.monitoring.metrics import daily_pnl_gauge, total_pnl_gauge
from src.utils.constants import OrderSide, PositionStatus

logger = get_logger(__name__)


def calculate_position_pnl(
    position: Dict[str, Any], current_price: float
) -> Dict[str, Any]:
    """Calculates P&L for a single position.

    Computes unrealized P&L based on current price relative to the
    average entry price, accounting for position side (BUY/SELL).

    Args:
        position: A position dictionary containing at minimum:
            - avg_entry_price (float): The average entry price.
            - quantity (int): The number of shares held.
            - side (OrderSide): BUY or SELL.
            - total_commission (float): Commissions paid so far.
            - symbol (str): The trading symbol.
        current_price: The current market price for the symbol.

    Returns:
        A dictionary containing:
            - symbol (str): The trading symbol.
            - entry_price (float): Average entry price.
            - current_price (float): Current market price.
            - quantity (int): Number of shares.
            - gross_pnl (float): P&L before commissions.
            - net_pnl (float): P&L after commissions.
            - pnl_pct (float): Percentage return on the position.
            - side (OrderSide): BUY or SELL.
    """
    try:
        symbol = position["symbol"]
        avg_entry = position["avg_entry_price"]
        quantity = position["quantity"]
        side = position["side"]
        commission = position.get("total_commission", 0.0)

        if side == OrderSide.BUY:
            gross_pnl = (current_price - avg_entry) * quantity
        else:
            gross_pnl = (avg_entry - current_price) * quantity

        net_pnl = gross_pnl - commission
        invested = avg_entry * quantity
        pnl_pct = (
            (gross_pnl / invested) * 100 if invested > 0 else 0.0
        )

        result = {
            "symbol": symbol,
            "entry_price": avg_entry,
            "current_price": current_price,
            "quantity": quantity,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "side": side,
        }

        logger.debug(
            "Position P&L calculated",
            extra={
                "symbol": symbol,
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
            },
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to calculate position P&L",
            exc_info=True,
            extra={
                "symbol": position.get("symbol", "UNKNOWN"),
                "error": str(e),
            },
        )
        return {
            "symbol": position.get("symbol", "UNKNOWN"),
            "entry_price": 0.0,
            "current_price": current_price,
            "quantity": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "pnl_pct": 0.0,
            "side": position.get("side", OrderSide.BUY),
        }


def calculate_portfolio_pnl(
    positions: Dict[str, Dict[str, Any]],
    current_prices: Dict[str, float],
) -> Dict[str, Any]:
    """Calculates aggregate P&L across all open positions.

    Iterates over all open positions, computes per-position P&L,
    and aggregates into portfolio-level totals.

    Args:
        positions: Dictionary mapping symbols to position details.
            Each position must have avg_entry_price, quantity, side,
            and total_commission fields.
        current_prices: Dictionary mapping symbols to their current
            market prices.

    Returns:
        A dictionary containing:
            - total_gross_pnl (float): Total P&L before commissions.
            - total_net_pnl (float): Total P&L after commissions.
            - total_invested (float): Total capital deployed.
            - portfolio_pnl_pct (float): Overall percentage return.
            - position_pnls (list): List of per-position P&L dicts.
            - best_position (dict or None): Highest P&L position.
            - worst_position (dict or None): Lowest P&L position.
    """
    try:
        position_pnls: List[Dict[str, Any]] = []
        total_gross = 0.0
        total_net = 0.0
        total_invested = 0.0

        for symbol, position in positions.items():
            price = current_prices.get(
                symbol, position.get("current_price", 0.0)
            )
            pnl = calculate_position_pnl(position, price)
            position_pnls.append(pnl)

            total_gross += pnl["gross_pnl"]
            total_net += pnl["net_pnl"]
            total_invested += (
                pnl["entry_price"] * pnl["quantity"]
            )

        portfolio_pnl_pct = (
            (total_gross / total_invested) * 100
            if total_invested > 0
            else 0.0
        )

        # Find best and worst positions
        best_position = None
        worst_position = None
        if position_pnls:
            best_position = max(
                position_pnls, key=lambda p: p["net_pnl"]
            )
            worst_position = min(
                position_pnls, key=lambda p: p["net_pnl"]
            )

        result = {
            "total_gross_pnl": round(total_gross, 2),
            "total_net_pnl": round(total_net, 2),
            "total_invested": round(total_invested, 2),
            "portfolio_pnl_pct": round(portfolio_pnl_pct, 4),
            "position_pnls": position_pnls,
            "best_position": best_position,
            "worst_position": worst_position,
        }

        logger.info(
            "Portfolio P&L calculated",
            extra={
                "total_gross_pnl": round(total_gross, 2),
                "total_net_pnl": round(total_net, 2),
                "positions_count": len(position_pnls),
            },
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to calculate portfolio P&L",
            exc_info=True,
            extra={"error": str(e)},
        )
        return {
            "total_gross_pnl": 0.0,
            "total_net_pnl": 0.0,
            "total_invested": 0.0,
            "portfolio_pnl_pct": 0.0,
            "position_pnls": [],
            "best_position": None,
            "worst_position": None,
        }


def calculate_realized_pnl(
    closed_positions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculates total realized P&L from all closed positions.

    Sums up the realized P&L across all historically closed trades
    and provides per-trade breakdowns.

    Args:
        closed_positions: List of closed position dictionaries. Each
            must contain:
                - realized_pnl (float): The realized P&L for that trade.
                - symbol (str): The trading symbol.
                - avg_entry_price (float): Entry price.
                - exit_price (float): Exit price.
                - quantity (int): Shares traded.

    Returns:
        A dictionary containing:
            - total_realized_pnl (float): Sum of all realized P&L.
            - total_trades (int): Number of closed trades.
            - winning_trades (int): Trades with positive P&L.
            - losing_trades (int): Trades with negative or zero P&L.
            - win_rate (float): Percentage of winning trades.
            - avg_win (float): Average P&L of winning trades.
            - avg_loss (float): Average P&L of losing trades.
            - largest_win (float): Largest single winning trade P&L.
            - largest_loss (float): Largest single losing trade P&L.
    """
    try:
        if not closed_positions:
            logger.info("No closed positions to calculate realized P&L")
            return {
                "total_realized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        total_realized = sum(
            p.get("realized_pnl", 0.0) for p in closed_positions
        )
        winners = [
            p for p in closed_positions
            if p.get("realized_pnl", 0.0) > 0
        ]
        losers = [
            p for p in closed_positions
            if p.get("realized_pnl", 0.0) <= 0
        ]

        total_trades = len(closed_positions)
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = (
            (winning_trades / total_trades) * 100
            if total_trades > 0
            else 0.0
        )

        avg_win = (
            sum(p["realized_pnl"] for p in winners)
            / winning_trades
            if winning_trades > 0
            else 0.0
        )
        avg_loss = (
            sum(p["realized_pnl"] for p in losers) / losing_trades
            if losing_trades > 0
            else 0.0
        )

        largest_win = (
            max(p["realized_pnl"] for p in winners)
            if winners
            else 0.0
        )
        largest_loss = (
            min(p["realized_pnl"] for p in losers)
            if losers
            else 0.0
        )

        result = {
            "total_realized_pnl": round(total_realized, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
        }

        # Update Prometheus gauge
        total_pnl_gauge.set(round(total_realized, 2))

        logger.info(
            "Realized P&L calculated",
            extra={
                "total_realized_pnl": round(total_realized, 2),
                "total_trades": total_trades,
                "win_rate": round(win_rate, 2),
            },
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to calculate realized P&L",
            exc_info=True,
            extra={"error": str(e)},
        )
        return {
            "total_realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }


def calculate_daily_pnl(
    portfolio_state: Dict[str, Any],
    prev_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates the daily P&L change between two portfolio states.

    Computes the difference in portfolio value, realized P&L, and
    unrealized P&L between the current and previous snapshots.

    Args:
        portfolio_state: Current portfolio state dictionary as
            returned by PortfolioManager.get_portfolio_state().
        prev_state: Previous portfolio state dictionary (typically
            from the prior trading day close).

    Returns:
        A dictionary containing:
            - daily_pnl (float): Change in portfolio value.
            - daily_pnl_pct (float): Percentage change in portfolio
                value.
            - daily_realized_pnl (float): Change in realized P&L.
            - daily_unrealized_pnl (float): Change in unrealized P&L.
            - prev_portfolio_value (float): Previous portfolio value.
            - curr_portfolio_value (float): Current portfolio value.
    """
    try:
        curr_value = portfolio_state.get("portfolio_value", 0.0)
        prev_value = prev_state.get("portfolio_value", 0.0)

        daily_pnl = curr_value - prev_value
        daily_pnl_pct = (
            (daily_pnl / prev_value) * 100
            if prev_value > 0
            else 0.0
        )

        daily_realized = portfolio_state.get(
            "total_realized_pnl", 0.0
        ) - prev_state.get("total_realized_pnl", 0.0)

        daily_unrealized = portfolio_state.get(
            "total_unrealized_pnl", 0.0
        ) - prev_state.get("total_unrealized_pnl", 0.0)

        result = {
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "daily_realized_pnl": round(daily_realized, 2),
            "daily_unrealized_pnl": round(daily_unrealized, 2),
            "prev_portfolio_value": round(prev_value, 2),
            "curr_portfolio_value": round(curr_value, 2),
        }

        # Update Prometheus gauge
        daily_pnl_gauge.set(round(daily_pnl, 2))

        logger.info(
            "Daily P&L calculated",
            extra={
                "daily_pnl": round(daily_pnl, 2),
                "daily_pnl_pct": round(daily_pnl_pct, 4),
            },
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to calculate daily P&L",
            exc_info=True,
            extra={"error": str(e)},
        )
        return {
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "daily_realized_pnl": 0.0,
            "daily_unrealized_pnl": 0.0,
            "prev_portfolio_value": 0.0,
            "curr_portfolio_value": 0.0,
        }

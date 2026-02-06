"""
Portfolio Manager

Purpose:
    Manages the paper trading portfolio including open and closed
    positions, cash balance, and sector allocations. Tracks position
    lifecycle from entry to exit and provides real-time portfolio
    state queries.

Dependencies:
    - src.monitoring.logger for structured logging
    - src.monitoring.metrics for Prometheus instrumentation
    - src.utils.config_loader for configuration
    - src.utils.constants for enums (OrderStatus, PositionStatus,
      OrderSide)

Logging:
    - Position opens at INFO
    - Position closes at INFO
    - Portfolio updates at DEBUG
    - Errors at ERROR
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    portfolio_value_gauge,
    active_positions_gauge,
    total_pnl_gauge,
)
from src.utils.config_loader import load_config, get_nested
from src.utils.constants import (
    OrderSide,
    OrderStatus,
    PositionStatus,
)

logger = get_logger(__name__)


class PortfolioManager:
    """Manages paper trading portfolio state.

    Tracks open and closed positions, cash balance, and sector
    allocations. Provides methods to open, close, and query
    positions, as well as portfolio-level state.

    Attributes:
        initial_capital: Starting cash balance in INR.
        cash_balance: Current available cash in INR.
        open_positions: Dict mapping symbol to position details
            for all currently open positions.
        closed_positions: List of all closed position records.
        sector_map: Dict mapping symbol to sector name.
    """

    def __init__(self, initial_capital: float = 1000000.0) -> None:
        """Initializes the PortfolioManager with starting capital.

        Args:
            initial_capital: Starting cash balance in INR. Defaults
                to 1,000,000 (10 Lakh).
        """
        try:
            config = load_config("paper_trading")
            self.initial_capital: float = get_nested(
                config,
                "portfolio.initial_capital",
                initial_capital,
            )
        except Exception:
            logger.warning(
                "Could not load paper_trading config, using defaults"
            )
            self.initial_capital = initial_capital

        self.cash_balance: float = self.initial_capital
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.closed_positions: List[Dict[str, Any]] = []
        self.sector_map: Dict[str, str] = {}

        logger.info(
            "PortfolioManager initialized",
            extra={"initial_capital": self.initial_capital},
        )

    def open_position(
        self,
        order_result: Dict[str, Any],
        strategy: str,
        sector: str = "Unknown",
    ) -> Optional[Dict[str, Any]]:
        """Opens a new position from a filled order result.

        Only processes orders that have EXECUTED status. Deducts the
        total cost from cash balance for BUY orders. If the symbol
        already has an open position, the existing position is
        averaged up/down.

        Args:
            order_result: Order execution result dictionary as
                returned by OrderSimulator.simulate_market_order or
                simulate_limit_order.
            strategy: Name of the strategy that generated the signal.
            sector: Market sector for the symbol (e.g., "IT",
                "Banking"). Defaults to "Unknown".

        Returns:
            The position dictionary if successfully opened, or None
            if the order was not executed or cash is insufficient.
        """
        try:
            if order_result.get("status") != OrderStatus.EXECUTED:
                logger.warning(
                    "Cannot open position from non-executed order",
                    extra={
                        "order_id": order_result.get("order_id"),
                        "status": str(
                            order_result.get("status")
                        ),
                    },
                )
                return None

            symbol = order_result["symbol"]
            side = order_result["side"]
            quantity = order_result["quantity"]
            executed_price = order_result["executed_price"]
            commission = order_result["commission"]
            total_cost = order_result["total_cost"]

            # Check cash sufficiency for BUY orders
            if side == OrderSide.BUY and total_cost > self.cash_balance:
                logger.warning(
                    "Insufficient cash to open position",
                    extra={
                        "symbol": symbol,
                        "required": total_cost,
                        "available": self.cash_balance,
                    },
                )
                return None

            # Deduct cash for BUY orders
            if side == OrderSide.BUY:
                self.cash_balance -= total_cost

            # Track sector mapping
            self.sector_map[symbol] = sector

            if symbol in self.open_positions:
                # Average into existing position
                existing = self.open_positions[symbol]
                old_qty = existing["quantity"]
                old_avg = existing["avg_entry_price"]
                new_qty = old_qty + quantity
                new_avg = (
                    (old_avg * old_qty) + (executed_price * quantity)
                ) / new_qty

                existing["quantity"] = new_qty
                existing["avg_entry_price"] = round(new_avg, 2)
                existing["total_invested"] += total_cost
                existing["total_commission"] += commission
                existing["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

                logger.info(
                    "Position averaged",
                    extra={
                        "symbol": symbol,
                        "new_quantity": new_qty,
                        "new_avg_price": round(new_avg, 2),
                    },
                )
                position = existing
            else:
                # Create new position
                position = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "avg_entry_price": executed_price,
                    "current_price": executed_price,
                    "total_invested": total_cost,
                    "total_commission": commission,
                    "unrealized_pnl": 0.0,
                    "status": PositionStatus.OPEN,
                    "strategy": strategy,
                    "sector": sector,
                    "order_id": order_result["order_id"],
                    "opened_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "updated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
                self.open_positions[symbol] = position

                logger.info(
                    "Position opened",
                    extra={
                        "symbol": symbol,
                        "quantity": quantity,
                        "entry_price": executed_price,
                        "strategy": strategy,
                        "sector": sector,
                    },
                )

            # Update Prometheus metrics
            active_positions_gauge.set(len(self.open_positions))

            return deepcopy(position)

        except Exception as e:
            logger.error(
                "Failed to open position",
                exc_info=True,
                extra={
                    "order_id": order_result.get("order_id"),
                    "error": str(e),
                },
            )
            return None

    def close_position(
        self, symbol: str, exit_price: float
    ) -> Optional[Dict[str, Any]]:
        """Closes an open position at the given exit price.

        Calculates realized P&L, records the closed position, and
        returns cash from the sale to the balance.

        Args:
            symbol: The symbol whose position should be closed.
            exit_price: The price at which to close the position.

        Returns:
            The closed position record dictionary, or None if no
            open position exists for the symbol.
        """
        try:
            if symbol not in self.open_positions:
                logger.warning(
                    "No open position to close",
                    extra={"symbol": symbol},
                )
                return None

            position = self.open_positions.pop(symbol)
            quantity = position["quantity"]
            avg_entry = position["avg_entry_price"]
            commission_entry = position["total_commission"]

            # Calculate exit commission (same rate as entry)
            exit_value = exit_price * quantity
            # Approximate exit commission using the same percentage
            exit_commission = round(
                exit_value * 0.0003, 2
            )

            # Calculate realized P&L
            if position["side"] == OrderSide.BUY:
                gross_pnl = (exit_price - avg_entry) * quantity
            else:
                gross_pnl = (avg_entry - exit_price) * quantity

            total_commission = commission_entry + exit_commission
            realized_pnl = round(gross_pnl - total_commission, 2)

            # Return cash from sale
            self.cash_balance += exit_value - exit_commission

            closed_record = {
                **position,
                "exit_price": exit_price,
                "exit_commission": exit_commission,
                "total_commission": total_commission,
                "realized_pnl": realized_pnl,
                "status": PositionStatus.CLOSED,
                "closed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
            self.closed_positions.append(closed_record)

            logger.info(
                "Position closed",
                extra={
                    "symbol": symbol,
                    "entry_price": avg_entry,
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "realized_pnl": realized_pnl,
                },
            )

            # Update Prometheus metrics
            active_positions_gauge.set(len(self.open_positions))
            total_realized = sum(
                p["realized_pnl"] for p in self.closed_positions
            )
            total_pnl_gauge.set(total_realized)

            return deepcopy(closed_record)

        except Exception as e:
            logger.error(
                "Failed to close position",
                exc_info=True,
                extra={"symbol": symbol, "error": str(e)},
            )
            return None

    def update_positions(
        self, current_prices: Dict[str, float]
    ) -> None:
        """Updates unrealized P&L for all open positions.

        Should be called periodically with the latest market prices
        to keep the portfolio state current.

        Args:
            current_prices: A dictionary mapping symbols to their
                current market prices.
        """
        try:
            for symbol, position in self.open_positions.items():
                if symbol in current_prices:
                    price = current_prices[symbol]
                    position["current_price"] = price
                    qty = position["quantity"]
                    avg_entry = position["avg_entry_price"]

                    if position["side"] == OrderSide.BUY:
                        unrealized = (price - avg_entry) * qty
                    else:
                        unrealized = (avg_entry - price) * qty

                    position["unrealized_pnl"] = round(
                        unrealized, 2
                    )
                    position["updated_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()

            # Update portfolio value metric
            total_value = self.cash_balance + sum(
                p["current_price"] * p["quantity"]
                for p in self.open_positions.values()
            )
            portfolio_value_gauge.set(round(total_value, 2))

            logger.debug(
                "Positions updated",
                extra={
                    "positions_updated": len(self.open_positions),
                    "portfolio_value": round(total_value, 2),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to update positions",
                exc_info=True,
                extra={"error": str(e)},
            )

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Returns a snapshot of the current portfolio state.

        Returns:
            A dictionary containing:
                - cash_balance (float): Available cash in INR.
                - portfolio_value (float): Total portfolio value
                    including cash and position market values.
                - initial_capital (float): Starting capital.
                - total_return_pct (float): Percentage return since
                    inception.
                - open_positions_count (int): Number of open positions.
                - closed_positions_count (int): Number of closed trades.
                - total_realized_pnl (float): Sum of all realized P&L.
                - total_unrealized_pnl (float): Sum of all
                    unrealized P&L on open positions.
                - timestamp (str): ISO-format UTC timestamp.
        """
        try:
            positions_value = sum(
                p["current_price"] * p["quantity"]
                for p in self.open_positions.values()
            )
            portfolio_value = self.cash_balance + positions_value
            total_realized = sum(
                p["realized_pnl"] for p in self.closed_positions
            )
            total_unrealized = sum(
                p.get("unrealized_pnl", 0.0)
                for p in self.open_positions.values()
            )
            total_return_pct = (
                (portfolio_value - self.initial_capital)
                / self.initial_capital
            ) * 100

            state = {
                "cash_balance": round(self.cash_balance, 2),
                "portfolio_value": round(portfolio_value, 2),
                "initial_capital": self.initial_capital,
                "total_return_pct": round(total_return_pct, 4),
                "open_positions_count": len(self.open_positions),
                "closed_positions_count": len(
                    self.closed_positions
                ),
                "total_realized_pnl": round(total_realized, 2),
                "total_unrealized_pnl": round(total_unrealized, 2),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            logger.debug(
                "Portfolio state retrieved",
                extra=state,
            )

            return state

        except Exception as e:
            logger.error(
                "Failed to get portfolio state",
                exc_info=True,
                extra={"error": str(e)},
            )
            return {
                "cash_balance": self.cash_balance,
                "portfolio_value": self.cash_balance,
                "initial_capital": self.initial_capital,
                "total_return_pct": 0.0,
                "open_positions_count": 0,
                "closed_positions_count": 0,
                "total_realized_pnl": 0.0,
                "total_unrealized_pnl": 0.0,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

    def get_sector_allocations(self) -> Dict[str, Dict[str, Any]]:
        """Returns sector-level allocation breakdown.

        Calculates the market value and weight of each sector based
        on currently open positions.

        Returns:
            A dictionary mapping sector names to allocation details:
                - market_value (float): Total market value in the
                    sector.
                - weight_pct (float): Percentage weight within the
                    portfolio.
                - positions (int): Number of positions in the sector.
                - symbols (list): Symbols held in the sector.
        """
        try:
            sector_data: Dict[str, Dict[str, Any]] = {}
            total_value = sum(
                p["current_price"] * p["quantity"]
                for p in self.open_positions.values()
            )

            for symbol, position in self.open_positions.items():
                sector = position.get("sector", "Unknown")
                mkt_val = (
                    position["current_price"] * position["quantity"]
                )

                if sector not in sector_data:
                    sector_data[sector] = {
                        "market_value": 0.0,
                        "weight_pct": 0.0,
                        "positions": 0,
                        "symbols": [],
                    }

                sector_data[sector]["market_value"] += mkt_val
                sector_data[sector]["positions"] += 1
                sector_data[sector]["symbols"].append(symbol)

            # Calculate weight percentages
            if total_value > 0:
                for sector in sector_data:
                    sector_data[sector]["weight_pct"] = round(
                        (
                            sector_data[sector]["market_value"]
                            / total_value
                        )
                        * 100,
                        2,
                    )
                    sector_data[sector]["market_value"] = round(
                        sector_data[sector]["market_value"], 2
                    )

            logger.debug(
                "Sector allocations computed",
                extra={"sectors": list(sector_data.keys())},
            )

            return sector_data

        except Exception as e:
            logger.error(
                "Failed to compute sector allocations",
                exc_info=True,
                extra={"error": str(e)},
            )
            return {}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Returns a list of all currently open positions.

        Returns:
            A list of position dictionaries, each containing symbol,
            side, quantity, entry price, current price, unrealized
            P&L, strategy, and sector information.
        """
        try:
            positions = [
                deepcopy(pos)
                for pos in self.open_positions.values()
            ]

            logger.debug(
                "Open positions retrieved",
                extra={"count": len(positions)},
            )

            return positions

        except Exception as e:
            logger.error(
                "Failed to retrieve open positions",
                exc_info=True,
                extra={"error": str(e)},
            )
            return []

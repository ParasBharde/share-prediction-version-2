"""
Order Simulator

Purpose:
    Simulates order execution for paper trading with realistic
    market conditions including slippage, commissions, and
    order rejection scenarios.

Dependencies:
    - src.monitoring.logger for structured logging
    - src.monitoring.metrics for Prometheus instrumentation
    - src.utils.config_loader for configuration
    - src.utils.constants for enums (OrderType, OrderSide, OrderStatus)

Logging:
    - Order submissions at INFO
    - Order fills at INFO
    - Rejections at WARNING
    - Errors at ERROR

Defaults:
    - Slippage: 0.1% (10 bps)
    - Commission: 0.03% (3 bps)
    - Rejection probability based on quantity thresholds
"""

import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    job_success_counter,
    job_failure_counter,
)
from src.utils.config_loader import load_config, get_nested
from src.utils.constants import (
    OrderType,
    OrderSide,
    OrderStatus,
)

logger = get_logger(__name__)


class OrderSimulator:
    """Simulates order execution with realistic market conditions.

    Applies slippage, commissions, and random rejection logic to
    emulate broker-level order handling for paper trading.

    Attributes:
        slippage_pct: Percentage of slippage applied to market orders.
        commission_pct: Percentage commission charged per trade.
        max_quantity: Maximum order quantity before rejection is triggered.
        rejection_symbols: Set of symbols that are always rejected
            (e.g., suspended or illiquid stocks).
    """

    def __init__(
        self,
        slippage_pct: float = 0.001,
        commission_pct: float = 0.0003,
        max_quantity: int = 50000,
    ) -> None:
        """Initializes the OrderSimulator with trading cost parameters.

        Args:
            slippage_pct: Slippage as a decimal fraction. Defaults to
                0.001 (0.1%).
            commission_pct: Commission as a decimal fraction. Defaults
                to 0.0003 (0.03%).
            max_quantity: Maximum allowed order quantity. Orders
                exceeding this are rejected. Defaults to 50000.
        """
        try:
            config = load_config("paper_trading")
            self.slippage_pct: float = get_nested(
                config, "order_simulator.slippage_pct", slippage_pct
            )
            self.commission_pct: float = get_nested(
                config, "order_simulator.commission_pct", commission_pct
            )
            self.max_quantity: int = get_nested(
                config, "order_simulator.max_quantity", max_quantity
            )
            self.rejection_symbols: set = set(
                get_nested(
                    config, "order_simulator.rejection_symbols", []
                )
            )
        except Exception:
            logger.warning(
                "Could not load paper_trading config, using defaults"
            )
            self.slippage_pct = slippage_pct
            self.commission_pct = commission_pct
            self.max_quantity = max_quantity
            self.rejection_symbols = set()

        logger.info(
            "OrderSimulator initialized",
            extra={
                "slippage_pct": self.slippage_pct,
                "commission_pct": self.commission_pct,
                "max_quantity": self.max_quantity,
            },
        )

    def simulate_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        current_price: float,
    ) -> Dict[str, Any]:
        """Simulates a market order execution.

        Market orders are filled immediately at the current price
        adjusted for slippage.

        Args:
            symbol: The trading symbol (e.g., "RELIANCE").
            side: Order side, either BUY or SELL.
            quantity: Number of shares to trade.
            current_price: Current market price of the symbol.

        Returns:
            A dictionary containing order execution details:
                - order_id (str): Unique order identifier.
                - symbol (str): The trading symbol.
                - side (OrderSide): BUY or SELL.
                - order_type (OrderType): MARKET.
                - quantity (int): Number of shares.
                - requested_price (float): Price before slippage.
                - executed_price (float): Price after slippage.
                - commission (float): Commission amount in INR.
                - total_cost (float): Total cost including commission.
                - status (OrderStatus): EXECUTED or REJECTED.
                - timestamp (str): ISO-format UTC timestamp.
                - rejection_reason (str or None): Reason if rejected.
        """
        order_id = str(uuid.uuid4())

        logger.info(
            "Submitting market order",
            extra={
                "order_id": order_id,
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "current_price": current_price,
            },
        )

        try:
            # Check for rejection conditions
            rejection_reason = self._check_rejection(symbol, quantity)
            if rejection_reason is not None:
                logger.warning(
                    "Market order rejected",
                    extra={
                        "order_id": order_id,
                        "symbol": symbol,
                        "reason": rejection_reason,
                    },
                )
                job_failure_counter.labels(
                    job_name="paper_trade_order"
                ).inc()
                return {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "order_type": OrderType.MARKET,
                    "quantity": quantity,
                    "requested_price": current_price,
                    "executed_price": 0.0,
                    "commission": 0.0,
                    "total_cost": 0.0,
                    "status": OrderStatus.REJECTED,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "rejection_reason": rejection_reason,
                }

            # Apply slippage to get executed price
            executed_price = self._apply_slippage(current_price, side)
            commission = self._calculate_commission(
                executed_price, quantity
            )

            # Calculate total cost
            if side == OrderSide.BUY:
                total_cost = (executed_price * quantity) + commission
            else:
                total_cost = (executed_price * quantity) - commission

            logger.info(
                "Market order executed",
                extra={
                    "order_id": order_id,
                    "symbol": symbol,
                    "executed_price": round(executed_price, 2),
                    "commission": round(commission, 2),
                    "total_cost": round(total_cost, 2),
                },
            )
            job_success_counter.labels(
                job_name="paper_trade_order"
            ).inc()

            return {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": OrderType.MARKET,
                "quantity": quantity,
                "requested_price": current_price,
                "executed_price": round(executed_price, 2),
                "commission": round(commission, 2),
                "total_cost": round(total_cost, 2),
                "status": OrderStatus.EXECUTED,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rejection_reason": None,
            }

        except Exception as e:
            logger.error(
                "Failed to simulate market order",
                exc_info=True,
                extra={
                    "order_id": order_id,
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            job_failure_counter.labels(
                job_name="paper_trade_order"
            ).inc()
            return {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": OrderType.MARKET,
                "quantity": quantity,
                "requested_price": current_price,
                "executed_price": 0.0,
                "commission": 0.0,
                "total_cost": 0.0,
                "status": OrderStatus.REJECTED,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rejection_reason": f"Internal error: {e}",
            }

    def simulate_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        current_price: float,
    ) -> Dict[str, Any]:
        """Simulates a limit order execution.

        Limit orders are only filled if the current price meets the
        limit condition: for BUY orders the current price must be at
        or below the limit; for SELL orders it must be at or above.

        Args:
            symbol: The trading symbol (e.g., "RELIANCE").
            side: Order side, either BUY or SELL.
            quantity: Number of shares to trade.
            limit_price: The maximum (BUY) or minimum (SELL) price at
                which the order should fill.
            current_price: Current market price of the symbol.

        Returns:
            A dictionary containing order execution details. Same
            structure as simulate_market_order return, with
            order_type set to LIMIT. Status will be PENDING if the
            limit condition is not met.
        """
        order_id = str(uuid.uuid4())

        logger.info(
            "Submitting limit order",
            extra={
                "order_id": order_id,
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "limit_price": limit_price,
                "current_price": current_price,
            },
        )

        try:
            # Check for rejection conditions
            rejection_reason = self._check_rejection(symbol, quantity)
            if rejection_reason is not None:
                logger.warning(
                    "Limit order rejected",
                    extra={
                        "order_id": order_id,
                        "symbol": symbol,
                        "reason": rejection_reason,
                    },
                )
                job_failure_counter.labels(
                    job_name="paper_trade_order"
                ).inc()
                return {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "order_type": OrderType.LIMIT,
                    "quantity": quantity,
                    "requested_price": limit_price,
                    "executed_price": 0.0,
                    "commission": 0.0,
                    "total_cost": 0.0,
                    "status": OrderStatus.REJECTED,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "rejection_reason": rejection_reason,
                }

            # Check limit price conditions
            can_fill = False
            if side == OrderSide.BUY and current_price <= limit_price:
                can_fill = True
            elif side == OrderSide.SELL and current_price >= limit_price:
                can_fill = True

            if not can_fill:
                logger.info(
                    "Limit order pending - price condition not met",
                    extra={
                        "order_id": order_id,
                        "symbol": symbol,
                        "side": side.value,
                        "limit_price": limit_price,
                        "current_price": current_price,
                    },
                )
                return {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "order_type": OrderType.LIMIT,
                    "quantity": quantity,
                    "requested_price": limit_price,
                    "executed_price": 0.0,
                    "commission": 0.0,
                    "total_cost": 0.0,
                    "status": OrderStatus.PENDING,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "rejection_reason": None,
                }

            # Fill at the limit price (better or equal)
            executed_price = self._apply_slippage(
                min(current_price, limit_price)
                if side == OrderSide.BUY
                else max(current_price, limit_price),
                side,
            )
            commission = self._calculate_commission(
                executed_price, quantity
            )

            if side == OrderSide.BUY:
                total_cost = (executed_price * quantity) + commission
            else:
                total_cost = (executed_price * quantity) - commission

            logger.info(
                "Limit order executed",
                extra={
                    "order_id": order_id,
                    "symbol": symbol,
                    "executed_price": round(executed_price, 2),
                    "commission": round(commission, 2),
                    "total_cost": round(total_cost, 2),
                },
            )
            job_success_counter.labels(
                job_name="paper_trade_order"
            ).inc()

            return {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": OrderType.LIMIT,
                "quantity": quantity,
                "requested_price": limit_price,
                "executed_price": round(executed_price, 2),
                "commission": round(commission, 2),
                "total_cost": round(total_cost, 2),
                "status": OrderStatus.EXECUTED,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rejection_reason": None,
            }

        except Exception as e:
            logger.error(
                "Failed to simulate limit order",
                exc_info=True,
                extra={
                    "order_id": order_id,
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            job_failure_counter.labels(
                job_name="paper_trade_order"
            ).inc()
            return {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": OrderType.LIMIT,
                "quantity": quantity,
                "requested_price": limit_price,
                "executed_price": 0.0,
                "commission": 0.0,
                "total_cost": 0.0,
                "status": OrderStatus.REJECTED,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rejection_reason": f"Internal error: {e}",
            }

    def _apply_slippage(
        self, price: float, side: OrderSide
    ) -> float:
        """Applies realistic slippage to a price.

        Slippage moves the price unfavourably: up for BUY orders,
        down for SELL orders. A random factor between 0 and 1 is
        applied so that slippage varies per execution.

        Args:
            price: The original price before slippage.
            side: Order side (BUY or SELL).

        Returns:
            The price adjusted for slippage.
        """
        slippage_factor = random.uniform(0, self.slippage_pct)

        if side == OrderSide.BUY:
            # Buyer pays more due to slippage
            adjusted_price = price * (1 + slippage_factor)
        else:
            # Seller receives less due to slippage
            adjusted_price = price * (1 - slippage_factor)

        logger.debug(
            "Slippage applied",
            extra={
                "original_price": price,
                "adjusted_price": round(adjusted_price, 2),
                "slippage_factor": round(slippage_factor, 6),
                "side": side.value,
            },
        )

        return adjusted_price

    def _calculate_commission(
        self, price: float, quantity: int
    ) -> float:
        """Calculates the commission for a trade.

        Args:
            price: The execution price per share.
            quantity: The number of shares traded.

        Returns:
            The total commission amount in INR.
        """
        trade_value = price * quantity
        commission = trade_value * self.commission_pct

        logger.debug(
            "Commission calculated",
            extra={
                "trade_value": round(trade_value, 2),
                "commission": round(commission, 2),
                "commission_pct": self.commission_pct,
            },
        )

        return round(commission, 2)

    def _check_rejection(
        self, symbol: str, quantity: int
    ) -> Optional[str]:
        """Checks whether an order should be rejected.

        Rejection conditions:
            - Symbol is in the rejection list (suspended/illiquid).
            - Quantity exceeds the configured maximum.
            - Random rejection to simulate exchange-level rejections
              (~0.5% probability).

        Args:
            symbol: The trading symbol.
            quantity: The order quantity.

        Returns:
            A rejection reason string if the order should be rejected,
            or None if the order can proceed.
        """
        if symbol in self.rejection_symbols:
            return (
                f"Symbol {symbol} is suspended or illiquid"
            )

        if quantity > self.max_quantity:
            return (
                f"Quantity {quantity} exceeds maximum "
                f"allowed {self.max_quantity}"
            )

        if quantity <= 0:
            return "Quantity must be a positive integer"

        # Random rejection to simulate exchange-level failures (~0.5%)
        if random.random() < 0.005:
            return "Simulated exchange rejection (random)"

        return None

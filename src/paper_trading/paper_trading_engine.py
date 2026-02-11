"""
Paper Trading Engine

Purpose:
    Orchestrates automated paper trading from scan signals.
    Calculates position sizes, places simulated orders, tracks
    portfolio state, and generates trade confirmation alerts.

Dependencies:
    - src.paper_trading.order_simulator for order execution
    - src.paper_trading.portfolio_manager for position tracking
    - src.paper_trading.pnl_calculator for P&L computation
    - src.paper_trading.performance_tracker for metrics
    - src.utils.config_loader for configuration
    - src.utils.constants for enums
    - src.utils.time_helpers for market time utilities
    - src.monitoring.logger for structured logging

Logging:
    - Trade placement at INFO
    - Position sizing at DEBUG
    - Errors at ERROR
"""

import math
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.monitoring.logger import get_logger
from src.paper_trading.order_simulator import OrderSimulator
from src.paper_trading.portfolio_manager import PortfolioManager
from src.paper_trading.performance_tracker import PerformanceTracker
from src.utils.config_loader import load_config, get_nested
from src.utils.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.utils.time_helpers import (
    get_next_trading_day,
    get_market_hours,
    is_market_open,
    now_ist,
)

logger = get_logger(__name__)


class PaperTradingEngine:
    """Orchestrates paper trading from signal generation to execution.

    Receives trading signals, calculates appropriate position sizes,
    executes simulated orders, and tracks the full portfolio lifecycle.

    Attributes:
        enabled: Whether paper trading is active.
        order_simulator: Handles simulated order execution.
        portfolio_manager: Tracks open/closed positions.
        performance_tracker: Computes performance metrics.
        config: Paper trading configuration dictionary.
    """

    def __init__(self) -> None:
        """Initialize the PaperTradingEngine from config."""
        try:
            self.config = load_config("paper_trading")
        except Exception:
            logger.warning(
                "Could not load paper_trading config, using defaults"
            )
            self.config = {}

        self.enabled: bool = get_nested(
            self.config, "enabled", True
        )

        # Position sizing config
        self.risk_per_trade_pct: float = get_nested(
            self.config,
            "position_sizing.risk_per_trade_percent",
            1.0,
        ) / 100.0
        self.max_position_pct: float = get_nested(
            self.config,
            "position_sizing.max_position_percent",
            20.0,
        ) / 100.0
        self.min_position_value: float = get_nested(
            self.config,
            "position_sizing.min_position_value",
            10000,
        )

        # Execution config
        self.min_confidence: float = get_nested(
            self.config,
            "execution.min_confidence_percent",
            70.0,
        )
        self.auto_place: bool = get_nested(
            self.config,
            "execution.auto_place_orders",
            True,
        )

        # Trading time config
        self.trading_time_config: Dict = get_nested(
            self.config, "trading_time", {}
        )

        # Initialize components
        initial_capital = get_nested(
            self.config, "portfolio.initial_capital", 1000000.0
        )
        self.order_simulator = OrderSimulator()
        self.portfolio_manager = PortfolioManager(initial_capital)
        self.performance_tracker = PerformanceTracker()

        # Track orders placed in current session
        self.session_orders: List[Dict[str, Any]] = []

        logger.info(
            "PaperTradingEngine initialized",
            extra={
                "enabled": self.enabled,
                "initial_capital": initial_capital,
                "risk_per_trade": self.risk_per_trade_pct,
                "min_confidence": self.min_confidence,
                "auto_place": self.auto_place,
            },
        )

    def process_signals(
        self,
        signals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process a list of trading signals and place paper trades.

        For each signal that meets the confidence threshold,
        calculates position size and places a simulated order.

        Args:
            signals: List of signal dictionaries. Each must have
                at minimum: symbol, signal_type, confidence,
                entry_price, stop_loss, target_price,
                strategy_name.

        Returns:
            List of trade result dictionaries containing order
            details, position info, and trade status.
        """
        if not self.enabled or not self.auto_place:
            logger.info(
                "Paper trading disabled or auto-place off, "
                "skipping order placement"
            )
            return []

        trade_results: List[Dict[str, Any]] = []
        max_positions = get_nested(
            self.config, "portfolio.max_positions", 10
        )

        for signal in signals:
            try:
                # Check confidence threshold
                confidence = signal.get("confidence", 0)
                # Handle both 0-1 and 0-100 scale
                conf_pct = (
                    confidence * 100 if confidence <= 1 else confidence
                )
                if conf_pct < self.min_confidence:
                    logger.debug(
                        f"Skipping {signal.get('symbol')}: "
                        f"confidence {conf_pct:.1f}% < "
                        f"{self.min_confidence}%"
                    )
                    continue

                # Check max positions
                current_positions = len(
                    self.portfolio_manager.open_positions
                )
                if current_positions >= max_positions:
                    logger.info(
                        f"Max positions ({max_positions}) reached, "
                        f"skipping {signal.get('symbol')}"
                    )
                    break

                # Check if already have position in this symbol
                symbol = signal.get("symbol", "")
                if symbol in self.portfolio_manager.open_positions:
                    logger.debug(
                        f"Already have position in {symbol}, "
                        f"skipping"
                    )
                    continue

                # Calculate position size
                entry_price = signal.get("entry_price", 0)
                stop_loss = signal.get("stop_loss", 0)
                if entry_price <= 0 or stop_loss <= 0:
                    continue

                quantity = self._calculate_position_size(
                    entry_price, stop_loss
                )
                if quantity <= 0:
                    logger.debug(
                        f"Position size too small for {symbol}"
                    )
                    continue

                # Determine order side
                signal_type = signal.get("signal_type", "BUY")
                if isinstance(signal_type, str):
                    side = (
                        OrderSide.BUY
                        if signal_type in ("BUY", "STRONG_BUY")
                        else OrderSide.SELL
                    )
                else:
                    side = (
                        OrderSide.BUY
                        if signal_type.value in ("BUY", "STRONG_BUY")
                        else OrderSide.SELL
                    )

                # Place simulated order
                order_result = self.order_simulator.simulate_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    current_price=entry_price,
                )

                # Open position if order executed
                strategy_name = signal.get(
                    "strategy_name", "Unknown"
                )
                sector = signal.get("sector", "Unknown")
                position = None

                if order_result["status"] == OrderStatus.EXECUTED:
                    position = self.portfolio_manager.open_position(
                        order_result, strategy_name, sector
                    )

                trade_result = {
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "order": order_result,
                    "position": position,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target_price": signal.get("target_price", 0),
                    "strategy": strategy_name,
                    "confidence": conf_pct,
                    "status": (
                        "PLACED"
                        if order_result["status"]
                        == OrderStatus.EXECUTED
                        else "FAILED"
                    ),
                }
                trade_results.append(trade_result)
                self.session_orders.append(trade_result)

                if order_result["status"] == OrderStatus.EXECUTED:
                    logger.info(
                        f"Paper trade placed: {side.value} "
                        f"{quantity} x {symbol} @ "
                        f"{order_result['executed_price']:.2f} "
                        f"(strategy: {strategy_name})",
                        extra={
                            "symbol": symbol,
                            "side": side.value,
                            "quantity": quantity,
                            "price": order_result["executed_price"],
                        },
                    )
                else:
                    logger.warning(
                        f"Paper trade failed for {symbol}: "
                        f"{order_result.get('rejection_reason')}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to process signal for "
                    f"{signal.get('symbol', 'UNKNOWN')}: {e}",
                    exc_info=True,
                )

        # Update performance tracker
        if trade_results:
            state = self.portfolio_manager.get_portfolio_state()
            self.performance_tracker.update(state)

        return trade_results

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> int:
        """Calculate position size based on risk per trade.

        Uses the risk-based method: risk amount divided by
        per-share risk (entry - stop loss) to get quantity.
        Caps at max_position_pct of portfolio.

        Args:
            entry_price: Entry price per share.
            stop_loss: Stop loss price per share.

        Returns:
            Number of shares to trade (integer).
        """
        capital = self.portfolio_manager.cash_balance
        risk_amount = capital * self.risk_per_trade_pct
        per_share_risk = abs(entry_price - stop_loss)

        if per_share_risk <= 0:
            return 0

        # Risk-based quantity
        quantity = int(risk_amount / per_share_risk)

        # Cap by max position value
        max_value = capital * self.max_position_pct
        max_qty_by_value = int(max_value / entry_price)
        quantity = min(quantity, max_qty_by_value)

        # Check minimum position value
        position_value = quantity * entry_price
        if position_value < self.min_position_value:
            return 0

        return quantity

    def get_trading_time_info(
        self,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get trading time details for a signal.

        Determines entry window, validity period, and holding
        period based on the strategy's timeframe configuration.

        Args:
            signal: Signal dictionary containing at minimum
                metadata with timeframe, or strategy_name.

        Returns:
            Dictionary with trading time details:
                - entry_session: When to enter (e.g. next_market_open)
                - entry_date: Specific date for entry
                - entry_window: Start-end time for entry
                - signal_validity: How long signal is valid
                - validity_expiry: Exact expiry datetime
                - holding_period: Min-max holding days
                - description: Human-readable timing description
        """
        # Determine timeframe from signal metadata
        metadata = signal.get("metadata", {})
        timeframe = metadata.get("timeframe", "1D")

        # Get individual signals to find timeframe
        individual = signal.get("individual_signals", [])
        for ind_sig in individual:
            ind_meta = ind_sig.get("metadata", {})
            if ind_meta.get("timeframe"):
                timeframe = ind_meta["timeframe"]
                break

        # Get timeframe config
        tf_config = self.trading_time_config.get(timeframe, {})
        if not tf_config:
            # Default to 1D config
            tf_config = self.trading_time_config.get("1D", {})

        current_time = now_ist()
        entry_session = tf_config.get(
            "entry_session", "next_market_open"
        )
        entry_window_start = tf_config.get(
            "entry_window_start", "09:15"
        )
        entry_window_end = tf_config.get(
            "entry_window_end", "10:30"
        )
        validity_days = tf_config.get("signal_validity_days", 2)
        hold_min = tf_config.get("holding_period_min_days", 1)
        hold_max = tf_config.get("holding_period_max_days", 30)
        description = tf_config.get(
            "description",
            f"{timeframe} timeframe signal",
        )

        # Calculate entry date
        if entry_session == "next_market_open":
            entry_date = get_next_trading_day(current_time.date())
        elif entry_session == "immediate" and is_market_open():
            entry_date = current_time.date()
        else:
            entry_date = get_next_trading_day(current_time.date())

        # Calculate validity expiry
        if validity_days > 0:
            expiry_date = entry_date
            remaining = validity_days - 1
            while remaining > 0:
                expiry_date = get_next_trading_day(expiry_date)
                remaining -= 1
            validity_expiry = f"{expiry_date} {entry_window_end} IST"
        else:
            validity_expiry = (
                f"{entry_date} {entry_window_end} IST (Intraday)"
            )

        # Format entry date for display
        entry_date_str = entry_date.strftime("%d %b %Y (%A)")

        return {
            "timeframe": timeframe,
            "entry_session": entry_session,
            "entry_date": entry_date_str,
            "entry_window": (
                f"{entry_window_start} - {entry_window_end} IST"
            ),
            "signal_validity_days": validity_days,
            "validity_expiry": validity_expiry,
            "holding_period_min": hold_min,
            "holding_period_max": hold_max,
            "holding_period": f"{hold_min}-{hold_max} trading days",
            "description": description,
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio state for alert display.

        Returns:
            Portfolio summary dictionary suitable for inclusion
            in Telegram alerts.
        """
        state = self.portfolio_manager.get_portfolio_state()
        positions = self.portfolio_manager.get_open_positions()

        return {
            "capital": state["initial_capital"],
            "cash_balance": state["cash_balance"],
            "portfolio_value": state["portfolio_value"],
            "total_return_pct": state["total_return_pct"],
            "open_positions": state["open_positions_count"],
            "closed_trades": state["closed_positions_count"],
            "realized_pnl": state["total_realized_pnl"],
            "unrealized_pnl": state["total_unrealized_pnl"],
            "positions": positions,
            "session_trades": len(self.session_orders),
        }

    def get_session_trades_summary(self) -> List[Dict[str, Any]]:
        """Get summary of trades placed in the current session.

        Returns:
            List of simplified trade records for display.
        """
        return [
            {
                "symbol": t["symbol"],
                "side": (
                    t["signal_type"]
                    if isinstance(t["signal_type"], str)
                    else t["signal_type"].value
                ),
                "quantity": t["quantity"],
                "entry_price": t["entry_price"],
                "stop_loss": t["stop_loss"],
                "target_price": t["target_price"],
                "status": t["status"],
                "strategy": t["strategy"],
            }
            for t in self.session_orders
        ]

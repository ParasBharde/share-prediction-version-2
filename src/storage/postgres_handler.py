"""
PostgreSQL Handler

Purpose:
    CRUD operations for metadata, signals, alerts, and orders.
    Manages non-time-series data in PostgreSQL.

Dependencies:
    - SQLAlchemy
    - db_manager for connections

Logging:
    - Write operations at INFO
    - Read operations at DEBUG
    - Failures at ERROR
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import and_

from src.monitoring.logger import get_logger
from src.storage.db_manager import get_db_manager
from src.storage.models import (
    Alert,
    Company,
    Order,
    PerformanceMetric,
    Position,
    Signal,
)

logger = get_logger(__name__)


class PostgresHandler:
    """Handles PostgreSQL CRUD operations for metadata."""

    def __init__(self):
        """Initialize with database manager."""
        self.db = get_db_manager()

    # --- Company Operations ---

    def upsert_company(self, company_data: Dict) -> int:
        """
        Insert or update a company record.

        Args:
            company_data: Dictionary with company fields.

        Returns:
            Company ID.
        """
        try:
            with self.db.get_session() as session:
                existing = (
                    session.query(Company)
                    .filter(Company.symbol == company_data["symbol"])
                    .first()
                )

                if existing:
                    for key, value in company_data.items():
                        if hasattr(existing, key) and value is not None:
                            setattr(existing, key, value)
                    company_id = existing.id
                else:
                    company = Company(**company_data)
                    session.add(company)
                    session.flush()
                    company_id = company.id

                logger.debug(
                    f"Upserted company: {company_data['symbol']}"
                )
                return company_id

        except Exception as e:
            logger.error(
                f"Failed to upsert company: "
                f"{company_data.get('symbol')}",
                exc_info=True,
            )
            raise

    def get_company(self, symbol: str) -> Optional[Dict]:
        """
        Get company by symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Company data dictionary or None.
        """
        try:
            with self.db.get_session() as session:
                company = (
                    session.query(Company)
                    .filter(Company.symbol == symbol)
                    .first()
                )
                if company:
                    return {
                        "id": company.id,
                        "symbol": company.symbol,
                        "name": company.name,
                        "sector": company.sector,
                        "industry": company.industry,
                        "market_cap": company.market_cap,
                        "is_active": company.is_active,
                    }
                return None

        except Exception as e:
            logger.error(
                f"Failed to get company: {symbol}",
                exc_info=True,
            )
            return None

    def get_active_companies(self) -> List[Dict]:
        """
        Get all active companies.

        Returns:
            List of company dictionaries.
        """
        try:
            with self.db.get_session() as session:
                companies = (
                    session.query(Company)
                    .filter(Company.is_active.is_(True))
                    .all()
                )
                return [
                    {
                        "id": c.id,
                        "symbol": c.symbol,
                        "name": c.name,
                        "sector": c.sector,
                        "market_cap": c.market_cap,
                    }
                    for c in companies
                ]

        except Exception as e:
            logger.error(
                "Failed to get active companies",
                exc_info=True,
            )
            return []

    # --- Signal Operations ---

    def save_signal(self, signal_data: Dict) -> int:
        """
        Save a trading signal.

        Args:
            signal_data: Signal data dictionary.

        Returns:
            Signal ID.
        """
        try:
            with self.db.get_session() as session:
                signal = Signal(**signal_data)
                session.add(signal)
                session.flush()

                logger.info(
                    f"Saved signal for {signal_data['symbol']}: "
                    f"{signal_data['signal_type']} "
                    f"(confidence: {signal_data['confidence']})",
                    extra=signal_data,
                )
                return signal.id

        except Exception as e:
            logger.error(
                f"Failed to save signal for "
                f"{signal_data.get('symbol')}",
                exc_info=True,
            )
            raise

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        hours: int = 24,
    ) -> List[Dict]:
        """
        Get recent signals with optional filters.

        Args:
            symbol: Filter by symbol.
            strategy: Filter by strategy name.
            hours: Look back period in hours.

        Returns:
            List of signal dictionaries.
        """
        try:
            from datetime import timedelta

            cutoff = datetime.now(timezone.utc) - timedelta(
                hours=hours
            )

            with self.db.get_session() as session:
                query = session.query(Signal).filter(
                    Signal.generated_at >= cutoff
                )

                if symbol:
                    query = query.filter(Signal.symbol == symbol)
                if strategy:
                    query = query.filter(
                        Signal.strategy_name == strategy
                    )

                results = query.order_by(
                    Signal.generated_at.desc()
                ).all()

                return [
                    {
                        "id": s.id,
                        "symbol": s.symbol,
                        "strategy_name": s.strategy_name,
                        "signal_type": s.signal_type,
                        "confidence": s.confidence,
                        "entry_price": s.entry_price,
                        "target_price": s.target_price,
                        "stop_loss": s.stop_loss,
                        "generated_at": s.generated_at,
                    }
                    for s in results
                ]

        except Exception as e:
            logger.error(
                "Failed to get recent signals",
                exc_info=True,
            )
            return []

    # --- Alert Operations ---

    def save_alert(self, alert_data: Dict) -> int:
        """
        Save an alert record.

        Args:
            alert_data: Alert data dictionary.

        Returns:
            Alert ID.
        """
        try:
            with self.db.get_session() as session:
                alert = Alert(**alert_data)
                session.add(alert)
                session.flush()
                return alert.id

        except Exception as e:
            logger.error(
                "Failed to save alert",
                exc_info=True,
            )
            raise

    def get_pending_alerts(self) -> List[Dict]:
        """
        Get alerts that need to be retried.

        Returns:
            List of pending alert dictionaries.
        """
        try:
            with self.db.get_session() as session:
                alerts = (
                    session.query(Alert)
                    .filter(
                        Alert.status == "queued",
                        Alert.retry_count < 3,
                    )
                    .all()
                )
                return [
                    {
                        "id": a.id,
                        "signal_id": a.signal_id,
                        "channel": a.channel,
                        "priority": a.priority,
                        "message": a.message,
                        "retry_count": a.retry_count,
                    }
                    for a in alerts
                ]

        except Exception as e:
            logger.error(
                "Failed to get pending alerts",
                exc_info=True,
            )
            return []

    def update_alert_status(
        self,
        alert_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update alert delivery status.

        Args:
            alert_id: Alert ID.
            status: New status.
            error_message: Optional error message.
        """
        try:
            with self.db.get_session() as session:
                alert = session.query(Alert).get(alert_id)
                if alert:
                    alert.status = status
                    if status == "sent":
                        alert.sent_at = datetime.now(timezone.utc)
                    if error_message:
                        alert.error_message = error_message
                    if status == "failed":
                        alert.retry_count += 1

        except Exception as e:
            logger.error(
                f"Failed to update alert {alert_id}",
                exc_info=True,
            )

    # --- Order Operations ---

    def save_order(self, order_data: Dict) -> int:
        """Save a paper trading order."""
        try:
            with self.db.get_session() as session:
                order = Order(**order_data)
                session.add(order)
                session.flush()
                return order.id

        except Exception as e:
            logger.error("Failed to save order", exc_info=True)
            raise

    # --- Position Operations ---

    def save_position(self, position_data: Dict) -> int:
        """Save a paper trading position."""
        try:
            with self.db.get_session() as session:
                position = Position(**position_data)
                session.add(position)
                session.flush()
                return position.id

        except Exception as e:
            logger.error("Failed to save position", exc_info=True)
            raise

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            with self.db.get_session() as session:
                positions = (
                    session.query(Position)
                    .filter(Position.status == "OPEN")
                    .all()
                )
                return [
                    {
                        "id": p.id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "quantity": p.quantity,
                        "avg_entry_price": p.avg_entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "stop_loss": p.stop_loss,
                        "target_price": p.target_price,
                        "strategy_name": p.strategy_name,
                        "sector": p.sector,
                        "opened_at": p.opened_at,
                    }
                    for p in positions
                ]

        except Exception as e:
            logger.error(
                "Failed to get open positions",
                exc_info=True,
            )
            return []

    def get_all_positions(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Get positions with optional status filter."""
        try:
            with self.db.get_session() as session:
                query = session.query(Position)
                if status:
                    query = query.filter(Position.status == status)
                positions = (
                    query.order_by(Position.opened_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": p.id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "quantity": p.quantity,
                        "avg_entry_price": p.avg_entry_price,
                        "current_price": p.current_price,
                        "unrealized_pnl": p.unrealized_pnl,
                        "realized_pnl": p.realized_pnl,
                        "stop_loss": p.stop_loss,
                        "target_price": p.target_price,
                        "strategy_name": p.strategy_name,
                        "sector": p.sector,
                        "status": p.status,
                        "opened_at": p.opened_at,
                        "closed_at": p.closed_at,
                    }
                    for p in positions
                ]

        except Exception as e:
            logger.error(
                "Failed to get positions", exc_info=True
            )
            return []

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent orders (trade history)."""
        try:
            with self.db.get_session() as session:
                orders = (
                    session.query(Order)
                    .order_by(Order.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": o.id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "order_type": o.order_type,
                        "quantity": o.quantity,
                        "price": o.price,
                        "executed_price": o.executed_price,
                        "status": o.status,
                        "slippage": o.slippage,
                        "commission": o.commission,
                        "strategy_name": o.strategy_name,
                        "created_at": o.created_at,
                    }
                    for o in orders
                ]

        except Exception as e:
            logger.error(
                "Failed to get trade history", exc_info=True
            )
            return []

    def get_performance_history(
        self, limit: int = 30
    ) -> List[Dict]:
        """Get recent performance snapshots."""
        try:
            with self.db.get_session() as session:
                metrics = (
                    session.query(PerformanceMetric)
                    .order_by(PerformanceMetric.date.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "date": m.date,
                        "portfolio_value": m.portfolio_value,
                        "cash_balance": m.cash_balance,
                        "daily_pnl": m.daily_pnl,
                        "total_pnl": m.total_pnl,
                        "total_return_pct": m.total_return_pct,
                        "active_positions": m.active_positions,
                    }
                    for m in metrics
                ]

        except Exception as e:
            logger.error(
                "Failed to get performance history",
                exc_info=True,
            )
            return []

    def update_position_price(
        self,
        position_id: int,
        current_price: float,
        unrealized_pnl: float,
    ) -> None:
        """Update current price and unrealized P&L for a position."""
        try:
            with self.db.get_session() as session:
                position = session.query(Position).get(position_id)
                if position:
                    position.current_price = current_price
                    position.unrealized_pnl = unrealized_pnl

        except Exception as e:
            logger.error(
                f"Failed to update position {position_id}",
                exc_info=True,
            )

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str = "MANUAL",
    ) -> None:
        """Close an open position."""
        try:
            with self.db.get_session() as session:
                position = session.query(Position).get(position_id)
                if position and position.status == "OPEN":
                    position.status = "CLOSED"
                    position.current_price = exit_price
                    position.exit_price = exit_price
                    position.realized_pnl = realized_pnl
                    position.unrealized_pnl = 0.0
                    position.exit_reason = exit_reason
                    position.closed_at = datetime.now(timezone.utc)
                    logger.info(
                        f"Position closed: {position.symbol} "
                        f"@ {exit_price} ({exit_reason}), "
                        f"P&L: {realized_pnl:+.2f}"
                    )

        except Exception as e:
            logger.error(
                f"Failed to close position {position_id}",
                exc_info=True,
            )

    # --- Performance Operations ---

    def save_performance_metric(self, metric_data: Dict) -> int:
        """Save a daily performance metric."""
        try:
            with self.db.get_session() as session:
                metric = PerformanceMetric(**metric_data)
                session.add(metric)
                session.flush()
                return metric.id

        except Exception as e:
            logger.error(
                "Failed to save performance metric",
                exc_info=True,
            )
            raise

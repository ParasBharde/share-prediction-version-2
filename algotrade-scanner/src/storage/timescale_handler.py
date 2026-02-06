"""
TimescaleDB Handler

Purpose:
    CRUD operations for time-series OHLCV data.
    Leverages TimescaleDB hypertables for performance.

Dependencies:
    - SQLAlchemy
    - TimescaleDB extension

Logging:
    - Bulk inserts at INFO
    - Query performance at DEBUG
    - Failures at ERROR

Fallbacks:
    Falls back to standard PostgreSQL if TimescaleDB unavailable.
"""

from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.monitoring.logger import get_logger
from src.monitoring.metrics import data_fetch_latency
from src.storage.db_manager import get_db_manager
from src.storage.models import OHLCVData

logger = get_logger(__name__)


class TimescaleHandler:
    """Handles TimescaleDB operations for OHLCV data."""

    def __init__(self):
        """Initialize with database manager."""
        self.db = get_db_manager()

    def setup_hypertable(self) -> None:
        """
        Convert ohlcv_data table to a TimescaleDB hypertable.
        Should be called once during initial setup.
        """
        try:
            with self.db.engine.connect() as conn:
                conn.execute(
                    text(
                        "SELECT create_hypertable("
                        "'ohlcv_data', 'date', "
                        "if_not_exists => TRUE"
                        ")"
                    )
                )
                conn.commit()
            logger.info("Hypertable created for ohlcv_data")
        except Exception as e:
            logger.warning(
                f"Could not create hypertable (TimescaleDB may not "
                f"be installed): {e}"
            )

    def insert_ohlcv(
        self,
        symbol: str,
        company_id: int,
        data: List[Dict],
    ) -> int:
        """
        Bulk insert OHLCV data with upsert.

        Args:
            symbol: Stock symbol.
            company_id: Company ID in database.
            data: List of OHLCV dictionaries.

        Returns:
            Number of rows inserted/updated.
        """
        if not data:
            return 0

        try:
            with self.db.get_session() as session:
                rows_affected = 0

                for record in data:
                    stmt = insert(OHLCVData).values(
                        company_id=company_id,
                        symbol=symbol,
                        date=record["date"],
                        open=record["open"],
                        high=record["high"],
                        low=record["low"],
                        close=record["close"],
                        volume=record["volume"],
                        delivery_volume=record.get("delivery_volume"),
                        delivery_percent=record.get("delivery_percent"),
                        vwap=record.get("vwap"),
                        turnover=record.get("turnover"),
                        trades=record.get("trades"),
                        source=record.get("source", "unknown"),
                    )

                    # Upsert: update on conflict
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_ohlcv_symbol_date",
                        set_={
                            "open": record["open"],
                            "high": record["high"],
                            "low": record["low"],
                            "close": record["close"],
                            "volume": record["volume"],
                            "delivery_volume": record.get(
                                "delivery_volume"
                            ),
                            "delivery_percent": record.get(
                                "delivery_percent"
                            ),
                            "vwap": record.get("vwap"),
                            "turnover": record.get("turnover"),
                            "trades": record.get("trades"),
                            "source": record.get("source", "unknown"),
                        },
                    )

                    session.execute(stmt)
                    rows_affected += 1

                logger.info(
                    f"Inserted/updated {rows_affected} OHLCV "
                    f"records for {symbol}"
                )
                return rows_affected

        except Exception as e:
            logger.error(
                f"Failed to insert OHLCV data for {symbol}",
                exc_info=True,
            )
            raise

    def get_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol and date range.

        Args:
            symbol: Stock symbol.
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with OHLCV data, sorted by date.
        """
        try:
            with self.db.get_session() as session:
                results = (
                    session.query(OHLCVData)
                    .filter(
                        OHLCVData.symbol == symbol,
                        OHLCVData.date >= datetime.combine(
                            start_date, datetime.min.time()
                        ),
                        OHLCVData.date <= datetime.combine(
                            end_date, datetime.max.time()
                        ),
                    )
                    .order_by(OHLCVData.date)
                    .all()
                )

                if not results:
                    return pd.DataFrame()

                data = [
                    {
                        "date": r.date,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "volume": r.volume,
                        "delivery_volume": r.delivery_volume,
                        "delivery_percent": r.delivery_percent,
                        "vwap": r.vwap,
                        "turnover": r.turnover,
                    }
                    for r in results
                ]

                df = pd.DataFrame(data)
                df.set_index("date", inplace=True)
                return df

        except Exception as e:
            logger.error(
                f"Failed to fetch OHLCV data for {symbol}",
                exc_info=True,
            )
            raise

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent OHLCV record for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Dictionary with latest OHLCV data or None.
        """
        try:
            with self.db.get_session() as session:
                result = (
                    session.query(OHLCVData)
                    .filter(OHLCVData.symbol == symbol)
                    .order_by(OHLCVData.date.desc())
                    .first()
                )

                if result:
                    return {
                        "date": result.date,
                        "open": result.open,
                        "high": result.high,
                        "low": result.low,
                        "close": result.close,
                        "volume": result.volume,
                    }
                return None

        except Exception as e:
            logger.error(
                f"Failed to get latest price for {symbol}",
                exc_info=True,
            )
            return None

    def cleanup_old_data(self, retention_days: int = 730) -> int:
        """
        Remove data older than retention period.

        Args:
            retention_days: Number of days to retain.

        Returns:
            Number of rows deleted.
        """
        try:
            with self.db.get_session() as session:
                cutoff = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                from datetime import timedelta
                cutoff -= timedelta(days=retention_days)

                result = (
                    session.query(OHLCVData)
                    .filter(OHLCVData.date < cutoff)
                    .delete()
                )

                logger.info(
                    f"Cleaned up {result} old OHLCV records "
                    f"(older than {retention_days} days)"
                )
                return result

        except Exception as e:
            logger.error(
                "Failed to cleanup old OHLCV data",
                exc_info=True,
            )
            raise

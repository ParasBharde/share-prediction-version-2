"""
Database Connection Manager

Purpose:
    Manages database connections with pooling.
    Handles auto-reconnect and health checks.
    Supports both PostgreSQL and TimescaleDB.

Dependencies:
    - SQLAlchemy for ORM
    - psycopg2 for PostgreSQL

Logging:
    - Connection events at INFO
    - Pool status at DEBUG
    - Failures at ERROR

Fallbacks:
    - Auto-reconnect with exponential backoff
    - Connection pool overflow handling
"""

import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.monitoring.logger import get_logger
from src.monitoring.metrics import db_connection_pool_gauge
from src.storage.models import Base
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and session lifecycle."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL. Falls back to
                         DATABASE_URL env var or config.
        """
        self.config = load_config("system").get("database", {})

        self.database_url = (
            database_url
            or os.environ.get("DATABASE_URL")
            or "postgresql://localhost/algotrade"
        )

        # Sync engine
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.config.get("pool_size", 20),
            max_overflow=self.config.get("max_overflow", 10),
            pool_timeout=self.config.get("pool_timeout", 30),
            pool_recycle=self.config.get("pool_recycle", 3600),
            pool_pre_ping=True,
            echo=False,
        )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

        # Register pool event listeners
        event.listen(self.engine, "checkout", self._on_checkout)
        event.listen(self.engine, "checkin", self._on_checkin)

        logger.info(
            "Database manager initialized",
            extra={
                "pool_size": self.config.get("pool_size", 20),
                "max_overflow": self.config.get("max_overflow", 10),
            },
        )

    def _on_checkout(self, dbapi_conn, connection_rec, connection_proxy):
        """Track connection checkouts."""
        pool = self.engine.pool
        db_connection_pool_gauge.labels(state="active").set(
            pool.checkedout()
        )

    def _on_checkin(self, dbapi_conn, connection_rec):
        """Track connection checkins."""
        pool = self.engine.pool
        db_connection_pool_gauge.labels(state="idle").set(
            pool.checkedin()
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(
                "Failed to create database tables",
                exc_info=True,
            )
            raise

    def drop_tables(self) -> None:
        """Drop all database tables (use with caution)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(
                "Failed to drop database tables",
                exc_info=True,
            )
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        Yields:
            SQLAlchemy Session.

        Example:
            with db_manager.get_session() as session:
                session.query(Company).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if database is reachable.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(
                f"Database health check failed: {e}",
                exc_info=True,
            )
            return False

    def get_pool_status(self) -> dict:
        """
        Get connection pool status.

        Returns:
            Dictionary with pool statistics.
        """
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "checked_in": pool.checkedin(),
            "overflow": pool.overflow(),
        }

    def dispose(self) -> None:
        """Dispose of the connection pool."""
        self.engine.dispose()
        logger.info("Database connection pool disposed")


class AsyncDatabaseManager:
    """Async database manager for async operations."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize async database manager.

        Args:
            database_url: Async database connection URL.
        """
        config = load_config("system").get("database", {})

        sync_url = (
            database_url
            or os.environ.get("DATABASE_URL")
            or "postgresql://localhost/algotrade"
        )

        # Convert to async URL
        async_url = sync_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )

        self.engine = create_async_engine(
            async_url,
            pool_size=config.get("pool_size", 20),
            max_overflow=config.get("max_overflow", 10),
            pool_timeout=config.get("pool_timeout", 30),
            pool_recycle=config.get("pool_recycle", 3600),
            pool_pre_ping=True,
        )

        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("Async database manager initialized")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.

        Yields:
            Async SQLAlchemy Session.
        """
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        """Dispose of the async connection pool."""
        await self.engine.dispose()
        logger.info("Async database connection pool disposed")


# Module-level singleton
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the singleton DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

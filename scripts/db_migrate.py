"""
Database Migration Script

Purpose:
    Initializes database schema and runs migrations.
    Creates all tables, indexes, and TimescaleDB hypertables.

Usage:
    python scripts/db_migrate.py
    python scripts/db_migrate.py --drop  # Drop and recreate
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.logger import get_logger
from src.storage.db_manager import DatabaseManager
from src.storage.timescale_handler import TimescaleHandler

logger = get_logger(__name__)


def run_migrations(drop_existing: bool = False):
    """
    Run database migrations.

    Args:
        drop_existing: If True, drop all tables first.
    """
    logger.info("Starting database migration...")

    db = DatabaseManager()

    # Check connectivity
    if not db.health_check():
        logger.error(
            "Cannot connect to database. "
            "Check DATABASE_URL in .env"
        )
        sys.exit(1)

    logger.info("Database connection successful")

    if drop_existing:
        logger.warning(
            "Dropping all existing tables..."
        )
        db.drop_tables()
        logger.info("Tables dropped")

    # Create tables
    logger.info("Creating database tables...")
    db.create_tables()
    logger.info("Tables created successfully")

    # Setup TimescaleDB hypertable
    logger.info("Setting up TimescaleDB hypertable...")
    ts_handler = TimescaleHandler()
    ts_handler.setup_hypertable()

    logger.info("Database migration completed successfully")

    # Print table info
    pool_status = db.get_pool_status()
    logger.info(f"Connection pool: {pool_status}")

    db.dispose()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run database migrations"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all tables before creating",
    )

    args = parser.parse_args()
    run_migrations(drop_existing=args.drop)


if __name__ == "__main__":
    main()

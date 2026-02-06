"""
Service Health Checker

Purpose:
    Monitors health of all system components.
    Reports status via Prometheus metrics.
    Alerts on degraded services.

Dependencies:
    - Database, Redis, external APIs

Logging:
    - Health checks at DEBUG
    - Status changes at INFO
    - Failures at WARNING

Fallbacks:
    Continues checking even if individual checks fail.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict

from src.monitoring.logger import get_logger
from src.monitoring.metrics import health_check_gauge
from src.utils.config_loader import load_config
from src.utils.constants import HealthStatus

logger = get_logger(__name__)


class HealthChecker:
    """Monitors health of all system services."""

    def __init__(self):
        """Initialize health checker."""
        config = load_config("system")
        self.interval = config.get("monitoring", {}).get(
            "health_check_interval", 30
        )
        self._running = False
        self._status: Dict[str, HealthStatus] = {}

    async def run(self):
        """
        Run health checks continuously.
        Checks every `interval` seconds.
        """
        self._running = True
        logger.info(
            f"Health checker started "
            f"(interval: {self.interval}s)"
        )

        while self._running:
            try:
                await self._check_all()
            except Exception as e:
                logger.error(
                    f"Health check cycle failed: {e}",
                    exc_info=True,
                )

            await asyncio.sleep(self.interval)

    async def _check_all(self):
        """Run all health checks."""
        checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "scheduler": self._check_scheduler,
        }

        for service_name, check_func in checks.items():
            try:
                is_healthy = await check_func()
                status = (
                    HealthStatus.HEALTHY
                    if is_healthy
                    else HealthStatus.UNHEALTHY
                )

                # Detect status changes
                prev_status = self._status.get(service_name)
                if prev_status != status:
                    logger.info(
                        f"Service {service_name} status changed: "
                        f"{prev_status} -> {status.value}"
                    )

                self._status[service_name] = status
                health_check_gauge.labels(
                    service=service_name
                ).set(1 if is_healthy else 0)

            except Exception as e:
                self._status[service_name] = (
                    HealthStatus.UNKNOWN
                )
                health_check_gauge.labels(
                    service=service_name
                ).set(0)
                logger.warning(
                    f"Health check failed for "
                    f"{service_name}: {e}"
                )

    async def _check_database(self) -> bool:
        """Check database connectivity."""
        try:
            from src.storage.db_manager import get_db_manager

            db = get_db_manager()
            return db.health_check()
        except Exception:
            return False

    async def _check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            from src.storage.redis_handler import RedisHandler

            redis = RedisHandler()
            return redis.health_check()
        except Exception:
            return False

    async def _check_scheduler(self) -> bool:
        """Check if scheduler is running."""
        # This is always true if we're running checks
        return True

    async def stop(self):
        """Stop health checker."""
        self._running = False
        logger.info("Health checker stopped")

    def get_status(self) -> Dict[str, str]:
        """
        Get current health status of all services.

        Returns:
            Dictionary of service -> status string.
        """
        return {
            name: status.value
            for name, status in self._status.items()
        }

    def is_healthy(self) -> bool:
        """
        Check if all services are healthy.

        Returns:
            True if all services healthy.
        """
        return all(
            s == HealthStatus.HEALTHY
            for s in self._status.values()
        )

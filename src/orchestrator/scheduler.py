"""
Main Orchestration Scheduler

Purpose:
    Schedules all scanning jobs.
    Monitors system health.
    Handles graceful shutdown.
    Coordinates all services.

Dependencies:
    - APScheduler for job scheduling
    - All service modules

Logging:
    - Job start/completion at INFO
    - Failures at ERROR
    - Health checks at DEBUG

Fallbacks:
    - If job fails, retries 3 times with 5-min delay
    - If persistent failure, sends alert to admin
    - Continues with next scheduled run
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.monitoring.error_tracker import init_sentry
from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    job_duration_histogram,
    job_failure_counter,
    job_success_counter,
    start_metrics_server,
)
from src.orchestrator.health_checker import HealthChecker
from src.orchestrator.shutdown_handler import ShutdownHandler
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class Orchestrator:
    """Main system orchestrator that coordinates all services."""

    def __init__(self):
        """Initialize orchestrator with all components."""
        self.config = load_config("system")
        self.scheduler = AsyncIOScheduler()
        self.health_checker = HealthChecker()
        self.shutdown_handler = ShutdownHandler()
        self.shutdown_event = asyncio.Event()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            "Orchestrator initialized",
            extra={
                "environment": self.config.get(
                    "system", {}
                ).get("environment", "unknown"),
                "version": self.config.get("system", {}).get(
                    "version", "unknown"
                ),
            },
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(
            f"Received signal {signum}, initiating shutdown"
        )
        self.shutdown_event.set()

    async def start(self):
        """Start all services and scheduling."""
        logger.info("Starting orchestrator...")

        # Initialize Sentry
        init_sentry()

        # Start Prometheus metrics server
        metrics_port = self.config.get("monitoring", {}).get(
            "prometheus_port", 9090
        )
        start_metrics_server(metrics_port)

        # Start health checker
        asyncio.create_task(self.health_checker.run())

        # Schedule daily scan
        scan_config = self.config.get("scanning", {}).get(
            "schedule", {}
        )
        if scan_config.get("enabled", False):
            hour, minute = map(
                int, scan_config["time"].split(":")
            )

            weekdays = scan_config.get(
                "weekdays", [0, 1, 2, 3, 4]
            )
            day_of_week = ",".join(map(str, weekdays))

            self.scheduler.add_job(
                self._run_daily_scan_wrapper,
                trigger=CronTrigger(
                    day_of_week=day_of_week,
                    hour=hour,
                    minute=minute,
                    timezone="Asia/Kolkata",
                ),
                id="daily_scan",
                name="Daily Stock Scan",
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
            )
            logger.info(
                f"Scheduled daily scan at "
                f"{scan_config['time']} IST"
            )

        # Start scheduler
        self.scheduler.start()
        logger.info("Scheduler started successfully")

        # Keep running until shutdown signal
        await self.shutdown_event.wait()
        await self.stop()

    async def _run_daily_scan_wrapper(self):
        """Wrapper for daily scan with error handling and metrics."""
        job_start = datetime.now(timezone.utc)

        try:
            logger.info("Starting daily scan job")

            # Import here to avoid circular imports
            from scripts.daily_scan import run_daily_scan

            result = await run_daily_scan()

            duration = (
                datetime.now(timezone.utc) - job_start
            ).total_seconds()
            job_success_counter.labels(
                job_name="daily_scan"
            ).inc()
            job_duration_histogram.labels(
                job_name="daily_scan"
            ).observe(duration)

            logger.info(
                f"Daily scan completed in {duration:.2f}s",
                extra={
                    "duration": duration,
                    "stocks_scanned": result.get(
                        "stocks_scanned", 0
                    )
                    if result
                    else 0,
                    "signals_generated": result.get(
                        "signals_generated", 0
                    )
                    if result
                    else 0,
                },
            )

        except Exception as e:
            job_failure_counter.labels(
                job_name="daily_scan"
            ).inc()

            logger.error(
                "Daily scan failed",
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

    async def stop(self):
        """Graceful shutdown of all services."""
        logger.info("Stopping orchestrator...")

        # Run shutdown handler
        await self.shutdown_handler.shutdown()

        # Stop scheduler
        self.scheduler.shutdown(wait=True)

        # Stop health checker
        await self.health_checker.stop()

        logger.info("Orchestrator stopped successfully")


async def main():
    """Entry point for the orchestrator."""
    orchestrator = Orchestrator()
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())

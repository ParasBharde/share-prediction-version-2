"""
Graceful Shutdown Handler

Purpose:
    Manages graceful shutdown of all system components.
    Ensures in-flight operations complete before stopping.
    Cleans up resources (connections, files, etc).

Dependencies:
    - All service modules

Logging:
    - Shutdown steps at INFO
    - Cleanup failures at WARNING

Fallbacks:
    Force shutdown after timeout (60s).
"""

import asyncio
from typing import Callable, List

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class ShutdownHandler:
    """Manages graceful system shutdown."""

    def __init__(self, timeout: int = 60):
        """
        Initialize shutdown handler.

        Args:
            timeout: Maximum time to wait for shutdown (seconds).
        """
        self.timeout = timeout
        self._cleanup_tasks: List[Callable] = []
        self._is_shutting_down = False

    def register_cleanup(self, func: Callable) -> None:
        """
        Register a cleanup function to run during shutdown.

        Args:
            func: Async or sync cleanup function.
        """
        self._cleanup_tasks.append(func)
        logger.debug(
            f"Registered cleanup task: {func.__name__}"
        )

    async def shutdown(self) -> None:
        """
        Execute graceful shutdown sequence.

        Runs all registered cleanup tasks with timeout.
        """
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress")
            return

        self._is_shutting_down = True
        logger.info(
            f"Starting graceful shutdown "
            f"(timeout: {self.timeout}s)"
        )

        try:
            await asyncio.wait_for(
                self._run_cleanup_tasks(),
                timeout=self.timeout,
            )
            logger.info("Graceful shutdown completed")
        except asyncio.TimeoutError:
            logger.warning(
                f"Shutdown timed out after {self.timeout}s, "
                f"forcing cleanup"
            )

    async def _run_cleanup_tasks(self) -> None:
        """Run all registered cleanup tasks."""
        for task_func in reversed(self._cleanup_tasks):
            try:
                logger.info(
                    f"Running cleanup: {task_func.__name__}"
                )
                if asyncio.iscoroutinefunction(task_func):
                    await task_func()
                else:
                    task_func()
                logger.info(
                    f"Cleanup completed: {task_func.__name__}"
                )
            except Exception as e:
                logger.warning(
                    f"Cleanup task {task_func.__name__} "
                    f"failed: {e}",
                    exc_info=True,
                )

        # Close database connections
        try:
            from src.storage.db_manager import get_db_manager

            db = get_db_manager()
            db.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.warning(
                f"Failed to close database connections: {e}"
            )

        logger.info("All cleanup tasks completed")

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down

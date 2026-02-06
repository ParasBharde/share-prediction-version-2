"""
Alert Retry Queue

Purpose:
    Manages failed alert retries using PostgreSQL as a durable queue.
    Alerts that fail delivery are enqueued and retried on a periodic
    schedule (every 5 minutes) up to a configurable maximum number
    of attempts.

Dependencies:
    - PostgresHandler from src.storage.postgres_handler

Logging:
    - Enqueue operations at INFO
    - Retry attempts at INFO
    - Successful retries at INFO
    - Permanent failures at ERROR

Fallbacks:
    If PostgreSQL is unavailable, retry operations are skipped and
    errors are logged.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.monitoring.logger import get_alert_logger
from src.monitoring.metrics import (
    alert_failed_counter,
    alert_sent_counter,
)
from src.storage.postgres_handler import PostgresHandler
from src.utils.constants import DEFAULT_MAX_RETRIES

logger = get_alert_logger()

# Retry configuration
MAX_RETRIES = DEFAULT_MAX_RETRIES
RETRY_INTERVAL_MINUTES = 5


class RetryQueue:
    """Durable retry queue for failed alert deliveries backed by PostgreSQL."""

    def __init__(self) -> None:
        """
        Initialize the retry queue.

        Creates a ``PostgresHandler`` instance for persisting queued
        alerts in the ``alerts`` table.
        """
        try:
            self._db = PostgresHandler()
            logger.info(
                "RetryQueue initialized",
                extra={
                    "max_retries": MAX_RETRIES,
                    "retry_interval_minutes": RETRY_INTERVAL_MINUTES,
                },
            )
        except Exception as e:
            self._db = None  # type: ignore[assignment]
            logger.error(
                f"Failed to initialize RetryQueue database: {e}",
                exc_info=True,
            )

    def enqueue(self, alert_data: Dict[str, Any]) -> Optional[int]:
        """
        Enqueue a failed alert for later retry.

        The alert is stored in the ``alerts`` table with status
        ``"queued"`` and ``retry_count`` starting at 0.

        Args:
            alert_data: Dictionary describing the alert.  Expected keys::

                {
                    "signal_id": 42,
                    "channel": "telegram",
                    "priority": "HIGH",
                    "message": "BUY Signal - RELIANCE ...",
                    "error_message": "Telegram API timeout",
                }

        Returns:
            The alert row ID if enqueued successfully, None on failure.
        """
        if self._db is None:
            logger.error(
                "Cannot enqueue alert: database handler unavailable"
            )
            return None

        try:
            record = {
                "signal_id": alert_data.get("signal_id"),
                "channel": alert_data.get("channel", "telegram"),
                "priority": alert_data.get("priority", "MEDIUM"),
                "status": "queued",
                "message": alert_data.get("message", ""),
                "retry_count": 0,
                "error_message": alert_data.get("error_message"),
            }

            alert_id = self._db.save_alert(record)

            logger.info(
                f"Alert enqueued for retry (id={alert_id})",
                extra={
                    "alert_id": alert_id,
                    "channel": record["channel"],
                    "priority": record["priority"],
                },
            )
            return alert_id

        except Exception as e:
            logger.error(
                f"Failed to enqueue alert: {e}",
                exc_info=True,
                extra={"alert_data": alert_data},
            )
            return None

    def process_queue(
        self,
        send_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, int]:
        """
        Process all queued alerts that are eligible for retry.

        Eligible alerts have ``status='queued'`` and
        ``retry_count < MAX_RETRIES``.  Each alert is passed to
        ``_retry_alert`` (or the provided *send_fn*) for re-delivery.

        Args:
            send_fn: Optional callable that accepts an alert dict and
                returns True on success.  If not provided a default
                stub is used that simply marks the attempt.

        Returns:
            Summary dict with counts::

                {"processed": 5, "succeeded": 3, "failed": 2}
        """
        results = {"processed": 0, "succeeded": 0, "failed": 0}

        if self._db is None:
            logger.error(
                "Cannot process retry queue: database handler "
                "unavailable"
            )
            return results

        try:
            pending = self._db.get_pending_alerts()

            if not pending:
                logger.debug("Retry queue is empty, nothing to process")
                return results

            logger.info(
                f"Processing retry queue: {len(pending)} alert(s) pending"
            )

            for alert in pending:
                results["processed"] += 1
                success = self._retry_alert(alert, send_fn)

                if success:
                    results["succeeded"] += 1
                else:
                    results["failed"] += 1

            logger.info(
                "Retry queue processing complete",
                extra=results,
            )
            return results

        except Exception as e:
            logger.error(
                f"Error processing retry queue: {e}",
                exc_info=True,
            )
            return results

    def get_queue_size(self) -> int:
        """
        Return the number of alerts currently queued for retry.

        Returns:
            Count of pending alerts. Returns 0 if the database is
            unavailable.
        """
        if self._db is None:
            logger.error(
                "Cannot get queue size: database handler unavailable"
            )
            return 0

        try:
            pending = self._db.get_pending_alerts()
            size = len(pending)

            logger.debug(
                f"Retry queue size: {size}",
                extra={"queue_size": size},
            )
            return size

        except Exception as e:
            logger.error(
                f"Failed to get retry queue size: {e}",
                exc_info=True,
            )
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retry_alert(
        self,
        alert: Dict[str, Any],
        send_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> bool:
        """
        Attempt to re-deliver a single queued alert.

        On success the alert status is updated to ``"sent"``.  On
        failure the ``retry_count`` is incremented; if it reaches
        ``MAX_RETRIES`` the status becomes ``"failed"`` permanently.

        Args:
            alert: Alert dictionary as returned by
                ``PostgresHandler.get_pending_alerts()``.
            send_fn: Optional callable for delivery.

        Returns:
            True if re-delivery succeeded, False otherwise.
        """
        alert_id = alert.get("id")
        channel = alert.get("channel", "telegram")
        priority = alert.get("priority", "MEDIUM")
        retry_count = alert.get("retry_count", 0)

        logger.info(
            f"Retrying alert id={alert_id} "
            f"(attempt {retry_count + 1}/{MAX_RETRIES})",
            extra={
                "alert_id": alert_id,
                "channel": channel,
                "retry_count": retry_count,
            },
        )

        try:
            if send_fn is not None:
                success = send_fn(alert)
            else:
                # Default: mark as attempted without actual delivery.
                # The caller is expected to wire up a real send_fn.
                logger.warning(
                    "No send_fn provided for retry, "
                    "marking alert as failed",
                    extra={"alert_id": alert_id},
                )
                success = False

            if success:
                self._db.update_alert_status(
                    alert_id=alert_id,
                    status="sent",
                )

                alert_sent_counter.labels(
                    priority=priority,
                    channel=channel,
                ).inc()

                logger.info(
                    f"Alert id={alert_id} delivered on retry",
                    extra={"alert_id": alert_id},
                )
                return True

            # Delivery failed -- increment retry counter
            new_retry_count = retry_count + 1
            if new_retry_count >= MAX_RETRIES:
                self._db.update_alert_status(
                    alert_id=alert_id,
                    status="failed",
                    error_message=(
                        f"Permanently failed after "
                        f"{MAX_RETRIES} retries"
                    ),
                )

                alert_failed_counter.labels(
                    priority=priority,
                    channel=channel,
                ).inc()

                logger.error(
                    f"Alert id={alert_id} permanently failed "
                    f"after {MAX_RETRIES} retries",
                    extra={"alert_id": alert_id},
                )
            else:
                self._db.update_alert_status(
                    alert_id=alert_id,
                    status="failed",
                    error_message=(
                        f"Retry {new_retry_count}/{MAX_RETRIES} failed"
                    ),
                )
                # Re-queue by resetting status so the next cycle picks
                # it up again.  The retry_count was already incremented
                # inside update_alert_status when status == "failed".
                self._db.update_alert_status(
                    alert_id=alert_id,
                    status="queued",
                )

                logger.warning(
                    f"Alert id={alert_id} retry "
                    f"{new_retry_count}/{MAX_RETRIES} failed, "
                    f"re-queued",
                    extra={
                        "alert_id": alert_id,
                        "retry_count": new_retry_count,
                    },
                )

            return False

        except Exception as e:
            logger.error(
                f"Exception during alert retry id={alert_id}: {e}",
                exc_info=True,
                extra={"alert_id": alert_id},
            )

            try:
                self._db.update_alert_status(
                    alert_id=alert_id,
                    status="failed",
                    error_message=str(e),
                )
            except Exception as db_err:
                logger.error(
                    f"Failed to update alert status after error: "
                    f"{db_err}",
                    exc_info=True,
                )

            return False

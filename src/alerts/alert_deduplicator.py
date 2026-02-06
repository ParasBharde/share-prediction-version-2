"""
Alert Deduplicator

Purpose:
    Prevents duplicate alerts from being sent within a configurable
    time window. Uses Redis to track which (symbol, strategy) pairs
    have already triggered an alert.

Dependencies:
    - RedisHandler from src.storage.redis_handler

Logging:
    - Duplicate detection at DEBUG
    - Mark-sent at DEBUG
    - History queries at DEBUG
    - Redis failures at ERROR

Fallbacks:
    If Redis is unavailable, deduplication is skipped and the alert
    is allowed through (fail-open).
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.monitoring.logger import get_alert_logger
from src.monitoring.metrics import alert_deduplicated_counter
from src.storage.redis_handler import RedisHandler
from src.utils.constants import (
    ALERT_DEDUP_WINDOW,
    REDIS_PREFIX_ALERT_SEEN,
)

logger = get_alert_logger()


class AlertDeduplicator:
    """Tracks and prevents duplicate alert delivery via Redis."""

    def __init__(self, redis_handler: RedisHandler) -> None:
        """
        Initialize the deduplicator with a Redis connection.

        Args:
            redis_handler: An initialized ``RedisHandler`` instance used
                to read/write deduplication keys.
        """
        self._redis = redis_handler
        self._dedup_window = ALERT_DEDUP_WINDOW  # 24 hours in seconds

        logger.info(
            "AlertDeduplicator initialized",
            extra={"dedup_window_seconds": self._dedup_window},
        )

    def is_duplicate(
        self, symbol: str, strategy_name: str
    ) -> bool:
        """
        Check whether an alert for the given symbol and strategy has
        already been sent within the deduplication window.

        Args:
            symbol: Stock ticker symbol (e.g. ``"RELIANCE"``).
            strategy_name: Name of the strategy that generated the
                signal (e.g. ``"RSI_MACD_Crossover"``).

        Returns:
            True if a matching alert was already sent within the window,
            False otherwise.  Returns False when Redis is unavailable
            (fail-open).
        """
        key = self._build_key(symbol, strategy_name)

        try:
            if not self._redis.is_connected:
                logger.warning(
                    "Redis unavailable for dedup check, allowing alert",
                    extra={
                        "symbol": symbol,
                        "strategy": strategy_name,
                    },
                )
                return False

            is_seen = self._redis.is_seen(key)

            if is_seen:
                alert_deduplicated_counter.labels(
                    strategy_name=strategy_name,
                ).inc()

                logger.debug(
                    f"Duplicate alert suppressed: "
                    f"{symbol}/{strategy_name}",
                    extra={
                        "symbol": symbol,
                        "strategy": strategy_name,
                    },
                )

            return is_seen

        except Exception as e:
            logger.error(
                f"Dedup check failed for {symbol}/{strategy_name}: {e}",
                exc_info=True,
            )
            return False

    def mark_sent(
        self, symbol: str, strategy_name: str
    ) -> bool:
        """
        Record that an alert for the given symbol and strategy has been
        sent, preventing duplicates for the duration of the dedup window.

        Args:
            symbol: Stock ticker symbol.
            strategy_name: Strategy that generated the signal.

        Returns:
            True if the marker was stored successfully, False otherwise.
        """
        key = self._build_key(symbol, strategy_name)

        try:
            if not self._redis.is_connected:
                logger.warning(
                    "Redis unavailable, cannot mark alert as sent",
                    extra={
                        "symbol": symbol,
                        "strategy": strategy_name,
                    },
                )
                return False

            # Store a JSON payload so history queries can reconstruct
            # what was sent.
            payload = json.dumps({
                "symbol": symbol,
                "strategy_name": strategy_name,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            })

            self._redis.client.setex(
                key, self._dedup_window, payload
            )

            logger.debug(
                f"Marked alert sent: {symbol}/{strategy_name}",
                extra={
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "ttl": self._dedup_window,
                },
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to mark alert sent for "
                f"{symbol}/{strategy_name}: {e}",
                exc_info=True,
            )
            return False

    def get_recent_alerts(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all alerts recorded in Redis within the last *hours*.

        This scans keys matching the dedup prefix and returns their
        stored payloads.

        Args:
            hours: Look-back window in hours (default 24).

        Returns:
            List of alert record dictionaries, each containing
            ``symbol``, ``strategy_name``, and ``sent_at``.
            Returns an empty list if Redis is unavailable.
        """
        try:
            if not self._redis.is_connected:
                logger.warning(
                    "Redis unavailable, cannot retrieve recent alerts"
                )
                return []

            prefix = f"{REDIS_PREFIX_ALERT_SEEN}:"
            keys = self._redis.client.keys(f"{prefix}*")

            if not keys:
                return []

            cutoff = datetime.now(timezone.utc).timestamp() - (
                hours * 3600
            )

            recent: List[Dict[str, Any]] = []
            for key in keys:
                raw = self._redis.client.get(key)
                if not raw:
                    continue

                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                sent_at_str = data.get("sent_at")
                if sent_at_str:
                    try:
                        sent_ts = datetime.fromisoformat(
                            sent_at_str
                        ).timestamp()
                        if sent_ts < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

                recent.append(data)

            logger.debug(
                f"Retrieved {len(recent)} recent alerts "
                f"(last {hours}h)",
            )
            return recent

        except Exception as e:
            logger.error(
                f"Failed to retrieve recent alerts: {e}",
                exc_info=True,
            )
            return []

    def clear_history(self) -> int:
        """
        Delete all deduplication keys from Redis.

        Returns:
            Number of keys deleted.  Returns 0 if Redis is unavailable.
        """
        try:
            if not self._redis.is_connected:
                logger.warning(
                    "Redis unavailable, cannot clear dedup history"
                )
                return 0

            deleted = self._redis.flush_prefix(
                f"{REDIS_PREFIX_ALERT_SEEN}:"
            )

            logger.info(
                f"Cleared {deleted} deduplication key(s) from Redis"
            )
            return deleted

        except Exception as e:
            logger.error(
                f"Failed to clear dedup history: {e}",
                exc_info=True,
            )
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_key(symbol: str, strategy_name: str) -> str:
        """
        Build a Redis key for the deduplication entry.

        Args:
            symbol: Stock ticker symbol.
            strategy_name: Strategy name.

        Returns:
            Redis key string in the form
            ``alert_seen:<symbol>:<strategy_name>``.
        """
        return f"{REDIS_PREFIX_ALERT_SEEN}:{symbol}:{strategy_name}"

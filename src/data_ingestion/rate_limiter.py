"""
Rate Limiter (Token Bucket)

Purpose:
    Implements token bucket rate limiting.
    Uses Redis for distributed rate limiting.
    Prevents API abuse across all data sources.

Dependencies:
    - Redis for distributed state

Logging:
    - Rate limit hits at WARNING
    - Token replenishment at DEBUG

Fallbacks:
    If Redis unavailable, uses in-memory limiter.
"""

import asyncio
import time
from typing import Optional

from src.monitoring.logger import get_logger
from src.monitoring.metrics import rate_limit_remaining_gauge
from src.storage.redis_handler import RedisHandler
from src.utils.constants import REDIS_PREFIX_RATE_LIMIT

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter with Redis backend."""

    def __init__(
        self,
        source_name: str,
        max_per_minute: int = 40,
        max_per_hour: int = 2000,
        redis_handler: Optional[RedisHandler] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            source_name: Data source identifier.
            max_per_minute: Maximum requests per minute.
            max_per_hour: Maximum requests per hour.
            redis_handler: Optional Redis handler for distributed
                          rate limiting.
        """
        self.source_name = source_name
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.redis = redis_handler or RedisHandler()

        # In-memory fallback
        self._minute_tokens = max_per_minute
        self._hour_tokens = max_per_hour
        self._last_minute_refill = time.time()
        self._last_hour_refill = time.time()

        logger.info(
            f"Rate limiter initialized for {source_name}: "
            f"{max_per_minute}/min, {max_per_hour}/hr"
        )

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a rate limit token.

        Args:
            timeout: Maximum time to wait for a token.

        Returns:
            True if token acquired, False if timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._try_acquire():
                return True

            # Wait and retry
            wait_time = min(
                1.0, timeout - (time.time() - start_time)
            )
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        logger.warning(
            f"Rate limit timeout for {self.source_name} "
            f"after {timeout}s"
        )
        return False

    def _try_acquire(self) -> bool:
        """
        Try to acquire a token (Redis or in-memory).

        Returns:
            True if token available.
        """
        if self.redis.is_connected:
            return self._try_acquire_redis()
        return self._try_acquire_memory()

    def _try_acquire_redis(self) -> bool:
        """
        Try to acquire token using Redis.

        Returns:
            True if token acquired.
        """
        now = time.time()
        minute_key = (
            f"{REDIS_PREFIX_RATE_LIMIT}:{self.source_name}:"
            f"minute:{int(now / 60)}"
        )
        hour_key = (
            f"{REDIS_PREFIX_RATE_LIMIT}:{self.source_name}:"
            f"hour:{int(now / 3600)}"
        )

        try:
            # Check minute limit
            minute_count = self.redis.get_counter(minute_key)
            if minute_count >= self.max_per_minute:
                remaining = 60 - (now % 60)
                logger.debug(
                    f"Minute rate limit hit for "
                    f"{self.source_name}, "
                    f"reset in {remaining:.0f}s"
                )
                rate_limit_remaining_gauge.labels(
                    source=self.source_name
                ).set(0)
                return False

            # Check hour limit
            hour_count = self.redis.get_counter(hour_key)
            if hour_count >= self.max_per_hour:
                remaining = 3600 - (now % 3600)
                logger.debug(
                    f"Hour rate limit hit for "
                    f"{self.source_name}, "
                    f"reset in {remaining:.0f}s"
                )
                rate_limit_remaining_gauge.labels(
                    source=self.source_name
                ).set(0)
                return False

            # Acquire tokens
            pipe = self.redis.client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            pipe.execute()

            remaining = self.max_per_minute - minute_count - 1
            rate_limit_remaining_gauge.labels(
                source=self.source_name
            ).set(remaining)

            return True

        except Exception as e:
            logger.debug(
                f"Redis rate limit check failed: {e}, "
                f"falling back to memory"
            )
            return self._try_acquire_memory()

    def _try_acquire_memory(self) -> bool:
        """
        Try to acquire token using in-memory counter.

        Returns:
            True if token acquired.
        """
        now = time.time()

        # Refill minute tokens
        minute_elapsed = now - self._last_minute_refill
        if minute_elapsed >= 60:
            self._minute_tokens = self.max_per_minute
            self._last_minute_refill = now

        # Refill hour tokens
        hour_elapsed = now - self._last_hour_refill
        if hour_elapsed >= 3600:
            self._hour_tokens = self.max_per_hour
            self._last_hour_refill = now

        # Check availability
        if self._minute_tokens <= 0 or self._hour_tokens <= 0:
            return False

        # Consume token
        self._minute_tokens -= 1
        self._hour_tokens -= 1

        rate_limit_remaining_gauge.labels(
            source=self.source_name
        ).set(self._minute_tokens)

        return True

    def get_remaining(self) -> dict:
        """
        Get remaining rate limit tokens.

        Returns:
            Dictionary with remaining tokens.
        """
        if self.redis.is_connected:
            now = time.time()
            minute_key = (
                f"{REDIS_PREFIX_RATE_LIMIT}:"
                f"{self.source_name}:"
                f"minute:{int(now / 60)}"
            )
            hour_key = (
                f"{REDIS_PREFIX_RATE_LIMIT}:"
                f"{self.source_name}:"
                f"hour:{int(now / 3600)}"
            )

            minute_used = self.redis.get_counter(minute_key)
            hour_used = self.redis.get_counter(hour_key)

            return {
                "minute_remaining": max(
                    0, self.max_per_minute - minute_used
                ),
                "hour_remaining": max(
                    0, self.max_per_hour - hour_used
                ),
            }

        return {
            "minute_remaining": self._minute_tokens,
            "hour_remaining": self._hour_tokens,
        }

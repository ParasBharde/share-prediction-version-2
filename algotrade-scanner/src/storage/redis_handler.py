"""
Redis Cache Handler

Purpose:
    Manages Redis cache operations.
    Provides caching, locking, and rate limiting support.

Dependencies:
    - redis-py

Logging:
    - Cache operations at DEBUG
    - Connection events at INFO
    - Failures at ERROR

Fallbacks:
    If Redis unavailable, operations return None/False gracefully.
"""

import json
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import redis

from src.monitoring.logger import get_logger
from src.monitoring.metrics import cache_hit_counter, cache_miss_counter

logger = get_logger(__name__)


class RedisHandler:
    """Manages Redis cache and utility operations."""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis connection.

        Args:
            redis_url: Redis connection URL.
        """
        self.redis_url = (
            redis_url
            or os.environ.get("REDIS_URL")
            or "redis://localhost:6379/0"
        )

        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self.client.ping()
            self._connected = True
            logger.info("Redis connected successfully")
        except redis.ConnectionError as e:
            self._connected = False
            logger.warning(
                f"Redis connection failed: {e}. "
                f"Operating without cache."
            )

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected:
            return False
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            self._connected = False
            return False

    async def get_json(self, key: str) -> Optional[Dict]:
        """
        Get JSON data from cache.

        Args:
            key: Cache key.

        Returns:
            Parsed JSON data or None.
        """
        return self.get_json_sync(key)

    def get_json_sync(self, key: str) -> Optional[Dict]:
        """
        Synchronous version of get_json.

        Args:
            key: Cache key.

        Returns:
            Parsed JSON data or None.
        """
        if not self.is_connected:
            return None

        try:
            data = self.client.get(key)
            if data:
                cache_hit_counter.labels(cache_type="redis").inc()
                return json.loads(data)
            cache_miss_counter.labels(cache_type="redis").inc()
            return None
        except Exception as e:
            logger.debug(f"Cache get failed for {key}: {e}")
            return None

    async def set_json(
        self, key: str, data: Dict, ttl: int = 300
    ) -> bool:
        """
        Set JSON data in cache.

        Args:
            key: Cache key.
            data: Data to cache.
            ttl: Time-to-live in seconds.

        Returns:
            True if successful.
        """
        return self.set_json_sync(key, data, ttl)

    def set_json_sync(
        self, key: str, data: Dict, ttl: int = 300
    ) -> bool:
        """
        Synchronous version of set_json.

        Args:
            key: Cache key.
            data: Data to cache.
            ttl: Time-to-live in seconds.

        Returns:
            True if successful.
        """
        if not self.is_connected:
            return False

        try:
            serialized = json.dumps(data, default=str)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.debug(f"Cache set failed for {key}: {e}")
            return False

    def is_seen(self, key: str) -> bool:
        """
        Check if a key has been seen (for deduplication).

        Args:
            key: Dedup key.

        Returns:
            True if key exists.
        """
        if not self.is_connected:
            return False

        try:
            return bool(self.client.exists(key))
        except Exception:
            return False

    def mark_seen(self, key: str, ttl: int = 86400) -> bool:
        """
        Mark a key as seen (for deduplication).

        Args:
            key: Dedup key.
            ttl: Time-to-live in seconds.

        Returns:
            True if successful.
        """
        if not self.is_connected:
            return False

        try:
            self.client.setex(key, ttl, "1")
            return True
        except Exception:
            return False

    def acquire_lock(
        self, lock_name: str, timeout: int = 30
    ) -> Optional[redis.lock.Lock]:
        """
        Acquire a distributed lock.

        Args:
            lock_name: Name of the lock.
            timeout: Lock timeout in seconds.

        Returns:
            Lock object if acquired, None otherwise.
        """
        if not self.is_connected:
            return None

        try:
            lock = self.client.lock(
                f"lock:{lock_name}", timeout=timeout
            )
            if lock.acquire(blocking=False):
                return lock
            return None
        except Exception as e:
            logger.debug(
                f"Failed to acquire lock {lock_name}: {e}"
            )
            return None

    def release_lock(self, lock: redis.lock.Lock) -> None:
        """
        Release a distributed lock.

        Args:
            lock: Lock object to release.
        """
        try:
            lock.release()
        except Exception as e:
            logger.debug(f"Failed to release lock: {e}")

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            key: Counter key.
            amount: Increment amount.

        Returns:
            New counter value.
        """
        if not self.is_connected:
            return 0

        try:
            return self.client.incr(key, amount)
        except Exception:
            return 0

    def get_counter(self, key: str) -> int:
        """
        Get counter value.

        Args:
            key: Counter key.

        Returns:
            Counter value.
        """
        if not self.is_connected:
            return 0

        try:
            val = self.client.get(key)
            return int(val) if val else 0
        except Exception:
            return 0

    def health_check(self) -> bool:
        """
        Check Redis connectivity.

        Returns:
            True if Redis is reachable.
        """
        return self.is_connected

    def flush_prefix(self, prefix: str) -> int:
        """
        Delete all keys matching a prefix.

        Args:
            prefix: Key prefix to match.

        Returns:
            Number of keys deleted.
        """
        if not self.is_connected:
            return 0

        try:
            keys = self.client.keys(f"{prefix}*")
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.debug(
                f"Failed to flush prefix {prefix}: {e}"
            )
            return 0

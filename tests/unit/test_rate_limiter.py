"""Unit tests for rate limiter."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.data_ingestion.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for token bucket rate limiter."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter with in-memory backend."""
        mock_redis = MagicMock()
        mock_redis.is_connected = False
        return RateLimiter(
            source_name="test",
            max_per_minute=5,
            max_per_hour=100,
            redis_handler=mock_redis,
        )

    def test_init(self, limiter):
        """Test limiter initialization."""
        assert limiter.source_name == "test"
        assert limiter.max_per_minute == 5
        assert limiter.max_per_hour == 100

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful token acquisition."""
        result = await limiter.acquire(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_exhausts_minute_tokens(self, limiter):
        """Test that minute tokens get exhausted."""
        for _ in range(5):
            result = await limiter.acquire(timeout=0.1)
            assert result is True

        # 6th attempt should fail (within same minute)
        result = await limiter.acquire(timeout=0.5)
        assert result is False

    def test_memory_fallback_refill(self, limiter):
        """Test in-memory token refill."""
        # Exhaust all tokens
        limiter._minute_tokens = 0
        assert limiter._try_acquire_memory() is False

        # Simulate minute passing
        limiter._last_minute_refill = time.time() - 61
        assert limiter._try_acquire_memory() is True

    def test_get_remaining(self, limiter):
        """Test remaining tokens report."""
        remaining = limiter.get_remaining()
        assert "minute_remaining" in remaining
        assert "hour_remaining" in remaining
        assert remaining["minute_remaining"] == 5
        assert remaining["hour_remaining"] == 100

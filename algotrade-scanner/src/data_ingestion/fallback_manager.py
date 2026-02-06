"""
Data Source Fallback Manager

Purpose:
    Implements waterfall pattern for data fetching.
    Tries primary source -> fallback1 -> fallback2.
    Uses stale cache as last resort.
    Tracks failure rates for circuit breaking.

Dependencies:
    - nse_fetcher, yahoo_fetcher, alpha_vantage_fetcher
    - redis_handler for caching
    - rate_limiter for throttling

Logging:
    - Each fetch attempt at DEBUG
    - Fallback transitions at WARN
    - Complete failures at ERROR

Fallbacks:
    1. NSE Official API (primary)
    2. Yahoo Finance (fallback 1)
    3. Alpha Vantage (fallback 2)
    4. Stale cache (last resort, up to 1 hour old)
    5. Return None (unavailable)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.data_ingestion.alpha_vantage_fetcher import (
    AlphaVantageFetcher,
)
from src.data_ingestion.nse_fetcher import NSEFetcher
from src.data_ingestion.yahoo_fetcher import YahooFetcher
from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    data_fetch_failure_counter,
    data_fetch_success_counter,
    fallback_usage_counter,
)
from src.storage.redis_handler import RedisHandler
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class FallbackManager:
    """Manages data source fallback logic with circuit breakers."""

    def __init__(self):
        """Initialize with all fetchers and config."""
        self.config = load_config("data_sources")
        self.redis = RedisHandler()

        # Initialize fetchers based on config
        self.fetchers = {}
        if self.config.get("primary", {}).get("enabled", True):
            self.fetchers["primary"] = NSEFetcher()
        if self.config.get("fallback_1", {}).get("enabled", True):
            self.fetchers["fallback_1"] = YahooFetcher()
        if self.config.get("fallback_2", {}).get("enabled", True):
            self.fetchers["fallback_2"] = AlphaVantageFetcher()

        # Circuit breaker state per source
        self.circuit_breakers: Dict[str, Dict] = {
            name: {"failures": 0, "last_failure": None}
            for name in self.fetchers
        }

        logger.info(
            "FallbackManager initialized",
            extra={
                "enabled_fetchers": list(self.fetchers.keys())
            },
        )

    async def fetch_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data with fallback strategy.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE').
            start_date: Start date for data.
            end_date: End date for data.

        Returns:
            Dict with OHLCV data or None if all sources fail.
        """
        # Check cache first
        cache_key = (
            f"ohlcv:{symbol}:"
            f"{start_date.date()}:{end_date.date()}"
        )
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for {symbol}")
            return cached_data

        # Try sources in order
        for source_name, fetcher in self.fetchers.items():
            # Check circuit breaker
            if self._is_circuit_open(source_name):
                logger.debug(
                    f"Circuit breaker open for {source_name}, "
                    f"skipping"
                )
                continue

            try:
                logger.debug(
                    f"Attempting fetch from {source_name} "
                    f"for {symbol}"
                )

                data = await fetcher.fetch(
                    symbol, start_date, end_date
                )

                if data and data.get("records"):
                    # Success
                    self._record_success(source_name)
                    await self._cache_data(cache_key, data)

                    logger.info(
                        f"Successfully fetched {symbol} "
                        f"from {source_name}",
                        extra={
                            "source": source_name,
                            "symbol": symbol,
                            "records": data.get("count", 0),
                        },
                    )
                    data_fetch_success_counter.labels(
                        source=source_name
                    ).inc()

                    return data

            except Exception as e:
                self._record_failure(source_name)

                logger.warning(
                    f"Failed to fetch {symbol} from "
                    f"{source_name}: {e}",
                    exc_info=True,
                    extra={
                        "source": source_name,
                        "symbol": symbol,
                        "error_type": type(e).__name__,
                    },
                )

                data_fetch_failure_counter.labels(
                    source=source_name
                ).inc()
                fallback_usage_counter.labels(
                    from_source=source_name,
                    to_source="next",
                ).inc()

                # Wait before trying next source
                delay = (
                    self.config.get("fallback_strategy", {})
                    .get("escalation_delay", 5)
                )
                await asyncio.sleep(delay)

        # All sources failed - try stale cache
        fallback_config = self.config.get(
            "fallback_strategy", {}
        )
        if fallback_config.get("use_stale_cache", True):
            logger.warning(
                f"All sources failed for {symbol}, "
                f"trying stale cache"
            )
            stale_data = await self._get_stale_cache(cache_key)
            if stale_data:
                logger.info(
                    f"Returning stale cache data for {symbol}"
                )
                fallback_usage_counter.labels(
                    from_source="all_failed",
                    to_source="stale_cache",
                ).inc()
                return stale_data

        # Complete failure
        logger.error(
            f"Unable to fetch data for {symbol} from any source"
        )
        return None

    async def _get_from_cache(
        self, key: str
    ) -> Optional[Dict]:
        """Get fresh data from cache."""
        return await self.redis.get_json(key)

    async def _get_stale_cache(
        self, key: str
    ) -> Optional[Dict]:
        """
        Get stale data from cache.
        Uses a separate key with longer TTL.
        """
        stale_key = f"stale:{key}"
        return await self.redis.get_json(stale_key)

    async def _cache_data(
        self, key: str, data: Dict
    ) -> None:
        """Cache data with appropriate TTL."""
        ttl = self.config.get("cache_ttl_eod", 86400)
        await self.redis.set_json(key, data, ttl=ttl)

        # Also store as stale cache with longer TTL
        max_stale = (
            self.config.get("fallback_strategy", {})
            .get("max_stale_age", 3600)
        )
        stale_key = f"stale:{key}"
        await self.redis.set_json(
            stale_key, data, ttl=ttl + max_stale
        )

    def _is_circuit_open(self, source_name: str) -> bool:
        """
        Check if circuit breaker is open for a source.

        Args:
            source_name: Data source name.

        Returns:
            True if circuit is open (source should be skipped).
        """
        cb = self.circuit_breakers.get(source_name, {})

        if cb.get("failures", 0) == 0:
            return False

        # Open circuit after 5 consecutive failures
        if cb["failures"] >= 5:
            if cb.get("last_failure"):
                cooldown = timedelta(minutes=5)
                if datetime.now() - cb["last_failure"] > cooldown:
                    # Reset circuit
                    cb["failures"] = 0
                    cb["last_failure"] = None
                    logger.info(
                        f"Circuit breaker reset for "
                        f"{source_name}"
                    )
                    return False
                return True

        return False

    def _record_success(self, source_name: str) -> None:
        """Record successful fetch, reset circuit breaker."""
        cb = self.circuit_breakers.get(source_name, {})
        cb["failures"] = 0
        cb["last_failure"] = None

    def _record_failure(self, source_name: str) -> None:
        """Record failed fetch, increment circuit breaker."""
        cb = self.circuit_breakers.get(source_name, {})
        cb["failures"] = cb.get("failures", 0) + 1
        cb["last_failure"] = datetime.now()

    async def close(self) -> None:
        """Close all fetcher sessions."""
        for fetcher in self.fetchers.values():
            await fetcher.close()

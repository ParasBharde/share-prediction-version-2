"""
Abstract Base Fetcher

Purpose:
    Defines the interface for all data fetchers.
    Provides common retry and timeout logic.

Dependencies:
    - aiohttp for async HTTP

Logging:
    Fetch attempts at DEBUG, failures at WARNING.

Fallbacks:
    Retries with exponential backoff.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

from src.monitoring.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers."""

    def __init__(self, source_name: str):
        """
        Initialize base fetcher.

        Args:
            source_name: Name of the data source.
        """
        self.source_name = source_name
        self.config = load_config("data_sources")
        self._session: Optional[aiohttp.ClientSession] = None

    @abstractmethod
    async def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Stock symbol.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dictionary with OHLCV data or None on failure.
        """
        pass

    @abstractmethod
    async def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current quote for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Quote data dictionary or None.
        """
        pass

    @abstractmethod
    async def fetch_stock_list(self, index: str) -> List[str]:
        """
        Fetch list of stocks in an index.

        Args:
            index: Index name (e.g., 'NIFTY500').

        Returns:
            List of stock symbols.
        """
        pass

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=30, connect=5, sock_read=10
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _request_with_retry(
        self,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        backoff_max: float = 16.0,
        auth_failure_handler: Optional[
            Callable[[], Awaitable[bool]]
        ] = None,
    ) -> Optional[Dict]:
        """
        Make HTTP request with retry and backoff.

        Args:
            url: Request URL.
            headers: Optional HTTP headers.
            params: Optional query parameters.
            max_retries: Maximum retry attempts.
            backoff_factor: Backoff multiplier.
            backoff_max: Maximum backoff delay.

        Returns:
            Response JSON or None on failure.
        """
        session = await self._get_session()
        last_status = None

        for attempt in range(max_retries):
            try:
                async with session.get(
                    url, headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        try:
                            return await response.json(
                                content_type=None
                            )
                        except (
                            aiohttp.ContentTypeError,
                            json.JSONDecodeError,
                        ) as exc:
                            logger.warning(
                                f"Non-JSON response from "
                                f"{self.source_name}, retrying",
                                extra={
                                    "source": self.source_name,
                                    "url": url,
                                    "error": str(exc),
                                    "attempt": attempt + 1,
                                },
                            )
                            if (
                                auth_failure_handler
                                and attempt < max_retries - 1
                            ):
                                await auth_failure_handler()
                                delay = min(
                                    backoff_factor ** (attempt + 1),
                                    backoff_max,
                                )
                                await asyncio.sleep(delay)
                                continue
                            last_status = response.status
                            break
                    elif response.status == 429:
                        # Rate limited - retry with backoff
                        delay = min(
                            backoff_factor ** (attempt + 1),
                            backoff_max,
                        )
                        logger.warning(
                            f"Rate limited by {self.source_name}, "
                            f"retrying in {delay}s",
                            extra={
                                "source": self.source_name,
                                "attempt": attempt + 1,
                                "delay": delay,
                            },
                        )
                        await asyncio.sleep(delay)
                    elif response.status in (403, 401):
                        if (
                            auth_failure_handler
                            and attempt < max_retries - 1
                        ):
                            logger.warning(
                                f"HTTP {response.status} from "
                                f"{self.source_name} (attempting "
                                f"auth refresh)",
                                extra={
                                    "source": self.source_name,
                                    "status": response.status,
                                    "url": url,
                                },
                            )
                            refreshed = await auth_failure_handler()
                            if refreshed:
                                delay = min(
                                    backoff_factor ** (attempt + 1),
                                    backoff_max,
                                )
                                await asyncio.sleep(delay)
                                continue

                        # Auth/forbidden - no point retrying
                        logger.warning(
                            f"HTTP {response.status} from "
                            f"{self.source_name} (auth failure, "
                            f"not retrying)",
                            extra={
                                "source": self.source_name,
                                "status": response.status,
                                "url": url,
                            },
                        )
                        last_status = response.status
                        break
                    else:
                        last_status = response.status
                        logger.warning(
                            f"HTTP {response.status} from "
                            f"{self.source_name}",
                            extra={
                                "source": self.source_name,
                                "status": response.status,
                                "url": url,
                            },
                        )

            except (asyncio.CancelledError, KeyboardInterrupt):
                # Never swallow cancellation or interrupt
                raise

            except asyncio.TimeoutError:
                delay = min(
                    backoff_factor ** (attempt + 1), backoff_max
                )
                logger.warning(
                    f"Timeout from {self.source_name}, "
                    f"attempt {attempt + 1}/{max_retries}",
                    extra={
                        "source": self.source_name,
                        "attempt": attempt + 1,
                    },
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)

            except aiohttp.ClientError as e:
                delay = min(
                    backoff_factor ** (attempt + 1), backoff_max
                )
                logger.warning(
                    f"Client error from {self.source_name}: {e}",
                    extra={
                        "source": self.source_name,
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)

        logger.error(
            f"All {max_retries} attempts failed for "
            f"{self.source_name}"
            + (f" (last status: {last_status})" if last_status else ""),
            extra={"source": self.source_name, "url": url},
        )
        return None

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            # Give the underlying SSL transports time to close
            await asyncio.sleep(0.25)

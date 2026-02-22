"""
NSE Official API Fetcher (Primary Data Source)

Purpose:
    Fetches stock data from NSE India's official API.
    Handles session management and cookie rotation.

Dependencies:
    - aiohttp for HTTP requests
    - base_fetcher for interface

Logging:
    - Fetch attempts at DEBUG
    - Session refresh at INFO
    - Failures at WARNING/ERROR

Fallbacks:
    - Session refresh on 403/401
    - Retries with exponential backoff
"""

import asyncio
import csv
import random
from io import StringIO
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

try:
    # Optional dependency: more stable NSE symbol endpoints.
    from nsepython import nse_eq_symbols
except Exception:
    nse_eq_symbols = None

from src.data_ingestion.base_fetcher import BaseFetcher
from src.monitoring.logger import get_logger
from src.utils.constants import (
    NSE_API_BASE,
    NSE_ARCHIVE_BASE,
    NSE_BASE_URL,
    NSE_HEADERS,
    NSE_HOMEPAGE_HEADERS,
    NSE_INDEX_ARCHIVE_PATH,
    NSE_USER_AGENTS,
)

logger = get_logger(__name__)


class NSEFetcher(BaseFetcher):
    """Fetches stock data from NSE India API."""

    def __init__(self):
        """Initialize NSE fetcher."""
        super().__init__("nse_official")
        self._cookies: Optional[Dict] = None
        self._cookie_expiry: Optional[datetime] = None
        # Rotate user-agent per session to reduce WAF fingerprinting
        self._user_agent: str = random.choice(NSE_USER_AGENTS)

        source_config = self.config.get("primary", {})
        self.retry_config = source_config.get("retry", {})
        self.timeout_config = source_config.get("timeout", {})

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with cookie jar."""
        if self._session is None or self._session.closed:
            jar = aiohttp.CookieJar()
            timeout = aiohttp.ClientTimeout(
                total=30, connect=5, sock_read=10
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                cookie_jar=jar,
            )
        return self._session

    async def _refresh_session(self) -> bool:
        """
        Refresh NSE session cookies by visiting a multi-page warmup sequence.

        Strategy:
        - Rotate user-agent on each refresh cycle
        - Visit homepage → market-data page → option-chain page
          (mimics a real trader browsing NSE)
        - Add human-like jitter delays between page visits
        - Handle 429 rate-limit with Retry-After respect
        - Full session recreation on repeated 403s

        Returns:
            True if session was refreshed and cookies obtained.
        """
        # Rotate user-agent to reduce WAF fingerprinting
        self._user_agent = random.choice(NSE_USER_AGENTS)

        warmup_sequence = [
            (NSE_BASE_URL, None),
            (f"{NSE_BASE_URL}/market-data/live-equity-market", NSE_BASE_URL),
            (f"{NSE_BASE_URL}/option-chain", f"{NSE_BASE_URL}/market-data/live-equity-market"),
        ]

        try:
            session = await self._get_session()

            for url, referer in warmup_sequence:
                page_headers = {**NSE_HOMEPAGE_HEADERS, "User-Agent": self._user_agent}
                if referer:
                    page_headers["Referer"] = referer
                else:
                    page_headers.pop("Referer", None)
                    page_headers["Sec-Fetch-Site"] = "none"

                try:
                    async with session.get(
                        url,
                        headers=page_headers,
                        allow_redirects=True,
                    ) as response:
                        await response.read()
                        status = response.status
                        logger.debug(
                            f"NSE warmup {url} → HTTP {status}"
                        )

                        if status == 429:
                            retry_after = int(
                                response.headers.get("Retry-After", "10")
                            )
                            logger.warning(
                                f"NSE rate-limited (429) during warmup, "
                                f"sleeping {retry_after}s"
                            )
                            await asyncio.sleep(retry_after + random.uniform(1, 3))

                        elif status in (401, 403):
                            logger.warning(
                                f"NSE auth failure ({status}) on warmup {url}"
                            )
                            await self._reset_session()
                            return False

                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    logger.debug(f"NSE warmup request failed for {url}: {exc}")

                # Human-like delay between page visits
                await asyncio.sleep(random.uniform(0.7, 1.8))

            # Collect cookies
            self._cookies = {
                cookie.key: cookie.value
                for cookie in session.cookie_jar
            }
            self._cookie_expiry = datetime.now()

            if self._cookies:
                logger.info(
                    f"NSE session cookies refreshed "
                    f"(keys: {list(self._cookies.keys())})"
                )
                return True
            else:
                logger.warning(
                    "NSE warmup completed but no cookies were set"
                )
                return False

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.error(
                f"Failed to refresh NSE session: {e}",
                exc_info=True,
            )
            return False

    async def _reset_session(self) -> None:
        """Reset session and cookie state."""
        self._cookies = None
        self._cookie_expiry = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> None:
        """Ensure we have valid session cookies."""
        if self._cookies is None:
            await self._refresh_session()
        elif self._cookie_expiry:
            # Refresh cookies if older than 3 minutes
            elapsed = (
                datetime.now() - self._cookie_expiry
            ).total_seconds()
            if elapsed > 180:
                await self._refresh_session()

    async def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical OHLCV data from NSE.

        Args:
            symbol: NSE stock symbol.
            start_date: Start date for data.
            end_date: End date for data.

        Returns:
            Dictionary with OHLCV data or None.
        """
        await self._ensure_session()

        url = f"{NSE_API_BASE}/historical/cm/equity"
        params = {
            "symbol": symbol,
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        }

        # Use API headers with Referer pointing to the equity page
        headers = {**NSE_HEADERS}
        headers["Referer"] = (
            f"https://www.nseindia.com/get-quotes/"
            f"equity?symbol={symbol}"
        )

        data = await self._request_with_retry(
            url=url,
            headers=headers,
            params=params,
            max_retries=self.retry_config.get("max_attempts", 1),
            backoff_factor=self.retry_config.get(
                "backoff_factor", 2
            ),
            backoff_max=self.retry_config.get("backoff_max", 16),
            auth_failure_handler=self._refresh_session,
        )

        if data and "data" in data:
            return self._parse_historical_data(data["data"], symbol)

        # Try refreshing session and retrying once
        if data is None:
            logger.info(
                f"Retrying {symbol} after session refresh"
            )
            refreshed = await self._refresh_session()

            if not refreshed:
                logger.warning(
                    f"Session refresh failed, skipping retry "
                    f"for {symbol}"
                )
                return None

            data = await self._request_with_retry(
                url=url,
                headers=headers,
                params=params,
                max_retries=1,
            )

            if data and "data" in data:
                return self._parse_historical_data(
                    data["data"], symbol
                )

        return None

    async def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current quote from NSE.

        Args:
            symbol: NSE stock symbol.

        Returns:
            Quote data or None.
        """
        await self._ensure_session()

        url = f"{NSE_API_BASE}/quote-equity"
        params = {"symbol": symbol}

        headers = {**NSE_HEADERS}
        headers["Referer"] = (
            f"https://www.nseindia.com/get-quotes/"
            f"equity?symbol={symbol}"
        )

        data = await self._request_with_retry(
            url=url,
            headers=headers,
            params=params,
            auth_failure_handler=self._refresh_session,
        )
        if data and "priceInfo" in data:
            price_info = data["priceInfo"]
            return {
                "symbol": symbol,
                "open": price_info.get("open"),
                "high": price_info.get("intraDayHighLow", {}).get(
                    "max"
                ),
                "low": price_info.get("intraDayHighLow", {}).get(
                    "min"
                ),
                "close": price_info.get("lastPrice"),
                "previous_close": price_info.get("previousClose"),
                "change": price_info.get("change"),
                "change_pct": price_info.get("pChange"),
                "volume": data.get("securityWiseDP", {}).get(
                    "quantityTraded"
                ),
            }

        return None

    async def fetch_stock_list(self, index: str) -> List[str]:
        """
        Fetch list of stocks in an NSE index.

        Args:
            index: Index name (e.g., 'NIFTY 500').

        Returns:
            List of stock symbols.
        """
        # Prefer nsepython for full equity universe when available.
        library_symbols = await self._fetch_symbols_via_nsepython(index)
        if library_symbols:
            logger.info(
                f"Fetched {len(library_symbols)} symbols via nsepython "
                f"for {index}"
            )
            return library_symbols

        await self._ensure_session()

        url = f"{NSE_API_BASE}/equity-stockIndices"
        params = {"index": index}

        headers = {**NSE_HEADERS}
        headers["Referer"] = "https://www.nseindia.com/market-data/live-equity-market"

        data = await self._request_with_retry(
            url=url,
            headers=headers,
            params=params,
            auth_failure_handler=self._refresh_session,
        )
        # print(f"Index data for {index}: {data}")
        if data and "data" in data:
            return [
                stock["symbol"]
                for stock in data["data"]
                if "symbol" in stock
            ]

        archive_symbols = await self._fetch_index_archive(
            index
        )
        if archive_symbols:
            logger.info(
                f"Fetched {len(archive_symbols)} symbols from "
                f"NSE archive for {index}"
            )
            return archive_symbols

        return []

    async def _fetch_symbols_via_nsepython(
        self, index: str
    ) -> List[str]:
        """Fetch symbols using optional nsepython helper."""
        if nse_eq_symbols is None:
            return []

        try:
            symbols = await asyncio.to_thread(nse_eq_symbols)
            if not isinstance(symbols, list):
                return []

            cleaned = [str(s).strip() for s in symbols if s]
            if not cleaned:
                return []

            index_map = {
                "NIFTY 50": "NIFTY50",
                "NIFTY 100": "NIFTY100",
                "NIFTY 500": "NIFTY500",
            }
            target_index = index_map.get(index)

            # nsepython provides all equity symbols. For index-specific
            # scans, keep using API/archive path for precise membership.
            if target_index is None:
                return cleaned

            return []
        except Exception as exc:
            logger.debug(
                f"nsepython symbol fetch failed for {index}: {exc}"
            )
            return []

    async def _fetch_index_archive(
        self, index: str
    ) -> List[str]:
        """
        Fetch index constituents from NSE archive CSVs.

        Args:
            index: Index name (e.g., 'NIFTY 500').

        Returns:
            List of stock symbols.
        """
        archive_map = {
            "NIFTY 50": "ind_nifty50list.csv",
            "NIFTY 100": "ind_nifty100list.csv",
            "NIFTY 500": "ind_nifty500list.csv",
        }
        archive_file = archive_map.get(index)
        if not archive_file:
            return []

        url = (
            f"{NSE_ARCHIVE_BASE}{NSE_INDEX_ARCHIVE_PATH}"
            f"/{archive_file}"
        )
        headers = {
            **NSE_HEADERS,
            "Accept": "text/csv",
            "Referer": NSE_BASE_URL,
        }

        session = await self._get_session()
        logger.debug(f"Fetching index archive for {index} from {url}")
        try:
            async with session.get(
                url, headers=headers
            ) as response:
                if response.status != 200:
                    logger.warning(
                        f"NSE archive fetch got HTTP "
                        f"{response.status} for {index}"
                    )
                    return []
                body = await response.text()
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning(
                f"NSE archive fetch failed for {index}: {exc}"
            )
            return []

        reader = csv.DictReader(StringIO(body))
        symbols = [
            row["Symbol"].strip()
            for row in reader
            if row.get("Symbol")
        ]
        return symbols

    def _parse_historical_data(
        self, raw_data: List[Dict], symbol: str
    ) -> Dict[str, Any]:
        """
        Parse NSE historical data response.

        Args:
            raw_data: Raw data from NSE API.
            symbol: Stock symbol.

        Returns:
            Parsed OHLCV data dictionary.
        """
        records = []
        for row in raw_data:
            try:
                records.append(
                    {
                        "date": datetime.strptime(
                            row.get("CH_TIMESTAMP", ""),
                            "%Y-%m-%d",
                        ),
                        "open": float(
                            row.get("CH_OPENING_PRICE", 0)
                        ),
                        "high": float(
                            row.get("CH_TRADE_HIGH_PRICE", 0)
                        ),
                        "low": float(
                            row.get("CH_TRADE_LOW_PRICE", 0)
                        ),
                        "close": float(
                            row.get("CH_CLOSING_PRICE", 0)
                        ),
                        "volume": int(
                            row.get("CH_TOT_TRADED_QTY", 0)
                        ),
                        "turnover": float(
                            row.get("CH_TOT_TRADED_VAL", 0)
                        ),
                        "trades": int(
                            row.get("CH_TOTAL_TRADES", 0)
                        ),
                        "delivery_volume": int(
                            row.get("COP_DELIV_QTY", 0)
                        ),
                        "delivery_percent": float(
                            row.get("COP_DELIV_PERC", 0)
                        ),
                        "source": "nse_official",
                    }
                )
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"Skipping malformed row for {symbol}: {e}"
                )
                continue

        return {
            "symbol": symbol,
            "source": "nse_official",
            "records": records,
            "count": len(records),
        }

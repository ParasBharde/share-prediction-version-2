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
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from src.data_ingestion.base_fetcher import BaseFetcher
from src.monitoring.logger import get_logger
from src.utils.constants import (
    NSE_API_BASE,
    NSE_BASE_URL,
    NSE_HEADERS,
)

logger = get_logger(__name__)


class NSEFetcher(BaseFetcher):
    """Fetches stock data from NSE India API."""

    def __init__(self):
        """Initialize NSE fetcher."""
        super().__init__("nse_official")
        self._cookies: Optional[Dict] = None
        self._cookie_expiry: Optional[datetime] = None

        source_config = self.config.get("primary", {})
        self.retry_config = source_config.get("retry", {})
        self.timeout_config = source_config.get("timeout", {})

    async def _refresh_session(self) -> bool:
        """
        Refresh NSE session cookies by visiting main page.
        NSE requires valid cookies from the main site.

        Returns:
            True if session was refreshed successfully.
        """
        try:
            session = await self._get_session()
            async with session.get(
                NSE_BASE_URL, headers=NSE_HEADERS
            ) as response:
                if response.status == 200:
                    self._cookies = {
                        cookie.key: cookie.value
                        for cookie in response.cookies.values()
                    }
                    self._cookie_expiry = datetime.now()
                    logger.info("NSE session cookies refreshed")
                    return True
                else:
                    logger.warning(
                        f"NSE session refresh got HTTP "
                        f"{response.status}"
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

    async def _ensure_session(self) -> None:
        """Ensure we have valid session cookies."""
        if self._cookies is None:
            await self._refresh_session()
        elif self._cookie_expiry:
            # Refresh cookies if older than 5 minutes
            elapsed = (
                datetime.now() - self._cookie_expiry
            ).total_seconds()
            if elapsed > 300:
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

        headers = {**NSE_HEADERS}
        if self._cookies:
            cookie_str = "; ".join(
                f"{k}={v}" for k, v in self._cookies.items()
            )
            headers["Cookie"] = cookie_str

        data = await self._request_with_retry(
            url=url,
            headers=headers,
            params=params,
            max_retries=self.retry_config.get("max_attempts", 3),
            backoff_factor=self.retry_config.get(
                "backoff_factor", 2
            ),
            backoff_max=self.retry_config.get("backoff_max", 16),
        )

        if data and "data" in data:
            return self._parse_historical_data(data["data"], symbol)

        # Try refreshing session and retrying once
        # Only if session refresh actually succeeds
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

            if self._cookies:
                cookie_str = "; ".join(
                    f"{k}={v}" for k, v in self._cookies.items()
                )
                headers["Cookie"] = cookie_str

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
        if self._cookies:
            cookie_str = "; ".join(
                f"{k}={v}" for k, v in self._cookies.items()
            )
            headers["Cookie"] = cookie_str

        data = await self._request_with_retry(
            url=url,
            headers=headers,
            params=params,
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
        await self._ensure_session()

        url = f"{NSE_API_BASE}/equity-stockIndices"
        params = {"index": index}

        headers = {**NSE_HEADERS}
        if self._cookies:
            cookie_str = "; ".join(
                f"{k}={v}" for k, v in self._cookies.items()
            )
            headers["Cookie"] = cookie_str

        data = await self._request_with_retry(
            url=url,
            headers=headers,
            params=params,
        )

        if data and "data" in data:
            return [
                stock["symbol"]
                for stock in data["data"]
                if "symbol" in stock
            ]

        return []

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

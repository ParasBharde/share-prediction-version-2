"""
Alpha Vantage Fetcher (Fallback 2)

Purpose:
    Fetches stock data from Alpha Vantage API as second fallback.
    Lower rate limits but reliable data.

Dependencies:
    - aiohttp for HTTP requests
    - base_fetcher for interface

Logging:
    - Fetch attempts at DEBUG
    - Rate limit warnings at WARNING

Fallbacks:
    Retries with exponential backoff.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.data_ingestion.base_fetcher import BaseFetcher
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class AlphaVantageFetcher(BaseFetcher):
    """Fetches stock data from Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        """Initialize Alpha Vantage fetcher."""
        super().__init__("alpha_vantage")

        source_config = self.config.get("fallback_2", {})
        self.api_key = os.environ.get(
            "ALPHA_VANTAGE_KEY",
            source_config.get("api_key", ""),
        )
        self.retry_config = source_config.get("retry", {})

    async def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical OHLCV data from Alpha Vantage.

        Args:
            symbol: Stock symbol with BSE/NSE suffix.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dictionary with OHLCV data or None.
        """
        if not self.api_key:
            logger.warning(
                "Alpha Vantage API key not configured"
            )
            return None

        # Alpha Vantage uses BSE symbol format for Indian stocks
        av_symbol = f"{symbol}.BSE"

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }

        data = await self._request_with_retry(
            url=self.BASE_URL,
            params=params,
            max_retries=self.retry_config.get("max_attempts", 2),
            backoff_factor=self.retry_config.get(
                "backoff_factor", 3
            ),
            backoff_max=self.retry_config.get("backoff_max", 20),
        )

        if not data:
            return None

        # Check for API errors
        if "Error Message" in data:
            logger.warning(
                f"Alpha Vantage error for {symbol}: "
                f"{data['Error Message']}"
            )
            return None

        if "Note" in data:
            logger.warning(
                f"Alpha Vantage rate limit: {data['Note']}"
            )
            return None

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return None

        records = []
        for date_str, values in time_series.items():
            try:
                record_date = datetime.strptime(
                    date_str, "%Y-%m-%d"
                )

                # Filter by date range
                if record_date < start_date or record_date > end_date:
                    continue

                records.append(
                    {
                        "date": record_date,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"]),
                        "source": "alpha_vantage",
                    }
                )
            except (ValueError, KeyError) as e:
                logger.debug(
                    f"Skipping malformed row for {symbol}: {e}"
                )
                continue

        # Sort by date
        records.sort(key=lambda x: x["date"])

        logger.debug(
            f"Fetched {len(records)} records from Alpha Vantage "
            f"for {symbol}"
        )

        return {
            "symbol": symbol,
            "source": "alpha_vantage",
            "records": records,
            "count": len(records),
        }

    async def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current quote from Alpha Vantage.

        Args:
            symbol: Stock symbol.

        Returns:
            Quote data or None.
        """
        if not self.api_key:
            return None

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": f"{symbol}.BSE",
            "apikey": self.api_key,
        }

        data = await self._request_with_retry(
            url=self.BASE_URL, params=params
        )

        if data and "Global Quote" in data:
            quote = data["Global Quote"]
            return {
                "symbol": symbol,
                "open": float(quote.get("02. open", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
                "close": float(quote.get("05. price", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "previous_close": float(
                    quote.get("08. previous close", 0)
                ),
                "change": float(quote.get("09. change", 0)),
                "change_pct": float(
                    quote.get("10. change percent", "0%").rstrip(
                        "%"
                    )
                ),
            }

        return None

    async def fetch_stock_list(self, index: str) -> List[str]:
        """
        Alpha Vantage doesn't support index constituent listing.

        Args:
            index: Index name.

        Returns:
            Empty list.
        """
        logger.debug(
            "Alpha Vantage doesn't support index listing"
        )
        return []

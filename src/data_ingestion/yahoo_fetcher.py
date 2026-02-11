"""
Yahoo Finance Fetcher (Fallback 1)

Purpose:
    Fetches stock data from Yahoo Finance as first fallback.
    Uses yfinance library for reliable data access.

Dependencies:
    - yfinance library
    - base_fetcher for interface

Logging:
    - Fetch attempts at DEBUG
    - Failures at WARNING

Fallbacks:
    Retries with exponential backoff.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf

from src.data_ingestion.base_fetcher import BaseFetcher
from src.monitoring.logger import get_logger
from src.utils.constants import YAHOO_NSE_SUFFIX

logger = get_logger(__name__)


class YahooFetcher(BaseFetcher):
    """Fetches stock data from Yahoo Finance."""

    def __init__(self):
        """Initialize Yahoo Finance fetcher."""
        super().__init__("yahoo_finance")
        self.source_config = self.config.get("fallback_1", {})

    def _nse_to_yahoo(self, symbol: str) -> str:
        """
        Convert NSE symbol to Yahoo Finance format.

        Args:
            symbol: NSE symbol (e.g., 'RELIANCE').

        Returns:
            Yahoo format (e.g., 'RELIANCE.NS').
        """
        return f"{symbol}{YAHOO_NSE_SUFFIX}"

    async def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: NSE stock symbol.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dictionary with OHLCV data or None.
        """
        yahoo_symbol = self._nse_to_yahoo(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            # yfinance excludes end_date, so add 1 day to include it
            inclusive_end = end_date + timedelta(days=1)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=inclusive_end.strftime("%Y-%m-%d"),
            )

            if df.empty:
                logger.warning(
                    f"No data from Yahoo for {symbol}"
                )
                return None

            records = []
            for idx, row in df.iterrows():
                records.append(
                    {
                        "date": idx.to_pydatetime(),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                        "source": "yahoo_finance",
                    }
                )

            logger.debug(
                f"Fetched {len(records)} records from Yahoo "
                f"for {symbol}"
            )

            return {
                "symbol": symbol,
                "source": "yahoo_finance",
                "records": records,
                "count": len(records),
            }

        except Exception as e:
            logger.warning(
                f"Yahoo Finance fetch failed for {symbol}: {e}",
                extra={
                    "source": "yahoo_finance",
                    "symbol": symbol,
                    "error": str(e),
                },
            )
            return None

    async def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current quote from Yahoo Finance.

        Args:
            symbol: NSE stock symbol.

        Returns:
            Quote data or None.
        """
        yahoo_symbol = self._nse_to_yahoo(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            if not info:
                return None

            return {
                "symbol": symbol,
                "open": info.get("open"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "close": info.get("currentPrice")
                or info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }

        except Exception as e:
            logger.warning(
                f"Yahoo quote fetch failed for {symbol}: {e}"
            )
            return None

    async def fetch_stock_list(self, index: str) -> List[str]:
        """
        Fetch stock list from Yahoo (limited support).

        Args:
            index: Index name.

        Returns:
            List of symbols (may be empty for unsupported indices).
        """
        # Yahoo Finance doesn't directly support NSE index listings
        # This would need a separate data source for index constituents
        logger.debug(
            f"Yahoo Finance doesn't support index listing for "
            f"{index}"
        )
        return []

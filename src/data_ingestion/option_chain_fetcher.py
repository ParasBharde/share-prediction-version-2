"""
Option Chain Data Fetcher

Fetches option chain data from NSE for NIFTY, BANKNIFTY, and individual stocks.
Provides OI, IV, premium, Greeks data for options strategies.
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"
NSE_EQUITY_OPTION_URL = "https://www.nseindia.com/api/option-chain-equities"
NSE_BASE_URL = "https://www.nseindia.com"

# Standard headers to mimic browser
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


class OptionChainFetcher:
    """Fetches option chain data from NSE India."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cookies = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create authenticated HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=NSE_HEADERS,
            )
            # Warm up session with NSE homepage for cookies
            await self._warm_up_session()
        return self._session

    async def _warm_up_session(self):
        """Visit NSE pages to get valid session cookies."""
        warmup_urls = [
            NSE_BASE_URL,
            f"{NSE_BASE_URL}/option-chain",
        ]
        for url in warmup_urls:
            try:
                async with self._session.get(url) as resp:
                    await resp.read()
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Warmup request to {url} failed: {e}")

    async def fetch_option_chain(
        self,
        symbol: str = "NIFTY",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch full option chain data.

        Args:
            symbol: Index name (NIFTY, BANKNIFTY) or equity symbol.

        Returns:
            Parsed option chain data with OI, IV, premiums.
        """
        session = await self._get_session()

        # Index or equity URL
        if symbol.upper() in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            url = NSE_OPTION_CHAIN_URL
        else:
            url = NSE_EQUITY_OPTION_URL

        params = {"symbol": symbol.upper()}

        for attempt in range(3):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        return self._parse_option_chain(data, symbol)
                    elif resp.status in (401, 403):
                        logger.warning(f"Auth failed for option chain, refreshing session")
                        await self._warm_up_session()
                        await asyncio.sleep(2)
                    else:
                        logger.warning(f"HTTP {resp.status} from NSE option chain")
            except Exception as e:
                logger.warning(f"Option chain fetch error (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)

        return None

    def _parse_option_chain(
        self,
        raw_data: Dict,
        symbol: str,
    ) -> Dict[str, Any]:
        """Parse NSE option chain response into structured data."""
        records = raw_data.get("records", {})
        filtered = raw_data.get("filtered", {})

        # Current underlying price
        underlying_value = records.get("underlyingValue", 0)

        # Expiry dates
        expiry_dates = records.get("expiryDates", [])
        current_expiry = expiry_dates[0] if expiry_dates else ""

        # Parse CE and PE data per strike
        all_data = filtered.get("data", []) or records.get("data", [])
        strikes = []

        total_ce_oi = 0
        total_pe_oi = 0
        max_ce_oi = 0
        max_ce_oi_strike = 0
        max_pe_oi = 0
        max_pe_oi_strike = 0

        for row in all_data:
            strike_price = row.get("strikePrice", 0)
            ce = row.get("CE", {})
            pe = row.get("PE", {})

            ce_oi = ce.get("openInterest", 0) or 0
            pe_oi = pe.get("openInterest", 0) or 0
            ce_oi_change = ce.get("changeinOpenInterest", 0) or 0
            pe_oi_change = pe.get("changeinOpenInterest", 0) or 0
            ce_iv = ce.get("impliedVolatility", 0) or 0
            pe_iv = pe.get("impliedVolatility", 0) or 0
            ce_ltp = ce.get("lastPrice", 0) or 0
            pe_ltp = pe.get("lastPrice", 0) or 0
            ce_volume = ce.get("totalTradedVolume", 0) or 0
            pe_volume = pe.get("totalTradedVolume", 0) or 0

            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                max_ce_oi_strike = strike_price

            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                max_pe_oi_strike = strike_price

            strikes.append({
                "strike": strike_price,
                "ce_oi": ce_oi,
                "pe_oi": pe_oi,
                "ce_oi_change": ce_oi_change,
                "pe_oi_change": pe_oi_change,
                "ce_iv": ce_iv,
                "pe_iv": pe_iv,
                "ce_ltp": ce_ltp,
                "pe_ltp": pe_ltp,
                "ce_volume": ce_volume,
                "pe_volume": pe_volume,
            })

        # PCR calculation
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

        return {
            "symbol": symbol.upper(),
            "underlying_price": underlying_value,
            "current_expiry": current_expiry,
            "expiry_dates": expiry_dates,
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "pcr": round(pcr, 3),
            "max_ce_oi_strike": max_ce_oi_strike,
            "max_ce_oi": max_ce_oi,
            "max_pe_oi_strike": max_pe_oi_strike,
            "max_pe_oi": max_pe_oi,
            "support": max_pe_oi_strike,   # Max PUT OI = support
            "resistance": max_ce_oi_strike,  # Max CALL OI = resistance
            "strikes": strikes,
        }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.25)

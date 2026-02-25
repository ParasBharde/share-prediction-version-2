"""Unit tests for option_chain_fetcher fallback behavior."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# Provide a minimal aiohttp stub so module import works in constrained CI shells.
aiohttp_stub = types.SimpleNamespace(
    ClientSession=object,
    ClientTimeout=object,
    CookieJar=object,
    TCPConnector=object,
    ClientConnectorError=Exception,
)
sys.modules.setdefault("aiohttp", aiohttp_stub)

import src.data_ingestion.option_chain_fetcher as option_chain_module
from src.data_ingestion.option_chain_fetcher import OptionChainFetcher


def test_fetch_option_chain_stops_on_playwright_success():
    """If Playwright returns data, later fallbacks must not run."""
    fetcher = OptionChainFetcher()

    fake_raw = {
        "records": {
            "data": [{"strikePrice": 22500, "CE": {}, "PE": {}}],
            "expiryDates": ["27-Feb-2026"],
            "underlyingValue": 22510,
        },
        "filtered": {"data": [{"strikePrice": 22500, "CE": {}, "PE": {}}]},
    }

    async def _run():
        with patch.object(option_chain_module, "_PLAYWRIGHT_OK", True), patch.object(fetcher, "_fetch_via_nsepython", AsyncMock(return_value=None)), patch.object(
            fetcher, "_fetch_via_playwright", AsyncMock(return_value=fake_raw)
        ) as pw_mock, patch.object(
            fetcher, "_fetch_via_curl_cffi", AsyncMock(return_value=None)
        ) as curl_mock, patch.object(
            fetcher, "_ensure_warmed_up", AsyncMock(return_value=False)
        ) as warm_mock:
            parsed = await fetcher.fetch_option_chain("NIFTY")

        assert parsed is not None
        assert parsed["underlying_price"] == 22510
        assert pw_mock.await_count == 1
        assert curl_mock.await_count == 0
        assert warm_mock.await_count == 0

    asyncio.run(_run())


def test_close_closes_playwright_context_before_browser():
    """close() should close context and browser safely when present."""
    fetcher = OptionChainFetcher()

    context = MagicMock()
    context.close = AsyncMock()
    browser = MagicMock()
    browser.close = AsyncMock()

    fetcher._pw_context = context
    fetcher._pw_browser = browser
    fetcher._pw_playwright = None
    fetcher._session = None

    asyncio.run(fetcher.close())

    context.close.assert_awaited_once()
    browser.close.assert_awaited_once()
    assert fetcher._pw_context is None
    assert fetcher._pw_browser is None

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


def test_playwright_headless_mode_env_override():
    """NSE_PLAYWRIGHT_HEADLESS env must explicitly control mode."""
    fetcher = OptionChainFetcher()

    with patch.dict("os.environ", {"NSE_PLAYWRIGHT_HEADLESS": "1"}, clear=False):
        assert fetcher._playwright_headless_mode() is True

    with patch.dict("os.environ", {"NSE_PLAYWRIGHT_HEADLESS": "0"}, clear=False):
        assert fetcher._playwright_headless_mode() is False


def test_fetch_option_chain_uses_recent_cache_when_all_live_paths_fail(tmp_path):
    """When all live paths fail, recent cached parsed data should be returned."""
    fetcher = OptionChainFetcher()
    fetcher._cache_dir = str(tmp_path)

    cached_payload = {
        "symbol": "NIFTY",
        "underlying_price": 22444.0,
        "pcr": 1.01,
        "strikes": [],
    }
    fetcher._save_parsed_cache("NIFTY", cached_payload)

    async def _run():
        with patch.object(option_chain_module, "_PLAYWRIGHT_OK", True), patch.object(option_chain_module, "_CURL_CFFI_OK", True), patch.object(
            fetcher, "_fetch_via_nsepython", AsyncMock(return_value=None)
        ), patch.object(
            fetcher, "_fetch_via_playwright", AsyncMock(return_value=None)
        ), patch.object(
            fetcher, "_fetch_via_curl_cffi", AsyncMock(return_value=None)
        ), patch.object(
            fetcher, "_ensure_warmed_up", AsyncMock(return_value=False)
        ), patch.object(
            fetcher, "_get_session", AsyncMock(side_effect=RuntimeError("blocked"))
        ):
            result = await fetcher.fetch_option_chain("NIFTY")

        assert result is not None
        assert result["symbol"] == "NIFTY"
        assert result["underlying_price"] == 22444.0
        assert result.get("is_stale") is True

    asyncio.run(_run())


def test_fetch_option_chain_uses_jina_before_aiohttp():
    """Jina relay success should short-circuit aiohttp fallback path."""
    fetcher = OptionChainFetcher()

    fake_raw = {
        "records": {
            "data": [{"strikePrice": 22600, "CE": {}, "PE": {}}],
            "expiryDates": ["27-Feb-2026"],
            "underlyingValue": 22610,
        },
        "filtered": {"data": [{"strikePrice": 22600, "CE": {}, "PE": {}}]},
    }

    async def _run():
        with patch.object(option_chain_module, "_PLAYWRIGHT_OK", True), patch.object(option_chain_module, "_CURL_CFFI_OK", True), patch.object(
            fetcher, "_fetch_via_nsepython", AsyncMock(return_value=None)
        ), patch.object(
            fetcher, "_fetch_via_playwright", AsyncMock(return_value=None)
        ), patch.object(
            fetcher, "_fetch_via_curl_cffi", AsyncMock(return_value=None)
        ), patch.object(
            fetcher, "_fetch_via_jina_proxy", AsyncMock(return_value=fake_raw)
        ) as jina_mock, patch.object(
            fetcher, "_ensure_warmed_up", AsyncMock(return_value=False)
        ) as warm_mock:
            result = await fetcher.fetch_option_chain("NIFTY")

        assert result is not None
        assert result["underlying_price"] == 22610
        assert jina_mock.await_count == 1
        assert warm_mock.await_count == 0

    asyncio.run(_run())

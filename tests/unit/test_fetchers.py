"""Unit tests for data fetchers."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data_ingestion.nse_fetcher import NSEFetcher
from src.data_ingestion.yahoo_fetcher import YahooFetcher
from src.data_ingestion.alpha_vantage_fetcher import AlphaVantageFetcher
from src.data_ingestion.fallback_manager import FallbackManager


class TestNSEFetcher:
    """Tests for NSE API fetcher."""

    @pytest.fixture
    def fetcher(self):
        with patch("src.data_ingestion.nse_fetcher.load_config"):
            return NSEFetcher()

    def test_init(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.source_name == "nse_official"
        assert fetcher._cookies is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_failure(self, fetcher):
        """Test fetch returns None when API fails."""
        fetcher._request_with_retry = AsyncMock(return_value=None)
        fetcher._refresh_session = AsyncMock()
        fetcher._cookies = {"test": "cookie"}

        result = await fetcher.fetch(
            "RELIANCE",
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_parses_data(self, fetcher):
        """Test fetch correctly parses NSE response."""
        mock_response = {
            "data": [
                {
                    "CH_TIMESTAMP": "2024-01-15",
                    "CH_OPENING_PRICE": "2500",
                    "CH_TRADE_HIGH_PRICE": "2550",
                    "CH_TRADE_LOW_PRICE": "2480",
                    "CH_CLOSING_PRICE": "2530",
                    "CH_TOT_TRADED_QTY": "1000000",
                    "CH_TOT_TRADED_VAL": "2500000000",
                    "CH_TOTAL_TRADES": "50000",
                    "COP_DELIV_QTY": "600000",
                    "COP_DELIV_PERC": "60.0",
                }
            ]
        }

        fetcher._request_with_retry = AsyncMock(
            return_value=mock_response
        )
        fetcher._ensure_session = AsyncMock()
        fetcher._cookies = {"test": "cookie"}

        result = await fetcher.fetch(
            "RELIANCE",
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
        )

        assert result is not None
        assert result["symbol"] == "RELIANCE"
        assert result["source"] == "nse_official"
        assert len(result["records"]) == 1
        assert result["records"][0]["close"] == 2530.0


class TestYahooFetcher:
    """Tests for Yahoo Finance fetcher."""

    @pytest.fixture
    def fetcher(self):
        with patch("src.data_ingestion.yahoo_fetcher.load_config"):
            return YahooFetcher()

    def test_nse_to_yahoo_conversion(self, fetcher):
        """Test NSE symbol to Yahoo format conversion."""
        assert fetcher._nse_to_yahoo("RELIANCE") == "RELIANCE.NS"
        assert fetcher._nse_to_yahoo("TCS") == "TCS.NS"

    def test_init(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.source_name == "yahoo_finance"


class TestAlphaVantageFetcher:
    """Tests for Alpha Vantage fetcher."""

    @pytest.fixture
    def fetcher(self):
        with patch(
            "src.data_ingestion.alpha_vantage_fetcher.load_config"
        ):
            with patch.dict("os.environ", {"ALPHA_VANTAGE_KEY": "test"}):
                return AlphaVantageFetcher()

    def test_init(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.source_name == "alpha_vantage"


class TestFallbackManager:
    """Tests for fallback manager."""

    @pytest.fixture
    def manager(self):
        with patch(
            "src.data_ingestion.fallback_manager.load_config"
        ) as mock_config:
            mock_config.return_value = {
                "primary": {"enabled": False},
                "fallback_1": {"enabled": False},
                "fallback_2": {"enabled": False},
                "fallback_strategy": {
                    "escalation_delay": 0,
                    "use_stale_cache": False,
                },
            }
            with patch(
                "src.data_ingestion.fallback_manager.RedisHandler"
            ):
                return FallbackManager()

    def test_circuit_breaker_closed_initially(self, manager):
        """Test circuit breakers start closed."""
        for cb in manager.circuit_breakers.values():
            assert cb["failures"] == 0
            assert cb["last_failure"] is None

    def test_record_failure_increments(self, manager):
        """Test failure recording increments counter."""
        manager.circuit_breakers["primary"] = {
            "failures": 0,
            "last_failure": None,
        }
        manager._record_failure("primary")
        assert manager.circuit_breakers["primary"]["failures"] == 1

    def test_record_success_resets(self, manager):
        """Test success recording resets counter."""
        manager.circuit_breakers["primary"] = {
            "failures": 5,
            "last_failure": datetime.now(),
        }
        manager._record_success("primary")
        assert manager.circuit_breakers["primary"]["failures"] == 0

    def test_circuit_opens_after_threshold(self, manager):
        """Test circuit opens after 5 failures."""
        manager.circuit_breakers["primary"] = {
            "failures": 5,
            "last_failure": datetime.now(),
        }
        assert manager._is_circuit_open("primary") is True

"""Integration tests for the data pipeline."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.fallback_manager import FallbackManager


class TestDataPipeline:
    """Integration tests for data fetching pipeline."""

    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        """Test that fallback chain works when primary fails."""
        with patch(
            "src.data_ingestion.fallback_manager.load_config"
        ) as mock_config:
            mock_config.return_value = {
                "primary": {"enabled": True},
                "fallback_1": {"enabled": True},
                "fallback_2": {"enabled": False},
                "fallback_strategy": {
                    "escalation_delay": 0,
                    "use_stale_cache": False,
                },
            }
            with patch(
                "src.data_ingestion.fallback_manager.RedisHandler"
            ) as mock_redis:
                mock_redis_instance = MagicMock()
                mock_redis_instance.get_json = AsyncMock(
                    return_value=None
                )
                mock_redis.return_value = mock_redis_instance

                with patch(
                    "src.data_ingestion.fallback_manager.NSEFetcher"
                ) as mock_nse, patch(
                    "src.data_ingestion.fallback_manager.YahooFetcher"
                ) as mock_yahoo:
                    # NSE fails
                    nse_instance = AsyncMock()
                    nse_instance.fetch.side_effect = Exception(
                        "API Error"
                    )
                    mock_nse.return_value = nse_instance

                    # Yahoo succeeds
                    yahoo_instance = AsyncMock()
                    yahoo_instance.fetch.return_value = {
                        "symbol": "TEST",
                        "source": "yahoo",
                        "records": [{"close": 100}],
                        "count": 1,
                    }
                    mock_yahoo.return_value = yahoo_instance

                    manager = FallbackManager()
                    result = await manager.fetch_stock_data(
                        "TEST",
                        datetime(2024, 1, 1),
                        datetime(2024, 12, 31),
                    )

                    assert result is not None
                    assert result["source"] == "yahoo"

    def test_data_validation_pipeline(self):
        """Test data validation on raw records."""
        validator = DataValidator()

        records = [
            {
                "date": datetime(2024, 1, 1),
                "open": 100,
                "high": 110,
                "low": 95,
                "close": 105,
                "volume": 1000000,
            },
            {
                "date": datetime(2024, 1, 2),
                "open": 105,
                "high": 90,  # Invalid: high < low
                "low": 95,
                "close": 88,
                "volume": 1200000,
            },
            {
                "date": datetime(2024, 1, 3),
                "open": 108,
                "high": 115,
                "low": 106,
                "close": 112,
                "volume": 800000,
            },
        ]

        valid, invalid = validator.validate_records(
            records, "TEST"
        )

        assert len(valid) == 2
        assert len(invalid) == 1

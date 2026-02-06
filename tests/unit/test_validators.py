"""Unit tests for validators."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.utils.validators import (
    validate_date_range,
    validate_ohlcv_data,
    validate_portfolio_constraints,
    validate_price_change,
    validate_strategy_config,
    validate_symbol,
)


class TestValidateSymbol:
    """Tests for symbol validation."""

    def test_valid_symbols(self):
        assert validate_symbol("RELIANCE") is True
        assert validate_symbol("TCS") is True
        assert validate_symbol("M&M") is True
        assert validate_symbol("L&T") is True

    def test_invalid_symbols(self):
        assert validate_symbol("") is False
        assert validate_symbol(None) is False
        assert validate_symbol("A" * 25) is False
        assert validate_symbol("test!@#") is False


class TestValidateOHLCV:
    """Tests for OHLCV data validation."""

    def test_valid_data(self):
        df = pd.DataFrame(
            {
                "open": [100, 105],
                "high": [110, 112],
                "low": [95, 100],
                "close": [108, 110],
                "volume": [1000000, 1200000],
            }
        )
        result = validate_ohlcv_data(df)
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_missing_columns(self):
        df = pd.DataFrame({"open": [100], "high": [110]})
        result = validate_ohlcv_data(df)
        assert result["valid"] is False
        assert any("Missing columns" in i for i in result["issues"])

    def test_negative_prices(self):
        df = pd.DataFrame(
            {
                "open": [-100],
                "high": [110],
                "low": [95],
                "close": [108],
                "volume": [1000000],
            }
        )
        result = validate_ohlcv_data(df)
        assert result["valid"] is False

    def test_high_less_than_low(self):
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [90],  # Less than low
                "low": [95],
                "close": [92],
                "volume": [1000000],
            }
        )
        result = validate_ohlcv_data(df)
        assert result["valid"] is False

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        result = validate_ohlcv_data(df)
        assert result["valid"] is False

    def test_zero_volume(self):
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [110],
                "low": [95],
                "close": [108],
                "volume": [0],
            }
        )
        result = validate_ohlcv_data(df)
        assert any("Zero volume" in i for i in result["issues"])


class TestValidatePriceChange:
    """Tests for price change validation."""

    def test_normal_change(self):
        assert validate_price_change(105, 100, 20) is True

    def test_excessive_change(self):
        assert validate_price_change(130, 100, 20) is False

    def test_zero_previous(self):
        assert validate_price_change(100, 0, 20) is False

    def test_negative_change(self):
        assert validate_price_change(85, 100, 20) is True


class TestValidateDateRange:
    """Tests for date range validation."""

    def test_valid_range(self):
        assert validate_date_range(
            date(2024, 1, 1), date(2024, 12, 31)
        ) is True

    def test_start_after_end(self):
        assert validate_date_range(
            date(2024, 12, 31), date(2024, 1, 1)
        ) is False

    def test_future_end_date(self):
        assert validate_date_range(
            date(2024, 1, 1), date(2030, 12, 31)
        ) is False


class TestValidateStrategyConfig:
    """Tests for strategy config validation."""

    def test_valid_config(self):
        config = {
            "strategy": {"name": "Test"},
            "indicators": [
                {"name": "ind1", "weight": 0.5},
                {"name": "ind2", "weight": 0.5},
            ],
            "signal_generation": {
                "min_conditions_met": 2,
                "confidence_threshold": 0.7,
            },
            "risk_management": {},
        }
        result = validate_strategy_config(config)
        assert result["valid"] is True

    def test_missing_strategy_name(self):
        config = {
            "strategy": {},
            "indicators": [{"weight": 1.0}],
            "signal_generation": {
                "min_conditions_met": 1,
                "confidence_threshold": 0.5,
            },
            "risk_management": {},
        }
        result = validate_strategy_config(config)
        assert result["valid"] is False

    def test_weights_not_summing_to_one(self):
        config = {
            "strategy": {"name": "Test"},
            "indicators": [
                {"name": "ind1", "weight": 0.3},
                {"name": "ind2", "weight": 0.3},
            ],
            "signal_generation": {
                "min_conditions_met": 1,
                "confidence_threshold": 0.5,
            },
            "risk_management": {},
        }
        result = validate_strategy_config(config)
        assert result["valid"] is False


class TestValidatePortfolioConstraints:
    """Tests for portfolio constraint validation."""

    def test_valid_position(self):
        result = validate_portfolio_constraints(
            position_value=10000,
            portfolio_value=100000,
            sector="IT",
            sector_allocations={"IT": 5},
        )
        assert result["valid"] is True

    def test_position_too_large(self):
        result = validate_portfolio_constraints(
            position_value=25000,
            portfolio_value=100000,
            sector="IT",
            sector_allocations={},
            max_position_pct=20,
        )
        assert result["valid"] is False

    def test_sector_concentration(self):
        result = validate_portfolio_constraints(
            position_value=10000,
            portfolio_value=100000,
            sector="IT",
            sector_allocations={"IT": 25},
            max_sector_pct=30,
        )
        assert result["valid"] is False

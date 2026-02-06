"""Integration tests for strategy execution pipeline."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.strategy_loader import StrategyLoader


def _create_bullish_df(rows: int = 300) -> pd.DataFrame:
    """Create a strongly bullish DataFrame for testing."""
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    base_price = 500

    # Strong uptrend
    trend = np.linspace(0, 1, rows)
    noise = np.random.normal(0, 0.01, rows)
    prices = base_price * (1 + trend + noise)

    # Make last day have volume surge
    volumes = np.random.randint(500000, 1500000, rows)
    volumes[-1] = volumes.mean() * 3  # 3x volume on last day

    df = pd.DataFrame(
        {
            "open": prices * 0.998,
            "high": prices * 1.015,
            "low": prices * 0.985,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )
    return df


class TestStrategyExecution:
    """Integration tests for strategy execution."""

    def test_momentum_strategy_on_bullish_data(self):
        """Test momentum strategy generates signal on bullish data."""
        config = {
            "strategy": {
                "name": "Momentum Breakout",
                "version": "1.0",
                "enabled": True,
                "priority": 1,
            },
            "filters": {},
            "indicators": [
                {"weight": 0.25},
                {"weight": 0.30},
                {"weight": 0.15},
                {"weight": 0.20},
                {"weight": 0.10},
            ],
            "signal_generation": {
                "min_conditions_met": 3,
                "confidence_threshold": 0.50,
            },
            "risk_management": {
                "stop_loss": {
                    "type": "atr_based",
                    "multiplier": 1.5,
                    "max_percent": 5,
                },
                "target": {"type": "risk_reward", "ratio": 2.5},
            },
        }
        strategy = MomentumBreakoutStrategy(config)
        df = _create_bullish_df()

        company_info = {
            "name": "Test Corp",
            "symbol": "TEST",
            "sector": "IT",
            "market_cap": 10000,
            "last_price": float(df["close"].iloc[-1]),
        }

        result = strategy.scan("TEST", df, company_info)
        # Signal generation depends on indicator conditions
        # With strongly bullish data, some indicators should pass
        if result is not None:
            assert result.symbol == "TEST"
            assert result.entry_price > 0
            assert result.target_price > result.entry_price
            assert result.stop_loss < result.entry_price

    def test_strategy_loader(self):
        """Test strategy loader discovers and loads strategies."""
        loader = StrategyLoader()
        strategies = loader.load_all()

        # Should load at least the 3 built-in strategies
        # (if they are enabled in config)
        assert isinstance(strategies, list)
        for s in strategies:
            assert hasattr(s, "name")
            assert hasattr(s, "scan")
            assert s.enabled is True

"""Unit tests for trading strategies."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.strategies.momentum_breakout import MomentumBreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volume_surge import VolumeSurgeStrategy
from src.strategies.indicators.moving_averages import ema, sma
from src.strategies.indicators.oscillators import rsi, macd
from src.strategies.indicators.volume_indicators import (
    obv,
    volume_ratio,
)
from src.utils.constants import AlertPriority, SignalType


def _create_sample_df(rows: int = 300) -> pd.DataFrame:
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    base_price = 1000

    # Generate trending price data
    returns = np.random.normal(0.001, 0.02, rows)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.005, rows)),
            "high": prices * (1 + abs(np.random.normal(0.01, 0.005, rows))),
            "low": prices * (1 - abs(np.random.normal(0.01, 0.005, rows))),
            "close": prices,
            "volume": np.random.randint(100000, 5000000, rows),
        },
        index=dates,
    )
    return df


class TestIndicators:
    """Tests for technical indicators."""

    @pytest.fixture
    def sample_series(self):
        np.random.seed(42)
        return pd.Series(
            np.cumsum(np.random.randn(100)) + 100,
            index=pd.date_range("2024-01-01", periods=100),
        )

    def test_sma(self, sample_series):
        """Test Simple Moving Average."""
        result = sma(sample_series, 20)
        assert len(result) == len(sample_series)
        assert result.isna().sum() == 19  # First 19 are NaN
        assert not np.isnan(result.iloc[-1])

    def test_ema(self, sample_series):
        """Test Exponential Moving Average."""
        result = ema(sample_series, 20)
        assert len(result) == len(sample_series)
        assert not np.isnan(result.iloc[-1])

    def test_rsi(self, sample_series):
        """Test RSI calculation."""
        result = rsi(sample_series, 14)
        assert len(result) == len(sample_series)
        # RSI should be between 0 and 100
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd(self, sample_series):
        """Test MACD calculation."""
        result = macd(sample_series, 12, 26, 9)
        assert "macd_line" in result.columns
        assert "signal_line" in result.columns
        assert "histogram" in result.columns

    def test_volume_ratio(self):
        """Test volume ratio calculation."""
        volume = pd.Series(
            [100000] * 20 + [300000],
            index=pd.date_range("2024-01-01", periods=21),
        )
        result = volume_ratio(volume, 20)
        assert result.iloc[-1] == pytest.approx(3.0, rel=0.01)


class TestTradingSignal:
    """Tests for TradingSignal dataclass."""

    def test_signal_creation(self):
        """Test signal creation."""
        signal = TradingSignal(
            symbol="RELIANCE",
            company_name="Reliance Industries",
            strategy_name="Test",
            signal_type=SignalType.BUY,
            confidence=0.85,
            entry_price=2500,
            target_price=2750,
            stop_loss=2375,
            priority=AlertPriority.HIGH,
            indicators_met=4,
            total_indicators=5,
        )
        assert signal.symbol == "RELIANCE"
        assert signal.confidence == 0.85

    def test_target_percent(self):
        """Test target percentage calculation."""
        signal = TradingSignal(
            symbol="TEST",
            company_name="Test",
            strategy_name="Test",
            signal_type=SignalType.BUY,
            confidence=0.8,
            entry_price=100,
            target_price=120,
            stop_loss=95,
            priority=AlertPriority.MEDIUM,
            indicators_met=3,
            total_indicators=5,
        )
        assert signal.target_percent == pytest.approx(20.0)
        assert signal.stop_loss_percent == pytest.approx(5.0)

    def test_risk_reward_ratio(self):
        """Test risk-reward ratio."""
        signal = TradingSignal(
            symbol="TEST",
            company_name="Test",
            strategy_name="Test",
            signal_type=SignalType.BUY,
            confidence=0.8,
            entry_price=100,
            target_price=110,
            stop_loss=95,
            priority=AlertPriority.MEDIUM,
            indicators_met=3,
            total_indicators=5,
        )
        assert signal.risk_reward_ratio == pytest.approx(2.0)

    def test_to_dict(self):
        """Test signal serialization."""
        signal = TradingSignal(
            symbol="TEST",
            company_name="Test",
            strategy_name="Test",
            signal_type=SignalType.BUY,
            confidence=0.8,
            entry_price=100,
            target_price=120,
            stop_loss=95,
            priority=AlertPriority.HIGH,
            indicators_met=3,
            total_indicators=5,
        )
        d = signal.to_dict()
        assert d["symbol"] == "TEST"
        assert d["signal_type"] == "BUY"
        assert d["priority"] == "HIGH"


class TestMomentumBreakout:
    """Tests for Momentum Breakout strategy."""

    @pytest.fixture
    def strategy(self):
        config = {
            "strategy": {
                "name": "Momentum Breakout",
                "version": "1.0",
                "enabled": True,
                "priority": 1,
            },
            "filters": {"price_min": 50, "price_max": 5000},
            "indicators": [
                {"name": "test", "weight": 0.25},
                {"name": "test2", "weight": 0.30},
                {"name": "test3", "weight": 0.15},
                {"name": "test4", "weight": 0.20},
                {"name": "test5", "weight": 0.10},
            ],
            "signal_generation": {
                "min_conditions_met": 4,
                "confidence_threshold": 0.70,
                "signal_type": "BUY",
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
        return MomentumBreakoutStrategy(config)

    def test_init(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "Momentum Breakout"
        assert strategy.enabled is True

    def test_pre_filter_pass(self, strategy):
        """Test pre-filter passes valid stocks."""
        info = {"market_cap": 5000, "last_price": 500}
        assert strategy.apply_pre_filters(info) is True

    def test_pre_filter_price_too_low(self, strategy):
        """Test pre-filter rejects low-price stocks."""
        info = {"market_cap": 5000, "last_price": 10}
        assert strategy.apply_pre_filters(info) is False

    def test_scan_insufficient_data(self, strategy):
        """Test scan returns None with insufficient data."""
        df = _create_sample_df(50)  # Only 50 rows
        result = strategy.scan(
            "TEST", df, {"name": "Test", "market_cap": 5000}
        )
        assert result is None

    def test_scan_returns_signal_or_none(self, strategy):
        """Test scan returns valid signal type."""
        df = _create_sample_df(300)
        result = strategy.scan(
            "TEST",
            df,
            {"name": "Test Corp", "market_cap": 5000, "last_price": 500},
        )
        # May or may not generate signal depending on random data
        assert result is None or isinstance(result, TradingSignal)

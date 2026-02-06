"""Integration tests for the alert delivery flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.alerts.alert_deduplicator import AlertDeduplicator
from src.alerts.alert_formatter import AlertFormatter
from src.strategies.base_strategy import TradingSignal
from src.utils.constants import AlertPriority, SignalType


class TestAlertFlow:
    """Integration tests for alert generation and delivery."""

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return TradingSignal(
            symbol="INFY",
            company_name="Infosys Ltd",
            strategy_name="Momentum Breakout",
            signal_type=SignalType.BUY,
            confidence=0.85,
            entry_price=1450.50,
            target_price=1560.75,
            stop_loss=1398.25,
            priority=AlertPriority.HIGH,
            indicators_met=4,
            total_indicators=5,
            indicator_details={
                "rsi": {"value": 62.5, "passed": True},
                "volume_surge": {
                    "volume_ratio": 2.3,
                    "passed": True,
                },
            },
            metadata={"volume_surge": 2.3, "rsi": 62.5},
        )

    def test_alert_formatter(self, sample_signal):
        """Test alert formatting produces valid message."""
        formatter = AlertFormatter()
        message = formatter.format_buy_signal(sample_signal)

        assert message is not None
        assert isinstance(message, str)
        assert "INFY" in message or "Infosys" in message

    def test_deduplication(self):
        """Test alert deduplication."""
        mock_redis = MagicMock()
        mock_redis.is_seen.return_value = False
        mock_redis.mark_seen.return_value = True

        dedup = AlertDeduplicator(redis_handler=mock_redis)

        # First time should not be duplicate
        assert dedup.is_duplicate("INFY", "Momentum") is False

        # Mark as sent
        dedup.mark_sent("INFY", "Momentum")
        mock_redis.mark_seen.assert_called_once()

    def test_daily_summary_format(self):
        """Test daily summary formatting."""
        formatter = AlertFormatter()
        summary = formatter.format_daily_summary(
            {
                "date": "06 Feb 2025 15:35 IST",
                "stocks_scanned": 500,
                "signals_count": 7,
                "alerts_sent": 7,
                "scan_duration": 420,
                "active_positions": 3,
                "total_pnl_pct": 5.2,
                "top_signals": [
                    {"symbol": "INFY", "confidence": 85},
                    {"symbol": "TCS", "confidence": 78},
                ],
            }
        )

        assert summary is not None
        assert isinstance(summary, str)

"""
Alert Message Formatter

Purpose:
    Renders alert messages from Jinja2 templates loaded from
    config/alert_templates.yaml. Supports buy signals, sell signals,
    daily summaries, error alerts, and portfolio updates.

Dependencies:
    - Jinja2
    - PyYAML (via config_loader)

Logging:
    - Template loading at INFO
    - Render operations at DEBUG
    - Missing templates at WARNING
    - Render failures at ERROR

Fallbacks:
    If templates cannot be loaded, a plain-text fallback is used.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from jinja2 import BaseLoader, Environment, TemplateSyntaxError

from src.monitoring.logger import get_alert_logger
from src.monitoring.metrics import alert_sent_counter
from src.utils.config_loader import load_config
from src.utils.constants import CURRENCY_SYMBOL, SignalType

logger = get_alert_logger()

# Default fallback templates used when config/alert_templates.yaml is
# missing or a specific template key is absent.
_FALLBACK_TEMPLATES: Dict[str, str] = {
    "buy_signal": (
        "*BUY Signal - {{ symbol }}*\n"
        "Strategy: {{ strategy_name }}\n"
        "Price: {{ currency }}{{ entry_price }}\n"
        "Target: {{ currency }}{{ target_price }}\n"
        "Stop-Loss: {{ currency }}{{ stop_loss }}\n"
        "Confidence: {{ confidence }}%"
    ),
    "sell_signal": (
        "*SELL Signal - {{ symbol }}*\n"
        "Strategy: {{ strategy_name }}\n"
        "Exit Price: {{ currency }}{{ exit_price }}\n"
        "P&L: {{ pnl_pct }}%\n"
        "Reason: {{ reason }}"
    ),
    "daily_summary": (
        "*Daily Summary - {{ date }}*\n"
        "Signals: {{ signals_generated }}\n"
        "Portfolio: {{ currency }}{{ portfolio_value }}\n"
        "P&L: {{ daily_pnl }} ({{ daily_pnl_pct }}%)"
    ),
    "error_alert": (
        "*ERROR ALERT*\n"
        "Component: {{ component }}\n"
        "Error: {{ error_message }}\n"
        "Time: {{ timestamp }}"
    ),
    "portfolio_update": (
        "*Portfolio Update*\n"
        "Value: {{ currency }}{{ portfolio_value }}\n"
        "Positions: {{ active_positions }}\n"
        "Daily P&L: {{ daily_pnl }} ({{ daily_pnl_pct }}%)"
    ),
}


class AlertFormatter:
    """Renders alert messages using Jinja2 templates."""

    def __init__(self) -> None:
        """
        Initialize the formatter and load templates.

        Templates are read from ``config/alert_templates.yaml``.
        If the file is missing or unparseable the built-in fallback
        templates are used instead.
        """
        self._jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            keep_trailing_newline=True,
        )
        self._templates: Dict[str, str] = {}
        self._load_templates()

    # ------------------------------------------------------------------
    # Public formatting methods
    # ------------------------------------------------------------------

    def format_buy_signal(self, signal: Dict[str, Any]) -> str:
        """
        Format a buy signal alert message.

        Args:
            signal: Signal data dictionary.  Expected keys::

                {
                    "symbol": "RELIANCE",
                    "strategy_name": "RSI_MACD_Crossover",
                    "entry_price": 2450.50,
                    "target_price": 2600.00,
                    "stop_loss": 2380.00,
                    "confidence": 85.0,
                    "indicators_met": 4,
                    "total_indicators": 5,
                }

        Returns:
            Rendered message string.
        """
        context = {
            "symbol": signal.get("symbol", "UNKNOWN"),
            "strategy_name": signal.get("strategy_name", "N/A"),
            "entry_price": self._format_price(
                signal.get("entry_price")
            ),
            "target_price": self._format_price(
                signal.get("target_price")
            ),
            "stop_loss": self._format_price(
                signal.get("stop_loss")
            ),
            "confidence": signal.get("confidence", 0),
            "indicators_met": signal.get("indicators_met", "N/A"),
            "total_indicators": signal.get("total_indicators", "N/A"),
            "signal_type": SignalType.BUY.value,
            "currency": CURRENCY_SYMBOL,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        }
        return self._render_template("buy_signal", context)

    def format_sell_signal(self, signal_data: Dict[str, Any]) -> str:
        """
        Format a sell signal alert message.

        Args:
            signal_data: Sell signal data dictionary.  Expected keys::

                {
                    "symbol": "TCS",
                    "strategy_name": "RSI_MACD_Crossover",
                    "exit_price": 3400.00,
                    "entry_price": 3200.00,
                    "pnl": 200.00,
                    "pnl_pct": 6.25,
                    "reason": "Target hit",
                }

        Returns:
            Rendered message string.
        """
        context = {
            "symbol": signal_data.get("symbol", "UNKNOWN"),
            "strategy_name": signal_data.get("strategy_name", "N/A"),
            "exit_price": self._format_price(
                signal_data.get("exit_price")
            ),
            "entry_price": self._format_price(
                signal_data.get("entry_price")
            ),
            "pnl": signal_data.get("pnl", 0),
            "pnl_pct": signal_data.get("pnl_pct", 0),
            "reason": signal_data.get("reason", "N/A"),
            "signal_type": SignalType.SELL.value,
            "currency": CURRENCY_SYMBOL,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        }
        return self._render_template("sell_signal", context)

    def format_daily_summary(
        self, summary: Dict[str, Any]
    ) -> str:
        """
        Format a daily summary alert message.

        Args:
            summary: Daily summary data dictionary.  Expected keys::

                {
                    "date": "2025-01-15",
                    "signals_generated": 12,
                    "alerts_sent": 10,
                    "portfolio_value": 1050000.0,
                    "daily_pnl": 2500.0,
                    "daily_pnl_pct": 0.24,
                    "active_positions": 5,
                    "winning_trades": 3,
                    "losing_trades": 1,
                }

        Returns:
            Rendered message string.
        """
        context = {
            "date": summary.get("date", "N/A"),
            "signals_generated": summary.get("signals_generated", 0),
            "alerts_sent": summary.get("alerts_sent", 0),
            "portfolio_value": self._format_price(
                summary.get("portfolio_value")
            ),
            "daily_pnl": summary.get("daily_pnl", 0),
            "daily_pnl_pct": summary.get("daily_pnl_pct", 0),
            "active_positions": summary.get("active_positions", 0),
            "winning_trades": summary.get("winning_trades", 0),
            "losing_trades": summary.get("losing_trades", 0),
            "currency": CURRENCY_SYMBOL,
        }
        return self._render_template("daily_summary", context)

    def format_error_alert(
        self, error_data: Dict[str, Any]
    ) -> str:
        """
        Format an error alert message.

        Args:
            error_data: Error data dictionary.  Expected keys::

                {
                    "component": "data_ingestion",
                    "error_message": "NSE API timeout after 10s",
                    "severity": "ERROR",
                    "traceback": "...",
                }

        Returns:
            Rendered message string.
        """
        context = {
            "component": error_data.get("component", "unknown"),
            "error_message": error_data.get("error_message", "N/A"),
            "severity": error_data.get("severity", "ERROR"),
            "traceback": error_data.get("traceback", ""),
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        }
        return self._render_template("error_alert", context)

    def format_portfolio_update(
        self, portfolio_data: Dict[str, Any]
    ) -> str:
        """
        Format a portfolio update alert message.

        Args:
            portfolio_data: Portfolio data dictionary.  Expected keys::

                {
                    "portfolio_value": 1050000.0,
                    "cash_balance": 500000.0,
                    "active_positions": 5,
                    "daily_pnl": 2500.0,
                    "daily_pnl_pct": 0.24,
                    "total_pnl": 50000.0,
                }

        Returns:
            Rendered message string.
        """
        context = {
            "portfolio_value": self._format_price(
                portfolio_data.get("portfolio_value")
            ),
            "cash_balance": self._format_price(
                portfolio_data.get("cash_balance")
            ),
            "active_positions": portfolio_data.get(
                "active_positions", 0
            ),
            "daily_pnl": portfolio_data.get("daily_pnl", 0),
            "daily_pnl_pct": portfolio_data.get("daily_pnl_pct", 0),
            "total_pnl": portfolio_data.get("total_pnl", 0),
            "currency": CURRENCY_SYMBOL,
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        }
        return self._render_template("portfolio_update", context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_templates(self) -> None:
        """
        Load Jinja2 template strings from config/alert_templates.yaml.

        Falls back to built-in ``_FALLBACK_TEMPLATES`` on any failure.
        """
        try:
            config = load_config("alert_templates")

            if not config:
                logger.warning(
                    "alert_templates.yaml is empty or missing, "
                    "using fallback templates"
                )
                self._templates = dict(_FALLBACK_TEMPLATES)
                return

            templates = config.get("templates", config)
            if not isinstance(templates, dict):
                logger.warning(
                    "Unexpected template config structure, "
                    "using fallback templates"
                )
                self._templates = dict(_FALLBACK_TEMPLATES)
                return

            # Merge: user templates override fallbacks
            self._templates = dict(_FALLBACK_TEMPLATES)
            self._templates.update(templates)

            logger.info(
                f"Loaded {len(templates)} alert template(s) from config",
                extra={"template_names": list(templates.keys())},
            )

        except Exception as e:
            logger.error(
                f"Failed to load alert templates: {e}",
                exc_info=True,
            )
            self._templates = dict(_FALLBACK_TEMPLATES)

    def _render_template(
        self,
        template_name: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Render a named template with the given context.

        Args:
            template_name: Key identifying the template
                (e.g. ``buy_signal``).
            context: Dictionary of template variables.

        Returns:
            Rendered message string, or a plain-text fallback on error.
        """
        template_str = self._templates.get(template_name)

        if template_str is None:
            logger.warning(
                f"Template '{template_name}' not found, "
                f"returning raw context"
            )
            return str(context)

        try:
            template = self._jinja_env.from_string(template_str)
            rendered = template.render(**context)

            logger.debug(
                f"Rendered template '{template_name}'",
                extra={"template": template_name},
            )
            return rendered

        except TemplateSyntaxError as e:
            logger.error(
                f"Syntax error in template '{template_name}': {e}",
                exc_info=True,
            )
            return self._plain_text_fallback(template_name, context)

        except Exception as e:
            logger.error(
                f"Failed to render template '{template_name}': {e}",
                exc_info=True,
            )
            return self._plain_text_fallback(template_name, context)

    @staticmethod
    def _plain_text_fallback(
        template_name: str, context: Dict[str, Any]
    ) -> str:
        """
        Build a plain-text representation when template rendering fails.

        Args:
            template_name: Name of the template that failed.
            context: Context variables for the message.

        Returns:
            Plain-text fallback string.
        """
        lines = [f"[{template_name.upper()}]"]
        for key, value in context.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _format_price(value: Optional[float]) -> str:
        """
        Format a numeric price to a comma-separated string.

        Args:
            value: Numeric price value.

        Returns:
            Formatted price string or ``"N/A"``.
        """
        if value is None:
            return "N/A"
        try:
            return f"{float(value):,.2f}"
        except (ValueError, TypeError):
            return "N/A"

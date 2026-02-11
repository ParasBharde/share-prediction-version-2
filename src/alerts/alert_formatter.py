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

from typing import List

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
    "paper_trade_summary": (
        "*PAPER TRADING - Session Summary*\n"
        "Portfolio: {{ currency }}{{ portfolio_value }}\n"
        "Positions: {{ open_positions }}\n"
        "Trades: {{ session_trades }}"
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
        Format a buy signal alert message with rich detail.

        Args:
            signal: Signal data dictionary with all available fields.

        Returns:
            Rendered message string.
        """
        entry = signal.get("entry_price", 0) or 0
        target = signal.get("target_price", 0) or 0
        sl = signal.get("stop_loss", 0) or 0

        # Compute percentages
        target_pct = 0.0
        sl_pct = 0.0
        risk_amt = 0.0
        reward_amt = 0.0
        rr_ratio = 0.0
        if entry > 0:
            target_pct = round((target - entry) / entry * 100, 1)
            sl_pct = round((entry - sl) / entry * 100, 1)
            risk_amt = round(entry - sl, 2)
            reward_amt = round(target - entry, 2)
            if risk_amt > 0:
                rr_ratio = round(reward_amt / risk_amt, 1)

        # Build indicator checks section
        indicators_summary = self._build_indicator_summary(signal)

        # Get strategy name
        strategy_name = signal.get("strategy_name", "")
        if not strategy_name:
            strategies = signal.get("contributing_strategies", [])
            strategy_name = ", ".join(strategies) if strategies else "N/A"

        sig_type = signal.get("signal_type", "BUY")

        company_name = signal.get("company_name", "")
        if not company_name or company_name == signal.get("symbol"):
            company_name = ""

        from src.utils.time_helpers import now_ist

        # Build trading time context
        trading_time = signal.get("trading_time", {})
        has_trading_time = bool(trading_time)

        context = {
            "symbol": signal.get("symbol", "UNKNOWN"),
            "company_name": company_name,
            "strategy_name": strategy_name,
            "signal_type": sig_type,
            "entry_price": self._format_price(entry),
            "target_price": self._format_price(target),
            "stop_loss": self._format_price(sl),
            "target_pct": target_pct,
            "stop_loss_pct": sl_pct,
            "risk_amount": self._format_price(risk_amt),
            "reward_amount": self._format_price(reward_amt),
            "rr_ratio": rr_ratio,
            "confidence": signal.get("confidence", 0),
            "indicators_met": signal.get("indicators_met", "N/A"),
            "total_indicators": signal.get("total_indicators", "N/A"),
            "indicators_summary": indicators_summary,
            "contributing_strategies": signal.get(
                "contributing_strategies", []
            ),
            "strategy_count": signal.get("strategy_count", 1),
            "currency": CURRENCY_SYMBOL,
            "timestamp": now_ist().strftime("%I:%M %p IST | %d %b %Y"),
            # Trading time fields
            "has_trading_time": has_trading_time,
            "timeframe": trading_time.get("timeframe", "1D"),
            "entry_date": trading_time.get("entry_date", ""),
            "entry_window": trading_time.get("entry_window", ""),
            "signal_validity_days": trading_time.get(
                "signal_validity_days", ""
            ),
            "validity_expiry": trading_time.get(
                "validity_expiry", ""
            ),
            "holding_period": trading_time.get(
                "holding_period", ""
            ),
            "trading_description": trading_time.get(
                "description", ""
            ),
        }
        return self._render_template("buy_signal", context)

    def _build_indicator_summary(
        self, signal: Dict[str, Any]
    ) -> str:
        """Build formatted indicator checks from signal data."""
        lines = []

        # Get indicator_details from individual_signals
        individual = signal.get("individual_signals", [])
        all_details = {}
        for ind_sig in individual:
            details = ind_sig.get("indicator_details", {})
            for k, v in details.items():
                # Prefix with strategy name to avoid collision
                strat = ind_sig.get("strategy_name", "")
                key = f"{strat}: {k}" if strat and len(individual) > 1 else k
                all_details[key] = v

        # Also check top-level indicator_details
        if signal.get("indicator_details"):
            all_details.update(signal["indicator_details"])

        if not all_details:
            return "  No indicator details available"

        for name, detail in all_details.items():
            if not isinstance(detail, dict):
                continue
            passed = detail.get("passed", False)
            icon = "PASS" if passed else "FAIL"
            # Replace underscores with spaces to avoid Telegram Markdown
            # interpreting _text_ as italic
            display_name = name.replace("_", " ").title()

            detail_parts = []
            for k, v in detail.items():
                if k == "passed":
                    continue
                # Replace underscores in keys for Telegram compatibility
                display_key = k.replace("_", " ").title()
                if isinstance(v, float):
                    detail_parts.append(f"{display_key}: {v:.2f}")
                elif isinstance(v, bool):
                    continue
                else:
                    detail_parts.append(f"{display_key}: {v}")

            detail_str = ", ".join(detail_parts)
            lines.append(f"  [{icon}] {display_name} ({detail_str})")

        return "\n".join(lines) if lines else "  No indicators"

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
            summary: Daily summary data dictionary.  All keys are
                passed through to the Jinja2 template so callers can
                include any data the template needs.

        Returns:
            Rendered message string.
        """
        # Start with defaults for backwards-compatible keys
        context: Dict[str, Any] = {
            "date": "N/A",
            "signals_generated": 0,
            "signals_count": 0,
            "stocks_scanned": 0,
            "alerts_sent": 0,
            "active_positions": 0,
            "total_pnl_pct": 0,
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "scan_duration": 0,
            "top_signals": [],
            "currency": CURRENCY_SYMBOL,
        }
        # Overlay with everything the caller provided
        context.update(summary)

        # Format portfolio_value if present
        if "portfolio_value" in summary:
            context["portfolio_value"] = self._format_price(
                summary.get("portfolio_value")
            )

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

    def format_paper_trade_summary(
        self,
        portfolio: Dict[str, Any],
        trades: List[Dict[str, Any]],
    ) -> str:
        """
        Format a paper trading session summary.

        Args:
            portfolio: Portfolio state dictionary from
                PaperTradingEngine.get_portfolio_summary().
            trades: List of trade records from
                PaperTradingEngine.get_session_trades_summary().

        Returns:
            Rendered message string.
        """
        context = {
            "portfolio_value": self._format_price(
                portfolio.get("portfolio_value", 0)
            ),
            "cash_balance": self._format_price(
                portfolio.get("cash_balance", 0)
            ),
            "open_positions": portfolio.get("open_positions", 0),
            "total_return_pct": portfolio.get(
                "total_return_pct", 0
            ),
            "session_trades": portfolio.get("session_trades", 0),
            "trades": trades,
            "currency": CURRENCY_SYMBOL,
        }
        return self._render_template(
            "paper_trade_summary", context
        )

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

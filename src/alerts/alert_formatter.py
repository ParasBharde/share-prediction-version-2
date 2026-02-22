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
        "{{ signal_type }} SIGNAL - {{ symbol }}\n"
        "Strategy: {{ strategy_name }}\n"
        "Entry: {{ currency }}{{ entry_price }}\n"
        "SL: {{ currency }}{{ stop_loss }} ({{ stop_loss_pct }}%)\n"
        "Target: {{ currency }}{{ target_price }} "
        "({{ target_pct }}%)\n"
        "R:R = 1:{{ rr_ratio }}\n"
        "{% if suggested_qty > 0 %}"
        "Position: {{ suggested_qty }} shares @ "
        "{{ currency }}{{ entry_price }} "
        "= {{ currency }}{{ suggested_investment }}\n"
        "{% endif %}"
        "Confidence: {{ confidence }}% "
        "({{ indicators_met }}/{{ total_indicators }})\n"
        "{{ indicators_summary }}"
    ),
    "sell_signal": (
        "SELL SIGNAL - {{ symbol }}\n"
        "Strategy: {{ strategy_name }}\n"
        "Exit Price: {{ currency }}{{ exit_price }}\n"
        "P&L: {{ pnl_pct }}%\n"
        "Reason: {{ reason }}"
    ),
    "daily_summary": (
        "DAILY SUMMARY - {{ date }}\n"
        "Scanned: {{ stocks_scanned }} | "
        "Signals: {{ signals_count }}\n"
        "Alerts: {{ alerts_sent }}"
        "{% if paper_trades_placed is defined and "
        "paper_trades_placed > 0 %}"
        " | Paper Trades: {{ paper_trades_placed }}"
        "{% endif %}\n"
        "Positions: {{ active_positions }} | "
        "P&L: {{ total_pnl_pct }}%\n"
        "{% for s in top_signals %}"
        "{{ loop.index }}. {{ s.symbol }} {{ s.confidence }}%\n"
        "{% endfor %}"
        "Scan: {{ scan_duration }}s"
    ),
    "error_alert": (
        "ERROR ALERT\n"
        "Component: {{ component }}\n"
        "Error: {{ error_message }}\n"
        "Time: {{ timestamp }}"
    ),
    "portfolio_update": (
        "Portfolio Update\n"
        "Value: {{ currency }}{{ portfolio_value }}\n"
        "Positions: {{ active_positions }}\n"
        "Daily P&L: {{ daily_pnl }} ({{ daily_pnl_pct }}%)"
    ),
    "paper_trade_summary": (
        "PAPER TRADES PLACED\n"
        "Portfolio: {{ currency }}{{ portfolio_value }}\n"
        "Cash: {{ currency }}{{ cash_balance }}\n"
        "Positions: {{ open_positions }} | "
        "Return: {{ total_return_pct }}%\n"
        "{% for t in trades %}"
        "{{ loop.index }}. {{ t.symbol }} {{ t.side }} "
        "x{{ t.quantity }} @ {{ currency }}{{ t.entry_price }}\n"
        "{% endfor %}"
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

        # Compute percentages (use abs() for amounts, signed for display)
        target_pct = ""
        sl_pct = ""
        risk_amt = 0.0
        reward_amt = 0.0
        rr_ratio = 0.0
        if entry > 0:
            raw_target_pct = round(
                (target - entry) / entry * 100, 1
            )
            raw_sl_pct = round(
                (entry - sl) / entry * 100, 1
            )
            # Format with sign: SL always shows as loss, target as gain
            sl_pct = f"-{abs(raw_sl_pct)}"
            target_pct = f"+{abs(raw_target_pct)}"
            risk_amt = round(abs(entry - sl), 2)
            reward_amt = round(abs(target - entry), 2)
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

        # Position sizing guidance
        # qty = (capital × risk_pct%) / risk_per_share
        sizing_config = load_config("system").get(
            "position_sizing", {}
        )
        default_capital = sizing_config.get("default_capital", 1_000_000)
        risk_pct = sizing_config.get("risk_per_trade_pct", 1.0)
        risk_budget = default_capital * risk_pct / 100.0  # e.g. ₹10,000
        if risk_amt > 0 and entry > 0:
            suggested_qty = max(1, int(risk_budget / risk_amt))
            suggested_investment = self._format_price(
                suggested_qty * entry
            )
        else:
            suggested_qty = 0
            suggested_investment = "N/A"
        # Show capital in lakh (e.g. "10L") for readability
        capital_lakh = int(default_capital / 100_000)

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
            # Position sizing
            "suggested_qty": suggested_qty,
            "suggested_investment": suggested_investment,
            "default_capital_lakh": capital_lakh,
            "risk_per_trade_pct": risk_pct,
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
            "max_entry_price": self._format_price(
                trading_time.get("max_entry_price", 0)
            ),
            "atr_pct": trading_time.get("atr_pct", 0),
            "volatility": trading_time.get("volatility", "N/A"),
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
            if passed:
                icon_emoji = "\u2705"
            else:
                icon_emoji = "\u274c"
            val_str = f" `{detail_str}`" if detail_str else ""
            lines.append(
                f"{icon_emoji} {display_name}{val_str}"
            )

        return "\n".join(lines) if lines else "  No indicators"

    def format_options_signal(self, signal: Dict[str, Any]) -> str:
        """
        Format an options trading alert message (CE / PE).

        Produces a concise, Telegram-friendly message that clearly shows:
          - Whether to buy CE or PE and at which strike
          - Spot price entry, target, and stop-loss
          - PCR, OI levels, Supertrend direction (where available)
          - Indicator checks (PASS/FAIL)
          - Exit rule and confidence

        Args:
            signal: Signal dict from TradingSignal.to_dict() or equivalent.
                    Must contain ``metadata`` with ``option_type`` and
                    ``atm_strike``.

        Returns:
            Rendered alert string (≤4096 chars for Telegram).
        """
        from src.utils.time_helpers import now_ist

        meta = signal.get("metadata", {})
        option_type = meta.get("option_type", "")  # "BUY_CE" or "BUY_PE"
        atm_strike = meta.get("atm_strike", 0)
        symbol = signal.get("symbol", "UNKNOWN")
        strategy = signal.get("strategy_name", "")

        # Derive readable option label
        if "CE" in option_type:
            action_line = f"📗 BUY CALL (CE)  →  {symbol} {atm_strike} CE"
            action_emoji = "📗"
        elif "PE" in option_type:
            action_line = f"📕 BUY PUT (PE)   →  {symbol} {atm_strike} PE"
            action_emoji = "📕"
        else:
            action_line = f"📊 OPTIONS SIGNAL → {symbol}"
            action_emoji = "📊"

        entry = signal.get("entry_price", 0) or 0
        target = signal.get("target_price", 0) or 0
        sl = signal.get("stop_loss", 0) or 0
        confidence = signal.get("confidence", 0)

        risk = abs(entry - sl)
        reward = abs(target - entry)
        rr = round(reward / risk, 1) if risk > 0 else 0

        # Key metadata lines
        extra_lines: List[str] = []
        if meta.get("pcr"):
            pcr_val = meta["pcr"]
            pcr_icon = (
                "🟢" if pcr_val > 1.0
                else "🔴" if pcr_val < 0.8
                else "🟡"
            )
            extra_lines.append(f"  PCR            : {pcr_icon} {pcr_val:.3f}")
        if meta.get("oi_resistance"):
            extra_lines.append(
                f"  OI Resistance  : ₹{meta['oi_resistance']:,} (max CE OI)"
            )
        if meta.get("oi_support"):
            extra_lines.append(
                f"  OI Support     : ₹{meta['oi_support']:,} (max PE OI)"
            )
        if meta.get("vwap_value"):
            extra_lines.append(
                f"  VWAP           : ₹{meta['vwap_value']:,.2f}"
            )
        if meta.get("supertrend_value"):
            st_dir = meta.get("supertrend_direction", "")
            st_icon = "🟢" if st_dir == "bullish" else "🔴"
            extra_lines.append(
                f"  Supertrend     : {st_icon} ₹{meta['supertrend_value']:,.2f}"
            )
        if meta.get("exit_rule"):
            extra_lines.append(f"  Exit Rule      : {meta['exit_rule']}")
        if meta.get("sentiment"):
            sent = meta["sentiment"].upper()
            extra_lines.append(f"  PCR Sentiment  : {sent}")

        # Indicator summary
        ind_summary = self._build_indicator_summary(signal)

        lines = [
            f"{'─'*38}",
            f"{action_line}",
            f"{'─'*38}",
            f"  Strategy       : {strategy}",
            f"  Spot Entry     : ₹{entry:,.2f}",
            f"  Target (Spot)  : ₹{target:,.2f}",
            f"  Stop Loss      : ₹{sl:,.2f}",
            f"  R:R Ratio      : 1:{rr}",
            f"  Confidence     : {confidence:.0f}%"
            f"  ({signal.get('indicators_met', 'N/A')}/{signal.get('total_indicators', 'N/A')})",
        ]
        if extra_lines:
            lines.append("")
            lines.extend(extra_lines)

        lines += [
            "",
            "  Indicators:",
            ind_summary,
            "",
            f"  ⏰ {now_ist().strftime('%H:%M IST  %d %b %Y')}",
            f"  🚀 ENTER NOW",
            f"{'─'*38}",
        ]

        return "\n".join(lines)

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

    def format_portfolio_update(
        self,
        positions: List[Dict[str, Any]],
        closed: List[Dict[str, Any]],
        sl_hits: List[Dict[str, Any]],
        target_hits: List[Dict[str, Any]],
    ) -> str:
        """Format a live portfolio P&L update for Telegram."""
        # Calculate totals
        total_invested = 0.0
        total_unrealized = 0.0
        for p in positions:
            entry = p.get("avg_entry_price", 0)
            qty = p.get("quantity", 0)
            total_invested += entry * qty
            total_unrealized += p.get("unrealized_pnl", 0) or 0

        total_realized = sum(
            (c.get("realized_pnl", 0) or 0) for c in closed
        )
        overall = total_unrealized + total_realized
        overall_icon = "\U0001f7e2" if overall >= 0 else "\U0001f534"

        winning = [
            c for c in closed
            if (c.get("realized_pnl", 0) or 0) > 0
        ]
        win_rate = (
            (len(winning) / len(closed)) * 100
            if closed else 0
        )

        lines = [
            "\U0001f4ca *PORTFOLIO UPDATE*",
            "",
            f"{overall_icon} *Overall P&L:* `{CURRENCY_SYMBOL}"
            f"{overall:+,.2f}`",
            f"\U0001f4b0 Unrealized: `{CURRENCY_SYMBOL}"
            f"{total_unrealized:+,.2f}`",
            f"\u2705 Realized: `{CURRENCY_SYMBOL}"
            f"{total_realized:+,.2f}`",
            f"\U0001f3af Win Rate: `{win_rate:.0f}%` "
            f"({len(winning)}W / "
            f"{len(closed) - len(winning)}L)",
            "",
        ]

        # SL/Target hits (important - show first)
        if sl_hits:
            lines.append(
                f"\U0001f6d1 *STOP LOSS HIT ({len(sl_hits)})*"
            )
            for h in sl_hits:
                lines.append(
                    f"  \u274c `{h['symbol']:12s}` "
                    f"`{CURRENCY_SYMBOL}{h.get('pnl', 0):>+10,.2f}`"
                )
            lines.append("")

        if target_hits:
            lines.append(
                f"\U0001f389 *TARGET HIT ({len(target_hits)})*"
            )
            for h in target_hits:
                lines.append(
                    f"  \U0001f4b5 `{h['symbol']:12s}` "
                    f"`{CURRENCY_SYMBOL}{h.get('pnl', 0):>+10,.2f}`"
                )
            lines.append("")

        # Open positions sorted by P&L
        if positions:
            sorted_pos = sorted(
                positions,
                key=lambda x: x.get("unrealized_pnl", 0) or 0,
                reverse=True,
            )
            lines.append(
                f"\U0001f4c8 *OPEN POSITIONS ({len(positions)})*"
            )
            lines.append("`\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500`")
            for p in sorted_pos:
                entry = p.get("avg_entry_price", 0)
                current = p.get("current_price", entry)
                pnl = p.get("unrealized_pnl", 0) or 0
                pnl_pct = (
                    ((current - entry) / entry) * 100
                    if entry > 0
                    else 0
                )
                icon = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
                lines.append(
                    f"{icon} `{p['symbol']:12s}` "
                    f"`{pnl_pct:>+6.1f}%` "
                    f"`{CURRENCY_SYMBOL}{pnl:>+10,.2f}`"
                )
            lines.append("`\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500`")

        now_str = datetime.now(timezone.utc).strftime(
            "%d %b %Y, %H:%M UTC"
        )
        lines.append(f"\n\U0001f552 _{now_str}_")

        return "\n".join(lines)

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

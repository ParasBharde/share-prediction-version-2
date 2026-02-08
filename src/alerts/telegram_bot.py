"""
Telegram Bot Integration

Purpose:
    Sends trading alerts via Telegram using python-telegram-bot (v20+).
    Supports inline buttons, retry logic, and priority-based delivery.
    Queues failed messages for later retry.

Dependencies:
    - python-telegram-bot (v20+, async)

Logging:
    - Successful sends at INFO
    - Retries at WARNING
    - Failures at ERROR

Fallbacks:
    If Telegram API unreachable, messages are enqueued for retry.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.error import (
        NetworkError,
        RetryAfter,
        TelegramError,
        TimedOut,
    )
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False
    Bot = None
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None

    # Define stub exception classes for when telegram is not installed
    class TelegramError(Exception):
        pass

    class RetryAfter(TelegramError):
        def __init__(self, retry_after=0):
            self.retry_after = retry_after

    class TimedOut(TelegramError):
        pass

    class NetworkError(TelegramError):
        pass

from src.monitoring.logger import get_alert_logger
from src.monitoring.metrics import (
    alert_failed_counter,
    alert_sent_counter,
    health_check_gauge,
)
from src.utils.constants import (
    AlertPriority,
    DEFAULT_MAX_RETRIES,
)

logger = get_alert_logger()

# Maximum number of send attempts per message
MAX_RETRIES = DEFAULT_MAX_RETRIES

# Base delay between retries in seconds (exponential backoff)
RETRY_BASE_DELAY = 2


class TelegramBot:
    """Telegram bot client for delivering trading alerts."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        """
        Initialize the Telegram bot.

        Args:
            bot_token: Telegram Bot API token from BotFather.
            chat_id: Target chat/channel ID for alert delivery.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._failed_queue: List[Dict[str, Any]] = []

        if TELEGRAM_AVAILABLE and Bot is not None:
            self._bot = Bot(token=self.bot_token)
        else:
            self._bot = None
            logger.warning(
                "python-telegram-bot not available, "
                "alerts will be logged only"
            )

        logger.info(
            "TelegramBot initialized",
            extra={"chat_id": self.chat_id},
        )

    async def send_alert(
        self,
        message: str,
        priority: str = AlertPriority.MEDIUM.value,
        inline_buttons: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """
        Send an alert message to Telegram with optional inline buttons.

        Retries up to MAX_RETRIES times with exponential backoff on
        transient failures. Failed messages are queued for later retry.

        Args:
            message: The alert message text (supports Markdown).
            priority: Alert priority level (HIGH, MEDIUM, LOW, CRITICAL).
            inline_buttons: Optional list of button dicts with keys
                ``text`` and ``callback_data``.  Example::

                    [{"text": "View Chart", "callback_data": "chart_RELIANCE"}]

        Returns:
            True if the message was delivered successfully, False otherwise.
        """
        if self._bot is None:
            logger.info(
                f"[DRY RUN] Alert ({priority}): {message[:200]}",
            )
            return True

        reply_markup = None
        if inline_buttons and TELEGRAM_AVAILABLE:
            reply_markup = self._build_inline_keyboard(inline_buttons)

        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self._bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                    disable_web_page_preview=True,
                )

                alert_sent_counter.labels(
                    priority=priority,
                    channel="telegram",
                ).inc()

                logger.info(
                    "Alert sent via Telegram",
                    extra={
                        "priority": priority,
                        "attempt": attempt,
                    },
                )
                return True

            except RetryAfter as e:
                wait = e.retry_after
                logger.warning(
                    f"Rate-limited by Telegram, retrying after "
                    f"{wait}s (attempt {attempt}/{MAX_RETRIES})",
                    extra={"retry_after": wait},
                )
                await asyncio.sleep(wait)
                last_error = e

            except TimedOut as e:
                delay = RETRY_BASE_DELAY ** attempt
                logger.warning(
                    f"Telegram request timed out, retrying in "
                    f"{delay}s (attempt {attempt}/{MAX_RETRIES})",
                    extra={"delay": delay},
                )
                await asyncio.sleep(delay)
                last_error = e

            except NetworkError as e:
                delay = RETRY_BASE_DELAY ** attempt
                logger.warning(
                    f"Network error sending alert, retrying in "
                    f"{delay}s (attempt {attempt}/{MAX_RETRIES})",
                    extra={"delay": delay, "error": str(e)},
                )
                await asyncio.sleep(delay)
                last_error = e

            except TelegramError as e:
                logger.error(
                    f"Telegram API error: {e}",
                    exc_info=True,
                    extra={"priority": priority, "attempt": attempt},
                )
                last_error = e
                break

        # All retries exhausted or non-retryable error
        alert_failed_counter.labels(
            priority=priority,
            channel="telegram",
        ).inc()

        self._enqueue_failed(message, priority, inline_buttons, last_error)

        logger.error(
            "Failed to send alert after all retries",
            extra={
                "priority": priority,
                "retries": MAX_RETRIES,
                "last_error": str(last_error),
            },
        )
        return False

    async def send_daily_summary(
        self, summary_data: Dict[str, Any]
    ) -> bool:
        """
        Send a formatted daily summary to Telegram.

        Args:
            summary_data: Dictionary containing daily summary fields.
                Expected keys::

                    {
                        "date": "2025-01-15",
                        "signals_generated": 12,
                        "alerts_sent": 10,
                        "portfolio_value": 1050000.0,
                        "daily_pnl": 2500.0,
                        "daily_pnl_pct": 0.24,
                        "active_positions": 5,
                        "top_gainer": {"symbol": "RELIANCE", "pct": 3.2},
                        "top_loser": {"symbol": "TCS", "pct": -1.1},
                    }

        Returns:
            True if the summary was delivered successfully.
        """
        try:
            date = summary_data.get("date", "N/A")
            signals = summary_data.get("signals_generated", 0)
            alerts = summary_data.get("alerts_sent", 0)
            portfolio = summary_data.get("portfolio_value", 0.0)
            daily_pnl = summary_data.get("daily_pnl", 0.0)
            daily_pnl_pct = summary_data.get("daily_pnl_pct", 0.0)
            positions = summary_data.get("active_positions", 0)

            pnl_emoji_marker = "+" if daily_pnl >= 0 else ""

            top_gainer = summary_data.get("top_gainer", {})
            top_loser = summary_data.get("top_loser", {})

            message_lines = [
                f"*Daily Summary - {date}*",
                "",
                f"Signals Generated: {signals}",
                f"Alerts Sent: {alerts}",
                f"Active Positions: {positions}",
                "",
                f"Portfolio Value: Rs {portfolio:,.2f}",
                f"Daily P&L: {pnl_emoji_marker}{daily_pnl:,.2f} "
                f"({pnl_emoji_marker}{daily_pnl_pct:.2f}%)",
            ]

            if top_gainer:
                message_lines.append(
                    f"Top Gainer: {top_gainer.get('symbol', 'N/A')} "
                    f"({top_gainer.get('pct', 0):.2f}%)"
                )
            if top_loser:
                message_lines.append(
                    f"Top Loser: {top_loser.get('symbol', 'N/A')} "
                    f"({top_loser.get('pct', 0):.2f}%)"
                )

            message = "\n".join(message_lines)
            return await self.send_alert(
                message=message,
                priority=AlertPriority.LOW.value,
            )

        except Exception as e:
            logger.error(
                f"Failed to build daily summary message: {e}",
                exc_info=True,
            )
            return False

    def _build_inline_keyboard(
        self, buttons: List[Dict[str, str]]
    ) -> InlineKeyboardMarkup:
        """
        Build a Telegram inline keyboard from button definitions.

        Args:
            buttons: List of button dictionaries, each containing:
                - ``text``: Display label for the button.
                - ``callback_data``: Data sent on button press.
                - ``url`` (optional): URL to open instead of callback.

        Returns:
            InlineKeyboardMarkup instance ready for attachment.
        """
        keyboard_rows: List[List[InlineKeyboardButton]] = []

        for btn in buttons:
            if "url" in btn:
                keyboard_rows.append([
                    InlineKeyboardButton(
                        text=btn["text"],
                        url=btn["url"],
                    )
                ])
            else:
                keyboard_rows.append([
                    InlineKeyboardButton(
                        text=btn["text"],
                        callback_data=btn.get("callback_data", "noop"),
                    )
                ])

        return InlineKeyboardMarkup(keyboard_rows)

    async def health_check(self) -> bool:
        """
        Verify bot connectivity by calling the Telegram ``getMe`` API.

        Updates the Prometheus health gauge for the ``telegram`` service.

        Returns:
            True if the bot is reachable and authenticated.
        """
        try:
            me = await self._bot.get_me()
            is_healthy = me is not None

            health_check_gauge.labels(service="telegram").set(
                1 if is_healthy else 0
            )

            if is_healthy:
                logger.debug(
                    f"Telegram health check passed: @{me.username}"
                )
            else:
                logger.warning("Telegram health check returned None")

            return is_healthy

        except TelegramError as e:
            health_check_gauge.labels(service="telegram").set(0)
            logger.error(
                f"Telegram health check failed: {e}",
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue_failed(
        self,
        message: str,
        priority: str,
        inline_buttons: Optional[List[Dict[str, str]]],
        error: Optional[Exception],
    ) -> None:
        """
        Queue a failed message for later retry.

        Args:
            message: Original message text.
            priority: Alert priority.
            inline_buttons: Optional inline button definitions.
            error: The exception that caused delivery failure.
        """
        self._failed_queue.append({
            "message": message,
            "priority": priority,
            "inline_buttons": inline_buttons,
            "error": str(error) if error else None,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            "Failed alert enqueued for retry",
            extra={"queue_size": len(self._failed_queue)},
        )

    def get_failed_queue(self) -> List[Dict[str, Any]]:
        """
        Return a copy of the current in-memory failed message queue.

        Returns:
            List of queued alert dictionaries.
        """
        return list(self._failed_queue)

    def clear_failed_queue(self) -> None:
        """Clear all messages from the in-memory failed queue."""
        count = len(self._failed_queue)
        self._failed_queue.clear()
        logger.info(
            f"Cleared {count} messages from failed queue"
        )

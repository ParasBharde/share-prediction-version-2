"""
Sentry Error Tracking Integration

Purpose:
    Integrates with Sentry for error tracking and alerting.
    Captures unhandled exceptions and provides context.

Dependencies:
    - sentry_sdk

Logging:
    - Sentry init at INFO
    - Capture events at ERROR

Fallbacks:
    If Sentry unavailable, errors are logged locally only.
"""

import os
from typing import Any, Dict, Optional

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

from src.monitoring.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def init_sentry() -> None:
    """
    Initialize Sentry error tracking.

    Reads configuration from system.yaml and environment variables.
    """
    try:
        config = load_config("system")
        sentry_config = config.get("monitoring", {}).get("sentry", {})

        if not sentry_config.get("enabled", False):
            logger.info("Sentry is disabled in config")
            return

        dsn = os.environ.get(
            "SENTRY_DSN", sentry_config.get("dsn", "")
        )

        if not dsn:
            logger.warning(
                "Sentry DSN not configured, skipping initialization"
            )
            return

        sentry_logging = LoggingIntegration(
            level=None,
            event_level=None,
        )

        sentry_sdk.init(
            dsn=dsn,
            integrations=[sentry_logging],
            traces_sample_rate=sentry_config.get(
                "traces_sample_rate", 0.1
            ),
            environment=config.get("system", {}).get(
                "environment", "production"
            ),
            release=config.get("system", {}).get(
                "version", "unknown"
            ),
        )

        logger.info("Sentry initialized successfully")

    except Exception as e:
        logger.error(
            f"Failed to initialize Sentry: {e}",
            exc_info=True,
        )


def capture_exception(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Capture an exception and send to Sentry.

    Args:
        error: The exception to capture.
        context: Additional context to attach.

    Returns:
        Sentry event ID if captured, None otherwise.
    """
    try:
        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_extra(key, value)
                event_id = sentry_sdk.capture_exception(error)
        else:
            event_id = sentry_sdk.capture_exception(error)

        if event_id:
            logger.info(
                f"Exception captured in Sentry: {event_id}",
                extra={"event_id": event_id},
            )
        return event_id

    except Exception:
        logger.error(
            "Failed to capture exception in Sentry",
            exc_info=True,
        )
        return None


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Capture a message and send to Sentry.

    Args:
        message: Message to capture.
        level: Severity level.
        context: Additional context.

    Returns:
        Sentry event ID if captured, None otherwise.
    """
    try:
        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_extra(key, value)
                event_id = sentry_sdk.capture_message(
                    message, level=level
                )
        else:
            event_id = sentry_sdk.capture_message(
                message, level=level
            )

        return event_id

    except Exception:
        logger.error(
            "Failed to capture message in Sentry",
            exc_info=True,
        )
        return None


def set_user_context(
    user_id: str,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> None:
    """
    Set user context for Sentry events.

    Args:
        user_id: User identifier.
        username: Optional username.
        email: Optional email.
    """
    sentry_sdk.set_user(
        {
            "id": user_id,
            "username": username,
            "email": email,
        }
    )


def add_breadcrumb(
    message: str,
    category: str = "default",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add a breadcrumb for debugging context.

    Args:
        message: Breadcrumb message.
        category: Category of the breadcrumb.
        level: Severity level.
        data: Additional data.
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {},
    )

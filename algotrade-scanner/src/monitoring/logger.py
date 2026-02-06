"""
Centralized Structured Logging

Purpose:
    Single source of truth for all logging.
    JSON-structured logs for machine parsing.
    Daily rotating files with retention.
    Separate logs per component.

Log Levels:
    - DEBUG: Detailed diagnostic info (disabled in prod)
    - INFO: General informational messages
    - WARNING: Warning messages (recoverable errors)
    - ERROR: Error messages (operation failed)
    - CRITICAL: Critical errors (system failure)

Log Files:
    - logs/daily.log - Main system log
    - logs/strategies.log - Strategy execution
    - logs/alerts.log - Alert delivery
    - logs/data.log - Data fetching
    - logs/errors.log - All errors across system

Retention:
    - 7 days for DEBUG/INFO logs
    - 30 days for ERROR/CRITICAL logs
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string of the log entry.
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        # Add extra fields from record
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno",
            "funcName", "pathname", "filename", "module",
            "thread", "threadName", "process", "processName",
            "levelname", "levelno", "message", "msecs",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_data[key] = value

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        message = record.getMessage()

        formatted = (
            f"{color}[{timestamp}] "
            f"{record.levelname:8s} "
            f"{record.name}: {message}{self.RESET}"
        )

        if record.exc_info and record.exc_info[0] is not None:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup logger with multiple handlers.

    Args:
        name: Logger name (usually __name__).
        log_file: Optional specific log file path.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # Daily rotating file handler
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file or "logs/daily.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Error file handler (errors only)
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename="logs/errors.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    logger.addHandler(error_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return setup_logger(name)


def get_strategy_logger() -> logging.Logger:
    """Logger for strategy execution."""
    return setup_logger("strategies", log_file="logs/strategies.log")


def get_alert_logger() -> logging.Logger:
    """Logger for alert delivery."""
    return setup_logger("alerts", log_file="logs/alerts.log")


def get_data_logger() -> logging.Logger:
    """Logger for data operations."""
    return setup_logger("data", log_file="logs/data.log")

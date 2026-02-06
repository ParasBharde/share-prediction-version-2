"""
Time and Market Hours Utilities

Purpose:
    Provides market-aware time functions.
    Handles IST timezone conversions.
    Determines market open/close status.

Dependencies:
    - pytz for timezone handling
    - python-dateutil for date parsing

Logging:
    None (utility functions)

Fallbacks:
    Uses UTC if IST timezone unavailable.
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, Tuple

import pytz
from dateutil import parser as date_parser

from src.utils.constants import (
    DEFAULT_TIMEZONE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)

IST = pytz.timezone(DEFAULT_TIMEZONE)
UTC = pytz.UTC

# NSE holidays for 2025 (update annually)
NSE_HOLIDAYS_2025 = [
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Janmashtami
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 21),  # Diwali (Laxmi Pujan)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
]


def now_ist() -> datetime:
    """Get current time in IST."""
    return datetime.now(IST)


def now_utc() -> datetime:
    """Get current time in UTC."""
    return datetime.now(UTC)


def to_ist(dt: datetime) -> datetime:
    """
    Convert datetime to IST.

    Args:
        dt: Datetime object (can be naive or aware).

    Returns:
        IST-aware datetime.
    """
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    return dt.astimezone(IST)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.

    Args:
        dt: Datetime object (can be naive or aware).

    Returns:
        UTC-aware datetime.
    """
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(UTC)


def parse_date(date_str: str) -> date:
    """
    Parse date string into date object.

    Args:
        date_str: Date string (various formats accepted).

    Returns:
        Parsed date object.
    """
    return date_parser.parse(date_str).date()


def parse_datetime(dt_str: str) -> datetime:
    """
    Parse datetime string into IST datetime.

    Args:
        dt_str: Datetime string.

    Returns:
        IST-aware datetime object.
    """
    dt = date_parser.parse(dt_str)
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(IST)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if the market is currently open.

    Args:
        dt: Datetime to check (defaults to now IST).

    Returns:
        True if market is open.
    """
    if dt is None:
        dt = now_ist()
    else:
        dt = to_ist(dt)

    # Check if it's a weekday
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Check if it's a holiday
    if dt.date() in NSE_HOLIDAYS_2025:
        return False

    # Check market hours
    market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)

    return market_open <= dt.time() <= market_close


def is_trading_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a given date is a trading day.

    Args:
        check_date: Date to check (defaults to today IST).

    Returns:
        True if it's a trading day.
    """
    if check_date is None:
        check_date = now_ist().date()

    # Check weekend
    if check_date.weekday() >= 5:
        return False

    # Check holiday
    if check_date in NSE_HOLIDAYS_2025:
        return False

    return True


def get_market_hours(
    check_date: Optional[date] = None,
) -> Tuple[datetime, datetime]:
    """
    Get market open and close times for a date.

    Args:
        check_date: Date to get hours for (defaults to today).

    Returns:
        Tuple of (market_open, market_close) as IST datetimes.
    """
    if check_date is None:
        check_date = now_ist().date()

    market_open = IST.localize(
        datetime.combine(
            check_date,
            time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE),
        )
    )
    market_close = IST.localize(
        datetime.combine(
            check_date,
            time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE),
        )
    )

    return market_open, market_close


def get_previous_trading_day(
    check_date: Optional[date] = None,
) -> date:
    """
    Get the most recent previous trading day.

    Args:
        check_date: Reference date (defaults to today).

    Returns:
        Previous trading day date.
    """
    if check_date is None:
        check_date = now_ist().date()

    prev_date = check_date - timedelta(days=1)
    while not is_trading_day(prev_date):
        prev_date -= timedelta(days=1)

    return prev_date


def get_next_trading_day(
    check_date: Optional[date] = None,
) -> date:
    """
    Get the next trading day.

    Args:
        check_date: Reference date (defaults to today).

    Returns:
        Next trading day date.
    """
    if check_date is None:
        check_date = now_ist().date()

    next_date = check_date + timedelta(days=1)
    while not is_trading_day(next_date):
        next_date += timedelta(days=1)

    return next_date


def trading_days_between(
    start_date: date, end_date: date
) -> int:
    """
    Count trading days between two dates.

    Args:
        start_date: Start date (inclusive).
        end_date: End date (inclusive).

    Returns:
        Number of trading days.
    """
    count = 0
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime for display in alerts.

    Args:
        dt: Datetime to format (defaults to now IST).

    Returns:
        Formatted string like '06 Feb 2025 15:35 IST'.
    """
    if dt is None:
        dt = now_ist()
    else:
        dt = to_ist(dt)

    return dt.strftime("%d %b %Y %H:%M IST")

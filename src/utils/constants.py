"""
System-wide Constants

Purpose:
    Defines all constant values used across the system.
    No hardcoded values should exist outside this file and config.

Dependencies:
    None (leaf module)

Logging:
    None (constants only)
"""

from enum import Enum

# Application metadata
APP_NAME = "AlgoTrade Scanner"
APP_VERSION = "1.0.0"

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
PRE_MARKET_OPEN_HOUR = 9
PRE_MARKET_OPEN_MINUTE = 0
POST_MARKET_CLOSE_HOUR = 15
POST_MARKET_CLOSE_MINUTE = 40

# Default timezone
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Market indices
NIFTY50_INDEX = "NIFTY 50"
NIFTY100_INDEX = "NIFTY 100"
NIFTY500_INDEX = "NIFTY 500"

# NSE API endpoints
NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_BASE = "https://www.nseindia.com/api"
NSE_ARCHIVE_BASE = "https://archives.nseindia.com"
NSE_INDEX_ARCHIVE_PATH = "/content/indices"
NSE_EQUITY_QUOTE = "/quote-equity"
NSE_TRADE_INFO = "/quote-equity?info=trade-info"
NSE_HISTORICAL = "/historical/cm/equity"
NSE_INDEX_DATA = "/equity-stockIndices"
NSE_MARKET_STATUS = "/marketStatus"

# ── NSE User-Agent rotation pool ─────────────────────────────────────────────
# Rotate through multiple realistic browser UA strings to avoid WAF fingerprinting.
NSE_USER_AGENTS = [
    # Chrome 124 on Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    # Chrome 123 on macOS
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    # Firefox 125 on Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    # Edge 124 on Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
    ),
    # Chrome 131 on Windows (legacy constant – kept for compatibility)
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
]

# Default UA (first in pool) – kept for backward compatibility
_DEFAULT_UA = NSE_USER_AGENTS[0]

# Default headers for NSE (must match real browser to avoid WAF blocks)
NSE_HEADERS = {
    "User-Agent": _DEFAULT_UA,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": (
        '"Google Chrome";v="124", "Chromium";v="124", '
        '"Not_A Brand";v="99"'
    ),
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}

# Headers for NSE homepage visit (to get cookies)
NSE_HOMEPAGE_HEADERS = {
    "User-Agent": _DEFAULT_UA,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": NSE_HEADERS["Sec-Ch-Ua"],
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}

# Yahoo Finance suffix for NSE stocks
YAHOO_NSE_SUFFIX = ".NS"

# Currency
CURRENCY_SYMBOL = "\u20b9"  # ₹
CURRENCY_CODE = "INR"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class AlertPriority(Enum):
    """Alert priority levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


class OrderType(Enum):
    """Order types for paper trading."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(Enum):
    """Position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


class DataSource(Enum):
    """Data source identifiers."""
    NSE = "nse_official"
    YAHOO = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    CACHE = "cache"
    STALE_CACHE = "stale_cache"


class HealthStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# Database table names
TABLE_OHLCV = "ohlcv_data"
TABLE_COMPANIES = "companies"
TABLE_SIGNALS = "signals"
TABLE_ALERTS = "alerts"
TABLE_ORDERS = "orders"
TABLE_POSITIONS = "positions"
TABLE_PORTFOLIO = "portfolio"
TABLE_PERFORMANCE = "performance_metrics"

# Redis key prefixes
REDIS_PREFIX_OHLCV = "ohlcv"
REDIS_PREFIX_RATE_LIMIT = "rate_limit"
REDIS_PREFIX_ALERT_SEEN = "alert_seen"
REDIS_PREFIX_LOCK = "lock"
REDIS_PREFIX_CIRCUIT = "circuit"

# Default timeouts (seconds)
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 10
DEFAULT_DB_TIMEOUT = 30

# Retry defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 2
DEFAULT_BACKOFF_MAX = 16

# Alert deduplication window (seconds)
# 3 trading days — prevents the same symbol/direction from re-alerting
# across consecutive daily scans (Mon→Wed).  BUY and SELL on the same
# symbol are still treated as separate slots.
ALERT_DEDUP_WINDOW = 259200  # 72 hours (3 trading days)

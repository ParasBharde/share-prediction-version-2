"""
Prometheus Metrics

Purpose:
    Defines and exposes all Prometheus metrics.
    Provides metric helpers for instrumentation.

Dependencies:
    - prometheus_client (optional)

Logging:
    Metric registration at DEBUG level.

Fallbacks:
    If Prometheus unavailable, metrics are no-ops.
"""

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except Exception:
    _PROMETHEUS_AVAILABLE = False
    logger.debug(
        "prometheus_client not installed, metrics will be no-ops"
    )

    # No-op stubs so the rest of the codebase can call
    # metric.labels(...).inc() / .observe() / .set() without errors.

    class _NoOpMetric:
        """Stub metric that silently ignores all operations."""

        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    Counter = _NoOpMetric
    Gauge = _NoOpMetric
    Histogram = _NoOpMetric
    Summary = _NoOpMetric
    REGISTRY = None

    def start_http_server(*args, **kwargs):
        pass


# --- Job Metrics ---

job_success_counter = Counter(
    "algotrade_job_success_total",
    "Total successful job executions",
    ["job_name"],
)

job_failure_counter = Counter(
    "algotrade_job_failure_total",
    "Total failed job executions",
    ["job_name"],
)

job_duration_histogram = Histogram(
    "algotrade_job_duration_seconds",
    "Job execution duration in seconds",
    ["job_name"],
    buckets=[10, 30, 60, 120, 300, 600, 900],
)

# --- Data Fetch Metrics ---

data_fetch_success_counter = Counter(
    "algotrade_data_fetch_success_total",
    "Successful data fetches",
    ["source"],
)

data_fetch_failure_counter = Counter(
    "algotrade_data_fetch_failure_total",
    "Failed data fetches",
    ["source"],
)

data_fetch_latency = Histogram(
    "algotrade_data_fetch_latency_seconds",
    "Data fetch latency in seconds",
    ["source"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

fallback_usage_counter = Counter(
    "algotrade_fallback_usage_total",
    "Fallback source usage count",
    ["from_source", "to_source"],
)

# --- Strategy Metrics ---

strategy_scan_counter = Counter(
    "algotrade_strategy_scan_total",
    "Stocks scanned per strategy",
    ["strategy_name"],
)

signal_generated_counter = Counter(
    "algotrade_signal_generated_total",
    "Signals generated per strategy",
    ["strategy_name", "signal_type"],
)

strategy_execution_time = Histogram(
    "algotrade_strategy_execution_seconds",
    "Strategy execution time in seconds",
    ["strategy_name"],
    buckets=[1, 5, 10, 30, 60, 120],
)

# --- Alert Metrics ---

alert_sent_counter = Counter(
    "algotrade_alert_sent_total",
    "Total alerts sent",
    ["priority", "channel"],
)

alert_failed_counter = Counter(
    "algotrade_alert_failed_total",
    "Total failed alert deliveries",
    ["priority", "channel"],
)

alert_deduplicated_counter = Counter(
    "algotrade_alert_deduplicated_total",
    "Alerts skipped due to deduplication",
    ["strategy_name"],
)

# --- Portfolio Metrics ---

portfolio_value_gauge = Gauge(
    "algotrade_portfolio_value",
    "Current portfolio value in INR",
)

active_positions_gauge = Gauge(
    "algotrade_active_positions",
    "Number of active positions",
)

total_pnl_gauge = Gauge(
    "algotrade_total_pnl",
    "Total P&L in INR",
)

daily_pnl_gauge = Gauge(
    "algotrade_daily_pnl",
    "Daily P&L in INR",
)

# --- System Health Metrics ---

health_check_gauge = Gauge(
    "algotrade_health_check",
    "Health check status (1=healthy, 0=unhealthy)",
    ["service"],
)

rate_limit_remaining_gauge = Gauge(
    "algotrade_rate_limit_remaining",
    "Remaining rate limit tokens",
    ["source"],
)

db_connection_pool_gauge = Gauge(
    "algotrade_db_pool_connections",
    "Database connection pool status",
    ["state"],  # active, idle, overflow
)

cache_hit_counter = Counter(
    "algotrade_cache_hits_total",
    "Cache hit count",
    ["cache_type"],
)

cache_miss_counter = Counter(
    "algotrade_cache_misses_total",
    "Cache miss count",
    ["cache_type"],
)


def start_metrics_server(port: int = 9090) -> None:
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to expose metrics on.
    """
    if not _PROMETHEUS_AVAILABLE:
        logger.info(
            "Prometheus not available, metrics server not started"
        )
        return

    try:
        start_http_server(port)
        logger.info(
            f"Prometheus metrics server started on port {port}"
        )
    except Exception as e:
        logger.error(
            f"Failed to start metrics server: {e}",
            exc_info=True,
        )

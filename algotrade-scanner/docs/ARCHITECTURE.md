# AlgoTrade Scanner - System Architecture

## Overview

AlgoTrade Scanner is a production-grade NSE stock scanning and alerting system built with Python 3.11+. It follows a microservices-ready, horizontally scalable architecture.

## Core Principles

1. **Everything configurable** - Zero hardcoded values
2. **Fail-safe with automatic fallbacks** - Circuit breakers, retry logic
3. **Observable** - Comprehensive logging, Prometheus metrics, Sentry
4. **Testable** - High test coverage with unit and integration tests
5. **Documented** - Inline docs + external guides

## Architecture Layers

### 1. Orchestrator Layer
- **Scheduler** (`src/orchestrator/scheduler.py`): APScheduler-based job scheduling
- **Health Checker** (`src/orchestrator/health_checker.py`): Monitors all services every 30s
- **Shutdown Handler** (`src/orchestrator/shutdown_handler.py`): Graceful shutdown logic

### 2. Data Ingestion Layer
- **NSE Fetcher** (Primary): Official NSE API with session management
- **Yahoo Finance** (Fallback 1): yfinance library
- **Alpha Vantage** (Fallback 2): REST API with API key
- **Rate Limiter**: Redis token bucket (40 req/min for NSE)
- **Data Validator**: Price sanity checks, volume validation
- **Fallback Manager**: Waterfall pattern with circuit breakers

### 3. Storage Layer
- **TimescaleDB**: OHLCV time-series data with hypertables
- **PostgreSQL**: Metadata, signals, alerts, orders
- **Redis**: Cache (5-min TTL), locks, rate limiting counters
- **Connection Pool**: Max 20 connections with auto-reconnect

### 4. Strategy Engine
- **Strategy Loader**: Dynamic plugin system from YAML configs
- **Parallel Executor**: Multiprocessing with CPU cores - 1 workers
- **Built-in Strategies**: Momentum Breakout, Mean Reversion, Volume Surge

### 5. Signal Processing
- **Signal Aggregator**: Weighted confidence scores, conflict voting
- **Ranking Engine**: Multi-factor scoring, sector diversification
- **Risk Filter**: Correlation checks, position limits, VIX filter

### 6. Alert Layer
- **Telegram Bot**: Primary delivery with inline buttons
- **Alert Formatter**: Jinja2 templates for rich messages
- **Deduplicator**: Redis-based, 24-hour window
- **Retry Queue**: PostgreSQL-backed, max 3 retries

### 7. Paper Trading
- **Order Simulator**: Slippage (0.1%) and commission (0.03%)
- **Portfolio Manager**: Position and cash tracking
- **P&L Calculator**: Realized + unrealized
- **Performance Tracker**: Sharpe, drawdown, win rate

### 8. Monitoring
- **Prometheus**: Metrics on port 9090
- **Grafana**: Visual dashboards
- **Sentry**: Error tracking
- **Structured Logging**: JSON format, daily rotation

## Data Flow

```
Stock Universe -> Data Fetching -> Validation -> Strategy Execution
-> Signal Generation -> Aggregation -> Ranking -> Risk Filtering
-> Alert Formatting -> Telegram Delivery
```

## Configuration

All configuration is in YAML files under `config/`:
- `system.yaml` - System-wide settings
- `data_sources.yaml` - API endpoints and rate limits
- `risk_management.yaml` - Portfolio constraints
- `alert_templates.yaml` - Telegram message templates
- `strategies/*.yaml` - Individual strategy configs

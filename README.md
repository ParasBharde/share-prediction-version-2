# AlgoTrade Scanner

Production-grade NSE stock scanning and alerting system with multi-strategy analysis, intelligent fallbacks, and Telegram alerts.

## Features

- **Multi-Strategy Scanning**: Momentum Breakout, Mean Reversion, Volume Surge
- **Intelligent Data Fetching**: NSE API (primary) with Yahoo Finance and Alpha Vantage fallbacks
- **Circuit Breakers**: Automatic failover with exponential backoff
- **Signal Processing**: Weighted confidence scoring, sector diversification, risk filtering
- **Telegram Alerts**: Rich formatted alerts with inline buttons
- **Paper Trading**: Simulated trading with slippage and commission modeling
- **Monitoring**: Prometheus metrics, Grafana dashboards, Sentry error tracking
- **Structured Logging**: JSON logs with daily rotation

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys

# 2. Start with Docker
cd deployments/docker
docker-compose up -d

# 3. Initialize database
docker-compose exec app python scripts/db_migrate.py

# 4. Start scanner
docker-compose exec app python -m src.orchestrator.scheduler
```

## Architecture

```
Orchestrator -> Data Ingestion -> Strategy Engine -> Signal Processing -> Alerts
                     |                                      |
                 Storage Layer                         Paper Trading
              (TimescaleDB/Redis)                    (Order Simulator)
```

## Configuration

All settings in `config/` directory:
- `system.yaml` - System-wide configuration
- `data_sources.yaml` - API endpoints and rate limits
- `risk_management.yaml` - Portfolio constraints
- `strategies/*.yaml` - Trading strategy configs

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Setup Guide](docs/SETUP.md)
- [Strategy Guide](docs/STRATEGIES.md)
- [API Reference](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Testing

```bash
pytest                              # Run all tests
pytest --cov=src                    # With coverage
pytest tests/unit/test_strategies.py # Specific test
```

## Tech Stack

- **Language**: Python 3.11+
- **Database**: PostgreSQL 14+ / TimescaleDB
- **Cache**: Redis 7+
- **Scheduling**: APScheduler
- **Monitoring**: Prometheus + Grafana
- **Alerts**: Telegram Bot API
- **Containerization**: Docker + Docker Compose

## License

Private - All Rights Reserved

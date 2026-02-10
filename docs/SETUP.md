# AlgoTrade Scanner - Setup Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ with TimescaleDB extension
- Redis 7+
- Docker & Docker Compose (recommended)

## Option 1: Docker Setup (Recommended)

### Step 1: Clone Repository
```bash
git clone <repo-url>
cd algotrade-scanner
```

### Step 2: Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Step 3: Start Services
```bash
cd deployments/docker
docker-compose up -d
```

This starts: TimescaleDB (5432), Redis (6379), Prometheus (9091), Grafana (3000)

### Step 4: Initialize Database
```bash
docker-compose exec app python scripts/db_migrate.py
```

### Step 5: Start Scanner
```bash
docker-compose exec app python -m src.orchestrator.scheduler
```

### Step 6: FOr FLush Redish
```bash
docker-compose exec redis redis-cli FLUSHALL
```

## Option 2: Manual Setup

### Step 1: Install Dependencies
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Setup PostgreSQL + TimescaleDB
```bash
sudo -u postgres psql
CREATE DATABASE algotrade;
\c algotrade
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

### Step 3: Setup Redis
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

### Step 4: Configure
```bash
cp .env.example .env
# Set DATABASE_URL, REDIS_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

### Step 5: Initialize Database
```bash
python scripts/db_migrate.py
```

### Step 6: Run
```bash
python -m src.orchestrator.scheduler
```

## Testing
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/unit/test_strategies.py
```

## Monitoring
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9091

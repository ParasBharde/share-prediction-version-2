# Runbook: Commands for Every Strategy and Every Script

This guide lists practical commands to run **all major functionality** in this repo:
- setup and migrations
- every script in `scripts/`
- every strategy (daily, BTST, intraday, options)
- full backtests and single-strategy backtests

> Run from repository root:
>
> `cd /workspace/share-prediction-version-2`

## 0) Environment and setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Optional env vars (Telegram/DB APIs etc.):
```bash
cp .env.example .env  # if available in your environment
# then edit .env values
```

## 1) Core system functionality

### Database migration
```bash
python scripts/db_migrate.py
python scripts/db_migrate.py --drop
```

### Start orchestrator scheduler
```bash
python -m src.orchestrator.scheduler
```

### Daily scan (all enabled daily strategies)
```bash
python scripts/daily_scan.py
python scripts/daily_scan.py --force
```

## 2) Commands for every strategy

## 2.1 Daily strategies

Run any combination using `daily_scan.py --strategies`:

```bash
python scripts/daily_scan.py --force --strategies "Darvas Box"
python scripts/daily_scan.py --force --strategies "Flag Pattern"
python scripts/daily_scan.py --force --strategies "Symmetrical Triangle"
python scripts/daily_scan.py --force --strategies "Descending Channel"
python scripts/daily_scan.py --force --strategies "Mother Candle V2"
python scripts/daily_scan.py --force --strategies "Momentum Breakout"
python scripts/daily_scan.py --force --strategies "Mean Reversion"
python scripts/daily_scan.py --force --strategies "Volume Surge"
```

Utility daily-scan modes:
```bash
python scripts/daily_scan.py --mother-v2-only
python scripts/daily_scan.py --btst-only
python scripts/daily_scan.py --test-symbol ICICIBANK
```

> Note: some strategies are disabled by default in `config/strategies/*.yaml`. Set `strategy.enabled: true` before running them.

## 2.2 BTST strategies (all 4)

```bash
python scripts/btst_scan.py --force
python scripts/btst_scan.py --force --dry-run
python scripts/btst_scan.py --force --strategies "Darvas Box"
python scripts/btst_scan.py --force --strategies "Flag Pattern" "Symmetrical Triangle" "Descending Channel"
python scripts/btst_scan.py --force --chart-dir ./charts/btst --verbose-stats
```

## 2.3 Intraday strategies

```bash
python scripts/live_scan.py --symbol RELIANCE --interval 15m --bypass-time
python scripts/live_scan.py --watchlist RELIANCE,INFY,TCS --interval 5m --bypass-time
python scripts/live_scan.py --universe NIFTY50 --interval 15m --bypass-time
python scripts/live_scan.py --universe NIFTY100 --interval 15m --min-confidence 0.8 --bypass-time
python scripts/live_scan.py --universe NIFTY50 --repeat 5 --telegram --chart-dir /tmp --bypass-time
```

Intraday strategy names loaded by `live_scan.py`:
- Intraday Momentum
- Intraday Volume Surge
- Intraday Mean Reversion

## 2.4 Options strategies

### Online options scanner (`options_scan.py`)
```bash
python scripts/options_scan.py --symbol NIFTY
python scripts/options_scan.py --symbol NIFTY --symbol BANKNIFTY --interval 5m
python scripts/options_scan.py --symbol NIFTY --telegram --chart-dir /tmp
python scripts/options_scan.py --symbol BANKNIFTY --dry-run
python scripts/options_scan.py --symbol NIFTY --repeat 5
```

Options strategies loaded by `options_scan.py`:
- Options OI Breakout
- Options VWAP Supertrend
- Options PCR Sentiment

### Offline options scanner (`options_scan_offline.py`)
```bash
python scripts/options_scan_offline.py --index BANKNIFTY
python scripts/options_scan_offline.py --index NIFTY --strikes 23500 23600 23700
python scripts/options_scan_offline.py --index FINNIFTY --expiry 25-Feb-2026
python scripts/options_scan_offline.py --index BANKNIFTY --repeat 5
```

## 2.5 Backtesting all/specific strategies

### Run all enabled strategies
```bash
python scripts/backtest_runner.py --all
python scripts/backtest_runner.py --all --universe NIFTY100 --start 2023-01-01 --end 2024-12-31
```

### Run one strategy at a time (all supported names)
```bash
python scripts/backtest_runner.py --strategy "Momentum Breakout"
python scripts/backtest_runner.py --strategy "Mean Reversion"
python scripts/backtest_runner.py --strategy "Volume Surge"
python scripts/backtest_runner.py --strategy "Intraday Momentum"
python scripts/backtest_runner.py --strategy "Intraday Volume Surge"
python scripts/backtest_runner.py --strategy "Intraday Mean Reversion"
python scripts/backtest_runner.py --strategy "Options OI Breakout"
python scripts/backtest_runner.py --strategy "Options VWAP Supertrend"
python scripts/backtest_runner.py --strategy "Options PCR Sentiment"
python scripts/backtest_runner.py --strategy "Mother Candle V2"
python scripts/backtest_runner.py --strategy "Darvas Box"
python scripts/backtest_runner.py --strategy "Flag Pattern"
python scripts/backtest_runner.py --strategy "Symmetrical Triangle"
python scripts/backtest_runner.py --strategy "Descending Channel"
```

Useful backtest options:
```bash
python scripts/backtest_runner.py --strategy "Flag Pattern" --symbols RELIANCE,TCS,INFY --start 2024-01-01 --end 2024-12-31 --capital 1000000 --output-dir backtest_results
```

## 3) Commands for every script in `scripts/`

### `scripts/backtest_runner.py`
```bash
python scripts/backtest_runner.py --all
python scripts/backtest_runner.py --strategy "Momentum Breakout"
```

### `scripts/btst_scan.py`
```bash
python scripts/btst_scan.py --force
python scripts/btst_scan.py --force --dry-run
```

### `scripts/daily_scan.py`
```bash
python scripts/daily_scan.py --force
python scripts/daily_scan.py --mother-v2-only
```

### `scripts/db_migrate.py`
```bash
python scripts/db_migrate.py
python scripts/db_migrate.py --drop
```

### `scripts/deploy_strategy.py`
```bash
python scripts/deploy_strategy.py --config config/strategies/mother_candle_v2.yaml --validate-only
python scripts/deploy_strategy.py --config config/strategies/mother_candle_v2.yaml
```

### `scripts/gemini_code.py`
```bash
python scripts/gemini_code.py
```

### `scripts/generate_report.py`
```bash
python scripts/generate_report.py --period daily
python scripts/generate_report.py --period weekly --output reports/weekly_report.txt
python scripts/generate_report.py --period monthly
```

### `scripts/live_scan.py`
```bash
python scripts/live_scan.py --symbol RELIANCE --bypass-time
python scripts/live_scan.py --universe NIFTY50 --repeat 5 --telegram --bypass-time
```

### `scripts/options_scan.py`
```bash
python scripts/options_scan.py --symbol NIFTY
python scripts/options_scan.py --symbol NIFTY --telegram
```

### `scripts/options_scan_offline.py`
```bash
python scripts/options_scan_offline.py --index BANKNIFTY
python scripts/options_scan_offline.py --index NIFTY --repeat 5
```

### `scripts/portfolio_tracker.py`
```bash
python scripts/portfolio_tracker.py
python scripts/portfolio_tracker.py --update
python scripts/portfolio_tracker.py --update --telegram
python scripts/portfolio_tracker.py --closed
python scripts/portfolio_tracker.py --report
python scripts/portfolio_tracker.py --reset
```

### `scripts/test_chart.py`
```bash
python scripts/test_chart.py
```

### `scripts/view_trades.py`
```bash
python scripts/view_trades.py
python scripts/view_trades.py --summary
python scripts/view_trades.py --orders
python scripts/view_trades.py --all
python scripts/view_trades.py --performance
```

## 4) Testing and quality checks

```bash
pytest
pytest --cov=src
pytest tests/unit/test_strategies.py
```

## 5) Fast command discovery (optional)

List all script help pages quickly:
```bash
for f in scripts/*.py; do
  echo "\n===== $f ====="
  python "$f" --help || true
done
```


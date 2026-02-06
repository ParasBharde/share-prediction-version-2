# AlgoTrade Scanner - Troubleshooting Guide

## Common Issues

### NSE API Blocked / 403 Errors
**Cause:** NSE rate limiting or IP blocking
**Solution:**
1. Check rate limits in `config/data_sources.yaml`
2. Increase `backoff_factor` and `backoff_max`
3. System will automatically use Yahoo Finance fallback
4. Check `logs/data.log` for details

### Database Connection Failed
**Cause:** PostgreSQL not running or wrong credentials
**Solution:**
1. Verify PostgreSQL is running: `systemctl status postgresql`
2. Check `DATABASE_URL` in `.env`
3. Test connection: `psql $DATABASE_URL -c "SELECT 1"`
4. Check pool status in logs

### No Telegram Alerts
**Cause:** Invalid bot token or chat ID
**Solution:**
1. Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
2. Test bot: send `/start` to your bot in Telegram
3. Check `logs/alerts.log` for delivery errors
4. Check retry queue: failed alerts are retried 3 times

### Redis Connection Failed
**Cause:** Redis not running
**Solution:**
1. Check Redis: `redis-cli ping`
2. Verify `REDIS_URL` in `.env`
3. System will fall back to in-memory caching automatically

### No Signals Generated
**Cause:** Market conditions don't match strategy criteria
**Solution:**
1. Check strategy configs in `config/strategies/`
2. Lower `confidence_threshold` temporarily for testing
3. Run backtest to verify strategy logic
4. Check `logs/strategies.log` for scan details

### High Memory Usage
**Cause:** Loading too many stocks or large DataFrames
**Solution:**
1. Reduce `chunk_size` in `config/system.yaml`
2. Reduce `max_workers` for parallel processing
3. Lower `ohlcv_retention_days` to reduce data volume

### Slow Scans
**Cause:** Network latency or too many stocks
**Solution:**
1. Increase `max_workers` (up to CPU cores - 1)
2. Use smaller stock universe (NIFTY100 instead of NIFTY500)
3. Check data source latency in Prometheus metrics

## Log Locations
- Main: `logs/daily.log`
- Strategies: `logs/strategies.log`
- Alerts: `logs/alerts.log`
- Data: `logs/data.log`
- Errors: `logs/errors.log`

## Health Check
```bash
# Check all service health
curl http://localhost:9090/metrics | grep health_check
```

## Getting Help
- Check logs first (errors.log)
- Review Prometheus metrics for trends
- Open an issue with logs and config (redact API keys)

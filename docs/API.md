# AlgoTrade Scanner - API Documentation

## Core Classes

### FallbackManager
```python
from src.data_ingestion.fallback_manager import FallbackManager

manager = FallbackManager()
data = await manager.fetch_stock_data("RELIANCE", start_date, end_date)
```

### StrategyLoader
```python
from src.strategies.strategy_loader import StrategyLoader

loader = StrategyLoader()
strategies = loader.load_all()
```

### TradingSignal
```python
from src.strategies.base_strategy import TradingSignal
signal.symbol         # Stock symbol
signal.confidence     # 0.0 to 1.0
signal.entry_price    # Recommended entry
signal.target_price   # Price target
signal.stop_loss      # Stop loss level
signal.to_dict()      # Serialize to dict
```

### StrategyExecutor
```python
from src.engine.strategy_executor import StrategyExecutor

executor = StrategyExecutor()
signals = executor.execute_strategies(symbol, df, company_info, strategies)
```

### TelegramBot
```python
from src.alerts.telegram_bot import TelegramBot

bot = TelegramBot()
await bot.send_alert(message, priority="HIGH")
```

## Configuration API

```python
from src.utils.config_loader import load_config, get_nested

config = load_config("system")
pool_size = get_nested(config, "database.pool_size", default=20)
```

## Prometheus Metrics

Available at `http://localhost:9090/metrics`:
- `algotrade_job_success_total` - Successful job executions
- `algotrade_job_failure_total` - Failed job executions
- `algotrade_data_fetch_success_total` - Data fetch successes
- `algotrade_signal_generated_total` - Generated signals
- `algotrade_alert_sent_total` - Sent alerts
- `algotrade_portfolio_value` - Current portfolio value
- `algotrade_health_check` - Service health status

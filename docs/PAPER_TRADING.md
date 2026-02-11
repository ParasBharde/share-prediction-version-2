# Paper Trading - Complete Setup & Usage Guide

## What is Paper Trading?

Paper trading is **simulated trading** using virtual money. Your bot's algorithm generates BUY/SELL signals, and instead of placing real orders with a broker, the system places **fake orders** with a virtual portfolio of **Rs 10,00,000 (10 Lakh)**. This lets you:

- Test your strategies with **real market signals** without risking real money
- Track **win rate, P&L, Sharpe ratio, max drawdown** automatically
- See exactly how much profit/loss your strategies would have made
- Build confidence before going live

---

## Architecture Overview

```
Daily Scan (scripts/daily_scan.py)
    |
    v
Strategy Engine (Mother Candle, Momentum, Volume, Mean Rev)
    |
    v
Signal Generated (e.g. BUY ROLEXRINGS @ 131.57, confidence 100%)
    |
    +---> Telegram Alert (with Trading Time info)
    |
    +---> Paper Trading Engine (auto-places order)
              |
              +---> Order Simulator (applies slippage + commission)
              +---> Portfolio Manager (tracks positions + cash)
              +---> Performance Tracker (Sharpe, drawdown, win rate)
              |
              v
          Paper Trade Summary Alert (sent via Telegram)
```

---

## Files Involved

| File | Purpose |
|------|---------|
| `config/paper_trading.yaml` | All paper trading settings (capital, position sizing, trading times) |
| `src/paper_trading/paper_trading_engine.py` | Main engine - receives signals, calculates size, places orders |
| `src/paper_trading/order_simulator.py` | Simulates realistic order execution (slippage, commission, rejections) |
| `src/paper_trading/portfolio_manager.py` | Tracks open/closed positions, cash balance, sector allocations |
| `src/paper_trading/pnl_calculator.py` | Computes P&L at position and portfolio level |
| `src/paper_trading/performance_tracker.py` | Tracks Sharpe ratio, Sortino ratio, max drawdown, win rate |
| `config/alert_templates.yaml` | Telegram alert templates including Trading Time section |
| `src/alerts/alert_formatter.py` | Formats alerts with trading time and paper trade summary |

---

## Step-by-Step Setup

### Step 1: Prerequisites

Make sure your base system is running:

```bash
# Your .env file must have these set:
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://algotrade:algotrade@localhost:5432/algotrade
```

### Step 2: Paper Trading Config

The config file is at `config/paper_trading.yaml`. Default settings:

```yaml
enabled: true                              # Toggle paper trading on/off

portfolio:
  initial_capital: 1000000                 # Rs 10 Lakh virtual money
  max_positions: 10                        # Max 10 stocks at once
  max_per_sector: 2                        # Max 2 stocks from same sector

position_sizing:
  method: "risk_based"                     # Size based on risk per trade
  risk_per_trade_percent: 1.0              # Risk 1% of capital per trade (Rs 10,000)
  max_position_percent: 20.0               # Max 20% of capital in one stock (Rs 2 Lakh)
  min_position_value: 10000                # Minimum Rs 10,000 per position

execution:
  auto_place_orders: true                  # Auto-place orders from signals
  min_confidence_percent: 70.0             # Only trade signals with >= 70% confidence
  order_type: "MARKET"                     # MARKET or LIMIT orders
```

### Step 3: Customize Settings (Optional)

**Change starting capital:**
```yaml
portfolio:
  initial_capital: 500000   # Rs 5 Lakh
```

**Change risk per trade:**
```yaml
position_sizing:
  risk_per_trade_percent: 2.0   # More aggressive: 2% risk per trade
```

**Only auto-trade 100% confidence signals:**
```yaml
execution:
  min_confidence_percent: 100.0
```

**Disable auto-trading (alerts only, no paper trades):**
```yaml
execution:
  auto_place_orders: false
```

### Step 4: Run the Scanner

```bash
# Normal run (only on trading days)
python scripts/daily_scan.py

# Force run (weekends/holidays too, for testing)
python scripts/daily_scan.py --force

# Or via environment variable
FORCE_RUN=true python scripts/daily_scan.py

# Run only Mother Candle strategy
python scripts/daily_scan.py --force --strategies "Mother Candle"
```

### Step 5: What Happens When You Run

1. **Scan starts** - Fetches 2000+ stocks from NSE
2. **Strategies run** - Mother Candle (and any enabled strategies) scan each stock
3. **Signals generated** - e.g. BUY ROLEXRINGS @ 131.57 with 100% confidence
4. **Trading time calculated** - Entry date, window, validity based on timeframe
5. **Telegram alert sent** - With full details including TRADING TIME section
6. **Paper trade placed** - If confidence >= 70%, order auto-placed with position sizing
7. **Paper trade summary sent** - Shows all orders placed and portfolio state
8. **Daily summary sent** - Overall scan results + paper trading P&L

---

## How Position Sizing Works

The engine uses **risk-based position sizing**:

```
Risk Amount = Capital x Risk Per Trade %
            = 10,00,000 x 1% = Rs 10,000

Per Share Risk = Entry Price - Stop Loss
              = 131.57 - 124.31 = Rs 7.26

Quantity = Risk Amount / Per Share Risk
         = 10,000 / 7.26 = 1,377 shares

Position Value = 1,377 x 131.57 = Rs 1,81,153 (18.1% of capital)
```

**Caps applied:**
- Max position value = 20% of capital = Rs 2,00,000
- Min position value = Rs 10,000
- Max concurrent positions = 10

---

## How Order Simulation Works

The `OrderSimulator` makes paper trading realistic:

| Factor | Default | What It Does |
|--------|---------|-------------|
| **Slippage** | 0.1% (10 bps) | BUY price slightly higher, SELL price slightly lower |
| **Commission** | 0.03% (3 bps) | Broker commission deducted per trade |
| **Random Rejection** | 0.5% chance | Simulates exchange-level order failures |
| **Max Quantity** | 50,000 shares | Orders above this are rejected |

**Example:**
```
Signal: BUY ROLEXRINGS @ 131.57
Slippage: +0.05% = 131.64 (you pay slightly more)
Commission: 0.03% of (131.64 x 1377) = Rs 54.35
Total Cost: Rs 1,81,249 + Rs 54.35 = Rs 1,81,303
```

---

## Trading Time Rules

Every alert now shows **when and how to trade**. Rules are based on the strategy's timeframe:

### Daily (1D) - Default for all current strategies:
```
Entry Date:     Next trading day (e.g. 11 Feb 2026, Wednesday)
Entry Window:   09:15 - 10:30 IST (first 75 minutes of market)
Signal Valid:   2 trading days
Holding Period: 1-30 trading days
```

### 4-Hour (4H):
```
Entry Date:     Next trading day
Entry Window:   09:15 - 15:30 IST (any time during market)
Signal Valid:   1 trading day
Holding Period: 1-15 trading days
```

### 1-Hour (1H):
```
Entry Window:   09:15 - 15:30 IST
Signal Valid:   1 trading day
Holding Period: 0-5 trading days
```

### 30-Minute (30m):
```
Entry Window:   09:15 - 15:30 IST (immediate entry)
Signal Valid:   Intraday only
Holding Period: 0-2 trading days
```

### 15-Minute (15m):
```
Entry Window:   09:15 - 15:00 IST (immediate entry)
Signal Valid:   Intraday only
Holding Period: 0-1 trading day (intraday)
```

To change trading time rules, edit `config/paper_trading.yaml` under `trading_time:`.

---

## Alert Format (What You'll See on Telegram)

### Buy Signal Alert:
```
Paras Trading Bot:
Mother Candle
ROLEXRINGS

Signal: BUY
Price: Rs 131.57

---- INDICATOR CHECKS ----
  [PASS] Mother Candle (Baby Count: 3)
  [PASS] Volume (Value: 2.27)
  [PASS] Trend ()
  [PASS] Rsi (Value: 63.43)
  [PASS] Adx (Value: 34.57)

---- TRADE SETUP ----
Entry: Rs 131.57
Stop Loss: Rs 124.31 (-5.5%)
Target: Rs 146.09 (+11.0%)
Risk:Reward = 1:2.0
Risk: Rs 7.26
Reward: Rs 14.52

---- TRADING TIME ----
Timeframe: 1D
Entry Date: 11 Feb 2026 (Wednesday)
Entry Window: 09:15 - 10:30 IST
Signal Valid For: 2 trading day(s)
Valid Until: 12 Feb 2026 10:30 IST
Holding Period: 1-30 trading days
Note: Daily timeframe - Enter at market open, valid for 2 trading days

---- CONFIDENCE: 100.0% ----
Indicators: 5/5

01:29 AM IST | 11 Feb 2026
```

### Paper Trade Summary Alert:
```
PAPER TRADING - Session Summary

Portfolio Value: Rs 10,00,000.00
Cash Balance: Rs 8,18,697.00
Open Positions: 3
Total Return: 0.00%

---- TRADES PLACED (3) ----
1. ROLEXRINGS | BUY | Qty: 1377
   Entry: Rs 131.57 | SL: Rs 124.31 | Target: Rs 146.09
   Strategy: Mother Candle | Status: PLACED

2. GRINDWELL | BUY | Qty: 101
   Entry: Rs 1,698.20 | SL: Rs 1,600.00 | Target: Rs 1,894.60
   Strategy: Mother Candle | Status: PLACED

3. CAMPUS | BUY | Qty: 578
   Entry: Rs 282.50 | SL: Rs 265.20 | Target: Rs 317.10
   Strategy: Mother Candle | Status: PLACED
```

---

## Portfolio Tracking

### What Is Tracked Automatically:

| Metric | Description |
|--------|-------------|
| **Cash Balance** | Remaining virtual cash after trades |
| **Portfolio Value** | Cash + market value of all open positions |
| **Open Positions** | Currently held stocks with unrealized P&L |
| **Closed Positions** | Completed trades with realized P&L |
| **Total Return %** | Overall portfolio performance |
| **Realized P&L** | Profit/loss from closed trades |
| **Unrealized P&L** | Paper profit/loss from open positions |
| **Win Rate** | % of profitable closed trades |
| **Sharpe Ratio** | Risk-adjusted return (higher = better) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Profit Factor** | Gross profits / Gross losses |

### Viewing Portfolio State

The portfolio state is included in the **Daily Summary** alert automatically. It shows:
- Active positions count
- Total P&L percentage
- Paper trades placed in the session

---

## Configuration Reference

### config/paper_trading.yaml - Full Reference

```yaml
# Master switch
enabled: true                        # true = paper trading active, false = disabled

# Portfolio settings
portfolio:
  initial_capital: 1000000           # Starting virtual money (INR)
  max_positions: 10                  # Max concurrent open positions
  max_per_sector: 2                  # Max stocks from same sector

# How to calculate position size
position_sizing:
  method: "risk_based"               # "risk_based" or "fixed"
  risk_per_trade_percent: 1.0        # % of capital to risk per trade
  max_position_percent: 20.0         # Max % of capital in single stock
  min_position_value: 10000          # Min position value (INR)

# Order simulation costs
order_simulator:
  slippage_pct: 0.001                # 0.1% slippage
  commission_pct: 0.0003             # 0.03% commission
  max_quantity: 50000                # Max shares per order
  rejection_symbols: []              # Symbols to always reject

# Auto-execution rules
execution:
  auto_place_orders: true            # Auto-place orders from signals
  min_confidence_percent: 70.0       # Min confidence to auto-trade
  order_type: "MARKET"               # "MARKET" or "LIMIT"
  limit_offset_percent: 0.05         # For LIMIT orders only

# Exit management
exit_management:
  auto_exit_on_target: true          # Auto-close when target hit
  auto_exit_on_stop_loss: true       # Auto-close when SL hit
  trailing_stop_enabled: false       # Enable trailing stop loss
  trailing_stop_atr_multiplier: 2.0  # ATR multiplier for trailing SL

# Performance calculation
performance:
  risk_free_rate: 0.065              # 6.5% for Sharpe/Sortino
  trading_days_per_year: 252         # For annualization

# Trading time rules per timeframe
trading_time:
  "1D":
    entry_session: "next_market_open"
    entry_window_start: "09:15"
    entry_window_end: "10:30"
    signal_validity_days: 2
    holding_period_min_days: 1
    holding_period_max_days: 30
    description: "Daily timeframe - Enter at market open, valid for 2 trading days"
  # ... (4H, 1H, 30m, 15m also available)
```

---

## Troubleshooting

### Paper trades not being placed?

1. Check `config/paper_trading.yaml` has `enabled: true` and `execution.auto_place_orders: true`
2. Check signal confidence is >= `min_confidence_percent` (default 70%)
3. Check max positions not reached (default 10)
4. Check logs: `grep "Paper trade" logs/algotrade.log`

### No Trading Time section in alerts?

1. Make sure your strategy config has `data.timeframe` set (e.g. `"1D"`)
2. Check `config/paper_trading.yaml` has `trading_time` section
3. The paper trading engine must be initialized (check logs for "Paper trading engine initialized")

### Position size is 0?

1. Stop loss too close to entry (per-share risk too small, quantity would be huge)
2. Position value below `min_position_value` (Rs 10,000)
3. Not enough cash (check `max_positions` and `max_position_percent`)

### Alerts not sending?

1. Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in your `.env`
2. Run `python -c "from src.alerts.telegram_bot import TelegramBot; print('OK')"` to verify
3. Check Redis is running: `redis-cli ping` should return `PONG`

---

## Current Limitations

1. **In-memory portfolio** - Portfolio state resets when the scanner restarts. For persistence, you would need database integration (tables already exist in `src/storage/models.py`)
2. **No real-time exit monitoring** - The system scans once daily; it doesn't monitor positions intraday for SL/target hits
3. **No broker integration** - This is pure simulation. To go live, you'd need a broker API (Zerodha Kite, Angel One, etc.)

---

## Extending to Live Trading (Future)

When you're ready to move from paper to real trading:

1. Replace `OrderSimulator` with a real broker API wrapper (e.g. Zerodha Kite Connect)
2. Add `BROKER_API_KEY` and `BROKER_API_SECRET` to `.env`
3. Keep the same `PaperTradingEngine` interface - just swap the order execution backend
4. Start with small capital and the same risk management rules

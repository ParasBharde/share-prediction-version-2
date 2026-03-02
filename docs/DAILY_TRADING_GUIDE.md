# Daily Trading Guide — What to Run, When, and Why

> **Run all commands from the project root:**
> ```
> cd F:\Projects\share-prediction-version-2
> ```

---

## Quick Reference — Daily Schedule

| Time (IST) | Command | Purpose |
|---|---|---|
| 9:15 AM | Start options scanner | Live F&O signals |
| 3:15 PM | BTST scanner | Buy-today-sell-tomorrow signals |
| 3:35 PM+ | Daily scanner | End-of-day stock signals |
| Any time | Portfolio tracker | Check open positions |

---

## 1. OPTIONS SCANNER — Run during market hours (9:15 AM – 3:30 PM)

### What it does
Scans NIFTY and SENSEX option chains every N minutes. Reads live OI, PCR,
premiums from SmartAPI (Angel One broker). Applies 3 strategies:
- **OI Breakout** — large OI buildup at a strike → directional move expected
- **PCR Sentiment** — PCR < 0.90 = bearish (buy PE), PCR > 1.10 = bullish (buy CE)
- **VWAP + Supertrend** — trend + VWAP bias alignment for entry

For each signal it shows: which strike to buy, premium, lot size, entry zone,
Target 1, Target 2, stop loss, and a full checklist of reasons to trade or wait.

### Commands

```powershell
# NIFTY only — repeat every 5 min, send Telegram alerts (RECOMMENDED)
python scripts/options_scan.py --symbol NIFTY --repeat 5 --telegram

# Both NIFTY and SENSEX
python scripts/options_scan.py --symbol NIFTY --symbol SENSEX --repeat 5 --telegram

# Test without sending Telegram (preview mode)
python scripts/options_scan.py --symbol NIFTY --dry-run

# Single scan (no repeat)
python scripts/options_scan.py --symbol NIFTY
```

### When to run
- Start: **9:15 AM IST** (market opens)
- Stop: **3:20 PM IST** (no new entries after 3 PM, scanner enforces this)
- Press **Ctrl+C** to stop

### How signals work
```
SCORE ≥ 65 + Confidence ≥ 65%  →  🟢 BUY  — Telegram alert sent
SCORE < 65 or Conf < 65%        →  ⚠️  WAIT — shown in console only
```
Each signal includes:
- Strike to buy (e.g., NIFTY 24700 PE)
- Current premium + entry zone (±3%)
- Target 1 (exit half lots), Target 2 (exit rest)
- Stop loss (−50% of premium paid)
- MTF analysis (15m + 5m trend)
- PCR, OI levels, expiry day warning

### Notes
- BANKNIFTY: skipped automatically if no near-term weekly options found in broker
- SENSEX: uses BSE monthly expiry (up to 35 DTE allowed)
- Signals have 30-min cooldown — same strike won't spam Telegram
- Expiry day: −10% confidence penalty applied, still runs (uses next expiry data)

---

## 2. BTST SCANNER — Run near market close (3:15 PM)

### What it does
Scans 900+ stocks (NIFTY 500 + Midcap + Smallcap) for breakout patterns.
Buy today at market close, sell tomorrow at open or during the day.

Strategies used:
- **Darvas Box** — price breaks above the top of a Darvas ceiling box
- **Flag Pattern** — strong pole + tight consolidation flag + breakout
- **Symmetrical Triangle** — converging highs/lows + breakout above resistance
- **Descending Channel** — pullback in uptrend + breakout above falling channel

### Command

```powershell
# Standard run with Telegram alerts (RECOMMENDED — run at 3:15 PM)
python scripts/btst_scan.py --telegram --verbose-stats

# Console only — no Telegram (preview mode)
python scripts/btst_scan.py --verbose-stats

# Test without Telegram on non-trading day
python scripts/btst_scan.py --force --dry-run

# Run specific strategies only
python scripts/btst_scan.py --strategies "Darvas Box" "Flag Pattern" --telegram
```

### When to run
- **3:15 PM IST** — gives 15 minutes to review signals and place orders before 3:30 PM close
- Run once per day, not repeated

### What the output shows
```
BTST SCAN RESULTS — 2026-03-02 15:18 IST
Stocks scanned : 934
Signals found  : 3

 1. 📦 RELIANCE     [Darvas Box]       Entry: Rs.1,280.00  Target: Rs.1,344.00  SL: Rs.1,228.80  Conf: 78%  R:R 1:2.0
 2. 🚩 ICICIBANK    [Flag Pattern]     Entry: Rs.1,150.00  Target: Rs.1,207.50  SL: Rs.1,109.50  Conf: 72%  R:R 1:1.5
 3. △  HDFCBANK     [Sym. Triangle]    Entry: Rs.1,730.00  Target: Rs.1,816.50  SL: Rs.1,665.00  Conf: 70%  R:R 1:1.8
```

### Why 3:15 PM and not after close
BTST orders must be placed BEFORE 3:30 PM so the buy executes at today's close.
After 3:30 PM the data is complete but it's too late to place the order for today.

---

## 3. DAILY SCANNER — Run after market close (3:35 PM or evening)

### What it does
Runs on the complete 1-day closing bar for all enabled strategies.
Uses full confirmed prices (not partial intraday). Scans 900+ stocks.

Active strategies (1D chart):
- **Mother Candle V2** — large range "mother" candle + consolidation babies + breakout
- **Darvas Box** — same as BTST but on confirmed daily close
- **Flag Pattern** — same as BTST but on confirmed daily close
- **Symmetrical Triangle** — same as BTST but on confirmed daily close
- **Descending Channel** — same as BTST but on confirmed daily close

### Commands

```powershell
# Full scan — all enabled strategies (RECOMMENDED — run after 3:35 PM)
python scripts/daily_scan.py --force --telegram

# Mother Candle V2 only
python scripts/daily_scan.py --force --strategies "Mother Candle V2" --telegram

# All BTST pattern strategies on confirmed close
python scripts/daily_scan.py --force --btst-only --telegram

# Single stock test
python scripts/daily_scan.py --force --test-symbol ICICIBANK

# Skip Telegram (log only)
python scripts/daily_scan.py --force --dry-run
```

### When to run
- **After 3:35 PM IST** — market is fully closed, all daily bars complete
- Can also run in the **evening (6–9 PM)** — same results, just more leisurely
- Do NOT run before 3:30 PM (today's bar will be incomplete → possible false signals)

### Mother Candle V2 filters (7 checks must all pass)
```
1. Trend      — Price above EMA50 + EMA50 not declining
2. ATR        — Volatility ≥ 1% ATR (skips low-vol stocks)
3. Pattern    — Mother candle range ≥ 1.5x previous 5-candle avg
4. Age        — Mother candle ≤ 10 bars old (stale patterns rejected)
5. Breakout   — Today's close 0.3%–2.0% above Mother High
6. Volume     — Mother volume 1.3x avg AND breakout volume 1.2x avg
7. Strong close — Breakout candle closes in top 25% of its range
```

---

## 4. PORTFOLIO TRACKER — Any time

### What it does
Tracks open positions entered from scanner signals. Shows P&L, targets hit,
stop losses triggered.

### Commands

```powershell
# View open positions
python scripts/portfolio_tracker.py

# Update prices (fetch latest)
python scripts/portfolio_tracker.py --update

# Update + send Telegram summary
python scripts/portfolio_tracker.py --update --telegram

# View closed/completed trades
python scripts/portfolio_tracker.py --closed

# Full P&L report
python scripts/portfolio_tracker.py --report
```

---

## 5. LIVE INTRADAY SCANNER — During market hours

### What it does
Scans stocks on 5m or 15m candles for intraday momentum, volume surge,
and mean reversion setups.

### Commands

```powershell
# Single stock on 15m
python scripts/live_scan.py --symbol RELIANCE --interval 15m --bypass-time

# Watch a list of stocks
python scripts/live_scan.py --watchlist RELIANCE,INFY,TCS --interval 5m --bypass-time

# Full NIFTY 50 scan every 5 min with Telegram
python scripts/live_scan.py --universe NIFTY50 --repeat 5 --telegram --bypass-time
```

---

## Full Day Workflow (copy-paste ready)

### Morning (9:00 AM) — Start options scanner

```powershell
python scripts/options_scan.py --symbol NIFTY --symbol SENSEX --repeat 5 --telegram
```
Leave this running in a terminal. Press Ctrl+C at 3:20 PM.

---

### Afternoon (3:15 PM) — BTST scan (new terminal)

```powershell
python scripts/btst_scan.py --telegram --verbose-stats
```
Review signals. Place buy orders before 3:30 PM.

---

### Evening (3:35 PM or later) — Daily scan

```powershell
python scripts/daily_scan.py --force --telegram
```
Signals for swing/positional trades to enter next day at open.

---

### Any time — Check portfolio

```powershell
python scripts/portfolio_tracker.py --update --telegram
```

---

## Environment Variables Required

Set these in your `.env` file (already configured):

```
TELEGRAM_BOT_TOKEN=...      # Bot token from @BotFather
TELEGRAM_CHAT_ID=...        # Your channel/group ID

SMARTAPI_API_KEY=...         # Angel One API key (for options chain data)
SMARTAPI_CLIENT_CODE=...     # Angel One client ID
SMARTAPI_PIN=...             # Angel One MPIN
SMARTAPI_TOTP_SECRET=...     # TOTP secret for 2FA login
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `nsepython returned {}` | Normal — NSE blocks direct requests. SmartAPI fallback takes over automatically |
| `BANKNIFTY skipped (85 days)` | Normal — broker scrip master lacks near-term BANKNIFTY weekly options. NIFTY works fine |
| `SENSEX skipped` | Increase `--repeat` interval; SENSEX uses monthly options (31-day expiry) |
| `Telegram not sending` | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| `Strategy disabled` | Edit `config/strategies/<name>.yaml` and set `enabled: true` |
| Chart error (kaleido) | Charts fall back to matplotlib automatically — still sent to Telegram |

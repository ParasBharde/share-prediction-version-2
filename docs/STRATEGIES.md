# AlgoTrade Scanner — Strategy Guide

## Table of Contents

1. [Running the Scanner](#running-the-scanner)
2. [BTST Strategies (New)](#btst-strategies)
   - [Darvas Box](#1-darvas-box)
   - [Flag Pattern](#2-flag-pattern)
   - [Symmetrical Triangle](#3-symmetrical-triangle)
   - [Descending Channel](#4-descending-channel)
3. [Classic Daily Strategies](#classic-daily-strategies)
   - [Mother Candle V2](#5-mother-candle-v2)
   - [Momentum Breakout](#6-momentum-breakout)
   - [Mean Reversion](#7-mean-reversion)
   - [Volume Surge](#8-volume-surge)
4. [Intraday Strategies](#intraday-strategies)
5. [Options Strategies](#options-strategies)
6. [Multi-Strategy Convergence](#multi-strategy-convergence)
7. [Creating Custom Strategies](#creating-custom-strategies)

---

## Running the Scanner

### Full daily scan (all strategies)
```bash
python scripts/daily_scan.py
```

### BTST-only scan (dedicated script — recommended at 15:15 IST)
```bash
python scripts/btst_scan.py
```

### BTST scan options
```bash
# Force-run on non-trading days (dev / back-test)
python scripts/btst_scan.py --force

# Dry-run: log signals without sending Telegram alerts
python scripts/btst_scan.py --dry-run

# Run only specific BTST strategies
python scripts/btst_scan.py --strategies "Darvas Box" "Flag Pattern"

# Save charts to a custom directory (default: /tmp)
python scripts/btst_scan.py --chart-dir ./charts/btst

# Show per-strategy rejection breakdown after the scan
python scripts/btst_scan.py --verbose-stats

# Combine options
python scripts/btst_scan.py --force --dry-run --chart-dir ./charts/btst --verbose-stats
```

### Daily-scan CLI flags
```bash
# Run only BTST strategies via daily_scan.py
python scripts/daily_scan.py --btst-only

# Run only Mother Candle V2
python scripts/daily_scan.py --mother-v2-only

# Run specific strategies by name
python scripts/daily_scan.py --strategies "Darvas Box" "Symmetrical Triangle"

# Test a single symbol (debug mode)
python scripts/daily_scan.py --test-symbol ICICIBANK

# Force run on non-trading days
python scripts/daily_scan.py --force

# Combine flags
python scripts/daily_scan.py --btst-only --force
```

### Environment variables
```bash
export TELEGRAM_BOT_TOKEN="<your-token>"
export TELEGRAM_CHAT_ID="<your-chat-id>"
FORCE_RUN=true python scripts/btst_scan.py
```

---

## BTST Strategies

> **BTST Timing**: All four BTST strategies are designed to be run at **15:15 IST**.
> With ~2700 stocks processed in chunks of 50, signals are ready by ~15:25 IST —
> a 5-minute window to place Buy Today, Sell Tomorrow orders before the 15:30 close.
>
> Scheduler is configured in `config/system.yaml` → `scanning.schedule.time: "15:15"`.

All BTST strategies share these common properties:
- **Trend filter**: Price must be above the 50-period EMA (configurable via `require_uptrend`)
- **Volume confirmation**: Breakout bar must exceed the volume multiplier threshold
- **Stop-loss**: Uses the pattern's natural structure (box bottom, flag low, support line) or a fixed % — whichever is tighter within `max_stop_loss_pct`
- **Chart overlay**: Each strategy generates a 60-day Plotly candlestick PNG with annotated entry, target, SL, and pattern-specific overlays

---

### 1. Darvas Box

**File**: `src/strategies/darvas_box.py`
**Config**: `config/strategies/darvas_box.yaml`
**Weight**: 0.38 (in signal aggregator)

**Theory**: Nicolas Darvas's "Box Theory" identifies stocks undergoing institutional accumulation. Price forms a ceiling (resistance) that holds for 3+ days, creating a well-defined box. A breakout above the ceiling on elevated volume signals resumption of the prior uptrend.

**Algorithm**:
1. Scan backwards from the second-to-last candle for a ceiling (highest recent high)
2. Verify that no subsequent candle (up to the breakout bar) exceeded the ceiling — confirming the box held
3. Check that the last close breaks above the ceiling
4. Volume on breakout bar must be ≥ 1.5× the 20-day average
5. Box depth (ceiling − box bottom) must be within `max_box_depth_pct` (default 12%)
6. Stop-loss at box bottom (or fixed 3%); target at +7% from entry

**Key parameters** (editable in YAML):
| Parameter | Default | Description |
|---|---|---|
| `box_hold_days` | 3 | Min days ceiling must hold |
| `max_lookback` | 20 | Max candles to search back for ceiling |
| `breakout_vol_multiplier` | 1.5 | Required volume vs 20-day avg |
| `max_box_depth_pct` | 12.0 | Max box height as % of ceiling |
| `target_pct` | 7.0 | Fixed target % from entry |
| `use_box_bottom_sl` | true | Use box bottom as SL |

**Chart overlay**: Horizontal resistance (box top) and support (box bottom) lines spanning the box period.

---

### 2. Flag Pattern

**File**: `src/strategies/flag_pattern.py`
**Config**: `config/strategies/flag_pattern.yaml`
**Weight**: 0.42 (highest BTST weight)

**Theory**: Bull Flag — one of the highest-probability continuation patterns. A sharp vertical "pole" (5%+ gain in 3–10 days) is followed by a tight, low-volatility "flag" consolidation (3–10 days). The measured-move target is `flag_high + pole_height`, giving a clear and objective profit objective.

**Algorithm**:
1. Trend filter: close > 50 EMA
2. Search for a tight flag (candles with range < `max_flag_range_pct` = 8%) ending at the second-to-last bar
3. Verify a qualifying pole immediately before the flag: gain ≥ `min_pole_gain_pct` (5%)
4. Flag retracement ≤ 50% of pole height
5. Last close must be above flag high (breakout confirmed)
6. Breakout volume ≥ 1.3× 20-day average
7. Stop-loss: flag low (or fixed 3%); target: `flag_high + pole_height` (measured move)

**Key parameters**:
| Parameter | Default | Description |
|---|---|---|
| `min_pole_gain_pct` | 5.0 | Min pole gain in % |
| `min_pole_days` / `max_pole_days` | 3 / 10 | Pole length range |
| `min_flag_days` / `max_flag_days` | 3 / 10 | Flag length range |
| `max_flag_range_pct` | 8.0 | Max flag height as % (tightness) |
| `max_flag_retrace_pct` | 50.0 | Max flag retrace as % of pole |
| `use_measured_move_target` | true | Use pole-height target |

**Chart overlay**: Pole shaded area, flag consolidation box (flag high / flag low), entry arrow.

---

### 3. Symmetrical Triangle

**File**: `src/strategies/symmetrical_triangle.py`
**Config**: `config/strategies/symmetrical_triangle.yaml`
**Weight**: 0.35

**Theory**: Symmetrical triangles represent a period of indecision ("coiling") where both buyers and sellers become less aggressive. The converging trendlines create a defined breakout point. Upside breakout with volume confirms a bullish resolution.

**Algorithm**:
1. Trend filter: close > 50 EMA
2. Find pivot highs (descending) and pivot lows (ascending) in the last `pattern_lookback` (50) candles using a `pivot_order`-candle window
3. Fit linear regression through ≥ 2 pivot highs (slope must be < −0.05) and ≥ 2 pivot lows (slope must be > +0.05)
4. Both regression fits must have R² ≥ 0.70 (quality gate)
5. Last close > projected resistance line value at last bar (breakout)
6. Breakout volume ≥ 1.3× 20-day average
7. Stop-loss: support line value at last bar (or fixed 3%); target: +8% from entry

**Key parameters**:
| Parameter | Default | Description |
|---|---|---|
| `pattern_lookback` | 50 | Candles to search for pivots |
| `pivot_order` | 3 | Candles each side for pivot detection |
| `min_r_squared` | 0.70 | Min R² for trendline quality |
| `resistance_max_slope` | −0.05 | Max slope for resistance (must be falling) |
| `support_min_slope` | 0.05 | Min slope for support (must be rising) |

**Chart overlay**: Descending resistance trendline (red), ascending support trendline (teal), pivot markers.

---

### 4. Descending Channel

**File**: `src/strategies/descending_channel.py`
**Config**: `config/strategies/descending_channel.yaml`
**Weight**: 0.33

**Theory**: A descending channel within a broader uptrend represents a healthy, orderly pullback. When price breaks above the upper channel boundary with volume, it signals the resumption of the primary uptrend — a low-risk entry with a well-defined SL at the lower channel line.

**Algorithm**:
1. Trend filter: close > 50 EMA (primary uptrend)
2. Fit pivot highs and pivot lows over the last `channel_lookback` (30) candles
3. Both channel lines must slope between `min_channel_slope` (−0.5) and `max_channel_slope` (−0.05) — confirming a short-term downtrend within the broader uptrend
4. Channel width must be ≥ `min_channel_width_pct` (3%) — avoiding noise
5. R² ≥ 0.65 for both lines
6. Last close > projected upper channel line at last bar (breakout)
7. Breakout volume ≥ 1.2× 20-day average
8. Stop-loss: lower channel line value at last bar (or fixed 3%); target: +6% from entry

**Key parameters**:
| Parameter | Default | Description |
|---|---|---|
| `channel_lookback` | 30 | Candles to fit channel |
| `pivot_order` | 2 | Candles each side for pivot detection |
| `min_r_squared` | 0.65 | Min R² quality gate |
| `min_channel_slope` | −0.5 | Steepest allowed channel slope |
| `max_channel_slope` | −0.05 | Shallowest allowed channel slope |
| `min_channel_width_pct` | 3.0 | Min channel width as % |

**Chart overlay**: Upper channel line (red), lower channel line (teal), pivot markers.

---

## Classic Daily Strategies

### 5. Mother Candle V2

**File**: `src/strategies/mother_candle_v2.py`
**Config**: `config/strategies/mother_candle_v2.yaml`
**Weight**: 0.40

Identifies a large "mother candle" followed by 2–5 smaller "baby candles" that stay within the mother's range (inside-bar compression). Signals BUY when price breaks above the mother candle high on elevated volume.

**Chart overlay**: Mother candle high/low box (yellow), entry arrow.

---

### 6. Momentum Breakout

Identifies stocks breaking 52-week highs with volume confirmation.

**Indicators:**
- 52-Week High Proximity (within 2%)
- Volume Surge (2× average)
- RSI 50–70 range
- EMA Alignment (20 > 50 > 200)
- Relative Strength vs NIFTY

**Signal:** BUY when 4/5 indicators pass with confidence ≥ 0.70

---

### 7. Mean Reversion

Targets oversold quality stocks at Bollinger Band lows.

**Indicators:**
- RSI ≤ 30 (oversold)
- Price at lower Bollinger Band
- Volume confirmation (1.5×)
- Above 200 DMA support
- MACD histogram improving

**Signal:** BUY when 3/5 indicators pass with confidence ≥ 0.65

---

### 8. Volume Surge

Detects unusual institutional volume activity.

**Indicators:**
- Volume ≥ 3× average
- Bullish candle (≥ 2% gain)
- Delivery % ≥ 60%
- Above VWAP
- EMA 20 > EMA 50 (trend)

**Signal:** BUY when 3/5 indicators pass with confidence ≥ 0.65

---

## Intraday Strategies

| Strategy | Description |
|---|---|
| Intraday Momentum | Short-term momentum breakouts on 5/15-min charts |
| Intraday Volume Surge | Intraday unusual volume detection |
| Intraday Mean Reversion | Intraday oversold bounce setups |

---

## Options Strategies

| Strategy | Description |
|---|---|
| Options OI Breakout | Open-interest buildup + price breakout |
| Options VWAP Supertrend | VWAP + Supertrend combined signal |
| Options PCR Sentiment | Put/Call ratio sentiment filter |

---

## Multi-Strategy Convergence

When multiple strategies fire on the same symbol, the signal aggregator combines them into a single `AggregatedSignal` with a weighted confidence score. Convergence of 2+ strategies on the same stock is a high-conviction setup.

**Strategy weights** (higher = more influence):

| Strategy | Weight |
|---|---|
| Flag Pattern | 0.42 |
| Mother Candle V2 | 0.40 |
| Options OI Breakout | 0.40 |
| Darvas Box | 0.38 |
| Momentum Breakout | 0.35 |
| Volume Surge | 0.35 |
| Intraday Momentum | 0.35 |
| Intraday Volume Surge | 0.35 |
| Symmetrical Triangle | 0.35 |
| Options VWAP Supertrend | 0.35 |
| Descending Channel | 0.33 |
| Mean Reversion | 0.30 |
| Intraday Mean Reversion | 0.30 |
| Options PCR Sentiment | 0.30 |

**Multi-strategy agreement bonus**: Up to +15% confidence boost when 2+ strategies agree (5% per additional strategy, capped at 15%).

**Example — high-conviction BTST setup**:
A stock where both Flag Pattern (0.42) and Darvas Box (0.38) fire simultaneously will produce an aggregated confidence score well above either individual signal, making it rise higher in the ranked output.

---

## Creating Custom Strategies

1. Copy `config/strategies/flag_pattern.yaml` as a starting template
2. Create `src/strategies/your_strategy.py` inheriting from `BaseStrategy`
3. Implement `scan(symbol, df, company_info) -> Optional[TradingSignal]`
4. Register in `src/strategies/strategy_loader.py` → `STRATEGY_REGISTRY`
5. Add a weight to `DEFAULT_STRATEGY_WEIGHTS` in `src/engine/signal_aggregator.py`

### Strategy Config Structure
```yaml
strategy:
  name: "Your Strategy"
  enabled: true
  priority: 5
  mode: "daily"          # daily | intraday | options

filters:
  market_cap_min: 1000
  min_price: 50
  max_price: 5000

indicators:
  - name: "Indicator Name"
    type: "price_comparison"
    params:
      condition: "close > ema_50"
    weight: 0.50
  - name: "Volume Check"
    type: "volume_analysis"
    params:
      multiplier: 1.5
    weight: 0.50             # weights across all indicators must sum to 1.0

signal_generation:
  min_conditions_met: 1
  confidence_threshold: 0.65

risk_management:
  stop_loss:
    type: "percentage"
    percent: 3.0
  target:
    type: "percentage"
    percent: 7.0
  max_stop_loss_pct: 5.0
```

### Available Indicator Types
- `price_comparison` — Compare price levels (close vs EMA, breakout levels)
- `volume_analysis` — Volume-based conditions (surge multipliers, delivery %)
- `oscillator` — RSI, MACD, Stochastic
- `trend` — Moving average alignment
- `volatility` — Bollinger Bands, ATR
- `comparative` — Relative strength vs benchmark

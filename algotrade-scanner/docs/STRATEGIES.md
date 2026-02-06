# AlgoTrade Scanner - Strategy Guide

## Built-in Strategies

### 1. Momentum Breakout
Identifies stocks breaking 52-week highs with volume confirmation.

**Indicators:**
- 52-Week High Proximity (within 2%)
- Volume Surge (2x average)
- RSI 50-70 range
- EMA Alignment (20 > 50 > 200)
- Relative Strength vs NIFTY

**Signal:** BUY when 4/5 indicators pass with confidence >= 0.70

### 2. Mean Reversion
Targets oversold quality stocks at Bollinger Band lows.

**Indicators:**
- RSI <= 30 (oversold)
- Price at lower Bollinger Band
- Volume confirmation (1.5x)
- Above 200 DMA support
- MACD histogram improving

**Signal:** BUY when 3/5 indicators pass with confidence >= 0.65

### 3. Volume Surge
Detects unusual institutional volume activity.

**Indicators:**
- Volume >= 3x average
- Bullish candle (>= 2% gain)
- Delivery % >= 60%
- Above VWAP
- EMA 20 > EMA 50 (trend)

**Signal:** BUY when 3/5 indicators pass with confidence >= 0.65

## Creating Custom Strategies

1. Copy `config/strategies/custom_strategy_template.yaml`
2. Customize indicators, filters, and risk management
3. Run `python scripts/deploy_strategy.py --config your_strategy.yaml`

### Strategy Config Structure
```yaml
strategy:
  name: "Your Strategy"
  enabled: true
  priority: 5

filters:
  market_cap_min: 1000

indicators:
  - name: "Indicator Name"
    type: "oscillator"
    params:
      condition: "rsi_14 <= 30"
    weight: 0.50

signal_generation:
  min_conditions_met: 2
  confidence_threshold: 0.60

risk_management:
  stop_loss:
    type: "percentage"
    percent: 5
  target:
    type: "risk_reward"
    ratio: 2.0
```

### Available Indicator Types
- `price_comparison`: Compare price levels
- `volume_analysis`: Volume-based conditions
- `oscillator`: RSI, MACD, Stochastic
- `trend`: Moving average alignment
- `volatility`: Bollinger Bands, ATR
- `comparative`: Relative strength vs benchmark

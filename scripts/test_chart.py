"""
Quick chart-generation smoke test.

Run from the project root:
    python scripts/test_chart.py

Prints PASS/FAIL and the exact error if it fails.
Does NOT require a database or Redis connection.
"""

import os
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

print("=" * 60)
print("Chart generation smoke test")
print("=" * 60)

# ── 1. Check plotly version ────────────────────────────────────
try:
    import plotly
    print(f"plotly    : {plotly.__version__}")
    major = int(plotly.__version__.split(".")[0])
    if major >= 6:
        print("FAIL: plotly >= 6.x detected — incompatible with kaleido 0.2.x")
        print("      Fix: pip install \"plotly>=5.17.0,<6.0.0\"")
        sys.exit(1)
except ImportError:
    print("FAIL: plotly not installed")
    sys.exit(1)

try:
    import kaleido
    print(f"kaleido   : {kaleido.__version__}")
except ImportError:
    print("FAIL: kaleido not installed")
    sys.exit(1)

# ── 2. Build a minimal synthetic OHLCV DataFrame ──────────────
print("\nBuilding synthetic OHLCV data ...")
n = 80
dates = [datetime(2025, 1, 2) + timedelta(days=i) for i in range(n)]
close = 1000 + np.cumsum(np.random.randn(n) * 5)
high  = close + abs(np.random.randn(n) * 3)
low   = close - abs(np.random.randn(n) * 3)
open_ = close + np.random.randn(n) * 2
vol   = np.random.randint(100_000, 2_000_000, n)

df = pd.DataFrame({
    "open":   open_,
    "high":   high,
    "low":    low,
    "close":  close,
    "volume": vol,
}, index=pd.to_datetime(dates))

# ── 3. Build a minimal TradingSignal ──────────────────────────
from src.utils.constants import AlertPriority, SignalType
from src.strategies.base_strategy import TradingSignal

signal = TradingSignal(
    symbol="TESTSTOCK",
    company_name="Test Company Ltd",
    strategy_name="Mother Candle V2",
    signal_type=SignalType.BUY,
    confidence=0.85,
    entry_price=float(close[-1]),
    target_price=float(close[-1] * 1.06),
    stop_loss=float(close[-1] * 0.97),
    priority=AlertPriority.HIGH,
    indicators_met=5,
    total_indicators=6,
    metadata={"atr_pct": 1.5},
)

# ── 4. Call save_signal_chart() directly ──────────────────────
from src.utils.visualizer import ChartVisualizer

out_path = os.path.join(tempfile.gettempdir(), "test_chart_smoke.png")
visualizer = ChartVisualizer()
print(f"\nOutput path : {out_path}")
print(f"Calling save_signal_chart() (timeout = {visualizer._KALEIDO_TIMEOUT}s) ...")
try:
    ok = visualizer.save_signal_chart(df, signal, out_path)
    if ok and os.path.isfile(out_path):
        size_kb = os.path.getsize(out_path) / 1024
        print(f"\nPASS: Chart written ({size_kb:.1f} KB)")
        print(f"      Open to verify: {out_path}")
    else:
        print("\nFAIL: save_signal_chart() returned False (check ERROR logs above)")
except Exception as exc:
    print(f"\nFAIL: Exception during chart generation:")
    traceback.print_exc()

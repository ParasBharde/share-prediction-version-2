"""
Market Context Checker

Purpose:
    Fetches and evaluates broad market conditions before running stock scans.
    Prevents generating bullish signals in bearish market regimes.

Checks:
    1. Nifty 50 trend — is price above/below its 20D EMA?
    2. India VIX level — reduce confidence / skip in fear regimes

Usage:
    context = await get_market_context()
    if context["regime"] == "BEARISH":
        # Only run SELL/short strategies, or skip BUY signals entirely
"""

import asyncio
from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


# VIX thresholds
VIX_NORMAL_MAX    = 15.0   # Below this: low fear, good for momentum
VIX_ELEVATED_MAX  = 20.0   # 15–20: moderate fear, reduce size
VIX_HIGH_MAX      = 30.0   # 20–30: high fear, be cautious
# Above 30: extreme fear, avoid new longs


async def get_market_context(fallback_manager=None) -> Dict[str, Any]:
    """
    Fetch Nifty 50 data and compute market regime.

    Market regime classification:
        BULLISH   – Nifty > 20D EMA and VIX < 20
        NEUTRAL   – Nifty near EMA (±0.5%) or VIX 20-30
        BEARISH   – Nifty < 20D EMA or VIX > 30

    Args:
        fallback_manager: Optional FallbackManager instance.
            If None, a basic yfinance fetch is attempted.

    Returns:
        Dict with:
            regime       : "BULLISH" | "NEUTRAL" | "BEARISH"
            nifty_close  : last Nifty 50 close
            nifty_ema20  : 20D EMA of Nifty
            nifty_vs_ema : % difference (positive = above EMA)
            vix          : India VIX value (0 if unavailable)
            vix_regime   : "LOW" | "MODERATE" | "HIGH" | "EXTREME"
            allow_buys   : bool — whether BUY signals should be sent
            allow_sells  : bool — whether SELL signals should be sent
            reason       : human-readable explanation
    """
    nifty_close: float = 0.0
    nifty_ema20: float = 0.0
    nifty_vs_ema: float = 0.0
    vix: float = 0.0

    # ── Fetch Nifty 50 data ───────────────────────────────────────────────────
    nifty_df: Optional[pd.DataFrame] = None

    # Try yfinance first (^NSEI = Nifty 50, ^INDIAVIX = India VIX)
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        ticker = yf.Ticker("^NSEI")
        raw = ticker.history(period="60d", interval="1d")
        if not raw.empty:
            raw.columns = [c.lower() for c in raw.columns]
            nifty_df = raw[["close"]].copy()
            nifty_df["close"] = nifty_df["close"].astype(float)
            logger.debug(
                f"Nifty 50 data fetched via yfinance "
                f"({len(nifty_df)} bars)"
            )

        # Fetch VIX
        vix_ticker = yf.Ticker("^INDIAVIX")
        vix_raw = vix_ticker.history(period="5d", interval="1d")
        if not vix_raw.empty:
            vix = float(vix_raw["Close"].iloc[-1])

    except Exception as exc:
        logger.warning(f"yfinance Nifty fetch failed: {exc}")

    # Fallback: try NSE via fallback_manager
    if nifty_df is None and fallback_manager is not None:
        try:
            from datetime import datetime, timedelta

            end = datetime.now()
            start = end - timedelta(days=90)
            data = await fallback_manager.fetch_stock_data(
                "NIFTY 50", start, end
            )
            if data and data.get("records"):
                records = data["records"]
                nifty_df = pd.DataFrame(records).set_index("date")[["close"]]
        except Exception as exc:
            logger.warning(
                f"FallbackManager Nifty fetch failed: {exc}"
            )

    # ── Compute regime ────────────────────────────────────────────────────────
    if nifty_df is not None and len(nifty_df) >= 20:
        ema20 = nifty_df["close"].ewm(span=20, adjust=False).mean()
        nifty_close = float(nifty_df["close"].iloc[-1])
        nifty_ema20 = float(ema20.iloc[-1])
        if nifty_ema20 > 0:
            nifty_vs_ema = (nifty_close - nifty_ema20) / nifty_ema20 * 100
    else:
        logger.warning(
            "Could not fetch Nifty 50 data — defaulting to NEUTRAL regime"
        )

    # ── VIX regime classification ─────────────────────────────────────────────
    if vix == 0.0:
        vix_regime = "UNKNOWN"
    elif vix < VIX_NORMAL_MAX:
        vix_regime = "LOW"
    elif vix < VIX_ELEVATED_MAX:
        vix_regime = "MODERATE"
    elif vix < VIX_HIGH_MAX:
        vix_regime = "HIGH"
    else:
        vix_regime = "EXTREME"

    # ── Overall market regime ─────────────────────────────────────────────────
    allow_buys = True
    allow_sells = True
    reason = ""

    if nifty_close == 0.0:
        # Data unavailable — don't block anything, just warn
        regime = "NEUTRAL"
        reason = "Nifty data unavailable; no market filter applied"

    elif vix >= VIX_HIGH_MAX:
        # Extreme fear: avoid new longs, allow shorts
        regime = "BEARISH"
        allow_buys = False
        reason = (
            f"India VIX={vix:.1f} (EXTREME fear ≥{VIX_HIGH_MAX}). "
            "Skipping all BUY signals."
        )

    elif nifty_vs_ema < -1.0:
        # Nifty more than 1% below its 20D EMA = bearish trend
        regime = "BEARISH"
        allow_buys = False
        reason = (
            f"Nifty ({nifty_close:.0f}) is {nifty_vs_ema:.1f}% below "
            f"its 20D EMA ({nifty_ema20:.0f}). Skipping BUY signals."
        )

    elif -1.0 <= nifty_vs_ema <= 0.5 or (
        VIX_ELEVATED_MAX <= vix < VIX_HIGH_MAX
    ):
        # Nifty near EMA or VIX elevated: neutral, allow both with caution
        regime = "NEUTRAL"
        reason = (
            f"Nifty {nifty_vs_ema:+.1f}% vs 20D EMA "
            f"| VIX={vix:.1f} ({vix_regime}). Neutral regime."
        )

    else:
        # Nifty above EMA and VIX low/moderate = bullish
        regime = "BULLISH"
        reason = (
            f"Nifty ({nifty_close:.0f}) is {nifty_vs_ema:.1f}% above "
            f"20D EMA ({nifty_ema20:.0f}) | VIX={vix:.1f} ({vix_regime}). "
            "Bullish regime."
        )

    context = {
        "regime":       regime,
        "nifty_close":  round(nifty_close, 2),
        "nifty_ema20":  round(nifty_ema20, 2),
        "nifty_vs_ema": round(nifty_vs_ema, 2),
        "vix":          round(vix, 2),
        "vix_regime":   vix_regime,
        "allow_buys":   allow_buys,
        "allow_sells":  allow_sells,
        "reason":       reason,
    }

    logger.info(
        f"Market context: {regime} | {reason}",
        extra=context,
    )

    return context

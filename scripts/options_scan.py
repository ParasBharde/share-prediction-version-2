"""
Options Trading Scanner

Purpose:
    Scans NIFTY/BANKNIFTY for options trading signals.
    Uses OI analysis, VWAP+Supertrend, and PCR sentiment strategies.
    Fetches option chain data from NSE and 5m candles from Yahoo.
    Generates annotated chart images and sends them to Telegram.

Usage:
    # Scan NIFTY options (single run)
    python scripts/options_scan.py --symbol NIFTY

    # Scan BANKNIFTY options
    python scripts/options_scan.py --symbol BANKNIFTY

    # Scan both indices
    python scripts/options_scan.py --symbol NIFTY --symbol BANKNIFTY

    # Auto-repeat every 5 minutes
    python scripts/options_scan.py --symbol NIFTY --repeat 5

    # With Telegram alerts + chart images
    python scripts/options_scan.py --symbol NIFTY --telegram

    # Dry run (print but don't send Telegram)
    python scripts/options_scan.py --symbol NIFTY --dry-run

    # Custom chart output dir (default /tmp)
    python scripts/options_scan.py --symbol NIFTY --telegram --chart-dir ./charts/options

Environment:
    TELEGRAM_BOT_TOKEN   — bot token
    TELEGRAM_CHAT_ID     — target chat / channel
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
import warnings

warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows so Rupee symbol and emoji print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.alerts.alert_formatter import AlertFormatter
from src.data_ingestion.option_chain_fetcher import OptionChainFetcher
from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import STRATEGY_REGISTRY
import contextlib
import io
import yfinance as yf

from src.utils.config_loader import load_strategy_config
from src.utils.visualizer import ChartVisualizer

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

try:
    from nsepython import nse_optionchain_scrapper
    _NSEPYTHON_AVAILABLE = True
except ImportError:
    _NSEPYTHON_AVAILABLE = False
    def nse_optionchain_scrapper(symbol, *args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("nsepython not installed — run: pip install nsepython")

IST = pytz.timezone("Asia/Kolkata")
logger = get_logger(__name__)

# ── Signal deduplication: don't re-alert the same trade within 30 min ────────
_signal_cooldown: Dict[str, datetime] = {}
_SIGNAL_COOLDOWN_MIN = 30  # minutes

# ── Market trading hours (IST) ────────────────────────────────────────────────
_MARKET_OPEN_H, _MARKET_OPEN_M   = 9, 15
_MARKET_CLOSE_H, _MARKET_CLOSE_M = 15, 30
_ENTRY_START_H, _ENTRY_START_M   = 9, 30   # skip first 15 min (choppy open)
_ENTRY_END_H,   _ENTRY_END_M     = 15, 0   # stop entering new trades at 3 PM


def _is_market_open() -> tuple:
    """Return (is_open: bool, reason: str) based on current IST time."""
    now = datetime.now(IST)
    # Weekend check
    if now.weekday() >= 5:
        return False, f"Market closed — {now.strftime('%A')}"
    h, m = now.hour, now.minute
    total_min = h * 60 + m
    open_min  = _MARKET_OPEN_H  * 60 + _MARKET_OPEN_M
    close_min = _MARKET_CLOSE_H * 60 + _MARKET_CLOSE_M
    if total_min < open_min:
        return False, f"Pre-market ({now.strftime('%H:%M')} IST)"
    if total_min >= close_min:
        return False, f"Market closed for today ({now.strftime('%H:%M')} IST)"
    return True, "Market open"


def _is_entry_allowed() -> tuple:
    """Return (allowed: bool, reason: str) — entry window 9:30–15:00 IST."""
    now = datetime.now(IST)
    h, m = now.hour, now.minute
    total_min = h * 60 + m
    start_min = _ENTRY_START_H * 60 + _ENTRY_START_M
    end_min   = _ENTRY_END_H   * 60 + _ENTRY_END_M
    if total_min < start_min:
        return False, "Wait — first 15 min choppy (opens 9:30)"
    if total_min >= end_min:
        return False, "Entry window closed — avoid new trades after 3 PM"
    return True, "Entry window open"


def _check_duplicate_signal(symbol: str, option_type: str) -> bool:
    """True if the same signal was sent within the cooldown window."""
    key = f"{symbol}_{option_type}"
    last = _signal_cooldown.get(key)
    if last is None:
        return False
    age_min = (datetime.now() - last).total_seconds() / 60
    return age_min < _SIGNAL_COOLDOWN_MIN


def _mark_signal_sent(symbol: str, option_type: str) -> None:
    key = f"{symbol}_{option_type}"
    _signal_cooldown[key] = datetime.now()


def _compute_ema(series: "pd.Series", period: int) -> "pd.Series":
    return series.ewm(span=period, adjust=False).mean()


def _compute_vwap(df: "pd.DataFrame") -> float:
    """
    Session VWAP from available bars.
    Falls back to mean typical price when volume is zero/NaN
    (index tickers like ^NSEI don't carry volume on Yahoo Finance).
    """
    try:
        tp = (df["high"].astype(float)
              + df["low"].astype(float)
              + df["close"].astype(float)) / 3
        vol = df["volume"].fillna(0).astype(float) if "volume" in df.columns \
              else pd.Series(0.0, index=df.index)
        total_vol = float(vol.sum())
        if total_vol > 0:
            return float((tp * vol).sum() / total_vol)
        # No volume data → use mean typical price as proxy
        return float(tp.mean())
    except Exception:
        return 0.0


def _compute_supertrend_direction(df: "pd.DataFrame", period: int = 10,
                                   mult: float = 3.0) -> str:
    """Return 'bullish' or 'bearish' based on Supertrend direction on last bar."""
    try:
        if len(df) < period + 2:
            return "neutral"
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        # ATR (Wilder)
        tr  = (high - low).combine(abs(high - close.shift()), max).combine(
                abs(low - close.shift()), max)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        hl2 = (high + low) / 2
        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr
        # Walk through bars
        st   = [0.0] * len(df)
        dir_ = [1]   * len(df)  # 1 = bull, -1 = bear
        for i in range(1, len(df)):
            ub_prev = upper_band.iloc[i - 1]
            lb_prev = lower_band.iloc[i - 1]
            ub_cur  = upper_band.iloc[i]
            lb_cur  = lower_band.iloc[i]
            ub_cur  = min(ub_cur, ub_prev) if close.iloc[i - 1] <= ub_prev else ub_cur
            lb_cur  = max(lb_cur, lb_prev) if close.iloc[i - 1] >= lb_prev else lb_cur
            upper_band.iloc[i] = ub_cur
            lower_band.iloc[i] = lb_cur
            if st[i - 1] == ub_prev:
                st[i]   = lb_cur if close.iloc[i] > ub_cur else ub_cur
                dir_[i] = 1     if close.iloc[i] > ub_cur else -1
            else:
                st[i]   = ub_cur if close.iloc[i] < lb_cur else lb_cur
                dir_[i] = -1    if close.iloc[i] < lb_cur else 1
        return "bullish" if dir_[-1] == 1 else "bearish"
    except Exception:
        return "neutral"


def _compute_mtf_bias(df_5m: "pd.DataFrame",
                       df_15m: "pd.DataFrame") -> dict:
    """
    Multi-Timeframe Bias  (inspired by John Carter 'Mastering the Trade').

    15m determines the trend; 5m provides the entry trigger.
    Returns:
        {
          'bias':    'bullish' | 'bearish' | 'neutral',
          'strength': 0-5,
          'details': { ... }
        }
    """
    result = {"bias": "neutral", "strength": 0, "details": {}}
    bull_score = 0
    bear_score = 0

    # ── 15m indicators ────────────────────────────────────────────────────
    if df_15m is not None and len(df_15m) >= 22:
        c15 = df_15m["close"].astype(float)
        ema9_15  = _compute_ema(c15, 9).iloc[-1]
        ema21_15 = _compute_ema(c15, 21).iloc[-1]
        price_15 = c15.iloc[-1]
        vwap_15  = _compute_vwap(df_15m)
        st_15    = _compute_supertrend_direction(df_15m)

        ema_trend_15 = "bullish" if ema9_15 > ema21_15 else "bearish"
        vwap_bias_15 = "bullish" if price_15 > vwap_15 else "bearish"

        result["details"]["15m_ema_trend"]  = ema_trend_15
        result["details"]["15m_vwap_bias"]  = vwap_bias_15
        result["details"]["15m_supertrend"] = st_15
        result["details"]["15m_ema9"]       = round(ema9_15, 2)
        result["details"]["15m_ema21"]      = round(ema21_15, 2)
        result["details"]["15m_vwap"]       = round(vwap_15, 2)

        # Weight: 15m counts 2x (higher timeframe = stronger signal)
        if ema_trend_15 == "bullish":  bull_score += 2
        else:                          bear_score += 2
        if vwap_bias_15 == "bullish":  bull_score += 1
        else:                          bear_score += 1
        if st_15 == "bullish":         bull_score += 1
        else:                          bear_score += 1
    else:
        result["details"]["15m_status"] = "insufficient data"

    # ── 5m indicators ─────────────────────────────────────────────────────
    if df_5m is not None and len(df_5m) >= 22:
        c5 = df_5m["close"].astype(float)
        ema9_5  = _compute_ema(c5, 9).iloc[-1]
        ema21_5 = _compute_ema(c5, 21).iloc[-1]
        ema_trend_5 = "bullish" if ema9_5 > ema21_5 else "bearish"
        result["details"]["5m_ema_trend"] = ema_trend_5

        if ema_trend_5 == "bullish":  bull_score += 1
        else:                         bear_score += 1

    total = bull_score + bear_score
    if total == 0:
        return result

    result["details"]["bull_score"] = bull_score
    result["details"]["bear_score"] = bear_score

    if bull_score > bear_score:
        result["bias"]     = "bullish"
        result["strength"] = bull_score
    elif bear_score > bull_score:
        result["bias"]     = "bearish"
        result["strength"] = bear_score
    else:
        result["bias"]     = "neutral"
        result["strength"] = 0

    return result


def _trade_quality_check(signal, option_chain: dict,
                          mtf: dict) -> dict:
    """
    Score a signal for trade quality.  Returns a dict with:
      - overall_ok: bool  (trade or no trade)
      - score: 0-100
      - reasons_to_trade: list[str]
      - reasons_to_avoid: list[str]
      - entry_zone: (low, high)
      - target1: float   (50% of full target, first exit)
      - target2: float   (full target, final exit)
    """
    opt_type    = signal.metadata.get("option_type", "BUY_CE")
    is_ce       = opt_type == "BUY_CE"
    entry_spot  = signal.entry_price
    target_spot = signal.target_price
    sl_spot     = signal.stop_loss
    atm_strike  = signal.metadata.get("atm_strike", 0)
    atm_ltp     = (signal.metadata.get("atm_ce_ltp", 0)
                   if is_ce else signal.metadata.get("atm_pe_ltp", 0))
    lot_size    = LOT_SIZES.get(signal.symbol, 25)
    pcr         = option_chain.get("pcr", 1.0)
    is_expiry   = option_chain.get("is_expiry_day", False)
    iv          = (signal.metadata.get("atm_ce_iv", 0)
                   if is_ce else signal.metadata.get("atm_pe_iv", 0))

    score          = int(signal.confidence * 100)
    reasons_trade  = []
    reasons_avoid  = []

    # ── MTF alignment ──────────────────────────────────────────────────────
    mtf_bias = mtf.get("bias", "neutral")
    signal_direction = "bullish" if is_ce else "bearish"
    if mtf_bias == signal_direction:
        score += 10
        reasons_trade.append(
            f"15m + 5m trend aligned ({mtf_bias.upper()})"
        )
    elif mtf_bias == "neutral":
        reasons_avoid.append("Multi-timeframe trend unclear — range market")
    else:
        score -= 20
        reasons_avoid.append(
            f"15m trend is {mtf_bias.upper()} but trade is {signal_direction.upper()} — COUNTER-TREND"
        )

    # ── PCR alignment ──────────────────────────────────────────────────────
    if is_ce and pcr >= 1.0:
        reasons_trade.append(f"PCR {pcr:.2f} — more PUT writers (bullish support)")
    elif not is_ce and pcr <= 1.0:
        reasons_trade.append(f"PCR {pcr:.2f} — more CALL writers (bearish pressure)")
    else:
        reasons_avoid.append(f"PCR {pcr:.2f} not ideal for {opt_type}")

    # ── IV filter ──────────────────────────────────────────────────────────
    if iv > 0:
        if iv > 35:
            score -= 10
            reasons_avoid.append(
                f"IV {iv:.1f}% is HIGH — option premium expensive, theta decay risk"
            )
        elif iv < 12:
            reasons_avoid.append(
                f"IV {iv:.1f}% very low — potential IV crush after event"
            )
        else:
            reasons_trade.append(f"IV {iv:.1f}% is in normal range — fair premium")

    # ── Expiry day caution ─────────────────────────────────────────────────
    if is_expiry:
        now_h = datetime.now(IST).hour
        if now_h < 11:
            score -= 15
            reasons_avoid.append(
                "Expiry day before 11 AM — extreme gamma/theta, very risky"
            )
        elif now_h >= 14:
            score -= 25
            reasons_avoid.append(
                "Expiry day after 2 PM — massive time decay, avoid"
            )
        else:
            reasons_trade.append("Expiry day 11 AM–2 PM — manageable risk window")

    # ── Entry time check ───────────────────────────────────────────────────
    entry_ok, entry_msg = _is_entry_allowed()
    if not entry_ok:
        score -= 30
        reasons_avoid.append(entry_msg)
    else:
        reasons_trade.append("Good entry time window (9:30 AM – 3:00 PM)")

    # ── Confidence gate ────────────────────────────────────────────────────
    if signal.confidence < 0.65:
        reasons_avoid.append(
            f"Signal confidence {signal.confidence:.0%} below threshold — wait for cleaner setup"
        )

    # ── Risk:Reward check ──────────────────────────────────────────────────
    risk  = abs(entry_spot - sl_spot)
    rewrd = abs(target_spot - entry_spot)
    rr    = rewrd / risk if risk > 0 else 0
    if rr >= 1.5:
        reasons_trade.append(f"R:R = 1:{rr:.1f} — favourable")
    else:
        reasons_avoid.append(f"R:R = 1:{rr:.1f} — too low (need min 1:1.5)")

    # ── OI levels as confirmation ──────────────────────────────────────────
    resistance = option_chain.get("resistance", 0)
    support    = option_chain.get("support",    0)
    if is_ce and resistance > entry_spot:
        reasons_trade.append(
            f"OI resistance wall at {resistance} — room to run upward"
        )
    elif not is_ce and support > 0 and support < entry_spot:
        reasons_trade.append(
            f"OI support floor at {support} — room to fall"
        )

    # ── Premium entry zone (±3%) ───────────────────────────────────────────
    entry_zone_lo = round(atm_ltp * 0.97, 1) if atm_ltp > 0 else 0
    entry_zone_hi = round(atm_ltp * 1.03, 1) if atm_ltp > 0 else 0

    # ── Two targets for option premium ────────────────────────────────────
    # T1 = half the spot target move, T2 = full spot target move
    # Use delta-based approximation: ATM delta ≈ 0.40 (conservative)
    # premium_gain ≈ spot_points × delta
    spot_points_full = abs(target_spot - entry_spot)
    spot_points_half = spot_points_full * 0.5
    assumed_delta    = 0.40          # ATM CE/PE delta ≈ 0.40-0.50
    target1_prem  = round(atm_ltp + spot_points_half * assumed_delta, 1)
    target2_prem  = round(atm_ltp + spot_points_full * assumed_delta, 1)
    sl_prem       = round(atm_ltp * 0.5, 1)   # SL: -50% of premium paid

    overall_ok = (
        score >= 55
        and entry_ok
        and signal.confidence >= 0.60
        and rr >= 1.2
        and not (is_expiry and datetime.now(IST).hour >= 14)
    )

    return {
        "overall_ok":     overall_ok,
        "score":          min(score, 100),
        "reasons_trade":  reasons_trade,
        "reasons_avoid":  reasons_avoid,
        "entry_zone":     (entry_zone_lo, entry_zone_hi),
        "target1_prem":   target1_prem,
        "target2_prem":   target2_prem,
        "sl_prem":        sl_prem,
        "atm_ltp":        atm_ltp,
        "lot_size":       lot_size,
        "premium_per_lot": round(atm_ltp * lot_size, 0),
        "rr":             round(rr, 1),
        "mtf":            mtf,
    }


def format_expert_telegram(signal, quality: dict, option_chain: dict,
                            strategies_agreed: int = 1) -> str:
    """
    Expert-grade Telegram alert with complete trade instructions.
    Inspired by: John Carter 'Mastering the Trade',
                 Mark Douglas 'Trading in the Zone'
    """
    opt_type   = signal.metadata.get("option_type", "BUY_CE")
    is_ce      = opt_type == "BUY_CE"
    opt_label  = "CALL (CE)" if is_ce else "PUT  (PE)"
    action     = "BUY" if quality["overall_ok"] else "WAIT / SKIP"
    emoji_top  = "🟢" if quality["overall_ok"] else "🔴"
    atm        = signal.metadata.get("atm_strike", 0)
    expiry     = option_chain.get("current_expiry", "N/A")
    spot       = signal.entry_price
    target_sp  = signal.target_price
    sl_sp      = signal.stop_loss
    atm_ltp    = quality["atm_ltp"]
    lot_size   = quality["lot_size"]
    score      = quality["score"]
    conf       = int(signal.confidence * 100)
    now_str    = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")
    mtf        = quality["mtf"]
    mtf_det    = mtf.get("details", {})
    pcr        = option_chain.get("pcr", 0)
    resistance = option_chain.get("resistance", 0)
    support    = option_chain.get("support",    0)
    is_expiry  = option_chain.get("is_expiry_day", False)

    # Strategy consensus badge
    agree_badge = ""
    if strategies_agreed >= 2:
        agree_badge = f"  ⚡ {strategies_agreed}/3 STRATEGIES AGREE — HIGH CONVICTION\n"

    # Indicator tick list
    ind_lines = []
    for name, det in (signal.indicator_details or {}).items():
        tick  = "✅" if det.get("passed") else "❌"
        label = name.replace("_", " ").title()
        # Build short note
        notes = []
        for k in ("value", "direction", "sentiment", "type", "volume_ratio"):
            v = det.get(k)
            if v is not None:
                notes.append(f"{k}: {v}")
        note  = ", ".join(notes[:2])
        ind_lines.append(f"  {tick} {label:<18} {note}")

    ind_block = "\n".join(ind_lines) if ind_lines else "  (no indicator details)"

    # MTF block
    mtf_bias  = mtf.get("bias", "neutral").upper()
    mtf_emoji = "🟢" if mtf_bias == "BULLISH" else ("🔴" if mtf_bias == "BEARISH" else "🟡")
    mtf_block = (
        f"  {mtf_emoji} 15m trend  : {mtf_det.get('15m_ema_trend','?').upper()}"
        f"  (EMA9 {mtf_det.get('15m_ema9','?')} vs EMA21 {mtf_det.get('15m_ema21','?')})\n"
        f"  {mtf_emoji} 15m VWAP   : {mtf_det.get('15m_vwap_bias','?').upper()}"
        f"  (VWAP {mtf_det.get('15m_vwap','?')})\n"
        f"  {mtf_emoji} 15m ST     : {mtf_det.get('15m_supertrend','?').upper()}\n"
        f"  {'🟢' if mtf_det.get('5m_ema_trend')=='bullish' else '🔴'}"
        f" 5m trend   : {mtf_det.get('5m_ema_trend','?').upper()}"
    )

    # Reasons block
    reasons_trade_block = "\n".join(
        f"  ✅ {r}" for r in quality["reasons_trade"]
    ) or "  (none)"
    reasons_avoid_block = "\n".join(
        f"  ⚠️  {r}" for r in quality["reasons_avoid"]
    ) or "  (none)"

    # Build full message
    sep = "━" * 38
    msg = (
        f"{sep}\n"
        f"{emoji_top}  {action} {opt_label}\n"
        f"    {signal.symbol}  {atm} {('CE' if is_ce else 'PE')}"
        f"  |  Expiry: {expiry}\n"
        f"{agree_badge}"
        f"    Score: {score}/100  |  Conf: {conf}%\n"
        f"{sep}\n"
        f"\n"
        f"📌 WHAT TO BUY:\n"
        f"  Strike   : {atm} {'CE (Call Option)' if is_ce else 'PE (Put Option)'}\n"
        f"  Premium  : Rs.{atm_ltp:.2f}  (price now)\n"
        f"  Lot Size : {lot_size} qty  →  Rs.{quality['premium_per_lot']:,.0f} per lot\n"
        f"\n"
        f"🟢 ENTRY ZONE (buy in this range):\n"
        f"  Premium : Rs.{quality['entry_zone'][0]} – Rs.{quality['entry_zone'][1]}\n"
        f"  Spot    : {'Above' if is_ce else 'Below'} Rs.{spot:,.2f}\n"
        f"\n"
        f"🎯 TARGETS (exit plan):\n"
        f"  Target 1: Rs.{quality['target1_prem']:.1f}  →  EXIT half your lots here\n"
        f"  Target 2: Rs.{quality['target2_prem']:.1f}  →  EXIT remaining lots here\n"
        f"  Spot T1 : Rs.{(spot + (target_sp-spot)*0.5):,.2f}\n"
        f"  Spot T2 : Rs.{target_sp:,.2f}\n"
        f"\n"
        f"🛑 STOP LOSS (exit immediately if hit):\n"
        f"  Premium SL : Rs.{quality['sl_prem']:.1f}  (-50% of premium paid)\n"
        f"  Spot SL    : Rs.{sl_sp:,.2f}\n"
        f"  R:R Ratio  : 1:{quality['rr']}\n"
        f"\n"
        f"❌ DO NOT BUY IF:\n"
        f"  → Premium above Rs.{quality['entry_zone'][1] * 1.1:.1f} (slippage too high)\n"
        f"  → {'Spot below' if is_ce else 'Spot above'} Rs.{sl_sp:,.2f} at entry\n"
        f"  → Time is after 3:00 PM IST\n"
        f"  → You already have an open {'CE' if is_ce else 'PE'} position\n"
        f"\n"
        f"📊 OI LEVELS:\n"
        f"  Resistance : {resistance}  (CALL writers — ceiling)\n"
        f"  Support    : {support}  (PUT writers — floor)\n"
        f"  PCR        : {pcr:.2f}  "
        f"({'bullish — more PUT writers' if pcr > 1.0 else 'bearish — more CALL writers'})\n"
        f"  Expiry day : {'YES ⚠️' if is_expiry else 'No'}\n"
        f"\n"
        f"📈 MULTI-TIMEFRAME ANALYSIS:\n"
        f"{mtf_block}\n"
        f"\n"
        f"🔍 INDICATORS  ({signal.indicators_met}/{signal.total_indicators} met):\n"
        f"{ind_block}\n"
        f"\n"
        f"{'✅ WHY TO TRADE:' if quality['reasons_trade'] else ''}\n"
        f"{reasons_trade_block}\n"
        f"\n"
        f"{'⚠️  CAUTION:' if quality['reasons_avoid'] else ''}\n"
        f"{reasons_avoid_block}\n"
        f"\n"
        f"📋 Strategy : {signal.strategy_name}\n"
        f"⏰ {now_str}\n"
        f"{sep}"
    )
    # Trim blank lines that appear when sections are empty
    import re
    msg = re.sub(r"\n{3,}", "\n\n", msg.strip())
    # Telegram limit 4096 chars
    if len(msg) > 4000:
        msg = msg[:3990] + "\n... (truncated)"
    return msg


def _console_signal_summary(signal) -> str:
    """One-block console summary for a TradingSignal."""
    opt = signal.metadata.get("option_type", "")
    atm = signal.metadata.get("atm_strike", 0)
    ltp = (signal.metadata.get("atm_ce_ltp", 0)
           if opt == "BUY_CE" else signal.metadata.get("atm_pe_ltp", 0))
    lot = LOT_SIZES.get(signal.symbol, 25)
    sep = "=" * 55
    return (
        f"\n{sep}\n"
        f"  SIGNAL: {opt}  |  {signal.symbol} {atm}\n"
        f"  Strategy : {signal.strategy_name}\n"
        f"  Spot     : Rs.{signal.entry_price:,.2f}"
        f"  |  Target: Rs.{signal.target_price:,.2f}"
        f"  |  SL: Rs.{signal.stop_loss:,.2f}\n"
        f"  Premium  : Rs.{ltp:.2f}  |  Lot: {lot} qty"
        f"  =  Rs.{ltp*lot:,.0f}/lot\n"
        f"  Conf     : {signal.confidence:.0%}"
        f"  |  Indicators: {signal.indicators_met}/{signal.total_indicators}\n"
        f"{sep}"
    )


def _compute_atr14(df: "pd.DataFrame") -> float:
    """Compute ATR-14 (Wilder) from an OHLCV DataFrame."""
    try:
        if len(df) < 15 or not {"high", "low", "close"}.issubset(df.columns):
            return 0.0
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = df["close"].astype(float).shift(1)
        tr = (
            (high - low)
            .combine(abs(high - prev_close), max)
            .combine(abs(low - prev_close), max)
        )
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return 0.0


# ============================================================================
# OPTIONS TRADING CONFIGURATION
# ============================================================================

OPTIONS_CONFIG = {
    # Risk management
    "max_premium_per_lot": 5000,        # Max ₹5000 premium per lot
    "min_premium_per_lot": 100,         # Min ₹100 premium (avoid illiquid)
    "max_delta_long": 0.7,              # Max delta 0.7 for long positions
    "min_delta_long": 0.3,              # Min delta 0.3 (avoid deep OTM)
    
    # Liquidity filters
    "min_oi": 100,                      # Minimum Open Interest
    "min_volume": 50,                   # Minimum volume today
    "max_bid_ask_spread_pct": 2.0,      # Max 2% spread
    
    # IV filters
    "min_iv_percentile": 30,            # Min IV percentile (for buying)
    "max_iv_percentile": 70,            # Max IV percentile (for selling)
    
    # Time decay
    "min_dte": 0,                       # Minimum days to expiry (0 = 0DTE allowed)
    "max_dte": 7,                       # Maximum days to expiry
    
    # Strike selection
    "strikes_around_spot": 10,          # Scan 10 strikes above/below spot
}

# Lot sizes — revised by SEBI / NSE+BSE (effective Sep 2024)
# Verify at: https://www.nseindia.com/regulations/lot-size
LOT_SIZES = {
    "NIFTY":      75,   # Revised from 50 → 75 (Sep 2024)
    "BANKNIFTY":  30,   # Revised from 15 → 30 (Sep 2024)
    "FINNIFTY":   65,   # Revised from 40 → 65 (Sep 2024)
    "MIDCPNIFTY": 120,  # Revised from 75 → 120 (Sep 2024)
    "SENSEX":     10,   # BSE SENSEX options lot size
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Options Trading Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol", type=str, action="append", default=[],
        help="Index symbol (NIFTY, BANKNIFTY). Can specify multiple."
    )
    parser.add_argument(
        "--interval", type=str, default="5m",
        choices=["5m", "15m"],
        help="Candle interval (default: 5m)"
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Auto-repeat every N minutes (0=once)"
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send alerts + chart images via Telegram"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print signals but do NOT send Telegram"
    )
    parser.add_argument(
        "--chart-dir", type=str, default="/tmp",
        metavar="DIR",
        help="Directory to save chart PNG files (default: /tmp)"
    )
    return parser.parse_args()

def get_spot_price(index: str) -> float:
    """
    Get current spot price of index.
    Uses option chain data (nsepython) as it contains the underlying value.
    """
    if not _NSEPYTHON_AVAILABLE:
        logger.warning("nsepython not installed — spot price unavailable. Run: pip install nsepython")
        return 0.0
    try:
        # Fetch option chain - it contains underlying (spot) price
        chain = nse_optionchain_scrapper(index)

        if not chain:
            return 0.0

        # Extract underlying value (spot price)
        records = chain.get("records", {})
        underlying_value = records.get("underlyingValue")

        if underlying_value:
            return float(underlying_value)

        # Alternative: Try strikePrices and find ATM
        # (spot is usually near ATM strike)
        data = records.get("data", [])
        if data and len(data) > 0:
            # Get middle strike as approximation
            strikes = [d.get("strikePrice", 0) for d in data]
            if strikes:
                return float(sorted(strikes)[len(strikes)//2])

        return 0.0

    except Exception as e:
        print(f"  ❌ Error fetching spot for {index}: {e}")
        # Last resort: Use fallback values (update these daily)
        fallback = {
            "NIFTY": 23500,
            "BANKNIFTY": 48000,
            "FINNIFTY": 22000,
        }
        print(f"  ⚠️  Using fallback spot: ₹{fallback.get(index, 0):,.2f}")
        return fallback.get(index, 0)


def get_expiry_dates(index: str) -> List[str]:
    """Get available expiry dates for index."""
    if not _NSEPYTHON_AVAILABLE:
        logger.warning("nsepython not installed — cannot fetch expiry dates. Run: pip install nsepython")
        return []
    try:
        data = nse_optionchain_scrapper(index)

        # Extract expiry dates
        expiries = data.get("records", {}).get("expiryDates", [])
        return expiries[:4]  # Return next 4 expiries

    except Exception as e:
        print(f"  ❌ Error fetching expiries for {index}: {e}")
        return []


def get_options_chain(index: str, expiry: str) -> pd.DataFrame:
    """
    Fetch options chain data for given index and expiry.
    
    Returns DataFrame with columns:
    - strike
    - CE_ltp, CE_bid, CE_ask, CE_oi, CE_volume, CE_iv, CE_delta
    - PE_ltp, PE_bid, PE_ask, PE_oi, PE_volume, PE_iv, PE_delta
    """
    if not _NSEPYTHON_AVAILABLE:
        print(f"    ❌ nsepython not installed — cannot fetch option chain. Run: pip install nsepython")
        return pd.DataFrame()
    try:
        # Fetch option chain
        print(f"    Fetching option chain for {index}...", end=" ")
        data = nse_optionchain_scrapper(index)

        if not data:
            print(f"No data returned")
            return pd.DataFrame()
        
        records = data.get("records", {}).get("data", [])
        
        if not records:
            print(f"No records in data")
            return pd.DataFrame()
        
        print(f"{len(records)} strikes fetched")
        
        # Parse into structured format
        rows = []
        for rec in records:
            # Check if this record is for the requested expiry
            rec_expiry = rec.get("expiryDate", "")
            if rec_expiry != expiry:
                continue
            
            strike = rec.get("strikePrice", 0)
            
            # Call data
            ce = rec.get("CE", {})
            ce_ltp = ce.get("lastPrice", 0)
            ce_bid = ce.get("bidprice", 0)
            ce_ask = ce.get("askPrice", 0)
            ce_oi = ce.get("openInterest", 0)
            ce_vol = ce.get("totalTradedVolume", 0)
            ce_iv = ce.get("impliedVolatility", 0)
            ce_delta = ce.get("delta", 0)
            
            # Put data
            pe = rec.get("PE", {})
            pe_ltp = pe.get("lastPrice", 0)
            pe_bid = pe.get("bidprice", 0)
            pe_ask = pe.get("askPrice", 0)
            pe_oi = pe.get("openInterest", 0)
            pe_vol = pe.get("totalTradedVolume", 0)
            pe_iv = pe.get("impliedVolatility", 0)
            pe_delta = pe.get("delta", 0)
            
            rows.append({
                "strike": strike,
                "CE_ltp": ce_ltp,
                "CE_bid": ce_bid,
                "CE_ask": ce_ask,
                "CE_oi": ce_oi,
                "CE_volume": ce_vol,
                "CE_iv": ce_iv,
                "CE_delta": ce_delta,
                "PE_ltp": pe_ltp,
                "PE_bid": pe_bid,
                "PE_ask": pe_ask,
                "PE_oi": pe_oi,
                "PE_volume": pe_vol,
                "PE_iv": pe_iv,
                "PE_delta": pe_delta,
            })
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            print(f"    ⚠️  No data for expiry {expiry}")
        else:
            print(f"    ✅ {len(df)} strikes for expiry {expiry}")
        
        return df
    
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# ============================================================================
# OPTIONS ANALYSIS & FILTERING
# ============================================================================

def calculate_moneyness(spot: float, strike: float, option_type: str) -> str:
    """
    Calculate if option is ITM/ATM/OTM.
    
    Returns: "ITM", "ATM", "OTM"
    """
    diff_pct = abs((strike - spot) / spot) * 100
    
    if diff_pct < 1:
        return "ATM"
    
    if option_type == "CE":
        if strike < spot:
            return "ITM"
        else:
            return "OTM"
    else:  # PE
        if strike > spot:
            return "ITM"
        else:
            return "OTM"


def calculate_bid_ask_spread(bid: float, ask: float) -> float:
    """Calculate bid-ask spread percentage."""
    if bid == 0 or ask == 0:
        return 100.0  # Invalid
    mid = (bid + ask) / 2
    if mid == 0:
        return 100.0
    return ((ask - bid) / mid) * 100


def estimate_delta(spot: float, strike: float, option_type: str, days_to_expiry: int) -> float:
    """
    Rough delta estimation (simplified Black-Scholes approximation).
    
    NOTE: This is a rough estimate. Real delta requires proper pricing model.
    """
    moneyness = (spot - strike) / spot
    
    # Time decay factor
    time_factor = max(0.1, days_to_expiry / 7)
    
    if option_type == "CE":
        # Call delta: 0 (deep OTM) → 1 (deep ITM)
        if strike <= spot - (spot * 0.05):  # Deep ITM
            return min(0.9, 0.7 + time_factor * 0.2)
        elif strike >= spot + (spot * 0.05):  # Deep OTM
            return max(0.1, 0.3 - time_factor * 0.2)
        else:  # ATM
            return 0.5
    else:  # PE
        # Put delta: -1 (deep ITM) → 0 (deep OTM)
        if strike >= spot + (spot * 0.05):  # Deep ITM
            return min(-0.1, -0.7 - time_factor * 0.2)
        elif strike <= spot - (spot * 0.05):  # Deep OTM
            return max(-0.9, -0.3 + time_factor * 0.2)
        else:  # ATM
            return -0.5


def filter_liquid_options(df: pd.DataFrame, option_type: str = "CE") -> pd.DataFrame:
    """
    Filter for liquid options based on OI, volume, spread.
    """
    prefix = option_type  # "CE" or "PE"
    
    # Filter by OI
    df = df[df[f"{prefix}_oi"] >= OPTIONS_CONFIG["min_oi"]].copy()
    
    # Filter by volume
    df = df[df[f"{prefix}_volume"] >= OPTIONS_CONFIG["min_volume"]].copy()
    
    # Filter by bid-ask spread
    df["spread"] = df.apply(
        lambda row: calculate_bid_ask_spread(
            row[f"{prefix}_bid"], row[f"{prefix}_ask"]
        ), axis=1
    )
    df = df[df["spread"] <= OPTIONS_CONFIG["max_bid_ask_spread_pct"]].copy()
    
    # Filter by premium range
    df = df[
        (df[f"{prefix}_ltp"] >= OPTIONS_CONFIG["min_premium_per_lot"]) &
        (df[f"{prefix}_ltp"] <= OPTIONS_CONFIG["max_premium_per_lot"])
    ].copy()
    
    return df


def calculate_days_to_expiry(expiry_str: str) -> int:
    """Calculate days to expiry from date string."""
    try:
        # Parse expiry date (format: "19-Feb-2026")
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y")
        today = datetime.now()
        days = (expiry_date - today).days
        return max(0, days)
    except Exception:
        return 0


# ============================================================================
# TRADING STRATEGIES
# ============================================================================

def strategy_trend_following(
    index: str,
    spot: float,
    df: pd.DataFrame,
    expiry: str,
    option_type: str = "CE"
) -> List[Dict]:
    """
    STRATEGY 1: Trend Following
    
    BUY ITM/ATM options when strong trend detected.
    - Look for high OI buildup at ITM strikes
    - Good IV (not too high)
    - Tight spread
    """
    signals = []
    dte = calculate_days_to_expiry(expiry)
    
    # Filter liquid options
    df_filtered = filter_liquid_options(df, option_type)
    
    if df_filtered.empty:
        return signals
    
    prefix = option_type
    
    for _, row in df_filtered.iterrows():
        strike = row["strike"]
        ltp = row[f"{prefix}_ltp"]
        oi = row[f"{prefix}_oi"]
        volume = row[f"{prefix}_volume"]
        iv = row[f"{prefix}_iv"]
        
        moneyness = calculate_moneyness(spot, strike, option_type)
        
        # Only ITM/ATM for trend following
        if moneyness not in ["ITM", "ATM"]:
            continue
        
        # Estimate delta
        delta = estimate_delta(spot, strike, option_type, dte)
        
        # Check delta range
        if option_type == "CE":
            if not (OPTIONS_CONFIG["min_delta_long"] <= delta <= OPTIONS_CONFIG["max_delta_long"]):
                continue
        else:
            if not (-OPTIONS_CONFIG["max_delta_long"] <= delta <= -OPTIONS_CONFIG["min_delta_long"]):
                continue
        
        # OI should be high (liquidity + interest)
        if oi < 500:
            continue
        
        # IV should not be too high (expensive)
        if iv > 50:
            continue
        
        # Calculate risk-reward
        # For calls: SL = premium, Target = 2x premium
        sl = ltp
        target = ltp * 2
        risk_reward = 2.0
        
        signals.append({
            "index": index,
            "strategy": "Trend Following",
            "type": option_type,
            "strike": strike,
            "moneyness": moneyness,
            "entry": ltp,
            "sl": 0,  # Premium loss (full)
            "target": target,
            "rr": risk_reward,
            "oi": oi,
            "volume": volume,
            "iv": iv,
            "delta": delta,
            "expiry": expiry,
            "dte": dte,
        })
    
    return signals


def strategy_breakout(
    index: str,
    spot: float,
    df: pd.DataFrame,
    expiry: str,
    option_type: str = "CE"
) -> List[Dict]:
    """
    STRATEGY 2: Breakout Trading
    
    BUY ATM options on strong directional moves.
    - High volume spike
    - ATM strikes
    - Tight stop loss
    """
    signals = []
    dte = calculate_days_to_expiry(expiry)
    
    df_filtered = filter_liquid_options(df, option_type)
    
    if df_filtered.empty:
        return signals
    
    prefix = option_type
    
    # Find ATM strike
    atm_strike = min(df_filtered["strike"], key=lambda x: abs(x - spot))
    
    row = df_filtered[df_filtered["strike"] == atm_strike]
    if row.empty:
        return signals
    
    row = row.iloc[0]
    
    ltp = row[f"{prefix}_ltp"]
    oi = row[f"{prefix}_oi"]
    volume = row[f"{prefix}_volume"]
    iv = row[f"{prefix}_iv"]
    
    # Volume should be high (breakout confirmation)
    if volume < 200:
        return signals
    
    delta = estimate_delta(spot, atm_strike, option_type, dte)
    
    signals.append({
        "index": index,
        "strategy": "Breakout",
        "type": option_type,
        "strike": atm_strike,
        "moneyness": "ATM",
        "entry": ltp,
        "sl": 0,
        "target": ltp * 1.5,
        "rr": 1.5,
        "oi": oi,
        "volume": volume,
        "iv": iv,
        "delta": delta,
        "expiry": expiry,
        "dte": dte,
    })
    
    return signals


# ============================================================================
# MAIN SCANNER
# ============================================================================

def scan_index_options(
    index: str,
    option_types: List[str] = ["CE", "PE"],
    expiry_filter: str = "weekly"
) -> List[Dict]:
    """
    Scan all options for given index.
    
    Args:
        index: "NIFTY", "BANKNIFTY", "FINNIFTY"
        option_types: ["CE", "PE"] or ["CE"] or ["PE"]
        expiry_filter: "0DTE", "weekly", "monthly", "all"
    
    Returns:
        List of trading signals
    """
    all_signals = []
    
    print(f"\n{'='*60}")
    print(f"  📊 SCANNING {index} OPTIONS")
    print(f"{'='*60}")
    
    # Get spot price
    spot = get_spot_price(index)
    if spot == 0:
        print(f"  ❌ Could not fetch spot price")
        return all_signals
    
    print(f"  Spot Price: ₹{spot:,.2f}")
    
    # Get expiry dates
    expiries = get_expiry_dates(index)
    if not expiries:
        print(f"  ❌ Could not fetch expiries")
        return all_signals
    
    print(f"  Expiries: {', '.join(expiries[:3])}")
    
    # Filter expiries based on preference
    if expiry_filter == "0DTE":
        # Only today's expiry
        expiries = [expiries[0]]
    elif expiry_filter == "weekly":
        # Next 2 weeklies
        expiries = expiries[:2]
    elif expiry_filter == "monthly":
        # Only monthly (last in list usually)
        expiries = [expiries[-1]]
    
    # Scan each expiry
    for expiry in expiries:
        dte = calculate_days_to_expiry(expiry)
        
        # Filter by DTE config
        if dte < OPTIONS_CONFIG["min_dte"] or dte > OPTIONS_CONFIG["max_dte"]:
            continue
        
        print(f"\n  Scanning expiry: {expiry} ({dte} DTE)")
        
        # Fetch options chain
        df = get_options_chain(index, expiry)
        
        if df.empty:
            print(f"    No data")
            continue
        
        print(f"    Strikes available: {len(df)}")
        
        # Apply strategies
        for opt_type in option_types:
            # Strategy 1: Trend Following
            signals = strategy_trend_following(index, spot, df, expiry, opt_type)
            all_signals.extend(signals)
            
            # Strategy 2: Breakout
            signals = strategy_breakout(index, spot, df, expiry, opt_type)
            all_signals.extend(signals)
    
    return all_signals


def format_signal_output(signal: Dict) -> str:
    """Format option signal for console output."""
    lines = [
        f"{'='*60}",
        f"  🎯 {signal['strategy'].upper()} - {signal['type']}",
        f"  {signal['index']} {signal['strike']} {signal['type']} ({signal['moneyness']})",
        f"  Expiry: {signal['expiry']} ({signal['dte']} DTE)",
        f"{'='*60}",
        f"  Entry    : ₹{signal['entry']:.2f}",
        f"  Target   : ₹{signal['target']:.2f} ({(signal['target']/signal['entry']-1)*100:.0f}%)",
        f"  R:R      : 1:{signal['rr']:.1f}",
        f"  Delta    : {signal['delta']:.2f}",
        f"",
        f"  📊 Greeks & Liquidity:",
        f"    IV       : {signal['iv']:.1f}%",
        f"    OI       : {signal['oi']:,}",
        f"    Volume   : {signal['volume']:,}",
        f"",
        f"  💰 Position Size:",
        f"    Lot Size : {LOT_SIZES.get(signal['index'], 0)} qty",
        f"    Premium  : ₹{signal['entry'] * LOT_SIZES.get(signal['index'], 0):,.0f} per lot",
        f"",
        f"  ⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


def format_telegram_signal(signal: Dict) -> str:
    """Format option signal for Telegram."""
    lot_size = LOT_SIZES.get(signal['index'], 0)
    premium_per_lot = signal['entry'] * lot_size
    
    return (
        f"🎯 <b>{signal['strategy'].upper()}</b>\n"
        f"<b>{signal['index']} {signal['strike']} {signal['type']}</b> ({signal['moneyness']})\n\n"
        f"💰 Entry: ₹{signal['entry']:.2f}\n"
        f"🎯 Target: ₹{signal['target']:.2f} (+{(signal['target']/signal['entry']-1)*100:.0f}%)\n"
        f"📊 R:R: 1:{signal['rr']:.1f}\n\n"
        f"📈 Delta: {signal['delta']:.2f}\n"
        f"📉 IV: {signal['iv']:.1f}%\n"
        f"📊 OI: {signal['oi']:,} | Vol: {signal['volume']:,}\n\n"
        f"💵 Premium: ₹{premium_per_lot:,.0f}/lot ({lot_size} qty)\n"
        f"📅 Expiry: {signal['expiry']} ({signal['dte']} DTE)\n\n"
        f"⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}"
    )


def format_option_chain_summary(oc: dict) -> str:
    """Return a concise one-block summary of a parsed option chain."""
    spot    = oc.get("underlying_price", 0)
    pcr     = oc.get("pcr", 0)
    expiry  = oc.get("current_expiry", "N/A")
    atm     = oc.get("atm_strike", 0)
    ce_ltp  = oc.get("atm_ce_ltp", 0)
    pe_ltp  = oc.get("atm_pe_ltp", 0)
    res     = oc.get("resistance", 0)
    sup     = oc.get("support", 0)
    stale   = " [STALE]" if oc.get("is_stale") else ""
    lines = [
        f"  Option Chain{stale} - expiry {expiry}",
        f"    Spot: Rs.{spot:,.2f}  |  ATM: {atm}  |  PCR: {pcr:.2f}",
        f"    ATM CE: Rs.{ce_ltp:.2f}  |  ATM PE: Rs.{pe_ltp:.2f}",
        f"    Resistance (max CE OI): {res}  |  Support (max PE OI): {sup}",
    ]
    return "\n".join(lines)


def _print_both_sides(symbol: str, option_chain: dict,
                       mtf: dict, symbol_signals: dict) -> None:
    """
    Always print a CE vs PE summary table after strategies run.
    This shows the user BOTH sides regardless of whether a signal fired.

    symbol_signals: {"BUY_CE": [...], "BUY_PE": [...]}
    """
    if not option_chain:
        return

    spot     = option_chain.get("underlying_price", 0)
    atm      = option_chain.get("atm_strike", 0)
    ce_ltp   = option_chain.get("atm_ce_ltp", 0)
    pe_ltp   = option_chain.get("atm_pe_ltp", 0)
    ce_oi    = option_chain.get("atm_ce_oi", 0)
    pe_oi    = option_chain.get("atm_pe_oi", 0)
    pcr      = option_chain.get("pcr", 1.0)
    expiry   = option_chain.get("current_expiry", "")
    mtf_bias = mtf.get("bias", "neutral")

    # Status for each side
    def _side_status(opt_type: str, direction: str) -> str:
        if opt_type in symbol_signals:
            best_conf = max(s.confidence for s in symbol_signals[opt_type])
            return f"SIGNAL ✅  ({best_conf:.0%} conf)"
        elif mtf_bias == direction:
            return "MONITOR ⚠️  (MTF aligned — no entry yet)"
        else:
            return "AVOID ❌  (MTF not aligned)"

    ce_status = _side_status("BUY_CE", "bullish")
    pe_status = _side_status("BUY_PE", "bearish")

    # PCR leaning
    pcr_hint = (
        "bullish (PUT writers dominate)" if pcr > 1.1
        else "bearish (CALL writers dominate)" if pcr < 0.9
        else "neutral"
    )

    sep = "─" * 55
    print(f"\n  {sep}")
    print(f"  BOTH SIDES  |  ATM {atm}  |  Expiry {expiry}")
    print(f"  Spot: Rs.{spot:,.2f}  |  PCR: {pcr:.2f}  ({pcr_hint})")
    print(f"  {sep}")
    print(f"  📈 CALL (CE)  Strike {atm}  |  Premium Rs.{ce_ltp:.2f}"
          + (f"  |  OI {ce_oi:,}" if ce_oi else ""))
    print(f"     Status  : {ce_status}")
    print(f"  📉 PUT  (PE)  Strike {atm}  |  Premium Rs.{pe_ltp:.2f}"
          + (f"  |  OI {pe_oi:,}" if pe_oi else ""))
    print(f"     Status  : {pe_status}")
    print(f"  {sep}")

    # If BOTH have signals — highlight it clearly
    if "BUY_CE" in symbol_signals and "BUY_PE" in symbol_signals:
        print(f"  ⚡ BOTH SIDES ACTIVE — consider STRADDLE")
        print(f"     Straddle cost: Rs.{ce_ltp + pe_ltp:.2f}  |  "
              f"Lot Rs.{(ce_ltp + pe_ltp) * LOT_SIZES.get(symbol, 75):,.0f}")
        print(f"     Break-even: spot moves > Rs.{ce_ltp + pe_ltp:.0f} "
              f"in either direction")
    print()


async def scan_options(
    symbols: List[str],
    interval: str,
    strategies: list,
    send_telegram: bool = False,
    dry_run: bool = False,
    chart_dir: str = "/tmp",
    oc_fetcher=None,          # Shared fetcher — pass from main() to reuse browser
) -> List[TradingSignal]:
    """Scan indices for options trading signals with chart images."""
    all_signals = []
    _own_fetcher = oc_fetcher is None
    if _own_fetcher:
        oc_fetcher = OptionChainFetcher()
    alert_formatter = AlertFormatter()
    visualizer = ChartVisualizer()
    Path(chart_dir).mkdir(parents=True, exist_ok=True)

    # Telegram setup
    telegram = None
    if send_telegram and not dry_run:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if TelegramBot and bot_token and chat_id:
            telegram = TelegramBot(bot_token, chat_id)
            logger.info("Telegram delivery enabled for options scan")
        else:
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "signals will be logged only"
            )
    elif dry_run:
        logger.info("Dry-run mode — Telegram delivery disabled")

    print(f"\nOptions Scanner — {len(symbols)} index/indices, {len(strategies)} strategies")
    print(f"Interval : {interval}  |  Chart dir: {chart_dir}")
    print(f"{'='*55}")

    for symbol in symbols:
        symbol = symbol.upper()
        print(f"\n  Scanning {symbol}...")

        # 1. Fetch option chain
        print("  Fetching option chain...")
        option_chain = await oc_fetcher.fetch_option_chain(symbol)

        if option_chain:
            print(format_option_chain_summary(option_chain))
            # DTE guard — skip if nearest expiry is >10 calendar days away
            # (far-out options have high premiums; not suitable for intraday)
            oc_expiry = option_chain.get("current_expiry", "")
            try:
                from datetime import date as _date_cls
                exp_date = datetime.strptime(oc_expiry, "%d-%b-%Y").date()
                dte_days = (exp_date - _date_cls.today()).days
            except Exception:
                dte_days = 0
            # DTE limits: weekly indices (NIFTY/BANKNIFTY) must be ≤10 days.
            # SENSEX trades monthly on BSE — allow up to 35 days.
            _SENSEX_INDICES = {"SENSEX", "BANKEX"}
            max_dte = 35 if symbol in _SENSEX_INDICES else 10
            if dte_days > max_dte:
                print(
                    f"  SKIP {symbol}: nearest expiry is {oc_expiry}"
                    f" ({dte_days} days) — too far for intraday options."
                    f"  (Near-term weekly options may not be listed in broker.)"
                )
                continue
        else:
            print(f"  WARNING: Could not fetch option chain for {symbol}")
            print("  (NSE may block requests. OI-based strategies will be skipped.)")
            option_chain = {}

        # 2. Fetch 5m AND 15m candles (multi-timeframe analysis)
        print(f"  Fetching 5m + 15m candle data...")
        df_5m  = fetch_index_data(symbol, "5m")
        df_15m = fetch_index_data(symbol, "15m")
        # Use 5m as primary df for strategies
        df = df_5m

        if df is None or len(df) < 20:
            print(f"  ERROR: Insufficient price data for {symbol}")
            continue

        print(
            f"  Got {len(df)} x5m candles, "
            f"latest close: Rs.{float(df['close'].iloc[-1]):,.2f}"
        )

        # 3. Compute multi-timeframe bias (15m + 5m)
        mtf = _compute_mtf_bias(df_5m, df_15m)
        mtf_emoji = "🟢" if mtf["bias"] == "bullish" else (
                    "🔴" if mtf["bias"] == "bearish" else "🟡")
        print(
            f"  MTF Bias: {mtf_emoji} {mtf['bias'].upper()}"
            f"  (15m: {mtf['details'].get('15m_ema_trend','?').upper()}"
            f" | ST: {mtf['details'].get('15m_supertrend','?').upper()})"
        )

        # 4. Build company_info with option chain
        company_info = {
            "name": symbol,
            "symbol": symbol,
            "market_cap": 0,
            "last_price": float(df["close"].iloc[-1]),
            "option_chain": option_chain,
        }

        # 5. Run strategies — collect signals per option_type for consensus
        symbol_signals: Dict[str, list] = {}   # option_type -> [signals]
        for strategy in strategies:
            try:
                signal = strategy.scan(symbol, df, company_info)
                if not signal:
                    _non_reason = {"total", "signals"}
                    stats = getattr(strategy, "_scan_stats", {})
                    skip_reason = next(
                        (k for k, v in stats.items()
                         if v > 0 and k not in _non_reason),
                        "conditions not met",
                    )
                    print(f"    [{strategy.name}] No signal — {skip_reason}")
                    continue

                opt_type = signal.metadata.get("option_type", "")
                symbol_signals.setdefault(opt_type, []).append(signal)
                print(
                    f"    [{strategy.name}] SIGNAL: {opt_type}"
                    f" | Conf: {signal.confidence:.0%}"
                )
            except Exception as e:
                logger.debug(f"{symbol}/{strategy.name}: {e}")

        # 5b. Always show BOTH CE and PE status so user sees the full picture
        _print_both_sides(symbol, option_chain, mtf, symbol_signals)

        # 6. Pick best signal per option_type; boost confidence if >1 strategy agrees
        for opt_type, sig_list in symbol_signals.items():
            # Sort by confidence descending — best signal leads
            sig_list.sort(key=lambda s: s.confidence, reverse=True)
            best = sig_list[0]
            n_agree = len(sig_list)

            # Confidence boost for consensus (Van Tharp: system edge compounds)
            if n_agree >= 2:
                best.confidence = min(best.confidence + 0.08 * (n_agree - 1), 1.0)
                best.metadata["strategies_agreed"] = n_agree
                best.metadata["strategies_names"] = (
                    ", ".join(s.strategy_name for s in sig_list)
                )
                print(
                    f"  ⚡ Consensus: {n_agree} strategies agree on {opt_type}"
                    f" — confidence boosted to {best.confidence:.0%}"
                )

            all_signals.append(best)

            # Enrich with ATR14
            if not best.metadata.get("atr_pct"):
                atr14 = _compute_atr14(df)
                if atr14 > 0:
                    best.metadata["atr"] = round(atr14, 4)
                    best.metadata["atr_pct"] = (
                        round(atr14 / best.entry_price * 100, 2)
                        if best.entry_price > 0 else 0.0
                    )

            # ── Quality check (MTF + conditions) ──────────────────────
            quality = _trade_quality_check(best, option_chain, mtf)

            # ── Signal deduplication ───────────────────────────────────
            if _check_duplicate_signal(symbol, opt_type):
                print(
                    f"  [SKIP] {symbol} {opt_type} already sent "
                    f"within {_SIGNAL_COOLDOWN_MIN} min — skipping Telegram"
                )
                # Still print console output
                output = _console_signal_summary(best)
                print(output)
                continue

            # ── Console output ─────────────────────────────────────────
            output = _console_signal_summary(best)
            print(output)
            print(
                f"  Trade Quality: {'✅ TRADE' if quality['overall_ok'] else '⚠️  WAIT'}"
                f"  Score: {quality['score']}/100"
            )
            if quality["reasons_avoid"]:
                for r in quality["reasons_avoid"]:
                    print(f"    ⚠️  {r}")

            # ── Generate chart image ───────────────────────────────────
            chart_path = str(
                Path(chart_dir)
                / f"options_{symbol}_{best.strategy_name.replace(' ', '_')}.png"
            )
            chart_saved = await asyncio.to_thread(
                visualizer.save_signal_chart, df, best, chart_path
            )

            # ── Expert Telegram message ────────────────────────────────
            tg_message = format_expert_telegram(
                best, quality, option_chain,
                strategies_agreed=n_agree,
            )

            # Always print the full Telegram template to console so user
            # can see exactly what would be sent (with or without --telegram)
            print("\n--- TELEGRAM MESSAGE PREVIEW ---")
            print(tg_message)
            print("--- END PREVIEW ---\n")

            # ── Send via Telegram ──────────────────────────────────────
            if telegram and not dry_run:
                try:
                    await telegram.send_alert(
                        tg_message,
                        best.priority.value,
                        image_path=chart_path if chart_saved else None,
                    )
                    _mark_signal_sent(symbol, opt_type)
                    logger.info(
                        f"Telegram alert sent: {symbol} {opt_type} "
                        f"{best.metadata.get('atm_strike', '')} "
                        f"conf={best.confidence:.0%}"
                    )
                except Exception as tg_exc:
                    logger.warning(f"Telegram send failed: {tg_exc}")
            else:
                # Not sending (no --telegram flag); still mark to avoid
                # duplicate console spam on next repeat scan
                _mark_signal_sent(symbol, opt_type)

            # Cleanup chart after delivery
            if chart_saved and (telegram or dry_run) and not dry_run:
                try:
                    os.remove(chart_path)
                except OSError:
                    pass

    if _own_fetcher:
        await oc_fetcher.close()
    return all_signals


# Map index names to their Yahoo Finance tickers
_INDEX_YAHOO_MAP = {
    "NIFTY":      "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "FINNIFTY":   "^CNXFIN",
    "MIDCPNIFTY": "^NSEMDCP50",
    "SENSEX":     "^BSESN",
}


def fetch_index_data(symbol: str, interval: str = "5m") -> Optional[pd.DataFrame]:
    """Fetch intraday OHLCV data for an index or stock via yfinance.

    NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY are mapped to their Yahoo Finance
    index tickers (^NSEI etc.).  Any other symbol is treated as an NSE equity
    and the '.NS' suffix is appended.
    """
    yahoo_symbol = _INDEX_YAHOO_MAP.get(symbol.upper(), f"{symbol}.NS")
    # Use short periods — we only need ~100-200 candles for indicators
    # (EMA21, MACD26, ATR14, Supertrend all need << 100 bars)
    period_map = {"1m": "3d", "5m": "5d", "15m": "5d", "30m": "10d", "1h": "60d"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period_map.get(interval, "60d"), interval=interval)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def load_options_strategies() -> list:
    """Load all options strategies from config files."""
    strategies = []
    for config_name in [
        "options_oi_breakout",
        "options_pcr_sentiment",
        "options_vwap_supertrend",
    ]:
        try:
            config = load_strategy_config(config_name)
            if not config:
                continue
            strategy_name = config.get("strategy", {}).get("name", "")
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_class:
                strategies.append(strategy_class(config))
            else:
                logger.debug(f"No class in registry for strategy: {strategy_name!r}")
        except Exception as exc:
            logger.debug(f"Failed to load {config_name}: {exc}")
    return strategies


async def main():
    args = parse_args()

    symbols = args.symbol if args.symbol else ["NIFTY"]

    strategies = load_options_strategies()
    if not strategies:
        print("Error: No options strategies loaded")
        sys.exit(1)

    print(f"\nStrategies: {', '.join(s.name for s in strategies)}")

    if args.repeat > 0:
        repeat_min = args.repeat
        print(f"\nAUTO-REPEAT: Scanning every {repeat_min} min  |  Press Ctrl+C to stop")
        print(f"Market hours: 9:15–15:30 IST  |  Entry window: 9:30–15:00 IST\n")

        scan_count   = 0
        last_status  = ""   # track last market-status message (avoid spam)

        # ONE shared fetcher — keeps SmartAPI session + browser alive
        shared_fetcher = OptionChainFetcher()
        try:
            while True:
                now_ist = datetime.now(IST)
                mkt_open, mkt_reason = _is_market_open()

                if not mkt_open:
                    # Print status once per change so log isn't flooded
                    if mkt_reason != last_status:
                        print(f"\n[{now_ist.strftime('%H:%M')} IST]  {mkt_reason} — waiting...")
                        last_status = mkt_reason
                    try:
                        # Sleep 1 min while market is closed, recheck
                        time.sleep(60)
                    except KeyboardInterrupt:
                        print("\nStopped.")
                        break
                    continue

                last_status = mkt_reason
                scan_count += 1
                print(f"\n{'#'*55}")
                print(
                    f"  OPTIONS SCAN #{scan_count} — "
                    f"{now_ist.strftime('%H:%M:%S IST')}"
                )
                # Show entry window status
                entry_ok, entry_msg = _is_entry_allowed()
                print(
                    f"  Entry Window: {'✅ OPEN' if entry_ok else '⛔ ' + entry_msg}"
                )
                print(f"{'#'*55}")

                signals = await scan_options(
                    symbols, args.interval, strategies,
                    send_telegram=args.telegram,
                    dry_run=args.dry_run,
                    chart_dir=args.chart_dir,
                    oc_fetcher=shared_fetcher,
                )

                if signals:
                    n_trade = sum(
                        1 for s in signals
                        if _trade_quality_check(
                            s,
                            {},
                            _compute_mtf_bias(None, None),
                        ).get("overall_ok", False)
                    )
                    print(
                        f"\n  SIGNALS: {len(signals)} found"
                        f"  ({n_trade} actionable)"
                    )
                else:
                    print(f"\n  No signals. Next scan in {repeat_min} min...")

                try:
                    time.sleep(repeat_min * 60)
                except KeyboardInterrupt:
                    print("\nStopped.")
                    break
        finally:
            await shared_fetcher.close()
    else:
        signals = await scan_options(
            symbols, args.interval, strategies,
            send_telegram=args.telegram,
            dry_run=args.dry_run,
            chart_dir=args.chart_dir,
        )

        print(f"\n{'='*55}")
        print(f"  OPTIONS SCAN SUMMARY")
        print(f"{'='*55}")
        print(f"  Indices Scanned : {len(symbols)}")
        print(f"  Strategies      : {len(strategies)}")
        print(f"  Signals Found   : {len(signals)}")

        if signals:
            for s in signals:
                option_type = s.metadata.get("option_type", "")
                atm = s.metadata.get("atm_strike", 0)
                print(
                    f"    {option_type}: {s.symbol} {atm} "
                    f"| {s.strategy_name} "
                    f"| Spot: Rs.{s.entry_price:,.2f} "
                    f"| Conf: {s.confidence*100:.0f}%"
                )
        else:
            print(f"  No options signals found. Market may be range-bound.")

        print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(main())
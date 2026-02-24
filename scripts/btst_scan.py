"""
BTST Scanner — Buy Today, Sell Tomorrow

Purpose:
    Dedicated runner for the 4 BTST pattern strategies. Designed to
    be launched at 15:15 IST so signals are ready before the 15:30
    market close.

Strategies:
    1. Darvas Box           (weight 0.38) — volume breakout above Darvas ceiling
    2. Flag Pattern         (weight 0.42) — pole + tight-flag measured move
    3. Symmetrical Triangle (weight 0.35) — converging trendlines breakout
    4. Descending Channel   (weight 0.33) — pullback channel breakout in uptrend

Usage:
    # Standard run (requires TELEGRAM_* env vars for delivery)
    python scripts/btst_scan.py

    # Force run on non-trading days (back-test / dev)
    python scripts/btst_scan.py --force

    # Dry-run: log signals without sending Telegram alerts
    python scripts/btst_scan.py --dry-run

    # Run only specific BTST strategies
    python scripts/btst_scan.py --strategies "Darvas Box" "Flag Pattern"

    # Save charts to a custom directory (default: /tmp)
    python scripts/btst_scan.py --chart-dir ./charts/btst

    # Combine options
    python scripts/btst_scan.py --force --dry-run --chart-dir ./charts/btst

    # Show per-symbol rejection summary at the end
    python scripts/btst_scan.py --verbose-stats

Environment variables:
    TELEGRAM_BOT_TOKEN   — Telegram bot token
    TELEGRAM_CHAT_ID     — Telegram chat / channel ID
    FORCE_RUN=true       — same as --force

Output:
    Signals printed to stdout and logger.
    PNG charts saved to --chart-dir (if Kaleido is installed).
    Alerts sent to Telegram (if token/chat configured and not --dry-run).
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.alert_deduplicator import AlertDeduplicator
from src.alerts.alert_formatter import AlertFormatter
from src.utils.visualizer import ChartVisualizer

try:
    from src.alerts.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None

from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.fallback_manager import FallbackManager
from src.engine.ranking_engine import rank_signals
from src.engine.risk_filter import filter_signals
from src.engine.signal_aggregator import aggregate_signals
from src.engine.strategy_executor import _process_single_stock
from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal
from src.strategies.strategy_loader import StrategyLoader
from src.storage.redis_handler import RedisHandler
from src.utils.config_loader import load_config
from src.utils.time_helpers import is_trading_day, now_ist

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
BTST_STRATEGIES = [
    "Darvas Box",
    "Flag Pattern",
    "Symmetrical Triangle",
    "Descending Channel",
]

_SIGNAL_ICONS = {
    "Darvas Box": "📦",
    "Flag Pattern": "🚩",
    "Symmetrical Triangle": "△",
    "Descending Channel": "↘",
}


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTST Scanner — Buy Today, Sell Tomorrow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/btst_scan.py\n"
            "  python scripts/btst_scan.py --force --dry-run\n"
            "  python scripts/btst_scan.py --strategies 'Darvas Box' 'Flag Pattern'\n"
            "  python scripts/btst_scan.py --chart-dir ./charts/btst --verbose-stats\n"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even on non-trading days (dev / back-test mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals but do NOT send Telegram alerts",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        metavar="STRATEGY",
        default=BTST_STRATEGIES,
        help=(
            "BTST strategies to run. "
            f"Defaults to all 4: {BTST_STRATEGIES}"
        ),
    )
    parser.add_argument(
        "--chart-dir",
        type=str,
        default="/tmp",
        metavar="DIR",
        help="Directory where PNG chart images are saved (default: /tmp)",
    )
    parser.add_argument(
        "--verbose-stats",
        action="store_true",
        help="Print per-strategy rejection stats after the scan",
    )
    return parser.parse_args()


# ── Main scan ──────────────────────────────────────────────────────────────

async def run_btst_scan(
    force_run: bool = False,
    dry_run: bool = False,
    strategy_names: Optional[List[str]] = None,
    chart_dir: str = "/tmp",
    verbose_stats: bool = False,
) -> Dict[str, Any]:
    """
    Execute the BTST scan pipeline.

    Args:
        force_run:      Skip trading-day check.
        dry_run:        Skip Telegram delivery (log only).
        strategy_names: Which BTST strategies to run.
        chart_dir:      Directory to write chart PNGs.
        verbose_stats:  Print per-strategy rejection stats.

    Returns:
        Summary dict with stocks_scanned, signals_generated, alerts_sent, errors.
    """
    start_time = time.time()
    scan_date = now_ist()
    names = strategy_names or BTST_STRATEGIES

    logger.info(
        f"[BTST] Starting scan for {scan_date.date()} "
        f"| strategies: {names}"
    )

    # ── Trading-day guard ──────────────────────────────────────────────────
    force_env = os.environ.get("FORCE_RUN", "").lower() in ("true", "1", "yes")
    should_force = force_run or force_env

    if not should_force and not is_trading_day(scan_date.date()):
        logger.info(
            "[BTST] Not a trading day — use --force or FORCE_RUN=true to override"
        )
        return {"status": "skipped", "reason": "not_trading_day"}

    if should_force and not is_trading_day(scan_date.date()):
        logger.info("[BTST] Non-trading day — running in force mode")

    results: Dict[str, Any] = {
        "scan_date": str(scan_date.date()),
        "strategies": names,
        "stocks_scanned": 0,
        "signals_generated": 0,
        "alerts_sent": 0,
        "errors": 0,
        "status": "running",
    }

    fallback_manager: Optional[FallbackManager] = None

    try:
        # ── Load config ────────────────────────────────────────────────────
        config = load_config("system")

        # ── Load strategies ────────────────────────────────────────────────
        loader = StrategyLoader()
        all_daily = loader.load_by_mode("daily")
        strategies = [s for s in all_daily if s.name in names]

        if not strategies:
            logger.error(
                f"[BTST] None of the requested strategies found: {names}. "
                "Check that YAML configs exist in config/strategies/."
            )
            return {**results, "status": "error", "reason": "no_strategies"}

        loaded_names = [s.name for s in strategies]
        logger.info(f"[BTST] Loaded {len(strategies)} strategies: {loaded_names}")

        # ── Init components ────────────────────────────────────────────────
        fallback_manager = FallbackManager()
        data_validator = DataValidator()
        alert_formatter = AlertFormatter()
        visualizer = ChartVisualizer()
        redis_handler = RedisHandler()
        deduplicator = AlertDeduplicator(redis_handler)

        # Ensure chart output directory exists
        Path(chart_dir).mkdir(parents=True, exist_ok=True)

        # Telegram (optional)
        telegram = None
        if not dry_run and TelegramBot is not None:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if bot_token and chat_id:
                telegram = TelegramBot(bot_token, chat_id)
                logger.info("[BTST] Telegram delivery enabled")
            else:
                logger.warning(
                    "[BTST] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                    "signals will be logged only"
                )
        elif dry_run:
            logger.info("[BTST] Dry-run mode — Telegram delivery disabled")

        # ── Stock universe ─────────────────────────────────────────────────
        from scripts.daily_scan import _compute_atr14, _get_stock_universe
        stock_list = await _get_stock_universe(fallback_manager, config)

        if not stock_list:
            logger.error("[BTST] Failed to retrieve stock universe")
            return {**results, "status": "error", "reason": "no_stocks"}

        logger.info(f"[BTST] Scanning {len(stock_list)} stocks")

        # ── Scan loop ──────────────────────────────────────────────────────
        import pandas as pd

        all_signals: List[TradingSignal] = []
        chart_paths: Dict[str, str] = {}
        symbol_dfs: Dict[str, Any] = {}   # symbol -> df.copy()
        symbol_sigs: Dict[str, Any] = {}  # symbol -> first TradingSignal
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        chunk_size = config.get("scanning", {}).get("chunk_size", 50)

        scan_stats = {
            "no_data": 0,
            "insufficient_records": 0,
            "strategy_scanned": 0,
            "signals_found": 0,
        }

        for i in range(0, len(stock_list), chunk_size):
            chunk = stock_list[i: i + chunk_size]
            chunk_num = i // chunk_size + 1
            logger.info(
                f"[BTST] Chunk {chunk_num}: "
                f"stocks {i + 1}–{min(i + chunk_size, len(stock_list))}"
            )

            chunk_signals = 0
            for symbol in chunk:
                try:
                    data = await fallback_manager.fetch_stock_data(
                        symbol, start_date, end_date
                    )
                    if not data or not data.get("records"):
                        scan_stats["no_data"] += 1
                        continue

                    clean_records = data_validator.clean_records(
                        data["records"], symbol
                    )
                    if len(clean_records) < 60:
                        scan_stats["insufficient_records"] += 1
                        continue

                    df = pd.DataFrame(clean_records)
                    if "date" in df.columns:
                        df.set_index("date", inplace=True)

                    last_price = (
                        float(df["close"].iloc[-1])
                        if "close" in df.columns
                        else 0.0
                    )
                    company_info = {
                        "name": symbol,
                        "symbol": symbol,
                        "sector": "Unknown",
                        "market_cap": 0,
                        "last_price": last_price,
                        # "profit_growth_pct": <float>  ← populate from a
                        # screener / fundamental data source to activate the
                        # Financial Health Check in apply_pre_filters().
                        # When absent the filter is skipped (safe default).
                    }

                    scan_stats["strategy_scanned"] += 1
                    signals = _process_single_stock(
                        symbol, df, company_info, strategies
                    )

                    if signals:
                        chunk_signals += len(signals)
                        scan_stats["signals_found"] += len(signals)
                        # Compute ATR14 once per symbol while df is in scope
                        atr14 = _compute_atr14(df)
                        for sig in signals:
                            icon = _SIGNAL_ICONS.get(
                                sig.strategy_name, "📈"
                            )
                            logger.info(
                                f"[BTST] {icon} SIGNAL: "
                                f"{sig.strategy_name} → {sig.symbol} "
                                f"entry={sig.entry_price:.2f} "
                                f"target={sig.target_price:.2f} "
                                f"SL={sig.stop_loss:.2f} "
                                f"conf={sig.confidence:.0%}"
                            )
                            # Enrich metadata with ATR if strategy didn't provide it
                            if atr14 > 0 and not sig.metadata.get("atr_pct"):
                                sig.metadata["atr"] = round(atr14, 4)
                                sig.metadata["atr_pct"] = (
                                    round(atr14 / sig.entry_price * 100, 2)
                                    if sig.entry_price > 0 else 0.0
                                )

                        # Cache df + signal; charts are generated later
                        # only for the signals that survive filtering.
                        if symbol not in symbol_dfs:
                            symbol_dfs[symbol] = df.copy()
                            symbol_sigs[symbol] = signals[0]

                    all_signals.extend(signals)
                    results["stocks_scanned"] += 1

                except (asyncio.CancelledError, KeyboardInterrupt):
                    logger.info("[BTST] Scan interrupted")
                    results["status"] = "interrupted"
                    break
                except Exception as e:
                    results["errors"] += 1
                    logger.error(
                        f"[BTST] Error processing {symbol}: {e}",
                        exc_info=True,
                    )
            else:
                logger.info(
                    f"[BTST] Chunk {chunk_num} done: "
                    f"{chunk_signals} signals"
                )
                continue
            break  # interrupted

        # ── Verbose per-strategy stats ─────────────────────────────────────
        if verbose_stats:
            print(f"\n{'─'*60}")
            print("  Per-strategy rejection stats")
            print(f"{'─'*60}")
            for s in strategies:
                if hasattr(s, "get_scan_stats"):
                    st = s.get_scan_stats()
                    print(
                        f"  {s.name:<24} "
                        f"total={st.get('total', 0):>5}  "
                        f"trend_rejected={st.get('trend_rejected', 0):>5}  "
                        f"overbought={st.get('overbought_rejected', 0):>5}  "
                        f"no_pattern={st.get('no_pattern', 0):>5}  "
                        f"vol_rejected={st.get('volume_rejected', 0):>5}  "
                        f"signals={st.get('signals', 0):>4}"
                    )
            print(f"{'─'*60}\n")

        # ── Aggregate → rank → filter ──────────────────────────────────────
        if not all_signals:
            elapsed = time.time() - start_time
            logger.info(
                f"[BTST] No signals found. "
                f"Scanned {results['stocks_scanned']} stocks "
                f"in {elapsed:.1f}s"
            )
            return {
                **results,
                "status": "ok",
                "elapsed_seconds": round(elapsed, 1),
            }

        aggregated = aggregate_signals(all_signals)
        ranked = rank_signals(aggregated)
        filtered = filter_signals(ranked)
        results["signals_generated"] = len(filtered)

        # ── Pretty-print summary ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  BTST SCAN RESULTS — {scan_date.strftime('%Y-%m-%d %H:%M IST')}")
        print(f"  Stocks scanned : {results['stocks_scanned']}")
        print(f"  Signals found  : {len(filtered)}")
        print(f"{'='*60}")

        for i, sig in enumerate(filtered, 1):
            strat = (
                sig.contributing_strategies[0]
                if sig.contributing_strategies
                else "Unknown"
            )
            icon = _SIGNAL_ICONS.get(strat, "📈")
            rr = None
            for ind in sig.individual_signals:
                rr = ind.get("metadata", {}).get("rr_ratio") or rr
            rr_str = f"R:R 1:{rr}" if rr else ""
            print(
                f"  {i:>2}. {icon} {sig.symbol:<12} "
                f"[{strat}]  "
                f"Entry: Rs.{sig.entry_price:>8.2f}  "
                f"Target: Rs.{sig.target_price:>8.2f}  "
                f"SL: Rs.{sig.stop_loss:>8.2f}  "
                f"Conf: {sig.weighted_confidence:.0%}  "
                f"{rr_str}"
            )
        print(f"{'='*60}\n")

        # ── Alert delivery ─────────────────────────────────────────────────
        for signal in filtered:
            try:
                signal_strategy = (
                    signal.contributing_strategies[0]
                    if signal.contributing_strategies
                    else "unknown"
                )
                if not force_run and deduplicator.is_duplicate(signal.symbol, signal_strategy):
                    logger.debug(
                        f"[BTST] Skipping duplicate: {signal.symbol}"
                    )
                    continue

                # ── Canonical prices: defined ONCE, passed to both text + chart ──
                entry_price = signal.entry_price
                target_price = signal.target_price
                stop_loss = signal.stop_loss

                signal_dict = signal.to_dict()
                raw_conf = 0.0
                best_met: Any = "N/A"
                best_total: Any = "N/A"
                for ind in signal.individual_signals:
                    ic = ind.get("confidence", 0)
                    if ic > raw_conf:
                        raw_conf = ic
                        best_met = ind.get("indicators_met", "N/A")
                        best_total = ind.get("total_indicators", "N/A")
                signal_dict["confidence"] = round(raw_conf * 100, 1)
                signal_dict["indicators_met"] = best_met
                signal_dict["total_indicators"] = best_total
                signal_dict["individual_signals"] = signal.individual_signals
                # Ensure text message always uses the canonical prices
                signal_dict["entry_price"] = entry_price
                signal_dict["target_price"] = target_price
                signal_dict["stop_loss"] = stop_loss

                message = alert_formatter.format_buy_signal(signal_dict)

                # Generate chart only for this final signal
                if signal.symbol not in chart_paths and signal.symbol in symbol_dfs:
                    _df = symbol_dfs[signal.symbol]
                    _sig = symbol_sigs[signal.symbol]
                    # Sync chart signal to canonical prices so both outputs match
                    _sig.entry_price = entry_price
                    _sig.target_price = target_price
                    _sig.stop_loss = stop_loss
                    _tmp = str(Path(chart_dir) / f"btst_{signal.symbol}.png")
                    _ok = await asyncio.to_thread(
                        visualizer.save_signal_chart, _df, _sig, _tmp
                    )
                    if _ok:
                        chart_paths[signal.symbol] = _tmp
                        logger.info(f"[BTST] Chart generated for {signal.symbol}")
                    else:
                        logger.warning(
                            f"[BTST] Chart generation failed for {signal.symbol}"
                        )

                chart_path = chart_paths.get(signal.symbol)

                if telegram and not dry_run:
                    sent = await telegram.send_alert(
                        message,
                        signal.priority.value,
                        image_path=chart_path,
                    )
                else:
                    logger.info(f"[BTST] ALERT (dry-run): {message}")
                    sent = True

                if sent:
                    deduplicator.mark_sent(signal.symbol, signal_strategy)
                    results["alerts_sent"] += 1
                    if chart_path and os.path.isfile(chart_path):
                        try:
                            os.remove(chart_path)
                        except OSError:
                            pass

            except Exception as e:
                logger.error(
                    f"[BTST] Alert delivery failed for {signal.symbol}: {e}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"[BTST] Fatal scan error: {e}", exc_info=True)
        results["status"] = "error"
        results["error_detail"] = str(e)
        return results

    finally:
        if fallback_manager:
            try:
                await fallback_manager.close()
            except Exception:
                pass

    elapsed = time.time() - start_time
    results["status"] = results.get("status", "ok") if results.get("status") != "running" else "ok"
    results["elapsed_seconds"] = round(elapsed, 1)

    logger.info(
        f"[BTST] Scan complete in {elapsed:.1f}s — "
        f"{results['stocks_scanned']} stocks, "
        f"{results['signals_generated']} signals, "
        f"{results['alerts_sent']} alerts sent"
    )
    return results


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_btst_scan(
            force_run=args.force,
            dry_run=args.dry_run,
            strategy_names=args.strategies,
            chart_dir=args.chart_dir,
            verbose_stats=args.verbose_stats,
        )
    )

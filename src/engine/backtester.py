"""
Historical Backtesting Framework

Purpose:
    Runs trading strategies against historical data to evaluate
    performance. Models real-world costs including commission
    and slippage. Calculates key performance metrics.

Dependencies:
    - pandas for time-series handling
    - numpy for statistical calculations
    - strategies.base_strategy for strategy interface

Logging:
    - Backtest start/end at INFO
    - Trade execution at DEBUG
    - Metric calculation at DEBUG

Fallbacks:
    If insufficient data for a date range, returns empty metrics.
    Individual trade errors do not abort the entire backtest.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    job_duration_histogram,
    job_success_counter,
    job_failure_counter,
)
from src.strategies.base_strategy import BaseStrategy, TradingSignal
from src.utils.config_loader import load_config, get_nested
from src.utils.constants import SignalType

logger = get_logger(__name__)

# Cost defaults
DEFAULT_COMMISSION_PCT = 0.03  # 0.03% per trade
DEFAULT_SLIPPAGE_PCT = 0.10   # 0.10% per trade
DEFAULT_INITIAL_CAPITAL = 1_000_000.0  # 10 lakh INR

# Risk-free rate for Sharpe calculation (annualized)
RISK_FREE_RATE = 0.065  # ~6.5% (Indian govt bond yield)

# Trading days per year (NSE)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class Trade:
    """Represents a single completed round-trip trade."""

    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    side: str = "BUY"
    commission: float = 0.0
    slippage_cost: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    strategy_name: str = ""
    exit_reason: str = ""
    # Signal-based exit levels (set at entry time from the strategy signal)
    signal_stop_loss: float = 0.0
    signal_target_price: float = 0.0

    @property
    def is_winner(self) -> bool:
        """Whether this trade was profitable."""
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "symbol": self.symbol,
            "entry_date": (
                self.entry_date.isoformat()
                if self.entry_date
                else None
            ),
            "entry_price": self.entry_price,
            "exit_date": (
                self.exit_date.isoformat()
                if self.exit_date
                else None
            ),
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "side": self.side,
            "commission": round(self.commission, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "pnl": round(self.pnl, 2),
            "return_pct": round(self.return_pct, 4),
            "strategy_name": self.strategy_name,
            "exit_reason": self.exit_reason,
            "signal_stop_loss": self.signal_stop_loss,
            "signal_target_price": self.signal_target_price,
        }


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest run."""

    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    final_portfolio_value: float = 0.0
    initial_capital: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return_pct": round(
                self.annualized_return_pct, 4
            ),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_commission": round(self.total_commission, 2),
            "total_slippage": round(self.total_slippage, 2),
            "final_portfolio_value": round(
                self.final_portfolio_value, 2
            ),
            "initial_capital": round(self.initial_capital, 2),
        }


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""

    metrics: BacktestMetrics = field(
        default_factory=BacktestMetrics
    )
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert full result to dictionary."""
        return {
            "metrics": self.metrics.to_dict(),
            "trade_count": len(self.trades),
            "trades": [t.to_dict() for t in self.trades],
        }


def run_backtest(
    strategy: BaseStrategy,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    df: Optional[pd.DataFrame] = None,
    company_info: Optional[Dict[str, Any]] = None,
) -> BacktestResult:
    """
    Run a historical backtest for a strategy on a single symbol.

    Walks through historical data day by day, calling the
    strategy's scan method. When a BUY signal is generated,
    a position is opened. The position is closed when the
    target is hit, stop-loss is triggered, or a SELL signal
    is produced. Commission and slippage are applied on each
    trade.

    Args:
        strategy: Initialized strategy instance to test.
        symbol: Stock symbol to backtest.
        start_date: Start date for the backtest period.
        end_date: End date for the backtest period.
        initial_capital: Starting capital in INR.
            Defaults to 10,00,000.
        df: Optional pre-loaded OHLCV DataFrame. If not
            provided, data must be fetched separately.
        company_info: Optional company metadata dictionary.

    Returns:
        BacktestResult with metrics, trades, and equity curve.
    """
    import time as _time

    config = load_config("scanner")
    bt_config = config.get("backtest", {})

    commission_pct = get_nested(
        bt_config, "commission_pct", DEFAULT_COMMISSION_PCT
    ) / 100.0
    slippage_pct = get_nested(
        bt_config, "slippage_pct", DEFAULT_SLIPPAGE_PCT
    ) / 100.0

    result = BacktestResult()
    result.metrics.initial_capital = initial_capital

    logger.info(
        f"Starting backtest: {strategy.name} on {symbol} "
        f"({start_date.date()} to {end_date.date()}, "
        f"capital={initial_capital:.0f})",
        extra={
            "strategy": strategy.name,
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": initial_capital,
        },
    )

    run_start = _time.time()

    if df is None or df.empty:
        logger.warning(
            f"{symbol}: No data provided for backtest",
            extra={"symbol": symbol},
        )
        return result

    info = company_info or {"name": symbol, "symbol": symbol}

    # Filter to date range
    try:
        df_bt = _prepare_data(df, start_date, end_date)
    except Exception as e:
        logger.error(
            f"{symbol}: Data preparation failed: {e}",
            exc_info=True,
        )
        return result

    if len(df_bt) < 50:
        logger.warning(
            f"{symbol}: Insufficient data for backtest "
            f"({len(df_bt)} rows, need at least 50)"
        )
        return result

    # Walk-forward simulation
    trades: List[Trade] = []
    cash = initial_capital
    position: Optional[Trade] = None
    equity_curve: List[float] = []

    # Lookback window needed for strategy indicators
    min_lookback = 200

    for i in range(min_lookback, len(df_bt)):
        # Provide the strategy with data up to current day
        window = df_bt.iloc[: i + 1]
        current_bar = df_bt.iloc[i]
        current_date = current_bar.name
        current_close = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        # Track equity
        if position is not None:
            portfolio_value = cash + (
                position.quantity * current_close
            )
        else:
            portfolio_value = cash
        equity_curve.append(portfolio_value)

        # If in a position, check exit conditions
        if position is not None:
            exit_price = None
            exit_reason = ""

            # Use signal stop-loss if available, else fall back to 5% below entry
            sl_level = (
                position.signal_stop_loss
                if position.signal_stop_loss > 0
                else position.entry_price * 0.95
            )
            # Use signal target if available, else fall back to 10% above entry
            target_level = (
                position.signal_target_price
                if position.signal_target_price > 0
                else position.entry_price * 1.10
            )

            # Check stop-loss (use low of the day)
            if current_low <= sl_level:
                exit_price = sl_level
                exit_reason = "stop_loss"

            # Check target (use high of the day)
            elif current_high >= target_level:
                exit_price = target_level
                exit_reason = "target_hit"

            else:
                # Check if strategy generates a SELL signal
                try:
                    signal = strategy.scan(symbol, window, info)
                    if (
                        signal is not None
                        and signal.signal_type
                        in (SignalType.SELL, SignalType.STRONG_SELL)
                    ):
                        exit_price = current_close
                        exit_reason = "sell_signal"
                except Exception as e:
                    logger.debug(
                        f"{symbol}@{current_date}: "
                        f"Strategy scan error (exit check): {e}"
                    )

            if exit_price is not None:
                # Apply slippage on exit
                exit_price_adj = exit_price * (1 - slippage_pct)
                exit_commission = (
                    exit_price_adj
                    * position.quantity
                    * commission_pct
                )

                position.exit_date = (
                    current_date
                    if isinstance(current_date, datetime)
                    else datetime.now(timezone.utc)
                )
                position.exit_price = round(exit_price_adj, 2)
                position.exit_reason = exit_reason

                # Calculate PnL
                gross_pnl = (
                    (exit_price_adj - position.entry_price)
                    * position.quantity
                )
                position.commission += exit_commission
                position.slippage_cost += (
                    abs(exit_price - exit_price_adj)
                    * position.quantity
                )
                position.pnl = round(
                    gross_pnl - position.commission, 2
                )
                position.return_pct = (
                    (position.pnl / (
                        position.entry_price * position.quantity
                    )) * 100.0
                    if position.entry_price > 0
                    and position.quantity > 0
                    else 0.0
                )

                cash += (
                    exit_price_adj * position.quantity
                    - exit_commission
                )
                trades.append(position)

                logger.debug(
                    f"{symbol}@{current_date}: EXIT "
                    f"({exit_reason}) at {exit_price_adj:.2f}, "
                    f"PnL={position.pnl:.2f}",
                )
                position = None
                continue

        # If not in a position, check for entry
        if position is None:
            try:
                signal = strategy.scan(symbol, window, info)
            except Exception as e:
                logger.debug(
                    f"{symbol}@{current_date}: "
                    f"Strategy scan error (entry check): {e}"
                )
                continue

            if signal is not None and signal.signal_type in (
                SignalType.BUY,
                SignalType.STRONG_BUY,
            ):
                # Apply slippage on entry
                entry_price = current_close * (1 + slippage_pct)
                entry_commission_per_share = (
                    entry_price * commission_pct
                )

                # Position sizing: use up to 95% of cash
                usable_cash = cash * 0.95
                cost_per_share = (
                    entry_price + entry_commission_per_share
                )
                quantity = int(usable_cash / cost_per_share)

                if quantity <= 0:
                    logger.debug(
                        f"{symbol}@{current_date}: "
                        f"Insufficient cash for entry "
                        f"(cash={cash:.2f})"
                    )
                    continue

                total_commission = (
                    entry_commission_per_share * quantity
                )
                total_cost = entry_price * quantity + total_commission
                slippage_cost = (
                    abs(entry_price - current_close) * quantity
                )

                # Use signal's actual SL and target for accurate exit simulation
                signal_sl = getattr(signal, "stop_loss", None)
                signal_tgt = getattr(signal, "target", None) or getattr(
                    signal, "target_price", None
                )

                position = Trade(
                    symbol=symbol,
                    entry_date=(
                        current_date
                        if isinstance(current_date, datetime)
                        else datetime.now(timezone.utc)
                    ),
                    entry_price=round(entry_price, 2),
                    quantity=quantity,
                    side="BUY",
                    commission=total_commission,
                    slippage_cost=slippage_cost,
                    strategy_name=strategy.name,
                    signal_stop_loss=signal.stop_loss if signal.stop_loss > 0 else 0.0,
                    signal_target_price=signal.target_price if signal.target_price > 0 else 0.0,
                )

                cash -= total_cost

                logger.debug(
                    f"{symbol}@{current_date}: ENTRY at "
                    f"{entry_price:.2f} x {quantity} "
                    f"(cost={total_cost:.2f})",
                )

    # Close any open position at the end
    if position is not None:
        final_close = float(df_bt.iloc[-1]["close"])
        exit_price_adj = final_close * (1 - slippage_pct)
        exit_commission = (
            exit_price_adj * position.quantity * commission_pct
        )

        position.exit_date = (
            df_bt.index[-1]
            if isinstance(df_bt.index[-1], datetime)
            else datetime.now(timezone.utc)
        )
        position.exit_price = round(exit_price_adj, 2)
        position.exit_reason = "backtest_end"
        gross_pnl = (
            (exit_price_adj - position.entry_price)
            * position.quantity
        )
        position.commission += exit_commission
        position.pnl = round(gross_pnl - position.commission, 2)
        position.return_pct = (
            (position.pnl / (
                position.entry_price * position.quantity
            )) * 100.0
            if position.entry_price > 0
            and position.quantity > 0
            else 0.0
        )

        cash += (
            exit_price_adj * position.quantity - exit_commission
        )
        trades.append(position)

        logger.debug(
            f"{symbol}: Closed open position at backtest end "
            f"(PnL={position.pnl:.2f})"
        )

    # Calculate daily returns from equity curve
    daily_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            daily_ret = (
                (equity_curve[i] - equity_curve[i - 1])
                / equity_curve[i - 1]
            )
            daily_returns.append(daily_ret)

    # Compute metrics
    result.trades = trades
    result.equity_curve = equity_curve
    result.daily_returns = daily_returns
    result.metrics = _calculate_metrics(
        trades, daily_returns, equity_curve, initial_capital
    )

    elapsed = _time.time() - run_start

    logger.info(
        f"Backtest complete: {strategy.name} on {symbol} "
        f"({len(trades)} trades, "
        f"return={result.metrics.total_return_pct:.2f}%, "
        f"sharpe={result.metrics.sharpe_ratio:.2f}, "
        f"max_dd={result.metrics.max_drawdown_pct:.2f}%, "
        f"win_rate={result.metrics.win_rate:.1f}%) "
        f"in {elapsed:.1f}s",
        extra={
            "strategy": strategy.name,
            "symbol": symbol,
            "elapsed": round(elapsed, 2),
            **result.metrics.to_dict(),
        },
    )

    job_duration_histogram.labels(job_name="backtest").observe(elapsed)
    job_success_counter.labels(job_name="backtest").inc()

    return result


def _prepare_data(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Filter and validate DataFrame for the backtest period.

    Ensures the DataFrame index is datetime-based, sorted
    ascending, and filtered to the requested date range.

    Args:
        df: Raw OHLCV DataFrame.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        Filtered and sorted DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # Filter to date range
    mask = (df.index >= pd.Timestamp(start_date)) & (
        df.index <= pd.Timestamp(end_date)
    )
    filtered = df.loc[mask]

    logger.debug(
        f"Data prepared: {len(filtered)} rows "
        f"({filtered.index[0].date()} to {filtered.index[-1].date()})"
        if len(filtered) > 0
        else "Data prepared: 0 rows"
    )

    return filtered


def _calculate_metrics(
    trades: List[Trade],
    daily_returns: List[float],
    equity_curve: List[float],
    initial_capital: float,
) -> BacktestMetrics:
    """
    Calculate comprehensive performance metrics.

    Computes total return, annualized return, Sharpe ratio,
    max drawdown, win rate, profit factor, and other statistics
    from the trade list and equity curve.

    Args:
        trades: List of completed Trade objects.
        daily_returns: List of daily portfolio return fractions.
        equity_curve: List of daily portfolio values.
        initial_capital: Starting capital amount.

    Returns:
        BacktestMetrics with all performance statistics.
    """
    metrics = BacktestMetrics(initial_capital=initial_capital)

    if not trades:
        logger.debug("No trades to calculate metrics from")
        metrics.final_portfolio_value = initial_capital
        return metrics

    # Final portfolio value
    final_value = equity_curve[-1] if equity_curve else initial_capital
    metrics.final_portfolio_value = final_value

    # Total return
    metrics.total_return_pct = (
        ((final_value - initial_capital) / initial_capital) * 100.0
    )

    # Annualized return
    trading_days = len(equity_curve) if equity_curve else 1
    years = trading_days / TRADING_DAYS_PER_YEAR
    if years > 0 and final_value > 0:
        metrics.annualized_return_pct = (
            ((final_value / initial_capital) ** (1.0 / years) - 1.0)
            * 100.0
        )

    # Sharpe ratio
    if daily_returns and len(daily_returns) > 1:
        returns_arr = np.array(daily_returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr, ddof=1)

        if std_return > 0:
            daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
            metrics.sharpe_ratio = (
                (mean_return - daily_rf)
                / std_return
                * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

    # Max drawdown
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        metrics.max_drawdown_pct = max_dd * 100.0

    # Trade statistics
    metrics.total_trades = len(trades)
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    metrics.winning_trades = len(winners)
    metrics.losing_trades = len(losers)

    # Win rate
    metrics.win_rate = (
        (len(winners) / len(trades)) * 100.0 if trades else 0.0
    )

    # Average win/loss
    if winners:
        metrics.avg_win_pct = (
            sum(t.return_pct for t in winners) / len(winners)
        )
    if losers:
        metrics.avg_loss_pct = (
            sum(t.return_pct for t in losers) / len(losers)
        )

    # Profit factor
    total_wins = sum(t.pnl for t in winners) if winners else 0.0
    total_losses = abs(sum(t.pnl for t in losers)) if losers else 0.0
    if total_losses > 0:
        metrics.profit_factor = total_wins / total_losses

    # Max consecutive wins/losses
    metrics.max_consecutive_wins = _max_consecutive(
        trades, is_winner=True
    )
    metrics.max_consecutive_losses = _max_consecutive(
        trades, is_winner=False
    )

    # Total costs
    metrics.total_commission = sum(t.commission for t in trades)
    metrics.total_slippage = sum(t.slippage_cost for t in trades)

    logger.debug(
        f"Metrics calculated: {metrics.total_trades} trades, "
        f"return={metrics.total_return_pct:.2f}%, "
        f"sharpe={metrics.sharpe_ratio:.2f}"
    )

    return metrics


def _max_consecutive(
    trades: List[Trade],
    is_winner: bool,
) -> int:
    """
    Count the maximum consecutive wins or losses.

    Args:
        trades: List of trades in chronological order.
        is_winner: True to count consecutive wins,
            False for consecutive losses.

    Returns:
        Maximum streak length.
    """
    max_streak = 0
    current_streak = 0

    for trade in trades:
        if trade.is_winner == is_winner:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak

"""
Performance Tracker

Purpose:
    Tracks and calculates portfolio performance metrics over time.
    Computes total return, Sharpe ratio, Sortino ratio, maximum
    drawdown, win rate, and generates daily performance reports.

Dependencies:
    - src.monitoring.logger for structured logging
    - src.monitoring.metrics for Prometheus instrumentation
    - src.utils.config_loader for configuration
    - src.utils.constants for enums

Logging:
    - Metric updates at DEBUG
    - Daily reports at INFO
    - Errors at ERROR
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.monitoring.logger import get_logger
from src.monitoring.metrics import (
    portfolio_value_gauge,
    total_pnl_gauge,
    daily_pnl_gauge,
)
from src.utils.config_loader import load_config, get_nested
from src.utils.constants import PositionStatus

logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks portfolio performance and computes trading metrics.

    Maintains an equity curve and trade history to calculate
    risk-adjusted returns, drawdown statistics, and win/loss
    ratios over time.

    Attributes:
        equity_curve: List of portfolio value snapshots over time.
        daily_returns: List of daily percentage returns.
        portfolio_snapshots: List of full portfolio state snapshots.
        trades: List of completed trade records.
        risk_free_rate: Annualized risk-free rate used for Sharpe
            and Sortino ratio calculations.
        trading_days_per_year: Number of trading days in a year,
            used for annualization.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.065,
        trading_days_per_year: int = 252,
    ) -> None:
        """Initializes the PerformanceTracker.

        Args:
            risk_free_rate: Annualized risk-free rate as a decimal.
                Defaults to 0.065 (6.5%, approximate Indian
                government bond yield).
            trading_days_per_year: Trading days per year for
                annualization. Defaults to 252.
        """
        try:
            config = load_config("paper_trading")
            self.risk_free_rate: float = get_nested(
                config,
                "performance.risk_free_rate",
                risk_free_rate,
            )
            self.trading_days_per_year: int = get_nested(
                config,
                "performance.trading_days_per_year",
                trading_days_per_year,
            )
        except Exception:
            logger.warning(
                "Could not load paper_trading config, using defaults"
            )
            self.risk_free_rate = risk_free_rate
            self.trading_days_per_year = trading_days_per_year

        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.portfolio_snapshots: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []

        logger.info(
            "PerformanceTracker initialized",
            extra={
                "risk_free_rate": self.risk_free_rate,
                "trading_days_per_year": self.trading_days_per_year,
            },
        )

    def update(self, portfolio_state: Dict[str, Any]) -> None:
        """Records a new portfolio state snapshot.

        Appends the portfolio value to the equity curve, calculates
        the daily return relative to the previous snapshot, and
        stores the full state for reporting.

        Args:
            portfolio_state: A portfolio state dictionary as returned
                by PortfolioManager.get_portfolio_state(). Must
                contain at minimum:
                    - portfolio_value (float)
                    - timestamp (str)
        """
        try:
            current_value = portfolio_state.get(
                "portfolio_value", 0.0
            )
            timestamp = portfolio_state.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            )

            # Record equity curve data point
            self.equity_curve.append(
                {
                    "value": current_value,
                    "timestamp": timestamp,
                }
            )

            # Calculate daily return
            if len(self.equity_curve) >= 2:
                prev_value = self.equity_curve[-2]["value"]
                if prev_value > 0:
                    daily_return = (
                        (current_value - prev_value) / prev_value
                    )
                    self.daily_returns.append(daily_return)
                else:
                    self.daily_returns.append(0.0)

            # Store full snapshot
            self.portfolio_snapshots.append(portfolio_state)

            # Update Prometheus
            portfolio_value_gauge.set(round(current_value, 2))

            logger.debug(
                "Performance tracker updated",
                extra={
                    "portfolio_value": round(current_value, 2),
                    "data_points": len(self.equity_curve),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to update performance tracker",
                exc_info=True,
                extra={"error": str(e)},
            )

    def calculate_sharpe_ratio(
        self,
        returns: Optional[List[float]] = None,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculates the annualized Sharpe ratio.

        The Sharpe ratio measures risk-adjusted return by comparing
        excess returns (over the risk-free rate) to return volatility.

        Args:
            returns: List of periodic (daily) returns as decimal
                fractions. If None, uses internally tracked
                daily_returns.
            risk_free_rate: Annualized risk-free rate. If None, uses
                the instance default.

        Returns:
            The annualized Sharpe ratio. Returns 0.0 if there is
            insufficient data (fewer than 2 data points) or zero
            volatility.
        """
        try:
            ret = returns if returns is not None else self.daily_returns
            rfr = (
                risk_free_rate
                if risk_free_rate is not None
                else self.risk_free_rate
            )

            if len(ret) < 2:
                logger.debug(
                    "Insufficient data for Sharpe ratio calculation",
                    extra={"data_points": len(ret)},
                )
                return 0.0

            # Convert annualized risk-free rate to daily
            daily_rfr = rfr / self.trading_days_per_year

            # Calculate excess returns
            excess_returns = [r - daily_rfr for r in ret]

            # Mean and standard deviation of excess returns
            mean_excess = sum(excess_returns) / len(excess_returns)
            variance = sum(
                (r - mean_excess) ** 2 for r in excess_returns
            ) / (len(excess_returns) - 1)
            std_dev = math.sqrt(variance)

            if std_dev == 0:
                logger.debug(
                    "Zero volatility, Sharpe ratio undefined"
                )
                return 0.0

            # Annualize
            sharpe = (
                mean_excess / std_dev
            ) * math.sqrt(self.trading_days_per_year)

            logger.debug(
                "Sharpe ratio calculated",
                extra={
                    "sharpe_ratio": round(sharpe, 4),
                    "data_points": len(ret),
                },
            )

            return round(sharpe, 4)

        except Exception as e:
            logger.error(
                "Failed to calculate Sharpe ratio",
                exc_info=True,
                extra={"error": str(e)},
            )
            return 0.0

    def calculate_sortino_ratio(
        self,
        returns: Optional[List[float]] = None,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculates the annualized Sortino ratio.

        The Sortino ratio is similar to the Sharpe ratio but only
        penalizes downside volatility, making it a better measure
        for strategies with asymmetric return distributions.

        Args:
            returns: List of periodic (daily) returns as decimal
                fractions. If None, uses internally tracked
                daily_returns.
            risk_free_rate: Annualized risk-free rate. If None, uses
                the instance default.

        Returns:
            The annualized Sortino ratio. Returns 0.0 if there is
            insufficient data or zero downside deviation.
        """
        try:
            ret = returns if returns is not None else self.daily_returns
            rfr = (
                risk_free_rate
                if risk_free_rate is not None
                else self.risk_free_rate
            )

            if len(ret) < 2:
                logger.debug(
                    "Insufficient data for Sortino ratio calculation",
                    extra={"data_points": len(ret)},
                )
                return 0.0

            daily_rfr = rfr / self.trading_days_per_year
            excess_returns = [r - daily_rfr for r in ret]
            mean_excess = sum(excess_returns) / len(excess_returns)

            # Downside deviation: only consider negative excess returns
            downside_returns = [
                r for r in excess_returns if r < 0
            ]
            if not downside_returns:
                logger.debug(
                    "No downside returns, Sortino ratio undefined"
                )
                return 0.0

            downside_variance = sum(
                r ** 2 for r in downside_returns
            ) / len(excess_returns)
            downside_dev = math.sqrt(downside_variance)

            if downside_dev == 0:
                return 0.0

            sortino = (
                mean_excess / downside_dev
            ) * math.sqrt(self.trading_days_per_year)

            logger.debug(
                "Sortino ratio calculated",
                extra={
                    "sortino_ratio": round(sortino, 4),
                    "data_points": len(ret),
                },
            )

            return round(sortino, 4)

        except Exception as e:
            logger.error(
                "Failed to calculate Sortino ratio",
                exc_info=True,
                extra={"error": str(e)},
            )
            return 0.0

    def calculate_max_drawdown(
        self,
        equity_curve: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculates the maximum drawdown from the equity curve.

        Maximum drawdown is the largest peak-to-trough decline in
        portfolio value, expressed as both an absolute amount and a
        percentage.

        Args:
            equity_curve: List of equity curve data points, each a
                dict with a "value" key. If None, uses the
                internally tracked equity curve.

        Returns:
            A dictionary containing:
                - max_drawdown_pct (float): Maximum drawdown as a
                    percentage (negative value).
                - max_drawdown_amount (float): Maximum drawdown in
                    absolute INR terms (negative value).
                - peak_value (float): The peak portfolio value before
                    the drawdown.
                - trough_value (float): The lowest portfolio value
                    during the drawdown.
                - peak_index (int): Index in the equity curve where
                    the peak occurred.
                - trough_index (int): Index where the trough occurred.
                - recovery_index (int or None): Index where the
                    portfolio recovered to the peak, or None if not
                    yet recovered.
        """
        try:
            curve = (
                equity_curve
                if equity_curve is not None
                else self.equity_curve
            )

            if len(curve) < 2:
                logger.debug(
                    "Insufficient data for drawdown calculation",
                    extra={"data_points": len(curve)},
                )
                return {
                    "max_drawdown_pct": 0.0,
                    "max_drawdown_amount": 0.0,
                    "peak_value": 0.0,
                    "trough_value": 0.0,
                    "peak_index": 0,
                    "trough_index": 0,
                    "recovery_index": None,
                }

            values = [point["value"] for point in curve]
            peak = values[0]
            peak_index = 0
            max_dd_pct = 0.0
            max_dd_amount = 0.0
            dd_peak_value = peak
            dd_trough_value = peak
            dd_peak_index = 0
            dd_trough_index = 0

            for i, value in enumerate(values):
                if value > peak:
                    peak = value
                    peak_index = i

                drawdown = value - peak
                dd_pct = (drawdown / peak) * 100 if peak > 0 else 0.0

                if dd_pct < max_dd_pct:
                    max_dd_pct = dd_pct
                    max_dd_amount = drawdown
                    dd_peak_value = peak
                    dd_trough_value = value
                    dd_peak_index = peak_index
                    dd_trough_index = i

            # Check for recovery
            recovery_index = None
            if dd_trough_index > 0:
                for i in range(dd_trough_index + 1, len(values)):
                    if values[i] >= dd_peak_value:
                        recovery_index = i
                        break

            result = {
                "max_drawdown_pct": round(max_dd_pct, 4),
                "max_drawdown_amount": round(max_dd_amount, 2),
                "peak_value": round(dd_peak_value, 2),
                "trough_value": round(dd_trough_value, 2),
                "peak_index": dd_peak_index,
                "trough_index": dd_trough_index,
                "recovery_index": recovery_index,
            }

            logger.debug(
                "Max drawdown calculated",
                extra={
                    "max_drawdown_pct": round(max_dd_pct, 4),
                    "peak_value": round(dd_peak_value, 2),
                    "trough_value": round(dd_trough_value, 2),
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to calculate max drawdown",
                exc_info=True,
                extra={"error": str(e)},
            )
            return {
                "max_drawdown_pct": 0.0,
                "max_drawdown_amount": 0.0,
                "peak_value": 0.0,
                "trough_value": 0.0,
                "peak_index": 0,
                "trough_index": 0,
                "recovery_index": None,
            }

    def calculate_win_rate(
        self, trades: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Calculates the win rate and trade statistics.

        Args:
            trades: List of closed trade dictionaries, each with a
                "realized_pnl" field. If None, uses the internally
                tracked trades list.

        Returns:
            A dictionary containing:
                - win_rate (float): Percentage of winning trades.
                - total_trades (int): Total number of trades.
                - winning_trades (int): Number of profitable trades.
                - losing_trades (int): Number of losing trades.
                - avg_win (float): Average winning trade P&L.
                - avg_loss (float): Average losing trade P&L.
                - profit_factor (float): Ratio of gross profits to
                    gross losses.
                - expectancy (float): Expected P&L per trade.
        """
        try:
            trade_list = (
                trades if trades is not None else self.trades
            )

            if not trade_list:
                logger.debug("No trades to calculate win rate")
                return {
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                    "expectancy": 0.0,
                }

            winners = [
                t
                for t in trade_list
                if t.get("realized_pnl", 0.0) > 0
            ]
            losers = [
                t
                for t in trade_list
                if t.get("realized_pnl", 0.0) <= 0
            ]

            total = len(trade_list)
            win_count = len(winners)
            loss_count = len(losers)
            win_rate = (win_count / total) * 100 if total > 0 else 0.0

            gross_profit = sum(
                t["realized_pnl"] for t in winners
            )
            gross_loss = abs(
                sum(t["realized_pnl"] for t in losers)
            )

            avg_win = gross_profit / win_count if win_count > 0 else 0.0
            avg_loss = (
                -gross_loss / loss_count if loss_count > 0 else 0.0
            )

            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else 0.0
            )

            total_pnl = sum(
                t.get("realized_pnl", 0.0) for t in trade_list
            )
            expectancy = total_pnl / total if total > 0 else 0.0

            result = {
                "win_rate": round(win_rate, 2),
                "total_trades": total,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 4),
                "expectancy": round(expectancy, 2),
            }

            logger.debug(
                "Win rate calculated",
                extra={
                    "win_rate": round(win_rate, 2),
                    "total_trades": total,
                    "profit_factor": round(profit_factor, 4),
                },
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to calculate win rate",
                exc_info=True,
                extra={"error": str(e)},
            )
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
            }

    def generate_daily_report(self) -> Dict[str, Any]:
        """Generates a comprehensive daily performance report.

        Combines the latest portfolio snapshot with all calculated
        metrics into a single report dictionary.

        Returns:
            A dictionary containing:
                - report_date (str): ISO-format date of the report.
                - portfolio_state (dict): Latest portfolio snapshot.
                - total_return_pct (float): Total return since
                    inception.
                - sharpe_ratio (float): Annualized Sharpe ratio.
                - sortino_ratio (float): Annualized Sortino ratio.
                - max_drawdown (dict): Maximum drawdown details.
                - win_rate_stats (dict): Win rate and trade
                    statistics.
                - daily_return (float): Most recent daily return.
                - data_points (int): Number of equity curve points.
        """
        try:
            # Latest portfolio state
            latest_state = (
                self.portfolio_snapshots[-1]
                if self.portfolio_snapshots
                else {}
            )

            # Total return
            if self.equity_curve and len(self.equity_curve) >= 2:
                initial_value = self.equity_curve[0]["value"]
                current_value = self.equity_curve[-1]["value"]
                total_return_pct = (
                    (
                        (current_value - initial_value)
                        / initial_value
                    )
                    * 100
                    if initial_value > 0
                    else 0.0
                )
            else:
                total_return_pct = latest_state.get(
                    "total_return_pct", 0.0
                )

            # Latest daily return
            latest_daily_return = (
                self.daily_returns[-1] * 100
                if self.daily_returns
                else 0.0
            )

            report = {
                "report_date": datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d"),
                "portfolio_state": latest_state,
                "total_return_pct": round(total_return_pct, 4),
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "sortino_ratio": self.calculate_sortino_ratio(),
                "max_drawdown": self.calculate_max_drawdown(),
                "win_rate_stats": self.calculate_win_rate(),
                "daily_return": round(latest_daily_return, 4),
                "data_points": len(self.equity_curve),
            }

            logger.info(
                "Daily performance report generated",
                extra={
                    "report_date": report["report_date"],
                    "total_return_pct": report["total_return_pct"],
                    "sharpe_ratio": report["sharpe_ratio"],
                    "max_drawdown_pct": report["max_drawdown"][
                        "max_drawdown_pct"
                    ],
                },
            )

            return report

        except Exception as e:
            logger.error(
                "Failed to generate daily report",
                exc_info=True,
                extra={"error": str(e)},
            )
            return {
                "report_date": datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d"),
                "portfolio_state": {},
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": {
                    "max_drawdown_pct": 0.0,
                    "max_drawdown_amount": 0.0,
                },
                "win_rate_stats": {"win_rate": 0.0},
                "daily_return": 0.0,
                "data_points": 0,
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Returns a concise summary of all performance metrics.

        Suitable for dashboard display or quick status checks.
        Provides the most important metrics without the full
        detail of the daily report.

        Returns:
            A dictionary containing:
                - total_return_pct (float): Total return since
                    inception.
                - sharpe_ratio (float): Annualized Sharpe ratio.
                - sortino_ratio (float): Annualized Sortino ratio.
                - max_drawdown_pct (float): Maximum drawdown
                    percentage.
                - win_rate (float): Win rate percentage.
                - total_trades (int): Total completed trades.
                - profit_factor (float): Ratio of gross profit to
                    gross loss.
                - current_value (float): Current portfolio value.
                - data_points (int): Number of equity observations.
                - timestamp (str): ISO-format UTC timestamp.
        """
        try:
            # Total return
            if self.equity_curve and len(self.equity_curve) >= 2:
                initial = self.equity_curve[0]["value"]
                current = self.equity_curve[-1]["value"]
                total_return = (
                    ((current - initial) / initial) * 100
                    if initial > 0
                    else 0.0
                )
                current_value = current
            else:
                total_return = 0.0
                current_value = (
                    self.equity_curve[-1]["value"]
                    if self.equity_curve
                    else 0.0
                )

            drawdown = self.calculate_max_drawdown()
            win_stats = self.calculate_win_rate()

            summary = {
                "total_return_pct": round(total_return, 4),
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "sortino_ratio": self.calculate_sortino_ratio(),
                "max_drawdown_pct": drawdown["max_drawdown_pct"],
                "win_rate": win_stats["win_rate"],
                "total_trades": win_stats["total_trades"],
                "profit_factor": win_stats["profit_factor"],
                "current_value": round(current_value, 2),
                "data_points": len(self.equity_curve),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            logger.info(
                "Performance summary generated",
                extra={
                    "total_return_pct": summary["total_return_pct"],
                    "sharpe_ratio": summary["sharpe_ratio"],
                    "win_rate": summary["win_rate"],
                },
            )

            return summary

        except Exception as e:
            logger.error(
                "Failed to generate performance summary",
                exc_info=True,
                extra={"error": str(e)},
            )
            return {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
                "current_value": 0.0,
                "data_points": 0,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

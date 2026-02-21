"""
Chart Visualizer for AlgoTrade Scanner

Purpose:
    Generates annotated 60-day candlestick PNG charts for every trading
    signal, one image per signal.  Supports five strategy types:

    ┌──────────────────────┬────────────────────────────────────────────┐
    │ Strategy             │ Annotation drawn                           │
    ├──────────────────────┼────────────────────────────────────────────┤
    │ Mother Candle V2     │ LightBlue box (consolidation zone)         │
    │ Darvas Box           │ Green box (ceiling + floor)                │
    │ Flag Pattern         │ Yellow pole highlight + orange flag box    │
    │ Symmetrical Triangle │ Red resistance + green support trendlines  │
    │ Descending Channel   │ Red upper + green lower channel lines      │
    └──────────────────────┴────────────────────────────────────────────┘

    All charts also include:
    - Horizontal target (green dashed) and stop-loss (red dashed) lines
    - Green "▲ Entry" annotation on the breakout bar

Dependencies:
    plotly >= 5.17.0
    kaleido >= 0.2.1   (for static PNG export)

Logging:
    Successful saves at INFO; missing dependencies / generation failures
    at WARNING / ERROR.
"""

import os
from typing import Any, List, Optional, Tuple

import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal

logger = get_logger(__name__)

# Number of trading days to show in every chart
CHART_DAYS = 60


class ChartVisualizer:
    """Generates annotated candlestick charts for trading signals."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_signal_chart(
        self,
        df: pd.DataFrame,
        signal: TradingSignal,
        output_path: str,
    ) -> bool:
        """
        Generate a 60-day annotated candlestick chart and save as PNG.

        Strategy-specific pattern overlays are drawn automatically based
        on ``signal.strategy_name`` and ``signal.metadata``.

        Args:
            df: Full OHLCV DataFrame, date-indexed, sorted ascending.
            signal: TradingSignal produced by any supported strategy.
            output_path: Filesystem path for the output PNG file.

        Returns:
            True if the chart was saved successfully, False otherwise.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning(
                "plotly is not installed — cannot generate chart. "
                "Install with: pip install plotly kaleido"
            )
            return False

        try:
            df_plot = df.iloc[-CHART_DAYS:].copy()
            df_plot.index = pd.to_datetime(df_plot.index)

            fig = go.Figure()

            # ── Candlestick ────────────────────────────────────────────
            fig.add_trace(
                go.Candlestick(
                    x=df_plot.index,
                    open=df_plot["open"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                    close=df_plot["close"],
                    name=signal.symbol,
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                    whiskerwidth=0.3,
                )
            )

            # ── Strategy-specific overlays ─────────────────────────────
            name = signal.strategy_name
            if name == "Mother Candle V2":
                self._draw_mother_candle_box(fig, df_plot, signal)
            elif name == "Darvas Box":
                self._draw_darvas_box(fig, df_plot, signal)
            elif name == "Flag Pattern":
                self._draw_flag_pattern(fig, df_plot, signal)
            elif name == "Symmetrical Triangle":
                self._draw_trendlines(
                    fig, df_plot, signal,
                    upper_key="resistance_points",
                    lower_key="support_points",
                    upper_color="#ef5350",
                    lower_color="#26a69a",
                    upper_label="Resistance",
                    lower_label="Support",
                )
            elif name == "Descending Channel":
                self._draw_trendlines(
                    fig, df_plot, signal,
                    upper_key="upper_line_points",
                    lower_key="lower_line_points",
                    upper_color="#ef5350",
                    lower_color="#26a69a",
                    upper_label="Upper Channel",
                    lower_label="Lower Channel",
                )

            # ── Horizontal target / SL lines ───────────────────────────
            self._draw_price_levels(fig, df_plot, signal)

            # ── Entry arrow ────────────────────────────────────────────
            self._draw_entry_arrow(fig, df_plot, signal)

            # ── Layout ─────────────────────────────────────────────────
            rr = signal.metadata.get("rr_ratio", signal.risk_reward_ratio)
            conf = signal.confidence * 100
            title = (
                f"<b>{signal.symbol}</b>  ·  {signal.strategy_name}"
                f"   |   Entry ₹{signal.entry_price:,.2f}"
                f"   Target ₹{signal.target_price:,.2f}"
                f"   SL ₹{signal.stop_loss:,.2f}"
                f"   R:R 1:{rr:.1f}"
                f"   Conf {conf:.0f}%"
            )
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=12, color="#d1d4dc"),
                    x=0.01,
                ),
                xaxis=dict(
                    type="date",
                    rangeslider=dict(visible=False),
                    tickformat="%d %b",
                    tickfont=dict(size=9, color="#9598a1"),
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.12)",
                ),
                yaxis=dict(
                    title="Price (₹)",
                    side="right",
                    tickfont=dict(size=9, color="#9598a1"),
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.12)",
                ),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                font=dict(color="#d1d4dc"),
                height=620,
                width=1050,
                margin=dict(l=10, r=80, t=50, b=50),
                showlegend=False,
            )

            # ── Save PNG ───────────────────────────────────────────────
            parent = os.path.dirname(os.path.abspath(output_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

            fig.write_image(output_path, format="png", scale=2)
            logger.info(
                "Chart saved",
                extra={"symbol": signal.symbol, "path": output_path},
            )
            return True

        except Exception as exc:
            logger.error(
                f"Failed to generate chart for {signal.symbol}: {exc}",
                exc_info=True,
            )
            return False

    def generate_pattern_chart(
        self,
        df: pd.DataFrame,
        signal: TradingSignal,
        output_path: str,
    ) -> bool:
        """Alias for save_signal_chart (backward compatibility)."""
        return self.save_signal_chart(df, signal, output_path)

    # ------------------------------------------------------------------
    # Strategy-specific overlay methods
    # ------------------------------------------------------------------

    def _draw_mother_candle_box(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """Semi-transparent LightBlue box for the Mother Candle zone."""
        meta = signal.metadata
        high = meta.get("mother_high")
        low = meta.get("mother_low")
        start = meta.get("mother_start_idx")
        end = meta.get("mother_end_idx", -2)

        if None in (high, low, start):
            return

        x0, x1 = self._neg_to_dates(df_plot, [start, end])
        if x0 is None or x1 is None:
            return

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=low, y1=high,
            fillcolor="LightBlue",
            opacity=0.22,
            line=dict(color="LightBlue", width=1.5),
        )
        self._box_label(fig, x1, high, "Mother Candle Range", "LightBlue")

    def _draw_darvas_box(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """Green semi-transparent rectangle for the Darvas Box."""
        meta = signal.metadata
        top = meta.get("box_top")
        bottom = meta.get("box_bottom")
        start = meta.get("box_start_idx")
        end = meta.get("box_end_idx", -2)

        if None in (top, bottom, start):
            return

        x0, x1 = self._neg_to_dates(df_plot, [start, end])
        if x0 is None or x1 is None:
            return

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=bottom, y1=top,
            fillcolor="#00e676",
            opacity=0.15,
            line=dict(color="#00e676", width=1.5),
        )
        self._box_label(fig, x1, top, "Darvas Box", "#00e676")

    def _draw_flag_pattern(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """Yellow pole highlight + orange flag box."""
        meta = signal.metadata
        pole_start = meta.get("pole_start_idx")
        pole_end = meta.get("pole_end_idx")
        flag_start = meta.get("flag_start_idx")
        flag_end = meta.get("flag_end_idx", -2)
        flag_high = meta.get("flag_high")
        flag_low = meta.get("flag_low")
        pole_start_price = meta.get("pole_start_price")

        if None in (pole_start, pole_end, flag_start, flag_high, flag_low):
            return

        n = len(df_plot)

        # Pole shading
        px0, px1 = self._neg_to_dates(df_plot, [pole_start, pole_end])
        if px0 and px1 and pole_start_price:
            pole_end_pos = _clamp(n + pole_end, 0, n - 1)
            pole_top = float(df_plot["close"].iloc[pole_end_pos])
            fig.add_shape(
                type="rect",
                x0=px0, x1=px1,
                y0=pole_start_price, y1=pole_top,
                fillcolor="#ffd600",
                opacity=0.12,
                line=dict(color="#ffd600", width=1),
            )
            self._box_label(fig, px1, pole_top, "Pole", "#ffd600")

        # Flag box
        fx0, fx1 = self._neg_to_dates(df_plot, [flag_start, flag_end])
        if fx0 and fx1:
            fig.add_shape(
                type="rect",
                x0=fx0, x1=fx1,
                y0=flag_low, y1=flag_high,
                fillcolor="#ff9800",
                opacity=0.18,
                line=dict(color="#ff9800", width=1.5),
            )
            self._box_label(fig, fx1, flag_high, "Flag", "#ff9800")

    def _draw_trendlines(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
        upper_key: str,
        lower_key: str,
        upper_color: str,
        lower_color: str,
        upper_label: str,
        lower_label: str,
    ) -> None:
        """
        Draw two regression trendlines through stored pivot points.
        Used for both Symmetrical Triangle and Descending Channel.
        """
        meta = signal.metadata
        self._draw_one_trendline(
            fig, df_plot, meta.get(upper_key, []),
            upper_color, upper_label
        )
        self._draw_one_trendline(
            fig, df_plot, meta.get(lower_key, []),
            lower_color, lower_label
        )

    def _draw_one_trendline(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        points: List,
        color: str,
        label: str,
    ) -> None:
        """Fit + draw one regression trendline extended to the last bar."""
        if not NUMPY_AVAILABLE or len(points) < 2:
            return

        n = len(df_plot)
        xs_plot, ys = [], []
        for pt in points:
            neg_idx, price = int(pt[0]), float(pt[1])
            pos = n + neg_idx
            if 0 <= pos < n:
                xs_plot.append(pos)
                ys.append(price)

        if len(xs_plot) < 2:
            return

        xs_arr = np.array(xs_plot, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        slope, intercept = float(
            np.polyfit(xs_arr, ys_arr, 1)[0]
        ), float(np.polyfit(xs_arr, ys_arr, 1)[1])

        x_start, x_end = int(xs_arr[0]), n - 1
        date_start = df_plot.index[x_start]
        date_end = df_plot.index[x_end]
        y_start = slope * x_start + intercept
        y_end = slope * x_end + intercept

        # Regression line
        fig.add_shape(
            type="line",
            x0=date_start, y0=y_start,
            x1=date_end, y1=y_end,
            line=dict(color=color, width=2, dash="dot"),
        )

        # Pivot markers
        pivot_dates = [
            df_plot.index[_clamp(n + int(pt[0]), 0, n - 1)]
            for pt in points
            if 0 <= n + int(pt[0]) < n
        ]
        pivot_prices = [
            float(pt[1]) for pt in points
            if 0 <= n + int(pt[0]) < n
        ]
        if pivot_dates:
            fig.add_trace(
                go.Scatter(
                    x=pivot_dates,
                    y=pivot_prices,
                    mode="markers",
                    marker=dict(
                        color=color, size=7,
                        line=dict(color="white", width=1),
                    ),
                    showlegend=False,
                )
            )

        # Label at midpoint of line
        mid_pos = _clamp((x_start + x_end) // 2, 0, n - 1)
        mid_date = df_plot.index[mid_pos]
        mid_y = slope * mid_pos + intercept
        fig.add_annotation(
            x=mid_date,
            y=mid_y,
            text=label,
            showarrow=False,
            yshift=10,
            font=dict(size=9, color=color),
            bgcolor="rgba(10,10,30,0.60)",
            borderpad=2,
        )

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _draw_price_levels(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """Horizontal dashed lines for target and stop-loss."""
        x0, x1 = df_plot.index[0], df_plot.index[-1]

        for price, color, label in (
            (signal.target_price, "#00e676", f"T ₹{signal.target_price:,.2f}"),
            (signal.stop_loss, "#ef5350", f"SL ₹{signal.stop_loss:,.2f}"),
        ):
            fig.add_shape(
                type="line",
                x0=x0, x1=x1, y0=price, y1=price,
                line=dict(color=color, width=1.2, dash="dash"),
            )
            fig.add_annotation(
                x=x1, y=price,
                text=label,
                showarrow=False,
                xanchor="left",
                xshift=5,
                font=dict(size=9, color=color),
                bgcolor="rgba(10,10,30,0.70)",
                borderpad=2,
            )

    def _draw_entry_arrow(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """Green upward annotation + arrow on the breakout (last) bar."""
        fig.add_annotation(
            x=df_plot.index[-1],
            y=signal.entry_price,
            text=f"<b>▲ Entry<br>₹{signal.entry_price:,.2f}</b>",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=2,
            arrowcolor="#00e676",
            font=dict(size=10, color="#00e676"),
            ax=0,
            ay=60,
            bgcolor="rgba(10,10,30,0.72)",
            bordercolor="#00e676",
            borderwidth=1,
            borderpad=4,
        )

    def _neg_to_dates(
        self,
        df_plot: pd.DataFrame,
        neg_indices: List[int],
    ) -> tuple:
        """Convert negative df offsets to dates within df_plot."""
        n = len(df_plot)
        out = []
        for neg in neg_indices:
            pos = n + neg
            out.append(df_plot.index[pos] if 0 <= pos < n else None)
        return tuple(out)

    def _box_label(
        self,
        fig: "go.Figure",
        x_right,
        y_top: float,
        text: str,
        color: str,
    ) -> None:
        """Annotation at the top-right of a rectangular shape."""
        fig.add_annotation(
            x=x_right, y=y_top,
            text=text,
            showarrow=False,
            xanchor="right",
            yshift=13,
            font=dict(size=9, color=color),
            bgcolor="rgba(10,10,30,0.65)",
            borderpad=2,
        )


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp value between lo and hi (inclusive)."""
    return max(lo, min(value, hi))

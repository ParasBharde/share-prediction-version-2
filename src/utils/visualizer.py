"""
Chart Visualizer for AlgoTrade Scanner

Purpose:
    Generates high-quality candlestick chart PNG images for trading signals.
    Annotates patterns such as the Mother Candle box and the breakout entry.

Dependencies:
    - plotly>=5.17.0
    - kaleido>=0.2.1  (for static PNG export)

Logging:
    - Successful chart saves at INFO
    - Missing dependencies or generation failures at WARNING/ERROR
"""

import os
from typing import TYPE_CHECKING

import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import TradingSignal

logger = get_logger(__name__)


class ChartVisualizer:
    """Generates annotated candlestick chart images for trading signals."""

    def generate_pattern_chart(
        self,
        df: pd.DataFrame,
        signal: TradingSignal,
        output_path: str,
    ) -> bool:
        """
        Generate a candlestick chart with pattern annotations and save as PNG.

        Renders the last 40 trading days.  For 'Mother Candle V2' signals the
        consolidation box (mother_high / mother_low) is drawn as a
        semi-transparent LightBlue rectangle, followed by a green arrow
        that marks the breakout entry price on the last (breakout) candle.

        Args:
            df: Full OHLCV DataFrame, date-indexed, sorted ascending.
            signal: TradingSignal produced by a strategy.
            output_path: Absolute or relative path for the output PNG file.

        Returns:
            True if the chart was saved successfully, False otherwise.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning(
                "plotly is not installed; cannot generate chart image. "
                "Install with: pip install plotly kaleido"
            )
            return False

        try:
            # ── Slice to last 40 candles ──────────────────────────────────
            df_plot = df.iloc[-40:].copy()
            df_plot.index = pd.to_datetime(df_plot.index)

            # ── Base candlestick trace ────────────────────────────────────
            fig = go.Figure()
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

            # ── Strategy-specific overlays ────────────────────────────────
            if signal.strategy_name == "Mother Candle V2":
                self._add_mother_candle_box(fig, df_plot, signal)

            # ── Entry arrow (green, always shown) ─────────────────────────
            self._add_entry_arrow(fig, df_plot, signal)

            # ── Chart layout ──────────────────────────────────────────────
            title_text = (
                f"<b>{signal.symbol}</b>  —  {signal.strategy_name}"
                f"   |   Entry ₹{signal.entry_price:.2f}"
                f"   Target ₹{signal.target_price:.2f}"
                f"   SL ₹{signal.stop_loss:.2f}"
            )
            fig.update_layout(
                title=dict(
                    text=title_text,
                    font=dict(size=13, color="#d1d4dc"),
                    x=0.01,
                ),
                xaxis=dict(
                    type="date",
                    rangeslider=dict(visible=False),
                    tickformat="%d %b",
                    tickfont=dict(size=9, color="#9598a1"),
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.15)",
                ),
                yaxis=dict(
                    title="Price (₹)",
                    side="right",
                    tickfont=dict(size=9, color="#9598a1"),
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.06)",
                    linecolor="rgba(255,255,255,0.15)",
                ),
                plot_bgcolor="#1e1e2e",
                paper_bgcolor="#1e1e2e",
                font=dict(color="#d1d4dc"),
                height=600,
                width=1000,
                margin=dict(l=10, r=70, t=55, b=50),
                showlegend=False,
            )

            # ── Save PNG ──────────────────────────────────────────────────
            parent_dir = os.path.dirname(os.path.abspath(output_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            fig.write_image(output_path, format="png", scale=2)
            logger.info(
                "Chart saved",
                extra={"symbol": signal.symbol, "path": output_path},
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to generate chart for {signal.symbol}: {e}",
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_mother_candle_box(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """
        Draw a semi-transparent LightBlue rectangle spanning the Mother
        Candle consolidation zone (from mother candle to last baby candle).

        Reads ``mother_high``, ``mother_low``, ``mother_start_idx``, and
        ``mother_end_idx`` from ``signal.metadata``.

        Args:
            fig: Plotly Figure to annotate.
            df_plot: 40-day OHLCV slice used for the chart.
            signal: TradingSignal with Mother Candle metadata.
        """
        meta = signal.metadata
        mother_high = meta.get("mother_high")
        mother_low = meta.get("mother_low")
        mother_start_idx = meta.get("mother_start_idx")
        mother_end_idx = meta.get("mother_end_idx", -2)

        if mother_high is None or mother_low is None or mother_start_idx is None:
            logger.debug(
                f"{signal.symbol}: Mother Candle metadata incomplete, "
                "skipping box annotation"
            )
            return

        n = len(df_plot)

        # Convert negative offsets to positions within df_plot.
        # mother_start_idx / mother_end_idx are relative to the END of the
        # *full* df; df_plot is the last n rows of that df, so the mapping is:
        #   position_in_df_plot = n + negative_idx
        start_pos = n + mother_start_idx if mother_start_idx < 0 else mother_start_idx
        end_pos = n + mother_end_idx if mother_end_idx < 0 else mother_end_idx

        # Clamp to valid range
        start_pos = max(0, min(start_pos, n - 1))
        end_pos = max(0, min(end_pos, n - 1))
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos

        x_start = df_plot.index[start_pos]
        x_end = df_plot.index[end_pos]

        # Semi-transparent LightBlue box
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x_start,
            x1=x_end,
            y0=mother_low,
            y1=mother_high,
            fillcolor="LightBlue",
            opacity=0.25,
            line=dict(color="LightBlue", width=1.5),
        )

        # Label centred at the top of the box
        mid_pos = (start_pos + end_pos) // 2
        x_mid = df_plot.index[mid_pos]
        fig.add_annotation(
            x=x_mid,
            y=mother_high,
            text="Mother Candle Range",
            showarrow=False,
            yshift=14,
            font=dict(size=10, color="LightBlue"),
            bgcolor="rgba(10,10,30,0.65)",
            borderpad=3,
        )

    def _add_entry_arrow(
        self,
        fig: "go.Figure",
        df_plot: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """
        Draw a green upward arrow at the entry (breakout) price on the last
        candle of the chart window.

        Args:
            fig: Plotly Figure to annotate.
            df_plot: 40-day OHLCV slice used for the chart.
            signal: TradingSignal carrying the entry_price.
        """
        x_entry = df_plot.index[-1]
        entry_price = signal.entry_price

        # Offset the arrow tail below the entry price for a clear upward cue
        y_tail_offset = (df_plot["high"].max() - df_plot["low"].min()) * 0.04

        fig.add_annotation(
            x=x_entry,
            y=entry_price,
            text=f"<b>▲ Entry<br>₹{entry_price:.2f}</b>",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=2,
            arrowcolor="#00e676",
            font=dict(size=10, color="#00e676"),
            # Tail sits below the annotation head
            ax=0,
            ay=55,
            bgcolor="rgba(10,10,30,0.70)",
            bordercolor="#00e676",
            borderwidth=1,
            borderpad=4,
        )

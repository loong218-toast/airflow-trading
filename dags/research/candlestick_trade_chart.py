from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()

if Path("/opt/airflow/airflow-trading").exists():
    PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
elif Path("/opt/airflow").exists():
    PROJECT_ROOT = Path("/opt/airflow")
else:
    PROJECT_ROOT = next(
        (p for p in HERE.parents if (p / "dags").exists() or (p / "data_lake").exists()),
        HERE.parents[1],
    )

DAGS_ROOT = PROJECT_ROOT / "dags" if (PROJECT_ROOT / "dags").exists() else Path("/opt/airflow/dags")
if str(DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGS_ROOT))


from research.exploration_config import CONFIG  # type: ignore
from research.trade_core import build_trade_universe, load_cache_df  # type: ignore


logger = logging.getLogger(__name__)

# Fallback defaults. You can also define these in exploration_config.py and
# the code below will pick them up automatically if present.
DEFAULT_CHART_LOOKBACK_DAYS = 180
DEFAULT_CHART_MAX_CANDLES = 3000
DEFAULT_MARKER_SIZE = 3


def _mix_rgb(start_rgb: tuple[int, int, int], end_rgb: tuple[int, int, int], t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    r = int(round(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t))
    g = int(round(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t))
    b = int(round(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t))
    return f"rgb({r},{g},{b})"


def _trade_marker_color(trade_r: float, positive_scale: float, negative_scale: float) -> str:
    if not np.isfinite(trade_r):
        return "rgb(120,120,120)"

    if trade_r >= 0:
        t = 1.0 if positive_scale <= 0 else float(np.clip(trade_r / positive_scale, 0.15, 1.0))
        return _mix_rgb((220, 245, 220), (0, 170, 0), t)

    t = 1.0 if negative_scale <= 0 else float(np.clip(abs(trade_r) / negative_scale, 0.15, 1.0))
    return _mix_rgb((250, 225, 225), (200, 0, 0), t)


def _trade_marker_symbol(side: int) -> str:
    return "circle"


def _resolve_chart_window(raw_df: pd.DataFrame, trade_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = raw_df.copy()
    trade_df = trade_df.copy()

    raw_df["dt_myt"] = pd.to_datetime(raw_df["time_ns"], unit="ns", utc=True).dt.tz_convert(CONFIG.timezone)

    if not trade_df.empty and "entry_time_myt" not in trade_df.columns:
        trade_df["entry_time_myt"] = pd.to_datetime(trade_df["time_ns"], unit="ns", utc=True).dt.tz_convert(CONFIG.timezone)

    raw_df = raw_df.sort_values("dt_myt")

    lookback_days = int(getattr(CONFIG, "chart_lookback_days", DEFAULT_CHART_LOOKBACK_DAYS) or 0)
    max_candles = int(getattr(CONFIG, "chart_max_candles", DEFAULT_CHART_MAX_CANDLES) or 0)

    cutoff = None
    if lookback_days > 0:
        cutoff = raw_df["dt_myt"].max() - pd.Timedelta(days=lookback_days)
        raw_df = raw_df.loc[raw_df["dt_myt"] >= cutoff].copy()

    if max_candles > 0 and len(raw_df) > max_candles:
        raw_df = raw_df.tail(max_candles).copy()
        cutoff = raw_df["dt_myt"].min()

    if cutoff is not None and not trade_df.empty:
        trade_df = trade_df.loc[trade_df["entry_time_myt"] >= cutoff].copy()

    return raw_df, trade_df


def build_candlestick_trade_chart(trade_df: pd.DataFrame, raw_df: pd.DataFrame):
    import plotly.graph_objects as go

    if raw_df.empty:
        raise RuntimeError("Raw OHLC data is empty.")

    raw_df, trade_df = _resolve_chart_window(raw_df, trade_df)

    if raw_df.empty:
        raise RuntimeError("No OHLC rows left after applying chart window.")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=raw_df["dt_myt"],
            open=raw_df["open"],
            high=raw_df["high"],
            low=raw_df["low"],
            close=raw_df["close"],
            name="OHLC",
            increasing_line_color="rgba(0,120,0,0.9)",
            decreasing_line_color="rgba(180,0,0,0.9)",
            increasing_fillcolor="rgba(0,120,0,0.25)",
            decreasing_fillcolor="rgba(180,0,0,0.25)",
        )
    )

    if not trade_df.empty:
        positive_series = trade_df.loc[trade_df["trade_r"] > 0, "trade_r"].to_numpy(dtype=float, copy=False)
        negative_series = np.abs(trade_df.loc[trade_df["trade_r"] < 0, "trade_r"].to_numpy(dtype=float, copy=False))

        positive_scale = float(np.nanpercentile(positive_series, 95)) if positive_series.size else 1.0
        negative_scale = float(np.nanpercentile(negative_series, 95)) if negative_series.size else 1.0

        colors = [
            _trade_marker_color(float(r), positive_scale=positive_scale, negative_scale=negative_scale)
            for r in trade_df["trade_r"].to_numpy(dtype=float, copy=False)
        ]
        symbols = [_trade_marker_symbol(int(side)) for side in trade_df["side"].to_numpy(dtype=int, copy=False)]

        # Smaller, simpler markers for lighter rendering.
        sizes = np.full(len(trade_df), int(getattr(CONFIG, "chart_marker_size", DEFAULT_MARKER_SIZE)), dtype=int)

        customdata = np.column_stack(
            [
                trade_df["side_label"].astype(str).to_numpy(),
                trade_df["trade_r"].to_numpy(dtype=float, copy=False),
                trade_df["trade_expectancy_pct"].to_numpy(dtype=float, copy=False),
                trade_df["exit_type"].astype(str).to_numpy(),
                trade_df["holding_minutes"].to_numpy(dtype=float, copy=False),
                trade_df["entry_bucket_label"].astype(str).to_numpy(),
            ]
        )

        fig.add_trace(
            go.Scattergl(
                x=trade_df["entry_time_myt"],
                y=trade_df["entry_price"],
                mode="markers",
                name="trade entries",
                marker=dict(
                    symbol="circle",
                    size=sizes,
                    color=colors,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Entry: %{x}<br>"
                    "Price: %{y:.6f}<br>"
                    "R: %{customdata[1]:.2f}<br>"
                    "Expectancy: %{customdata[2]:.4f}%<br>"
                    "Exit: %{customdata[3]}<br>"
                    "Held: %{customdata[4]:.1f} min<br>"
                    "Bucket: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{CONFIG.instrument} candlestick with trade entry markers",
        xaxis_title=f"Time ({CONFIG.timezone})",
        yaxis_title="Price",
        template="plotly_white",
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=30, t=70, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,120,120,0.12)")
    return fig


def run_candlestick_trade_chart() -> dict[str, str]:
    CONFIG.chart_output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_cache_df(CONFIG.cache_file)
    if raw_df.empty:
        raise RuntimeError(f"No cache data available at {CONFIG.cache_file}")

    trade_df = build_trade_universe(raw_df, CONFIG)
    if trade_df.empty:
        raise RuntimeError("No trades were generated for the selected configuration.")

    fig = build_candlestick_trade_chart(trade_df=trade_df, raw_df=raw_df)

    out_html = CONFIG.chart_output_dir / f"{CONFIG.prefix}_candlestick_trades.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)

    logger.info("Saved chart: %s", out_html)
    return {
        "chart_file": str(out_html),
        "cache_file": str(CONFIG.cache_file),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_candlestick_trade_chart()
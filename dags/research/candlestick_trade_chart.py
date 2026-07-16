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

try:
    from research.exploration_config import CONFIG  # type: ignore
    from research.trade_core import build_trade_universe, load_cache_df  # type: ignore
except ImportError:
    from exploration_config import CONFIG  # type: ignore
    from trade_core import build_trade_universe, load_cache_df  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_CHART_LOOKBACK_DAYS = 30
DEFAULT_CHART_MAX_CANDLES = 3000
DEFAULT_MARKER_SIZE = 10
DEFAULT_MARKER_COLOR = "orange"
DEFAULT_MARKER_LINE_COLOR = "black"


def _resolve_int_attr(name: str, default: int) -> int:
    value = getattr(CONFIG, name, default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_bool_attr(name: str, default: bool) -> bool:
    value = getattr(CONFIG, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _trade_marker_size() -> int:
    return max(4, _resolve_int_attr("chart_marker_size", DEFAULT_MARKER_SIZE))


def _trade_marker_color() -> str:
    return str(getattr(CONFIG, "chart_marker_color", DEFAULT_MARKER_COLOR))


def _trade_marker_line_color() -> str:
    return str(getattr(CONFIG, "chart_marker_line_color", DEFAULT_MARKER_LINE_COLOR))


def _resolve_chart_window(raw_df: pd.DataFrame, trade_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = raw_df.copy()
    trade_df = trade_df.copy()

    tz = getattr(CONFIG, "timezone", "Asia/Kuala_Lumpur")
    raw_df["dt_myt"] = pd.to_datetime(raw_df["time_ns"], unit="ns", utc=True).dt.tz_convert(tz)

    if not trade_df.empty and "entry_time_myt" not in trade_df.columns:
        trade_df["entry_time_myt"] = pd.to_datetime(trade_df["time_ns"], unit="ns", utc=True).dt.tz_convert(tz)

    raw_df = raw_df.sort_values("dt_myt").reset_index(drop=True)

    lookback_days = _resolve_int_attr("chart_lookback_days", DEFAULT_CHART_LOOKBACK_DAYS)
    max_candles = _resolve_int_attr("chart_max_candles", DEFAULT_CHART_MAX_CANDLES)

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
        sizes = np.full(len(trade_df), _trade_marker_size(), dtype=int)
        side_labels = trade_df["side_label"].astype(str).to_numpy() if "side_label" in trade_df.columns else np.array([""] * len(trade_df))
        trade_r = pd.to_numeric(trade_df["trade_r"], errors="coerce").to_numpy(dtype=float, copy=False) if "trade_r" in trade_df.columns else np.full(len(trade_df), np.nan)
        expectancy_pct = pd.to_numeric(trade_df["trade_expectancy_pct"], errors="coerce").to_numpy(dtype=float, copy=False) if "trade_expectancy_pct" in trade_df.columns else np.full(len(trade_df), np.nan)
        exit_type = trade_df["exit_type"].astype(str).to_numpy() if "exit_type" in trade_df.columns else np.array([""] * len(trade_df))
        holding_minutes = pd.to_numeric(trade_df["holding_minutes"], errors="coerce").to_numpy(dtype=float, copy=False) if "holding_minutes" in trade_df.columns else np.full(len(trade_df), np.nan)
        entry_bucket = trade_df["entry_bucket_label"].astype(str).to_numpy() if "entry_bucket_label" in trade_df.columns else np.array([""] * len(trade_df))
        entry_time_str = trade_df["entry_time_myt"].astype(str).to_numpy()
        entry_price_arr = pd.to_numeric(trade_df["entry_price"], errors="coerce").to_numpy(dtype=float, copy=False)

        customdata = np.column_stack(
            [
                side_labels,
                trade_r,
                expectancy_pct,
                exit_type,
                holding_minutes,
                entry_bucket,
                entry_time_str,
                entry_price_arr,
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=trade_df["entry_time_myt"],
                y=trade_df["entry_price"],
                mode="markers",
                name="trade entries",
                marker=dict(
                    symbol="circle",
                    size=sizes,
                    color=_trade_marker_color(),
                    opacity=1.0,
                    line=dict(width=1.4, color=_trade_marker_line_color()),
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Entry: %{x}<br>"
                    "Price: %{customdata[7]:.6f}<br>"
                    "R: %{customdata[1]:.2f}<br>"
                    "Expectancy: %{customdata[2]:.4f}%<br>"
                    "Exit: %{customdata[3]}<br>"
                    "Held: %{customdata[4]:.1f} min<br>"
                    "Bucket: %{customdata[5]}<br>"
                    "Entry time: %{customdata[6]}<extra></extra>"
                ),
            )
        )

        show_labels = _resolve_bool_attr("chart_show_trade_labels", False)
        if show_labels:
            max_labels = _resolve_int_attr("chart_max_trade_labels", 80)
            label_df = trade_df.tail(max_labels).copy()
            fig.add_trace(
                go.Scatter(
                    x=label_df["entry_time_myt"],
                    y=label_df["entry_price"],
                    mode="text",
                    text=[str(i) for i in label_df.index.tolist()],
                    textposition="top center",
                    textfont=dict(size=10, color="orange"),
                    name="trade labels",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{CONFIG.instrument} candlestick with trade entry markers",
        xaxis_title=f"Time ({getattr(CONFIG, 'timezone', 'Asia/Kuala_Lumpur')})",
        yaxis_title="Price",
        template="plotly_white",
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=30, t=70, b=40),
        hovermode="closest",
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
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)

    logger.info("Saved chart: %s", out_html)
    return {
        "chart_file": str(out_html),
        "cache_file": str(CONFIG.cache_file),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_candlestick_trade_chart()

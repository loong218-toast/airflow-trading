
# trailing_tp_compare_runner.py

from __future__ import annotations

import logging
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from research.exploration_config import CONFIG
from research.trade_core import build_trade_universe, load_cache_df, replace_file, safe_name  # type: ignore

logger = logging.getLogger(__name__)


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        out = list(value)
        return out if out else list(default)
    return [value]


def _first_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return value[0] if len(value) else default
    return default if value is None else value


def _root_name() -> str:
    return safe_name(
        f"{CONFIG.instrument}_tp_{str(CONFIG.target_pct).replace('.', 'p')}_"
        f"sl_{str(CONFIG.sl_pct).replace('.', 'p')}_h_{int(CONFIG.horizon_hours)}"
    )


def _core_label(cfg) -> str:
    sim = safe_name(str(cfg.simulation_mode))
    ma = "ma" if bool(cfg.use_ma_filter) else "no_ma"
    return f"{sim}_{ma}"


def _output_dirs() -> tuple[Path, Path, Path]:
    root = CONFIG.output_dir / _root_name()
    data_dir = root / "data"
    plot_dir = root / "plot"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return root, data_dir, plot_dir


def _combo_label(use_trailing_tp: bool, activation_r: float, pct: float, interval: int) -> str:
    if not use_trailing_tp:
        return "baseline"
    return f"trail_a{str(float(activation_r)).replace('.', 'p')}_p{str(float(pct)).replace('.', 'p')}_i{int(interval)}"


def _summarize_trade_df(trade_df: pd.DataFrame, cfg, setup_label: str) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    taken = int(len(trade_df))
    exit_type = trade_df["exit_type"].astype(str) if "exit_type" in trade_df.columns else pd.Series(["censored"] * taken)
    trade_r = pd.to_numeric(trade_df["trade_r"], errors="coerce") if "trade_r" in trade_df.columns else pd.Series(dtype=float)

    resolved_mask = exit_type.ne("censored")
    censored_mask = ~resolved_mask
    trailing_mask = exit_type.eq("trailing")

    resolved_count = int(resolved_mask.sum())
    censored_count = int(censored_mask.sum())
    trailing_count = int(trailing_mask.sum())

    resolved_rate_pct = float((resolved_count / taken) * 100.0) if taken > 0 else np.nan
    censored_rate_pct = float((censored_count / taken) * 100.0) if taken > 0 else np.nan

    resolved_r_mean = float(trade_r[resolved_mask].mean()) if resolved_count > 0 else np.nan
    net_r_mean = float(trade_r.mean()) if taken > 0 else np.nan
    forced_exit_r_mean = float(trade_r[censored_mask].mean()) if censored_count > 0 else np.nan

    expectancy_resolved_pct = resolved_r_mean * resolved_rate_pct * float(cfg.risk_pct) if np.isfinite(resolved_r_mean) else np.nan
    net_expectancy_pct = net_r_mean * float(cfg.risk_pct) * 100.0 if np.isfinite(net_r_mean) else np.nan

    def _mean_col(col: str) -> float:
        if col not in trade_df.columns:
            return np.nan
        s = pd.to_numeric(trade_df[col], errors="coerce").dropna()
        return float(s.mean()) if not s.empty else np.nan

    trailing_triggered = (
        pd.Series(trade_df["trailing_tp_triggered"], index=trade_df.index)
        if "trailing_tp_triggered" in trade_df.columns
        else pd.Series(False, index=trade_df.index)
    ).fillna(False).astype(bool)

    trailing_moved_r = (
        pd.to_numeric(trade_df["trailing_tp_moved_r"], errors="coerce")
        if "trailing_tp_moved_r" in trade_df.columns
        else pd.Series(np.nan, index=trade_df.index)
    )

    trailing_moved_r_resolved = trailing_moved_r.fillna(0.0)
    trailing_moved_r_triggered = trailing_moved_r[trailing_triggered]

    trailing_tp_trigger_count = int(trailing_triggered.sum())
    trailing_tp_trigger_rate_pct = float((trailing_tp_trigger_count / taken) * 100.0) if taken > 0 else np.nan

    row = {
        "instrument": cfg.instrument,
        "source_symbol": cfg.mt5_symbol,
        "pair": cfg.pair,
        "compare_mode": "trailing_tp",
        "sim_mode": cfg.simulation_mode,
        "side_mode": cfg.side_mode,
        "use_entry_time_window": bool(cfg.use_entry_time_window),
        "entry_window_start_hour": int(cfg.entry_window_start_hour_myt),
        "entry_window_end_hour": int(cfg.entry_window_end_hour_myt),
        "entry_window_label": f"{cfg.entry_window_start_hour_myt:02d}:00-{cfg.entry_window_end_hour_myt:02d}:00 MYT",
        "target_pct": float(cfg.target_pct),
        "sl_pct": float(cfg.sl_pct),
        "spread_pct": float(cfg.spread_pct),
        "horizon_hours": int(cfg.horizon_hours),
        "use_trailing_tp": bool(cfg.use_trailing_tp),
        "trailing_tp_activation_r": float(_first_value(cfg.trailing_tp_activation_r, 0.3)),
        "trailing_tp_distance_r": float(_first_value(cfg.trailing_tp_distance_r, 0.2)),
        "trailing_tp_interval": int(_first_value(cfg.trailing_tp_interval, 3)),
        "setup_label": setup_label,
        "trades_taken_count": taken,
        "resolved_trades_count": resolved_count,
        "resolved_trades_pct": resolved_rate_pct,
        "censored_trades_count": censored_count,
        "censored_trades_pct": censored_rate_pct,
        "trailing_tp_count": trailing_count,
        "trailing_tp_rate_pct": float((trailing_mask.mean() * 100.0)) if taken > 0 else np.nan,
        "trailing_tp_trigger_count": trailing_tp_trigger_count,
        "trailing_tp_trigger_rate_pct": trailing_tp_trigger_rate_pct,
        "trailing_tp_moved_r_mean_resolved": float(trailing_moved_r_resolved[resolved_mask].mean()) if resolved_count > 0 else np.nan,
        "trailing_tp_moved_r_mean_triggered": float(trailing_moved_r_triggered.mean()) if trailing_tp_trigger_count > 0 else np.nan,
        "trailing_tp_moved_r_median_triggered": float(trailing_moved_r_triggered.median()) if trailing_tp_trigger_count > 0 else np.nan,
        "expectancy_resolved_pct": float(expectancy_resolved_pct) if np.isfinite(expectancy_resolved_pct) else np.nan,
        "net_expectancy_pct": float(net_expectancy_pct) if np.isfinite(net_expectancy_pct) else np.nan,
        "target_minutes_mean": _mean_col("target_minutes"),
        "sl_minutes_mean": _mean_col("sl_minutes"),
        "first_event_minutes_mean": _mean_col("first_event_minutes"),
        "forced_exit_r_mean": float(forced_exit_r_mean) if np.isfinite(forced_exit_r_mean) else np.nan,
        "forced_exit_r_count": int(censored_count),
        "loss_streak_trigger_count": int(trade_df.attrs.get("loss_streak_trigger_count", 0)),
        "skipped_after_loss_count": int(trade_df.attrs.get("skipped_after_loss_count", 0)),
    }
    return pd.DataFrame([row])


def _save_bar_plot(summary_df: pd.DataFrame, metric: str, out_html: Path, title: str) -> None:
    df = summary_df.copy().sort_values(["setup_label"]).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["setup_label"].tolist(), y=pd.to_numeric(df[metric], errors="coerce").tolist(), name=metric))
    fig.update_layout(
        title=title,
        xaxis_title="setup",
        yaxis_title=metric,
        template="plotly_white",
        bargap=0.25,
        height=650,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def _save_trailing_moved_plot(trade_df: pd.DataFrame, out_html: Path, title: str) -> None:
    if trade_df.empty or "trailing_tp_moved_r" not in trade_df.columns:
        return

    df = trade_df.copy()
    df["trailing_tp_moved_r"] = pd.to_numeric(df["trailing_tp_moved_r"], errors="coerce").fillna(0.0)

    fig = go.Figure()
    for label, grp in df.groupby("setup_label", sort=True):
        fig.add_trace(go.Box(y=grp["trailing_tp_moved_r"].to_list(), name=str(label), boxmean=True))

    fig.update_layout(
        title=title,
        xaxis_title="setup",
        yaxis_title="trailing_tp_moved_r (R)",
        template="plotly_white",
        height=650,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def _save_3d_heatmap(summary_df: pd.DataFrame, metric: str, out_html: Path, title: str) -> None:
    df = summary_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["trailing_tp_activation_r", "trailing_tp_distance_r", "trailing_tp_interval", metric]).copy()
    if df.empty:
        return

    for col in ["trailing_tp_activation_r", "trailing_tp_distance_r", "trailing_tp_interval"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cmin = float(df[metric].min())
    cmax = float(df[metric].max())

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["trailing_tp_activation_r"],
                y=df["trailing_tp_distance_r"],
                z=df["trailing_tp_interval"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=df[metric],
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=metric),
                    opacity=0.9,
                ),
                text=df["setup_label"],
                hovertemplate=(
                    "setup=%{text}<br>"
                    "activation_r=%{x}<br>"
                    "distance_r=%{y}<br>"
                    "interval=%{z}<br>"
                    f"{metric}=%{{marker.color:.6f}}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="activation_r", yaxis_title="distance_r", zaxis_title="interval"),
        template="plotly_white",
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def _grid_complete(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return {"trailing_tp_activation_r", "trailing_tp_distance_r", "trailing_tp_interval"}.issubset(df.columns)


def run_trailing_tp_compare() -> dict[str, str]:
    _, data_dir, plot_dir = _output_dirs()

    df = load_cache_df(CONFIG.cache_file)
    if df.empty:
        raise RuntimeError(f"No cache data available at {CONFIG.cache_file}")

    activation_values = _as_list(CONFIG.trailing_tp_activation_r, [0.3])
    trailing_pct_values = _as_list(CONFIG.trailing_tp_distance_r, [0.2])
    trailing_interval_values = _as_list(CONFIG.trailing_tp_interval, [3])

    runs: List[Tuple[str, Any]] = []
    baseline_cfg = replace(
        CONFIG,
        use_trailing_tp=False,
        trailing_tp_activation_r=float(_first_value(activation_values, 0.3)),
        trailing_tp_distance_r=tuple(trailing_pct_values[:1]),
        trailing_tp_interval=tuple(trailing_interval_values[:1]),
    )
    runs.append(("baseline", baseline_cfg))

    for activation_r, pct, interval in product(activation_values, trailing_pct_values, trailing_interval_values):
        cfg = replace(
            CONFIG,
            use_trailing_tp=True,
            trailing_tp_activation_r=float(activation_r),
            trailing_tp_distance_r=(float(pct),),
            trailing_tp_interval=(int(interval),),
        )
        runs.append((_combo_label(True, float(activation_r), float(pct), int(interval)), cfg))

    summaries: List[pd.DataFrame] = []
    trade_exports: List[pd.DataFrame] = []

    for label, cfg in runs:
        logger.info("Running setup: %s", label)
        trade_df = build_trade_universe(df=df, config=cfg)
        if trade_df.empty:
            logger.warning("No trades for setup: %s", label)
            continue

        trade_df = trade_df.copy()
        trade_df["setup_label"] = label
        trade_df["trailing_tp_activation_r"] = float(_first_value(cfg.trailing_tp_activation_r, 0.3))
        trade_df["trailing_tp_distance_r"] = float(_first_value(cfg.trailing_tp_distance_r, 0.2))
        trade_df["trailing_tp_interval"] = int(_first_value(cfg.trailing_tp_interval, 3))

        summaries.append(_summarize_trade_df(trade_df, cfg, label))

        if CONFIG.save_trade_universe_csv:
            out_trade = data_dir / f"{safe_name(label)}_trade_universe.csv"
            replace_file(out_trade)
            trade_df.to_csv(out_trade, index=False)

        trade_exports.append(trade_df)

    if not summaries:
        raise RuntimeError("No summary rows were produced.")

    summary_df = pd.concat(summaries, ignore_index=True).round(6).sort_values(["setup_label"]).reset_index(drop=True)

    core = _core_label(CONFIG)
    summary_file = data_dir / f"{core}_trailing_tp_compare_summary.csv"
    replace_file(summary_file)
    summary_df.to_csv(summary_file, index=False)

    if trade_exports:
        all_trade_df = pd.concat(trade_exports, ignore_index=True)
        trade_file = data_dir / f"{core}_trailing_tp_trade_level_compare.csv"
        replace_file(trade_file)
        all_trade_df.to_csv(trade_file, index=False)
        moved_plot = plot_dir / f"{core}_trailing_tp_moved_r_by_setup.html"
        _save_trailing_moved_plot(all_trade_df, moved_plot, "Trailing TP moved in R by setup")
    else:
        trade_file = Path()
        moved_plot = Path()

    resolved_plot = plot_dir / f"{core}_trailing_tp_resolved_expectancy.html"
    net_plot = plot_dir / f"{core}_trailing_tp_net_expectancy.html"
    heatmap_plot = plot_dir / f"{core}_trailing_tp_3d_heatmap.html"

    _save_bar_plot(summary_df, "expectancy_resolved_pct", resolved_plot, "Resolved expectancy comparison")
    _save_bar_plot(summary_df, "net_expectancy_pct", net_plot, "Net expectancy comparison")

    heatmap_source = summary_df[summary_df["setup_label"].ne("baseline")].copy()
    if heatmap_source.empty:
        heatmap_source = summary_df.copy()

    if _grid_complete(heatmap_source):
        _save_3d_heatmap(heatmap_source, "net_expectancy_pct", heatmap_plot, "3D trailing-TP heatmap")

    logger.info("Saved summary: %s", summary_file)
    if trade_file:
        logger.info("Saved trade-level compare: %s", trade_file)
    if moved_plot:
        logger.info("Saved trailing moved plot: %s", moved_plot)
    logger.info("Saved plots: %s, %s", resolved_plot, net_plot)
    if heatmap_plot.exists():
        logger.info("Saved 3D heatmap: %s", heatmap_plot)

    return {
        "summary_file": str(summary_file),
        "trade_file": str(trade_file) if trade_file else "",
        "resolved_plot": str(resolved_plot),
        "net_plot": str(net_plot),
        "moved_plot": str(moved_plot) if moved_plot else "",
        "heatmap_plot": str(heatmap_plot) if heatmap_plot.exists() else "",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_trailing_tp_compare()


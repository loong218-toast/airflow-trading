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
from research.trade_core import (  # type: ignore
    bootstrap_mean_samples,
    build_trade_universe,
    load_cache_df,
    replace_file,
    safe_name,
)

logger = logging.getLogger(__name__)


def save_histogram(
    values: np.ndarray,
    out_html: Path,
    title: str,
    xlabel: str,
    observed: float,
    extra_vlines=None,
) -> None:
    import plotly.graph_objects as go

    extra_vlines = list(extra_vlines or [])
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=40,
            name="bootstrap sample means",
            opacity=0.85,
        )
    )

    fig.add_vline(x=observed, line_width=2, line_dash="solid", line_color="black")
    for x in extra_vlines:
        if np.isfinite(x):
            fig.add_vline(x=float(x), line_width=1.5, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.02,
        height=650,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)


def build_summary(trade_df: pd.DataFrame, bootstrap_samples_r: np.ndarray) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    attempted_count = int(trade_df.attrs.get("attempted_count", len(trade_df)))
    resolved_count = int(trade_df.attrs.get("resolved_count", len(trade_df)))
    censored_count = int(trade_df.attrs.get("censored_count", max(0, attempted_count - resolved_count)))

    resolved_rate_pct = float((resolved_count / attempted_count) * 100.0) if attempted_count > 0 else np.nan
    censored_rate_pct = float((censored_count / attempted_count) * 100.0) if attempted_count > 0 else np.nan
    bootstrap_used_count = int(len(trade_df))
    bootstrap_used_pct = float((bootstrap_used_count / attempted_count) * 100.0) if attempted_count > 0 else np.nan

    observed_mean_r = float(trade_df["trade_r"].mean())
    observed_median_r = float(trade_df["trade_r"].median())
    observed_std_r = float(trade_df["trade_r"].std(ddof=1)) if len(trade_df) > 1 else 0.0

    win_rate_pct = float((trade_df["trade_r"] > 0).mean() * 100.0)
    loss_rate_pct = float((trade_df["trade_r"] < 0).mean() * 100.0)
    flat_rate_pct = float((trade_df["trade_r"] == 0).mean() * 100.0)

    if bootstrap_samples_r.size > 0:
        b_mean = float(np.mean(bootstrap_samples_r))
        b_median = float(np.median(bootstrap_samples_r))
        b_std = float(np.std(bootstrap_samples_r, ddof=1)) if bootstrap_samples_r.size > 1 else 0.0
        b_p05 = float(np.percentile(bootstrap_samples_r, 5))
        b_p10 = float(np.percentile(bootstrap_samples_r, 10))
        b_p25 = float(np.percentile(bootstrap_samples_r, 25))
        b_p75 = float(np.percentile(bootstrap_samples_r, 75))
        b_p90 = float(np.percentile(bootstrap_samples_r, 90))
        b_p95 = float(np.percentile(bootstrap_samples_r, 95))
        prob_negative_pct = float((bootstrap_samples_r < 0).mean() * 100.0)
    else:
        b_mean = b_median = b_std = b_p05 = b_p10 = b_p25 = b_p75 = b_p90 = b_p95 = prob_negative_pct = np.nan

    row = {
        "instrument": CONFIG.instrument,
        "source_symbol": CONFIG.mt5_symbol,
        "pair": CONFIG.pair,
        "target_pct": float(CONFIG.target_pct),
        "sl_pct": float(CONFIG.sl_pct),
        "horizon_hours": int(CONFIG.horizon_hours),
        "risk_pct": float(CONFIG.risk_pct),
        "use_ma_filter": bool(CONFIG.use_ma_filter),
        "randomize_entry_price": bool(CONFIG.randomize_entry_price),
        "use_entry_bucket_hours": bool(CONFIG.use_entry_bucket_hours),
        "n_trades": int(len(trade_df)),
        "attempted_count": attempted_count,
        "resolved_count": resolved_count,
        "censored_count": censored_count,
        "resolved_rate_pct": resolved_rate_pct,
        "censored_rate_pct": censored_rate_pct,
        "bootstrap_used_count": bootstrap_used_count,
        "bootstrap_used_pct": bootstrap_used_pct,
        "observed_mean_r": observed_mean_r,
        "observed_median_r": observed_median_r,
        "observed_std_r": observed_std_r,
        "observed_expectancy_pct": observed_mean_r * CONFIG.risk_pct * 100.0,
        "observed_median_expectancy_pct": observed_median_r * CONFIG.risk_pct * 100.0,
        "win_rate_pct": win_rate_pct,
        "loss_rate_pct": loss_rate_pct,
        "flat_rate_pct": flat_rate_pct,
        "bootstrap_rounds": int(CONFIG.bootstrap_rounds),
        "bootstrap_block_size": int(CONFIG.bootstrap_block_size),
        "bootstrap_mean_r": b_mean,
        "bootstrap_median_r": b_median,
        "bootstrap_std_r": b_std,
        "bootstrap_p05_r": b_p05,
        "bootstrap_p10_r": b_p10,
        "bootstrap_p25_r": b_p25,
        "bootstrap_p75_r": b_p75,
        "bootstrap_p90_r": b_p90,
        "bootstrap_p95_r": b_p95,
        "bootstrap_prob_negative_pct": prob_negative_pct,
        "bootstrap_mean_expectancy_pct": b_mean * CONFIG.risk_pct * 100.0 if np.isfinite(b_mean) else np.nan,
        "bootstrap_median_expectancy_pct": b_median * CONFIG.risk_pct * 100.0 if np.isfinite(b_median) else np.nan,
        "bootstrap_p05_expectancy_pct": b_p05 * CONFIG.risk_pct * 100.0 if np.isfinite(b_p05) else np.nan,
        "bootstrap_p25_expectancy_pct": b_p25 * CONFIG.risk_pct * 100.0 if np.isfinite(b_p25) else np.nan,
        "bootstrap_p75_expectancy_pct": b_p75 * CONFIG.risk_pct * 100.0 if np.isfinite(b_p75) else np.nan,
        "bootstrap_p95_expectancy_pct": b_p95 * CONFIG.risk_pct * 100.0 if np.isfinite(b_p95) else np.nan,
    }
    return pd.DataFrame([row])


def run_bootstrap_expectancy_scan() -> dict[str, str]:
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cache_df(CONFIG.cache_file)
    if df.empty:
        raise RuntimeError(f"No cache data available at {CONFIG.cache_file}")

    logger.info("Using instrument: %s", CONFIG.instrument)
    logger.info("Cache file: %s", CONFIG.cache_file)
    logger.info(
        "Selected setup: TP=%.4f%% SL=%.4f%% Horizon=%dh",
        CONFIG.target_pct,
        CONFIG.sl_pct,
        CONFIG.horizon_hours,
    )

    trade_df = build_trade_universe(df=df, config=CONFIG)
    if trade_df.empty:
        raise RuntimeError("No trades were generated for the selected configuration.")

    if CONFIG.save_trade_universe_csv:
        trade_df.to_csv(CONFIG.output_dir / "trade_universe.csv", index=False)

    trade_r = trade_df["trade_r"].to_numpy(dtype=np.float64, copy=False)
    bootstrap_samples_r = bootstrap_mean_samples(
        values=trade_r,
        n_bootstrap=CONFIG.bootstrap_rounds,
        block_size=CONFIG.bootstrap_block_size,
        seed=CONFIG.bootstrap_seed,
    )

    summary_df = build_summary(trade_df, bootstrap_samples_r)
    if summary_df.empty:
        raise RuntimeError("No summary produced.")

    summary_file = CONFIG.output_dir / f"{CONFIG.prefix}_summary.csv"
    plot_file = CONFIG.output_dir / f"{CONFIG.prefix}_hist.html"

    replace_file(summary_file)
    summary_df.to_csv(summary_file, index=False)

    observed = float(summary_df.iloc[0]["observed_mean_r"])
    b_median = float(summary_df.iloc[0]["bootstrap_median_r"])
    b_p05 = float(summary_df.iloc[0]["bootstrap_p05_r"])
    b_p95 = float(summary_df.iloc[0]["bootstrap_p95_r"])

    save_histogram(
        values=bootstrap_samples_r,
        out_html=plot_file,
        title=(
            f"{CONFIG.instrument} | TP {CONFIG.target_pct:.4g}% | SL {CONFIG.sl_pct:.4g}% | "
            f"{int(CONFIG.horizon_hours)}h | bootstrap expectancy"
        ),
        xlabel="Bootstrap sample mean expectancy in R",
        observed=observed,
        extra_vlines=[b_median, b_p05, b_p95],
    )

    if CONFIG.save_bootstrap_samples_csv:
        pd.DataFrame({"bootstrap_sample_mean_r": bootstrap_samples_r}).to_csv(
            CONFIG.output_dir / f"{CONFIG.prefix}_bootstrap_samples.csv",
            index=False,
        )

    logger.info("Saved summary: %s", summary_file)
    logger.info("Saved plot: %s", plot_file)

    return {
        "summary_file": str(summary_file),
        "plot_file": str(plot_file),
        "cache_file": str(CONFIG.cache_file),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bootstrap_expectancy_scan()

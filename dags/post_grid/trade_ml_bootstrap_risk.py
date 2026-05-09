# trade_ml_bootstrap_risk.py

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import os
import numpy as np
import polars as pl

HERE = Path(__file__).resolve()
DAGS_ROOT = HERE.parents[1]  # .../dags
if str(DAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(DAGS_ROOT))

from research.bootstrap_core import (
    bootstrap_equity_drawdown_samples,
    describe_numeric_series,
    equity_curve_from_returns,
    max_drawdown_pct_from_equity,
    safe_name,
    save_histogram,
)
from post_grid.post_grid_config import (
    REGIME_ID,
    SESSION_NAME,        # This is the hardcoded one
    DATA_LAKE_ROOT,      # Add this import
    analysis_root,
    trade_ml_partitioned_root,
)

from airflow.sdk import dag, task, Variable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("trade_ml_bootstrap_risk")

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_BLOCK_SIZE = 20
BOOTSTRAP_SEED = 14121212

pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(500)
pl.Config.set_fmt_str_lengths(80)


def _regime_label() -> str:
    return f"regime_{REGIME_ID}" if REGIME_ID is not None else "all_regimes"

def _get_session_dir() -> Path:
    """
    Check Airflow Variable, then Env, then hardcoded config.
    """
    # 1. Try Airflow Variable
    # 2. Try Environment Variable
    # 3. Fallback to the SESSION_NAME imported from post_grid_config
    session = Variable.get("SESSION_NAME", default=os.getenv("SESSION_NAME", SESSION_NAME))
    
    if not session:
        raise ValueError("SESSION_NAME is required (Airflow Variable, env, or config)")
        
    return Path(DATA_LAKE_ROOT) / session

def _load_trade_ml_df() -> pl.DataFrame:
    base_path = _get_session_dir() / "trade_ml_partitioned"
    files = sorted(base_path.rglob("*.parquet"))

    logger.info(f"Session: {SESSION_NAME}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Found {len(files)} parquet files.")

    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in {base_path}")

    # 1. Scan the files
    lf = pl.scan_parquet([str(p) for p in files])

    # --- DEBUGGING: Check what regimes actually exist in the data ---
    # We collect a small sample to see what's available
    available_regimes = lf.select("regime_id").unique().collect().get_column("regime_id").to_list()
    logger.info(f"Available regimes in data: {available_regimes}")
    logger.info(f"Filtering for REGIME_ID: {REGIME_ID} (type: {type(REGIME_ID)})")
    # ----------------------------------------------------------------

    cols = [
        "regime_id", "signal_layer", "signal_scope_id", "era_int",
        "side", "SL", "TP", "SL_hit", "TP_hit",
        "signal_idx", "signal_time_ns", "signal_price",
        "order_idx", "order_time_ns", "order_price",
        "order_mode", "fill_status",
        "entry_idx", "entry_time_ns", "entry_price",
        "exit_idx", "exit_time_ns", "exit_price",
        "exit_reason", "fill_delay_bars", "pnl_pct",
    ]

    # 2. Apply Filter
    if REGIME_ID is not None:
        # We use cast(pl.Int64) to ensure types match during comparison
        lf = lf.filter(pl.col("regime_id").cast(pl.Int64) == int(REGIME_ID))

    df = lf.select(cols).collect(engine="streaming")

    if df.is_empty():
        # This provides a much more helpful error message
        raise RuntimeError(
            f"Filter result is empty! You requested REGIME_ID={REGIME_ID}, "
            f"but the files only contain regimes: {available_regimes}"
        )

    sort_cols = [c for c in ["era_int", "entry_time_ns", "entry_idx", "signal_idx", "exit_idx"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    return df


def _summary_row(df: pl.DataFrame, pnl: np.ndarray, actual_equity: np.ndarray, bootstrap: dict[str, np.ndarray]) -> dict[str, Any]:
    pnl_pct = pnl * 100.0
    total_rows = int(df.height)
    
    # Calculate actual streaks
    from research.bootstrap_core import get_max_streaks
    act_wins, act_losses = get_max_streaks(pnl)

    pnl_stats = describe_numeric_series(pnl_pct)
    terminal_balance = float(actual_equity[-1]) if actual_equity.size else 100.0
    actual_max_dd_pct = float(max_drawdown_pct_from_equity(actual_equity)) if actual_equity.size else np.nan

    term_stats = describe_numeric_series(bootstrap["terminal_balance"])
    dd_stats = describe_numeric_series(bootstrap["max_drawdown_pct"])
    win_streak_stats = describe_numeric_series(bootstrap["max_con_wins"])
    loss_streak_stats = describe_numeric_series(bootstrap["max_con_losses"])

    row = {
        "session_name": SESSION_NAME,
        "regime_id": int(REGIME_ID) if REGIME_ID is not None else -1,
        "actual_max_con_wins": act_wins,
        "actual_max_con_losses": act_losses,
        "bootstrap_max_con_wins_mean": win_streak_stats["mean"],
        "bootstrap_max_con_wins_p95": win_streak_stats["p95"],
        "bootstrap_max_con_losses_mean": loss_streak_stats["mean"],
        "bootstrap_max_con_losses_p95": loss_streak_stats["p95"],
        # ... (keep all your existing rows) ...
        "actual_terminal_balance": terminal_balance,
        "actual_max_drawdown_pct": actual_max_dd_pct,
    }
    # Merge existing stats into row (simplified for brevity, keep your existing logic)
    return row


def main() -> None:
    # 1. Load the dataframe 
    # (Note: pnl_pct here is now the already-scaled Account PnL from the grid search)
    df = _load_trade_ml_df()
    if df.is_empty():
        raise RuntimeError("No trade_ml rows found after loading/filtering.")

    out_dir = analysis_root() / "trade_ml_bootstrap" / _regime_label()
    out_dir.mkdir(parents=True, exist_ok=True)

    from research.bootstrap_core import get_max_streaks

    # 2. Extract PnL
    # We use the raw column directly because the grid search already applied the leverage
    pnl = df["pnl_pct"].cast(pl.Float64).to_numpy()
    pnl = pnl[np.isfinite(pnl)]
    
    actual_equity = equity_curve_from_returns(pnl, start_balance=100.0)
    act_wins, act_losses = get_max_streaks(pnl)

    bootstrap = bootstrap_equity_drawdown_samples(
        returns=pnl,
        n_bootstrap=BOOTSTRAP_ROUNDS,
        block_size=BOOTSTRAP_BLOCK_SIZE,
        seed=BOOTSTRAP_SEED,
        start_balance=100.0,
    )

    # 4. Generate Summary
    summary_row = _summary_row(df=df, pnl=pnl, actual_equity=actual_equity, bootstrap=bootstrap)
    summary_df = pl.DataFrame([summary_row])

    # 5. Save Results
    summary_out = out_dir / f"{safe_name(_regime_label())}_bootstrap_summary.csv"
    pnl_hist_out = out_dir / f"{safe_name(_regime_label())}_pnl_distribution.png"
    balance_hist_out = out_dir / f"{safe_name(_regime_label())}_bootstrap_terminal_balance_hist.png"
    dd_hist_out = out_dir / f"{safe_name(_regime_label())}_bootstrap_max_drawdown_hist.png"

    summary_df.write_csv(str(summary_out))

    # observed_pnl_mean here will now correctly represent Account Mean %
    pnl_plot_vals = pnl * 100.0
    observed_pnl_mean = float(np.mean(pnl_plot_vals)) if pnl_plot_vals.size else np.nan
    
    save_histogram(
        values=pnl_plot_vals,
        out_png=pnl_hist_out,
        title=f"{_regime_label()} | Account PnL distribution",
        xlabel="Account PnL (%)",
        observed=observed_pnl_mean,
        bins=60,
    )

    save_histogram(
        values=bootstrap["terminal_balance"],
        out_png=balance_hist_out,
        title=f"{_regime_label()} | Bootstrap terminal balance",
        xlabel="Terminal balance",
        observed=float(actual_equity[-1]) if actual_equity.size else np.nan,
        bins=50,
    )

    save_histogram(
        values=bootstrap["max_drawdown_pct"],
        out_png=dd_hist_out,
        title=f"{_regime_label()} | Bootstrap max drawdown",
        xlabel="Max drawdown (%)",
        observed=float(max_drawdown_pct_from_equity(actual_equity)) if actual_equity.size else np.nan,
        bins=50,
    )

    win_streak_hist_out = out_dir / f"{safe_name(_regime_label())}_bootstrap_max_con_wins_hist.png"
    loss_streak_hist_out = out_dir / f"{safe_name(_regime_label())}_bootstrap_max_con_losses_hist.png"

    save_histogram(
        values=bootstrap["max_con_wins"],
        out_png=win_streak_hist_out,
        title=f"{_regime_label()} | Bootstrap Max Consecutive Wins",
        xlabel="Consecutive Wins",
        observed=float(act_wins),
        bins=20,
    )

    save_histogram(
        values=bootstrap["max_con_losses"],
        out_png=loss_streak_hist_out,
        title=f"{_regime_label()} | Bootstrap Max Consecutive Losses",
        xlabel="Consecutive Losses",
        observed=float(act_losses),
        bins=20,
    )

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
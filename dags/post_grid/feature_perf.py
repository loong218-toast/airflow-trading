# post_grid/feature_perf.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import polars as pl

# 1. FIX: Use absolute imports relative to the 'dags' folder
from post_grid.feature_perf_io import compare_root, load_master_df, load_session_run_config, write_csv
from post_grid.feature_perf_metrics import (
    aggregate_master_combos,
    build_baseline_vs_signal_comparisons,
    get_risk_pct,
    prepare_master,
    summarize_comparisons,
)
from post_grid.post_grid_config import (
    DATA_LAKE_ROOT,
    MASTER_FILENAME,
    MIN_MASTER_BALANCE,
    MIN_TOTAL_POS,
    RUN_CONFIG_FILENAME,
    SESSION_NAME,
    TRADE_FLIP_ON_ENTRY_FILTER,
    TRADE_OVERLAP_FILTER,
)

pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(500)
pl.Config.set_fmt_str_lengths(80)

def main() -> None:
    # 2. FIX: Move Variable.get INSIDE main. 
    # This prevents Airflow from hitting the DB every 30 seconds during DAG parsing.
    from airflow.sdk import Variable
    current_session = Variable.get("SESSION_NAME", default=os.getenv("SESSION_NAME", SESSION_NAME))

    run_cfg, run_cfg_path, session_dir = load_session_run_config(
        session_name=current_session, 
        data_lake_root=DATA_LAKE_ROOT,
        run_config_filename=RUN_CONFIG_FILENAME,
    )
    risk_pct = get_risk_pct(run_cfg)

    print(f"Session: {session_dir}")
    print(f"Loaded run_config: {run_cfg_path}")
    print(f"risk_pct: {risk_pct}")

    df_master = load_master_df(session_dir, master_filename=MASTER_FILENAME)
    print(f"Master rows: {df_master.height}")

    df_master = prepare_master(
        df_master,
        risk_pct=risk_pct,
        trade_overlap_filter=TRADE_OVERLAP_FILTER,
        trade_flip_on_entry_filter=TRADE_FLIP_ON_ENTRY_FILTER,
    )

    if "total_pos" in df_master.columns:
        df_master = df_master.filter(pl.col("total_pos") >= MIN_TOTAL_POS)
    if "balance" in df_master.columns:
        df_master = df_master.filter(pl.col("balance") >= MIN_MASTER_BALANCE)

    combo_agg = aggregate_master_combos(df_master, risk_pct=risk_pct)
    cmp_df = build_baseline_vs_signal_comparisons(combo_agg)
    summary_df = summarize_comparisons(cmp_df)

    compare_dir = compare_root(session_dir)
    compare_dir.mkdir(parents=True, exist_ok=True)

    combo_out = compare_dir / "combo_aggregate_summary.csv"
    cmp_out = compare_dir / "baseline_vs_signal_comparisons.csv"
    summary_out = compare_dir / "baseline_vs_signal_summary.csv"

    write_csv(combo_agg, combo_out)
    write_csv(cmp_df, cmp_out)
    write_csv(summary_df, summary_out)

    print("\nSaved:")
    print(combo_out)
    print(cmp_out)
    print(summary_out)

if __name__ == "__main__":
    main()
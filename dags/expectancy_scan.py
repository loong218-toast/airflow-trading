# expectancy_scan.py

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pendulum
from airflow.sdk import dag, task, Variable

PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("airflow.task")


@dag(
    dag_id="expectancy_scan",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "loong", "retries": 0, "retry_delay": timedelta(minutes=10)},
    tags=["expectancy", "mt5"],
)
def expectancy_scan_dag():
    @task()
    def build_summary_task() -> str:
        from research.expectancy_analysis import build_expectancy_summary_from_cache, save_expectancy_summary_csv
        from research.expectancy_config import (
            CACHE_FILE,
            ENTRY_BUCKET_HOURS,
            HORIZON_HOURS_LIST,
            INSTRUMENT,
            MT5_SYMBOL,
            PAIR,
            SL_PCT_LIST,
            TARGET_PCT_LIST,
            USE_ENTRY_BUCKET_HOURS,
            USE_MA_FILTER,
        )

        bucket_mode = USE_ENTRY_BUCKET_HOURS
        time_bucket_hours = ENTRY_BUCKET_HOURS if bucket_mode else 24

        logger.info("Starting expectancy scan")
        logger.info("Cache file: %s", CACHE_FILE)

        summary_df = build_expectancy_summary_from_cache(
            cache_file=CACHE_FILE,
            instrument=INSTRUMENT,
            source_symbol=MT5_SYMBOL,
            pair=PAIR,
            target_pct_list=TARGET_PCT_LIST,
            sl_pct_list=SL_PCT_LIST,
            horizon_hours_list=HORIZON_HOURS_LIST,
            time_bucket_hours=time_bucket_hours,
            include_entry_bucket_hours=bucket_mode,
            use_ma_filter=USE_MA_FILTER,
        )

        if summary_df.empty:
            raise RuntimeError("No summary produced.")

        output_file = save_expectancy_summary_csv(summary_df=summary_df, instrument=INSTRUMENT)
        logger.info("Saved summary CSV to: %s", output_file)
        return str(output_file)

    @task()
    def save_mae_mfe_plots_task(summary_file: str) -> list[str]:
        from research.expectancy_analysis import save_mae_mfe_distribution_plots
        from research.expectancy_config import INSTRUMENT, MT5_SYMBOL, PAIR

        summary_df = pd.read_csv(summary_file)
        plot_files = save_mae_mfe_distribution_plots(
            summary_df=summary_df,
            instrument=INSTRUMENT,
            pair=PAIR,
            symbol=MT5_SYMBOL,
        )
        logger.info("MAE/MFE plots saved: %d", len(plot_files))
        return [str(p) for p in plot_files]

    @task()
    def save_heatmaps_task(summary_file: str) -> list[str]:
        from research.expectancy_analysis import save_net_expectancy_tp_sl_plot
        from research.expectancy_config import INSTRUMENT, MT5_SYMBOL, PAIR, USE_ENTRY_BUCKET_HOURS

        summary_df = pd.read_csv(summary_file)
        plot_files = save_net_expectancy_tp_sl_plot(
            summary_df=summary_df,
            instrument=INSTRUMENT,
            pair=PAIR,
            symbol=MT5_SYMBOL,
            include_entry_bucket_hours=USE_ENTRY_BUCKET_HOURS,
        )
        logger.info("Heatmaps saved: %d", len(plot_files))
        return [str(p) for p in plot_files]

    summary_file = build_summary_task()
    save_mae_mfe_plots_task(summary_file)
    save_heatmaps_task(summary_file)


expectancy_scan_dag = expectancy_scan_dag()

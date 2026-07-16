from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pendulum
from airflow.sdk import dag, task

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
    def build_summary_task() -> list[str]:
        from research.expectancy_analysis import (
            build_expectancy_summary_from_cache,
            save_expectancy_summary_csv,
        )
        from research.expectancy_config import (
            AFTER_LOSS_SKIP_TRADES_LIST,
            AFTER_LOSS_TRIGGER_COUNT_LIST,
            CACHE_FILE,
            ENTRY_WINDOW_END_HOUR_MYT,
            ENTRY_WINDOW_START_HOUR_MYT,
            HORIZON_HOURS_LIST,
            INSTRUMENT,
            LOOKBACK_DAYS_LIST,
            MT5_SYMBOL,
            PAIR,
            SL_PCT_LIST,
            TARGET_PCT_LIST,
            TRADES_PER_HOUR,
            USE_AFTER_LOSS_FILTER,
            USE_ENTRY_TIME_WINDOW,
            USE_MA_FILTER,
        )

        output_files: list[str] = []

        for lookback_days in LOOKBACK_DAYS_LIST:
            logger.info("Starting expectancy scan for lookback=%s", lookback_days)

            summary_df = build_expectancy_summary_from_cache(
                cache_file=CACHE_FILE,
                instrument=INSTRUMENT,
                source_symbol=MT5_SYMBOL,
                pair=PAIR,
                target_pct_list=TARGET_PCT_LIST,
                sl_pct_list=SL_PCT_LIST,
                horizon_hours_list=HORIZON_HOURS_LIST,
                use_entry_time_window=USE_ENTRY_TIME_WINDOW,
                entry_window_start_hour=ENTRY_WINDOW_START_HOUR_MYT,
                entry_window_end_hour=ENTRY_WINDOW_END_HOUR_MYT,
                trades_per_hour=TRADES_PER_HOUR,
                use_ma_filter=USE_MA_FILTER,
                use_after_loss_filter=USE_AFTER_LOSS_FILTER,
                after_loss_trigger_count_list=AFTER_LOSS_TRIGGER_COUNT_LIST,
                after_loss_skip_trades_list=AFTER_LOSS_SKIP_TRADES_LIST,
                lookback_days=lookback_days,
            )

            if summary_df.empty:
                raise RuntimeError(f"No summary produced for lookback={lookback_days}.")

            output_file = save_expectancy_summary_csv(
                summary_df=summary_df,
                instrument=INSTRUMENT,
                lookback_days=lookback_days,
            )

            logger.info("Saved summary CSV to: %s", output_file)
            output_files.append(str(output_file))

        return output_files

    @task()
    def save_mae_mfe_plots_task(summary_files: list[str]) -> list[str]:
        from research.expectancy_analysis import save_mae_mfe_distribution_plots
        from research.expectancy_config import INSTRUMENT, MT5_SYMBOL, PAIR

        all_plot_files: list[str] = []

        for summary_file in summary_files:
            summary_df = pd.read_csv(summary_file)
            lookback_days = None
            if not summary_df.empty and "lookback_d" in summary_df.columns:
                try:
                    lookback_days = int(summary_df["lookback_d"].iloc[0])
                except Exception:
                    lookback_days = None

            plot_files = save_mae_mfe_distribution_plots(
                summary_df=summary_df,
                instrument=INSTRUMENT,
                pair=PAIR,
                symbol=MT5_SYMBOL,
                lookback_days=lookback_days,
            )
            logger.info("MAE/MFE plots saved for %s: %d", summary_file, len(plot_files))
            all_plot_files.extend(str(p) for p in plot_files)

        return all_plot_files

    @task()
    def save_heatmaps_task(summary_files: list[str]) -> list[str]:
        from research.expectancy_analysis import save_net_expectancy_tp_sl_plot
        from research.expectancy_config import INSTRUMENT, MT5_SYMBOL, PAIR

        all_plot_files: list[str] = []

        for summary_file in summary_files:
            summary_df = pd.read_csv(summary_file)
            lookback_days = None
            if not summary_df.empty and "lookback_d" in summary_df.columns:
                try:
                    lookback_days = int(summary_df["lookback_d"].iloc[0])
                except Exception:
                    lookback_days = None

            plot_files = save_net_expectancy_tp_sl_plot(
                summary_df=summary_df,
                instrument=INSTRUMENT,
                pair=PAIR,
                symbol=MT5_SYMBOL,
                lookback_days=lookback_days,
            )
            logger.info("Heatmaps saved for %s: %d", summary_file, len(plot_files))
            all_plot_files.extend(str(p) for p in plot_files)

        return all_plot_files

    summary_files = build_summary_task()
    save_mae_mfe_plots_task(summary_files)
    save_heatmaps_task(summary_files)


expectancy_scan_dag = expectancy_scan_dag()

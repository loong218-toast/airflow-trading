from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path("/opt/airflow/airflow-trading")
DATA_LAKE_ROOT = PROJECT_ROOT / "data_lake"

SESSION_NAME = "Opt_Session_20260504_172618_01"
REGIME_ID: Optional[int] = 0

RUN_CONFIG_FILENAME = "run_config.json"
MASTER_FILENAME = "master_metrics.parquet"

ANALYSIS_ROOT_NAME = "analysis"
COMPARE_ROOT_NAME = "baseline_vs_signal"
TRADE_ML_PARTITIONED_DIRNAME = "trade_ml_partitioned"
TRADE_ML_ANALYSIS_ROOT_NAME = "trade_ml_analysis"

TRADE_OVERLAP_FILTER: Optional[bool] = False
TRADE_FLIP_ON_ENTRY_FILTER: Optional[bool] = None

MIN_MASTER_BALANCE = 100.0
MIN_TOTAL_POS = 1
TOP_N_ROWS_TO_PRINT = 40


def session_dir() -> Path:
    return DATA_LAKE_ROOT / SESSION_NAME


def analysis_root() -> Path:
    return session_dir() / ANALYSIS_ROOT_NAME


def compare_root() -> Path:
    return analysis_root() / COMPARE_ROOT_NAME


def trade_ml_partitioned_root() -> Path:
    return session_dir() / TRADE_ML_PARTITIONED_DIRNAME


def trade_ml_analysis_root() -> Path:
    base = analysis_root() / TRADE_ML_ANALYSIS_ROOT_NAME
    if REGIME_ID is None:
        return base / "all_regimes"
    return base / f"regime_{REGIME_ID}"
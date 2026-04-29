from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import polars as pl


ANALYSIS_ROOT_NAME = "analysis"
COMPARE_ROOT_NAME = "baseline_vs_signal"
RUN_CONFIG_FILENAME = "run_config.json"
MASTER_FILENAME = "master_metrics.parquet"


def analysis_root(session_dir: str | Path) -> Path:
    return Path(session_dir) / ANALYSIS_ROOT_NAME


def compare_root(session_dir: str | Path) -> Path:
    return analysis_root(session_dir) / COMPARE_ROOT_NAME


def write_csv(df: Optional[pl.DataFrame], path: Path) -> None:
    if df is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if df.is_empty():
        df.write_csv(str(path))
        return

    out = df
    for name, dtype in out.schema.items():
        dtype_name = str(dtype).lower()
        if "list" in dtype_name or "struct" in dtype_name or "array" in dtype_name:
            out = out.with_columns(
                pl.col(name)
                .map_elements(
                    lambda x: json.dumps(x, default=str) if x is not None else None,
                    return_dtype=pl.Utf8,
                )
                .alias(name)
            )

    out.write_csv(str(path))


def load_session_run_config(
    session_name: str,
    data_lake_root: str | Path,
    run_config_filename: str = RUN_CONFIG_FILENAME,
) -> tuple[dict[str, Any], Path, Path]:
    session_dir = Path(data_lake_root) / str(session_name)
    run_config_path = session_dir / run_config_filename

    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {run_config_path}")

    with open(run_config_path, "r", encoding="utf8") as fh:
        run_cfg = json.load(fh)

    if not isinstance(run_cfg, dict):
        raise ValueError(f"run_config.json must contain a JSON object: {run_config_path}")

    return run_cfg, run_config_path, session_dir


def load_master_df(session_dir: str | Path, master_filename: str = MASTER_FILENAME) -> pl.DataFrame:
    session_dir = Path(session_dir)
    candidates = [
        session_dir / master_filename,
        session_dir / "results" / master_filename,
        session_dir / "results" / "batch_master_metrics.parquet",
    ]

    for p in candidates:
        if p.exists():
            return pl.read_parquet(str(p))

    raise FileNotFoundError("master_metrics.parquet not found. Looked in session_dir and results/.")
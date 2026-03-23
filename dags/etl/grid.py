from __future__ import annotations
import logging
import os
import time
from datetime import timedelta
import json
import math
import gc
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
import secrets
from itertools import product

import numba as _numba
_numba_threads = int(os.getenv("NUMBA_NUM_THREADS", os.getenv("NUMBA_NUM_THREADS_OVERRIDE", "1")))
try:
    _numba.set_num_threads(_numba_threads)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", str(_numba_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_numba_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_numba_threads))

import psutil
from airflow.exceptions import AirflowFailException

from etl.transform import build_df_main_from_5m_polars, load_candles_from_db_polars
from etl.feature_helpers import (
    precompute_all_possible_features,
    normalize_signals_times,
    generate_filtered_signals,
    selected_gap_col_for_ma_int,
    get_ma_price_gaps_for_indices
)
from etl.db import get_engine

from etl.backtest import (
    backtest_signals_sl_tp_rets,
    fast_compound_equity,
    fast_compound_equity_gate,
    compute_pnl_pct_vectorized,
    warmup_numba_kernels,
)

from etl.cache import (
    load_signals_cached,
    stage_for_flush,
    load_backtest_cached,
    flush_all_buffers,
)

from etl.io_utils import FULL_LAKE_DIR, _atomic_write_parquet
from etl.merge_utils import combine_cache, combine_results_to_master
from etl.master_io_utils import (
    buffer_master_row,
    merge_parquet_files_fast,
    _flush_master_rows_buffer,
)

from etl.schema import enforce_schema, get_schema, cast_to_schema, classify_fragment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# DEFAULT_RUN_CONFIG omitted for brevity - keep same as before
DEFAULT_RUN_CONFIG = {
    "pair": "XXBTZUSD",
    "BASE_MINUTES": 5,
    "ma_periods": [50, 600],
    "sl_tp_in_pct": True,
    "min_rr": 3.0,
    "sl_range": {"min": 0.1, "max": 2.0, "step": 0.2},
    "tp_range": {"min": 2.0, "max": 8.0, "step": 0.2},
    "exit_windows": [1, 4, 12, 24, 48, 72, 168],
    "entry_lookback_units": [0, 1, 4, 8, 12, 16, 20, 24, 48, 72, 168],
    "ma_reversion": [False, True],
    "use_stochastic": False,
    "BTC_SETTINGS": {"spread": 0.0002},
    "risk_pct": 0.005,
    "funding_period_hours": 8,
    "funding_rate_unit": "per_period",
    "conservative_sl_first": True,
    "treat_no_hit_as_loss": True,
    "BATCH_SIZE": 80,
    "grid_start_date": "2024-09-01T00:00:00Z",
    "grid_end_date": "2025-09-15T23:59:59Z",
    "max_dd_threshold": 0.20,
    "PARQUET_FLUSH_ROWS": 50000,
    "EQUITY_PARTITION_BY": "era_int",
    "force_rebuild_cache": False,
    "CACHE_USE_STREAMING_MERGE": True,
    "CACHE_FLUSH_ROWS": 50000,
    "CACHE_MAX_INMEM_ROWS": 20000,
    "CACHE_TMP_DIR": None,
    "PARALLEL_WORKERS": 1,
}

# ---------------- utilities ----------------
def heartbeat_log(tag: str, extra: dict | None = None):
    proc = psutil.Process()
    mem = proc.memory_info().rss
    cpu = psutil.cpu_percent(interval=None)
    extra = extra or {}
    logger.info(f"💓 {tag} | cpu={cpu:.1f}% mem={mem//1024//1024}MB {extra}")

def _split_period_windows_from_pl(df: pl.DataFrame, months: int,
                                  min_dt: Optional[datetime.datetime] = None,
                                  max_dt: Optional[datetime.datetime] = None) -> List[tuple]:
    import datetime as _dt

    def _parse(dt_in):
        if dt_in is None:
            return None
        if isinstance(dt_in, str):
            s = dt_in
            if s.endswith("Z"):
                s = s[:-1]
            try:
                parsed = _dt.datetime.fromisoformat(s)
            except Exception:
                try:
                    parsed = _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_dt.timezone.utc)
            return parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        if isinstance(dt_in, _dt.datetime):
            d = dt_in
            if d.tzinfo is not None:
                return d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
            return d
        return None

    if (min_dt is None or max_dt is None) and (df is None or df.height == 0):
        return []

    if min_dt is None or max_dt is None:
        try:
            first = df[0, "time"]
            last = df[-1, "time"]
            if hasattr(first, "to_pydatetime"):
                min_dt = min_dt or first.to_pydatetime()
            else:
                min_dt = min_dt or first
            if hasattr(last, "to_pydatetime"):
                max_dt = max_dt or last.to_pydatetime()
            else:
                max_dt = max_dt or last
        except Exception:
            return []

    min_dt = _parse(min_dt)
    max_dt = _parse(max_dt)
    if min_dt is None or max_dt is None:
        return []

    start = _dt.datetime(min_dt.year, min_dt.month, 1)
    end_bound = _dt.datetime(max_dt.year, max_dt.month, 1)
    windows = []
    cur = start
    while cur <= end_bound:
        m = cur.month - 1 + int(months)
        y = cur.year + (m // 12)
        mm = (m % 12) + 1
        end = _dt.datetime(year=y, month=mm, day=1)
        windows.append((cur, end))
        cur = end
    return windows

def _list(x, default):
    if x is None:
        return list(default)
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]

def _stoch_opts(cfg, use_stoch_list):
    out = []

    ks = _list(cfg.get("stoch_k"), [12]) if "stoch_k" in cfg else _list(cfg.get("k"), [12])
    ds = _list(cfg.get("stoch_d"), [3]) if "stoch_d" in cfg else _list(cfg.get("d"), [3])
    ss = _list(cfg.get("stoch_s"), [3]) if "stoch_s" in cfg else _list(cfg.get("s"), [3])
    ths = _list(cfg.get("stoch_thresholds"), [[20, 80]]) if "stoch_thresholds" in cfg else _list(cfg.get("thresholds"), [[20, 80]])

    for use_stoch in use_stoch_list:
        if not use_stoch:
            out.append({
                "use_stochastic": False,
                "stoch_key": "OFF",
                "col": None,
                "low": None,
                "high": None,
            })
            continue

        for k, d, s, t in product(ks, ds, ss, ths):
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                raise ValueError(f"Invalid stoch_threshold entry: {t!r}")

            low = float(t[0])
            high = float(t[1])

            out.append({
                "use_stochastic": True,
                "stoch_key": f"k{k}_d{d}_s{s}_l{low:g}_u{high:g}",
                "col": f"stoch_k{k}_d{d}_s{s}",
                "low": low,
                "high": high,
            })

    return out

def _bbw_opts(cfg, use_bbw_list):
    out = []

    periods = [int(x) for x in _list(cfg.get("bbw_periods"), [96]) if int(x) > 0]
    stds = [float(x) for x in _list(cfg.get("bbw_std"), [2.5])]
    ths = [float(x) for x in _list(cfg.get("bbw_thresholds"), [50])]

    for use_bbw in use_bbw_list:
        if not use_bbw:
            out.append({
                "use_bbw": False,
                "bbw_periods": 0,
                "bbw_std": 0.0,
                "bbw_thresholds": 0.0,
            })
            continue

        for p, s, t in product(periods, stds, ths):
            out.append({
                "use_bbw": True,
                "bbw_periods": p,
                "bbw_std": s,
                "bbw_thresholds": t,
            })

    return out

def _write_batch(path: Path, batch_id: int, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump({"batch_id": batch_id, "regimes": rows}, f, indent=2)

def generate_configs(session_dir: Path, run_cfg: dict) -> list[Path]:
    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    exit_windows = _list(run_cfg.get("exit_windows"), [24])
    lookbacks = _list(run_cfg.get("entry_lookback_units"), [24])

    sl_vals, tp_vals = _expand_sl_tp(run_cfg)
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))

    ma_periods = run_cfg.get("ma_periods", []) or []
    if not isinstance(ma_periods, list):
        ma_periods = list(ma_periods)

    ma_periods_sorted = sorted({int(x) for x in ma_periods})
    n_bits = len(ma_periods_sorted)
    max_ma_int = (1 << n_bits) if n_bits > 0 else 1

    ma_reversion_list = _list(run_cfg.get("ma_reversion"), [False])
    use_stoch_list = _list(run_cfg.get("use_stochastic"), [False, True])
    use_bbw_list = _list(run_cfg.get("use_bbw"), [False, True])

    stoch_opts = _stoch_opts(run_cfg, use_stoch_list)
    bbw_opts = _bbw_opts(run_cfg, use_bbw_list)

    all_regime_configs = []
    idx = 0

    for ma_rev in ma_reversion_list:
        for ma_int in range(0, max_ma_int):
            for st in stoch_opts:
                for bw in bbw_opts:
                    for lb_h in lookbacks:
                        for exit_h in exit_windows:
                            regime = {
                                "regime_id": f"{idx:05d}",
                                "ma_int": int(ma_int),
                                "ma_reversion": bool(ma_rev),
                                "use_stochastic": bool(st["use_stochastic"]),
                                "stoch_key": st["stoch_key"],
                                "stoch_col": st["col"],
                                "stoch_lower": st["low"],
                                "stoch_upper": st["high"],
                                "use_bbw": bool(bw["use_bbw"]),
                                "bbw_periods": int(bw["bbw_periods"]),
                                "bbw_std": float(bw["bbw_std"]),
                                "bbw_thresholds": float(bw["bbw_thresholds"]),
                                "entry_lookback_units": int(lb_h),
                                "exit_window_h": int(exit_h),
                                "ma_periods": ma_periods_sorted,
                            }
                            all_regime_configs.append(regime)
                            idx += 1

    total_regimes = len(all_regime_configs)
    batch_size = int(run_cfg.get("BATCH_SIZE", 150))
    saved_batch_paths = []

    logger.info("=" * 60)
    logger.info("🚀 GRID CONFIG GENERATION (BATCH MODE)")
    logger.info("📈 Total Unique Regimes: %d", total_regimes)
    logger.info("📦 Batch Size: %d | Expected Tasks: %d", batch_size, math.ceil(total_regimes / batch_size))
    logger.info("🧪 SL/TP combos per regime: %d", len(combos))
    logger.info("📊 Total Backtests to be run: %d", total_regimes * len(combos))
    logger.info("=" * 60)

    for i in range(0, total_regimes, batch_size):
        batch_slice = all_regime_configs[i : i + batch_size]
        batch_num = i // batch_size
        batch_payload = {"batch_id": batch_num, "regimes": batch_slice}
        batch_filename = cfg_dir / f"batch_{batch_num:04d}.json"

        with open(batch_filename, "w", encoding="utf8") as f:
            json.dump(batch_payload, f, indent=2)

        saved_batch_paths.append(batch_filename)
        if len(saved_batch_paths) % 20 == 0 or i + batch_size >= total_regimes:
            logger.info("📝 Written %d batch files...", len(saved_batch_paths))

    return saved_batch_paths

def list_pending_config_paths(session_dir: Path) -> List[str]:
    cfg_dir = session_dir / "configs"
    results_dir = session_dir / "results"
    batch_files = sorted(cfg_dir.glob("batch_*.json"))
    pending_batches = []

    for batch_path in batch_files:
        try:
            with open(batch_path, "r", encoding="utf8") as f:
                batch_data = json.load(f)

            regimes = batch_data.get("regimes", [])

            is_batch_complete = True
            for r in regimes:
                regime_id = r.get("regime_id")
                result_file = results_dir / f"cfg_{regime_id}_summary.json"
                if not result_file.exists():
                    is_batch_complete = False
                    break
            if not is_batch_complete:
                pending_batches.append(str(batch_path))
        except Exception:
            pending_batches.append(str(batch_path))
    logger.info("🔍 Checked %d batches: %d still pending.", len(batch_files), len(pending_batches))
    return pending_batches

class EquityStager:
    """
    One temp parquet per (partition_key, batch_id, worker_id).
    Appends row groups to the same file instead of creating many files.
    """

    def __init__(
        self,
        equity_part_base: Path,
        batch_id: int,
        flush_rows: int,
        partition_by: str,
        max_total_rows: int = 200_000,
        max_partitions: int = 128,
        tmp_dir: Optional[Path] = None,
    ):
        self.base = Path(equity_part_base)
        self.batch_id = int(batch_id)
        self.flush_rows = int(flush_rows)
        self.partition_by = str(partition_by)

        self.max_total_rows = int(max_total_rows)
        self.max_partitions = int(max_partitions)

        self._tmp_dir = Path(tmp_dir) if tmp_dir is not None else (self.base / "_tmp")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.base.mkdir(parents=True, exist_ok=True)

        # Open writer per partition key and keep it open until flush_all()
        self._writers: Dict[str, pq.ParquetWriter] = {}
        self._paths: Dict[str, Path] = {}

    def _worker_id(self) -> str:
        return os.getenv("AIRFLOW_MAP_INDEX", "0")

    def _part_path(self, part_key: str) -> Path:
        worker_id = self._worker_id()
        return self._tmp_dir / (
            f"equity_{self.partition_by}={part_key}"
            f"_batch={self.batch_id}"
            f"_task={worker_id}.parquet"
        )

    def _ensure_writer(self, part_key: str, first_table: Optional[pl.DataFrame] = None) -> pq.ParquetWriter:
        writer = self._writers.get(part_key)
        if writer is not None:
            return writer

        out_path = self._part_path(part_key)
        self._paths[part_key] = out_path

        # Fresh run or stale partial file: overwrite cleanly
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        if first_table is None or first_table.height == 0:
            raise RuntimeError(f"Cannot open ParquetWriter for empty equity batch: part_key={part_key}")

        arrow_table = first_table.to_arrow()
        writer = pq.ParquetWriter(
            str(out_path),
            arrow_table.schema,
            compression="snappy",
        )
        self._writers[part_key] = writer
        return writer

    def _append_df(self, part_key: str, df_chunk: pl.DataFrame):
        if df_chunk is None or df_chunk.height == 0:
            return

        # Enforce canonical equity schema once before writing
        df_chunk = enforce_schema(df_chunk, "equity", strict=True)
        if df_chunk.height == 0:
            return

        writer = self._ensure_writer(part_key, df_chunk)
        writer.write_table(df_chunk.to_arrow())

    def stage(self, part_key: str, df_small: pl.DataFrame):
        """
        Append equity rows immediately, chunked by flush_rows to keep memory low.
        """
        if df_small is None or df_small.height == 0:
            return

        try:
            df_canonical = enforce_schema(df_small, "equity", strict=True)
        except Exception:
            logger.exception("EquityStager.stage: enforce_schema failed for part %s; skipping fragment", part_key)
            return

        if df_canonical.height == 0:
            return

        # Write in smaller chunks if the input batch is large
        n = int(df_canonical.height)
        step = max(1, int(self.flush_rows))

        for start in range(0, n, step):
            chunk = df_canonical.slice(start, step)
            try:
                self._append_df(part_key, chunk)
            except Exception:
                logger.exception("EquityStager.stage: write failed for part %s", part_key)
                raise

        # free temporary objects quickly
        try:
            del df_canonical
            gc.collect()
        except Exception:
            pass

    def flush(self, part_key: str):
        """
        No buffer to flush because writes are immediate.
        Kept as a no-op interface method.
        """
        return

    def flush_all(self):
        """
        Close all open Parquet writers.
        """
        for part_key, writer in list(self._writers.items()):
            try:
                writer.close()
            except Exception:
                logger.exception("EquityStager.flush_all: failed to close writer for %s", part_key)

        self._writers = {}
        self._paths = {}

# SL/TP helpers (unchanged)
def _expand_sl_tp(run_cfg: Dict) -> Tuple[List[float], List[float]]:
    def _get_vals(key):
        if f"{key}_values" in run_cfg:
            return [float(x) for x in run_cfg[f"{key}_values"]]
        if f"{key}_range" in run_cfg:
            r = run_cfg[f"{key}_range"]
            mn = float(r["min"]); mx = float(r["max"]); step = float(r["step"])
            vals = []
            v = mn
            while v <= mx + 1e-9:
                vals.append(round(float(v), 8))
                v += step
            return vals
        raise ValueError(f"Missing '{key}_range' or '{key}_values' in run_config.")
    sl_vals = _get_vals("sl")
    tp_vals = _get_vals("tp")
    return sl_vals, tp_vals

def _prune_by_min_rr(sl_vals: List[float], tp_vals: List[float], min_rr: float) -> List[tuple]:
    """
    Return list of (SL, TP) pairs where TP >= min_rr * SL.

    Implementation notes:
    - Avoids O(n*m) nested Python loops and large boolean matrices by sorting TP once
      and using np.searchsorted to find the first acceptable TP for each SL.
    - Filters out non-positive SL/TP values (they cannot satisfy a positive RR).
    - If min_rr <= 0, returns the Cartesian product (behaves like no pruning).
    """
    import numpy as np

    if not sl_vals or not tp_vals:
        return []

    sl_arr = np.asarray(sl_vals, dtype=np.float32)
    tp_arr = np.asarray(tp_vals, dtype=np.float32)

    # Fast path: if min_rr <= 0, nothing to prune (return full cartesian product)
    if float(min_rr) <= 0.0:
        return [(float(s), float(t)) for s in sl_arr for t in tp_arr]

    # Exclude non-positive values (can't satisfy positive RR)
    sl_arr = sl_arr[sl_arr > 0.0]
    tp_arr = tp_arr[tp_arr > 0.0]
    if sl_arr.size == 0 or tp_arr.size == 0:
        return []

    # Sort TP once and use searchsorted to find the first TP >= threshold for each SL
    tp_sorted = np.sort(tp_arr)
    combos: List[tuple] = []
    rr = float(min_rr)

    for s in sl_arr:
        threshold = rr * float(s)
        pos = int(np.searchsorted(tp_sorted, threshold, side="left"))
        if pos < tp_sorted.size:
            # extend with only acceptable TP values (keeps memory usage minimal)
            # convert to Python floats for downstream compatibility
            combos.extend([(float(s), float(t)) for t in tp_sorted[pos:]])

    return combos

def _partition_list(items: List, chunk_size: int) -> List[List]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def _to_numpy_ensure(arr_series: pl.Series, dtype):
    try:
        a = arr_series.to_numpy(allow_copy=False)
    except RuntimeError:
        a = arr_series.to_numpy(allow_copy=True)
    if a.dtype != dtype:
        try:
            return a.astype(dtype, copy=False)
        except Exception:
            return a.astype(dtype, copy=True)
    return a

def prepare_worker_data(session_dir: Path, run_cfg: dict):
    base_file = Path(FULL_LAKE_DIR) / "base_data_full.parquet"
    if not base_file.exists():
        raise RuntimeError(f"Base data file not found at {base_file}")

    # Read once (collect) then normalize
    logger.debug("prepare_worker_data: reading base file %s", str(base_file))
    df_main = pl.read_parquet(str(base_file))

    df_main = df_main.sort("time")

    # Normalize core columns and types
    norm_exprs = []
    if "time" in df_main.columns:
        norm_exprs.append(pl.col("time").dt.replace_time_zone(None).alias("time"))
    if "time_ns" in df_main.columns:
        norm_exprs.append(pl.col("time_ns").cast(pl.Int64).alias("time_ns"))
    else:
        if "time" in df_main.columns:
            norm_exprs.append(pl.col("time").cast(pl.Datetime("ns")).cast(pl.Int64).alias("time_ns"))
    if "close" in df_main.columns:
        norm_exprs.append(pl.col("close").cast(pl.Float32).alias("close"))
    else:
        raise RuntimeError("df_main missing required 'close' column")
    norm_exprs.append(
        pl.col("spread").fill_null(0.0).cast(pl.Float32).alias("spread")
        if "spread" in df_main.columns
        else pl.lit(0.0).cast(pl.Float32).alias("spread")
    )
    norm_exprs.append(
        pl.col("funding_rate").fill_null(0.0).cast(pl.Float32).alias("funding_rate")
        if "funding_rate" in df_main.columns
        else pl.lit(0.0).cast(pl.Float32).alias("funding_rate")
    )

    df_main = df_main.with_columns(norm_exprs).with_row_count("idx").with_columns(pl.col("idx").cast(pl.Int64))

    # Downcast any Float64 to Float32
    float64_cols = [c for c, dt in df_main.schema.items() if dt == pl.Float64]
    if float64_cols:
        df_main = df_main.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols])

    try:
        # Keep feature columns: non-strict enforcement (casts base cols but keeps extras)
        df_main = enforce_schema(df_main, "df_main", strict=False)
    except Exception as e:
        logger.error(f"CRITICAL: enforce_schema failed: {e}")
        raise

    base_cols = set(df_main.columns)

    # --- IMPORTANT: call the single-source precompute function to *add feature cols* ---
    # precompute_all_possible_features is the canonical place that creates ma_a/ma_b, pct_d_slow, breakout_Xh, etc.
    # prepare_worker_data should call it so df_main contains the precomputed feature columns.
    df_main, updated_cfg = precompute_all_possible_features(df_main, run_cfg)
    run_cfg["lookback_map"] = updated_cfg.get("lookback_map")
    logger.debug("prepare_worker_data: after precompute cols=%s", df_main.columns)

    # feature columns are anything not in DF_MAIN base set
    feature_cols = [c for c in df_main.columns if c not in base_cols]

    # Ensure numeric columns are Float32 to save memory
    df_main = df_main.with_columns(
        [pl.col(c).cast(pl.Float32) for c in df_main.columns if df_main.schema[c] in (pl.Float64, pl.Float32)]
    )

    logger.info(
        "prepare_worker_data: df_main rows=%d cols=%d sample_cols=%s",
        int(df_main.height),
        len(df_main.columns),
        df_main.columns[:20],
    )

    # ---------------------
    # Load lookback_map (HOURS) — MUST be produced by precompute step (no worker computation)
    #   - Try run_cfg["lookback_map"], then session_dir/lookback_map.json.
    #   - If missing, fail loudly and instruct operator to run precompute stage.
    # ---------------------
    lookback_map = None
    if "lookback_map" in run_cfg and isinstance(run_cfg["lookback_map"], dict) and run_cfg["lookback_map"]:
        lookback_map = dict(run_cfg["lookback_map"])
    else:
        lb_file = Path(session_dir) / "lookback_map.json"
        if lb_file.exists():
            try:
                lookback_map = json.loads(lb_file.read_text(encoding="utf8"))
            except Exception as e:
                logger.exception("Failed to read lookback_map from %s: %s", lb_file, e)
                lookback_map = None

    if not isinstance(lookback_map, dict) or not lookback_map:
        # fail loudly: precompute must produce lookback_map
        raise RuntimeError(
            "Missing required lookback_map (era_label -> hours). "
            "This must be produced by your precompute step (the single source of truth) and placed into run_cfg['lookback_map'] "
            "or saved as session_dir/lookback_map.json. Worker will not compute defaults."
        )

    # Normalize and validate lookback_map values (hours)
    normalized_lb_map: Dict[str, int] = {}
    for k, v in lookback_map.items():
        try:
            vh = int(v)
        except Exception:
            raise RuntimeError(f"Invalid lookback_map value for '{k}': must be integer hours, got {v!r}")
        if vh < 0:
            raise RuntimeError(f"Invalid lookback_map value for '{k}': hours must be non-negative, got {vh}")
        normalized_lb_map[str(k)] = vh

    # return context
    return {
        "df_main": df_main,
        "feature_cols": feature_cols,
        "base_cols": base_cols,
        "lookback_map": normalized_lb_map,  # hours
        "main_close_arr": _to_numpy_ensure(df_main["close"], np.float32),
        "main_time_ns_arr": _to_numpy_ensure(df_main["time_ns"], np.int64),
        "main_spread_arr": _to_numpy_ensure(df_main["spread"], np.float32),
        "main_funding_arr": _to_numpy_ensure(df_main["funding_rate"], np.float32),
    }

def _get_or_generate_signals(
    era_label: str,
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    start_idx: int,
    end_idx: int,
) -> pl.DataFrame:
    """
    Strict signal loader/generator.

    - data_ctx must include 'df_main' and 'feature_cols'.
    - start_idx/end_idx computed by caller (no lookback calculation here).
    - Caching behavior is controlled via run_cfg.
    - Persist canonical signals to cache (strict schema), return annotated df_signals for in-memory use.
    """
    # --- validate required context ---
    if "df_main" not in data_ctx:
        raise RuntimeError("data_ctx missing required key: 'df_main' (precomputed from prepare_worker_data)")
    if "feature_cols" not in data_ctx:
        raise RuntimeError("data_ctx missing required key: 'feature_cols' (precomputed from prepare_worker_data)")

    df_main: pl.DataFrame = data_ctx["df_main"]
    feature_cols: list = data_ctx["feature_cols"]

    # Strict read from run_cfg (no external overrides)
    months = int(run_cfg["sl_tp_interval_months"])  # intentionally KeyError if absent
    force_rebuild_cache = bool(run_cfg.get("force_rebuild_cache", False))

    # Validate indices
    if start_idx is None or end_idx is None:
        raise ValueError("start_idx and end_idx must be provided (no internal lookback computation allowed).")
    if end_idx <= start_idx:
        # empty era window -> return empty canonical signals DF
        return pl.DataFrame([], schema=get_schema("signals"))

    # 1) attempt to load canonical cached signals (strict schema)
    df_signals_cached = None
    if not force_rebuild_cache:
        try:
            df_signals_cached = load_signals_cached(months, era_label, str(regime_id))
        except Exception as e:
            logger.debug("signals cache read failed (will regenerate) cfg=%s era=%s: %s", regime_id, era_label, e)
            df_signals_cached = None

    # 2) if cached -> normalize, annotate with feature cols, and return
    if df_signals_cached is not None and not df_signals_cached.is_empty():
        df_signals_cached = normalize_signals_times(df_signals_cached, df_main=df_main)
        if feature_cols:
            # join feature columns for in-memory downstream work
            df_signals_cached = df_signals_cached.join(df_main.select(["idx"] + feature_cols), on="idx", how="left")
        return enforce_schema(df_signals_cached, "signals", strict=False)

    # 3) generate signals from df slice (strict: caller provided indices)
    slice_len = max(0, end_idx - start_idx)
    if slice_len == 0:
        return pl.DataFrame([], schema=get_schema("signals"))

    df_input_slice = df_main.slice(start_idx, slice_len)
    if df_input_slice is None or df_input_slice.height == 0:
        return pl.DataFrame([], schema=get_schema("signals"))

    try:
        df_signals = generate_filtered_signals(df_input_slice, {**run_cfg, **regime_cfg, "regime_id": regime_id}, df_main=df_main)
    except Exception as e:
        # generation failure is critical — raise so caller can decide
        raise RuntimeError(f"generate_filtered_signals failed for regime={regime_id} era={era_label}: {e}") from e

    # annotate signals with precomputed feature columns for in-memory use
    if not df_signals.is_empty() and feature_cols:
        df_signals = df_signals.join(df_main.select(["idx"] + feature_cols), on="idx", how="left")

    # Build canonical cache DF (strict schema) and stage it (so caches remain small)
    if not df_signals.is_empty():
        try:
            signals_for_cache = df_signals.select(["idx", "time_ns", "side", "regime_id"])
        except Exception:
            # fallback: enforce canonical schema via explicit construction
            signals_for_cache = df_signals.with_columns(
                [
                    pl.col("idx").cast(pl.Int64),
                    pl.col("time_ns").cast(pl.Int64),
                    pl.col("side").cast(pl.Int8),
                    pl.col("regime_id").cast(pl.Int32),
                ]
            ).select(["idx", "time_ns", "side", "regime_id"])

        signals_for_cache = enforce_schema(signals_for_cache, "signals", strict=True)

        try:
            stage_for_flush("signals", months, era_label, str(regime_id), signals_for_cache)
        except Exception as e:
            logger.debug("stage_for_flush(signals) failed: %s", e)

    # Return annotated DF for downstream (non-strict so features persist in-memory)
    return enforce_schema(df_signals, "signals", strict=False) if not df_signals.is_empty() else pl.DataFrame([], schema=get_schema("signals"))

def compute_equity_preview(
    closed_rets: np.ndarray,
    closed_entry_idxs: np.ndarray,
    closed_exit_idxs: np.ndarray,
    main_close_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    main_spread_arr: np.ndarray,
    main_funding_arr: np.ndarray,
    sl_val: float,
    tp_val: float,
    run_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    pnl_pct_arr, exit_times_ns = compute_pnl_pct_vectorized(
        closed_masked_rets=closed_rets,
        closed_entry_idxs=closed_entry_idxs,
        closed_exit_idxs=closed_exit_idxs,
        main_close_arr=main_close_arr,
        main_time_ns_arr=main_time_ns_arr,
        entry_prices_arr=None,
        spread_arr=main_spread_arr,
        funding_arr=main_funding_arr,
        sl_val=sl_val,
        tp_val=tp_val,
        risk_pct=float(run_cfg.get("risk_pct", 0.005)),
        sl_tp_in_pct=bool(run_cfg.get("sl_tp_in_pct", True)),
        funding_period_hours=int(run_cfg.get("funding_period_hours", 8)),
        funding_rate_unit=str(run_cfg.get("funding_rate_unit", "per_period")),
        spread_is_percent=bool(run_cfg.get("spread_is_percent", True)),
    )

    if pnl_pct_arr is None or pnl_pct_arr.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0.0,
            False,
        )

    pnl_pct_arr = np.nan_to_num(pnl_pct_arr, nan=0.0, posinf=1e37, neginf=-1e37)
    max_dd_threshold = float(run_cfg.get("max_dd_threshold", 0.5))
    equity_arr, max_dd, breached = fast_compound_equity_gate(pnl_pct_arr, 100.0, max_dd_threshold)
    equity_arr = np.nan_to_num(equity_arr, nan=0.0, posinf=1e37, neginf=-1e37)

    return pnl_pct_arr, exit_times_ns, equity_arr, float(max_dd), bool(breached)

def stage_equity_from_preview(
    pnl_pct_arr: np.ndarray,
    exit_times_ns: np.ndarray,
    equity_arr: np.ndarray,
    closed_entry_idxs: np.ndarray,
    closed_exit_idxs: np.ndarray,
    sl_val: float,
    tp_val: float,
    regime_id: int,
    era_int: int,
    side_flag: int,
    stager,
    max_dd: float,
    ma_p_gap_a_entry: Optional[np.ndarray] = None,
    ma_p_gap_b_entry: Optional[np.ndarray] = None,
    ma_p_gap_a_exit: Optional[np.ndarray] = None,
    ma_p_gap_b_exit: Optional[np.ndarray] = None,
) -> Tuple[float, int, float]:
    """
    Stage equity rows only after the DD gate already passed.
    Returns final_balance, win_pos, max_dd.
    """
    if equity_arr is None or equity_arr.size == 0:
        return 100.0, 0, float(max_dd)

    pnl_pct_arr = np.nan_to_num(pnl_pct_arr, nan=0.0, posinf=1e37, neginf=-1e37)
    equity_arr = np.nan_to_num(equity_arr, nan=0.0, posinf=1e37, neginf=-1e37)

    win_pos = int(np.count_nonzero(pnl_pct_arr > 0.0))
    final_balance = float(equity_arr[-1]) if equity_arr.size else 100.0

    def _prepare_gap(arr, target_len):
        if arr is None or arr.size == 0:
            return np.full(target_len, np.nan, dtype=np.float32)
        a = np.asarray(arr, dtype=np.float32)
        if a.shape[0] != target_len:
            if a.shape[0] > target_len:
                a = a[:target_len]
            else:
                pad = np.full(target_len - a.shape[0], np.nan, dtype=np.float32)
                a = np.concatenate([a, pad])
        return a

    gap_len = equity_arr.shape[0]
    gap_a_entry = _prepare_gap(ma_p_gap_a_entry, gap_len)
    gap_b_entry = _prepare_gap(ma_p_gap_b_entry, gap_len)
    gap_a_exit = _prepare_gap(ma_p_gap_a_exit, gap_len)
    gap_b_exit = _prepare_gap(ma_p_gap_b_exit, gap_len)

    equity_data = {
        "regime_id": int(regime_id),
        "era_int": int(era_int),
        "side": int(side_flag),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "time_ns": exit_times_ns.astype(np.int64),
        "entry_idx": closed_entry_idxs.astype(np.int64),
        "exit_idx": closed_exit_idxs.astype(np.int64),
        "pnl_pct": pnl_pct_arr.astype(np.float32),
        "equity": equity_arr.astype(np.float32),
        "ma_p_gap_a_entry": gap_a_entry,
        "ma_p_gap_b_entry": gap_b_entry,
        "ma_p_gap_a_exit": gap_a_exit,
        "ma_p_gap_b_exit": gap_b_exit,
    }

    try:
        equity_df = cast_to_schema(equity_data, "equity")
    except Exception as e:
        logger.exception("stage_equity_from_preview: failed to build equity DataFrame: %s", e)
        return final_balance, win_pos, float(max_dd)

    meta = classify_fragment(equity_df, "equity")
    if not meta.get("is_like", False):
        logger.error(
            "stage_equity_from_preview: refusing to stage fragment; not equity-like. meta=%s sample=%s",
            meta,
            equity_df.head(1).to_dicts() if equity_df.height else None,
        )
        return final_balance, win_pos, float(max_dd)

    partition_val = str(regime_id) if getattr(stager, "partition_by", "") == "regime_id" else str(era_int)
    try:
        stager.stage(partition_val, equity_df)
    except Exception:
        logger.exception("stage_equity_from_preview: stager.stage failed for partition %s", partition_val)

    return final_balance, win_pos, float(max_dd)

def _make_master_row(
    regime_id: int,
    era_int: int,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    total_pos: int,
    win_pos: int,
    balance: float,
    max_dd: float,
    regime_cfg: dict,
) -> dict:
    return {
        "balance": float(balance),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "win_pos": int(win_pos),
        "total_pos": int(total_pos),
        "side": int(side_flag),
        "exit_window_h": int(regime_cfg.get("exit_window_h", 0)),
        "era_int": int(era_int),
        "regime_id": int(regime_id),
        "ma_int": int(regime_cfg.get("ma_int", 0)),
        "ma_reversion": bool(regime_cfg.get("ma_reversion", False)),
        "entry_lookback_units": int(regime_cfg.get("entry_lookback_units", 0)),
        "use_bbw": bool(regime_cfg.get("use_bbw", False)),
        "bbw_periods": int(regime_cfg.get("bbw_periods", 0)),
        "bbw_std": float(regime_cfg.get("bbw_std", 0.0)),
        "bbw_thresholds": int(regime_cfg.get("bbw_thresholds", 0)),
        "use_stochastic": bool(regime_cfg.get("use_stochastic", False)),
        "stoch_key": str(regime_cfg.get("stoch_key", "OFF")),
        "max_drawdown": float(max_dd),
    }


def _run_backtest_grid(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    df_signals: pl.DataFrame,
    months: int,
    era_label: str,
    era_int: int,
    results_dir: Path,
    batch_id: int,
    stager,
    max_dd_threshold: float,
    current_idx: int,
    total_in_batch: int,
    combos: List[Tuple[float, float]],
):
    months = int(run_cfg["sl_tp_interval_months"])

    df_main = data_ctx["df_main"]
    main_close_arr = data_ctx["main_close_arr"]
    main_time_ns_arr = data_ctx["main_time_ns_arr"]
    main_spread_arr = data_ctx["main_spread_arr"]
    main_funding_arr = data_ctx["main_funding_arr"]

    if df_signals is not None and not df_signals.is_empty():
        buys = df_signals.filter(pl.col("side") == 1)
        sells = df_signals.filter(pl.col("side") == -1)
    else:
        buys = pl.DataFrame([], schema=get_schema("signals"))
        sells = pl.DataFrame([], schema=get_schema("signals"))

    bucket_map = {}
    if not bool(run_cfg.get("force_rebuild_cache", False)):
        try:
            bucket_df = load_backtest_cached(months, era_label, str(regime_id))
            if bucket_df is not None and not bucket_df.is_empty():
                for row in bucket_df.iter_rows(named=True):
                    key = (
                        int(row["sig_n"]),
                        int(row["sig_min_ns"]),
                        int(row["sig_max_ns"]),
                        round(float(row["SL"]), 6),
                        round(float(row["TP"]), 6),
                        int(row["side"]),
                        int(row["exit_window_h"]),
                    )
                    bucket_map[key] = row
        except Exception as e:
            logger.debug("load_backtest_cached failed for cfg=%s era=%s: %s", regime_id, era_label, e)
            bucket_map = {}

    logger.info("DEBUG: era=%s regime=%s total_signals=%d", era_label, regime_id, int(df_signals.height))

    dd_pass = 0
    dd_fail = 0
    cache_hit = 0
    cache_miss = 0
    staged_equity = 0
    master_rows_written = 0
    empty_rows_written = 0

    for sl_val, tp_val in combos:
        for side_df_pl, side_flag in ((buys, 1), (sells, -1)):
            total_pos = int(side_df_pl.height) if side_df_pl is not None and not side_df_pl.is_empty() else 0

            if total_pos == 0:
                master_row_raw = _make_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    total_pos=0,
                    win_pos=0,
                    balance=100.0,
                    max_dd=0.0,
                    regime_cfg=regime_cfg,
                )
                canonical = pl.DataFrame([master_row_raw]).pipe(enforce_schema, "master").to_dicts()[0]
                buffer_master_row(results_dir, batch_id, canonical)
                empty_rows_written += 1
                master_rows_written += 1
                continue

            sig_idxs = side_df_pl["idx"].to_numpy(allow_copy=True).astype(np.int64)
            sig_times_ns = side_df_pl["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
            sig_n = int(sig_idxs.size)
            sig_min_ns = int(sig_times_ns.min()) if sig_n > 0 else -1
            sig_max_ns = int(sig_times_ns.max()) if sig_n > 0 else -1

            entry_idx = None
            exit_idx = None
            rets = None

            if bucket_map:
                key = (
                    sig_n,
                    sig_min_ns,
                    sig_max_ns,
                    round(float(sl_val), 6),
                    round(float(tp_val), 6),
                    int(side_flag),
                    int(regime_cfg.get("exit_window_h", 0)),
                )
                hit = bucket_map.get(key)
                if hit is not None:
                    cache_hit += 1
                    entry_idx = np.asarray(hit["entry_idx"], dtype=np.int64)
                    exit_idx = np.asarray(hit["exit_idx"], dtype=np.int64)
                    rets = np.asarray(hit["ret"], dtype=np.float64)
                else:
                    cache_miss += 1

            if (sig_idxs < 0).any() or (sig_idxs >= len(main_close_arr)).any():
                logger.error("OOB Index: cfg=%s era=%s sig_idxs out of range", regime_id, era_label)
                continue

            if entry_idx is None:
                try:
                    res = backtest_signals_sl_tp_rets(
                        main_close_arr=main_close_arr,
                        main_time_ns_arr=main_time_ns_arr,
                        sig_idxs=sig_idxs,
                        sl=float(sl_val),
                        tp=float(tp_val),
                        sl_tp_in_pct=bool(run_cfg.get("sl_tp_in_pct", True)),
                        exit_window_h=int(regime_cfg.get("exit_window_h", 0)),
                        base_minutes=int(run_cfg.get("BASE_MINUTES", 5)),
                        spread=float(run_cfg.get("BTC_SETTINGS", {}).get("spread", 0.0)),
                        conservative_sl_first=bool(run_cfg.get("conservative_sl_first", True)),
                        treat_no_hit_as_loss=bool(run_cfg.get("treat_no_hit_as_loss", True)),
                        side_flag=side_flag,
                    )
                    entry_idx = np.asarray(res.get("entry_idx", []), dtype=np.int64)
                    exit_idx = np.asarray(res.get("exit_idx", []), dtype=np.int64)
                    rets = np.asarray(res.get("rets", res.get("ret", [])), dtype=np.float64)

                    res_df = pl.DataFrame(
                        {
                            "sig_n": [sig_n],
                            "sig_min_ns": [sig_min_ns],
                            "sig_max_ns": [sig_max_ns],
                            "SL": [float(sl_val)],
                            "TP": [float(tp_val)],
                            "side": [int(side_flag)],
                            "regime_id": [int(regime_id)],
                            "exit_window_h": [int(regime_cfg.get("exit_window_h", 0))],
                            "entry_idx": pl.Series([entry_idx], dtype=pl.List(pl.Int64)),
                            "exit_idx": pl.Series([exit_idx], dtype=pl.List(pl.Int64)),
                            "ret": pl.Series([rets], dtype=pl.List(pl.Float32)),
                        }
                    ).pipe(enforce_schema, "backtest", strict=True)

                    stage_for_flush("backtest", months, era_label, str(regime_id), res_df)
                except Exception as e:
                    logger.error("Backtest kernel error cfg=%s era=%s: %s", regime_id, era_label, e)
                    continue

            mask_closed = np.asarray([]) if exit_idx is None else (exit_idx >= 0)

            if not (mask_closed.size and mask_closed.any()):
                master_row_raw = _make_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    total_pos=total_pos,
                    win_pos=0,
                    balance=100.0,
                    max_dd=0.0,
                    regime_cfg=regime_cfg,
                )
                canonical = pl.DataFrame([master_row_raw]).pipe(enforce_schema, "master").to_dicts()[0]
                buffer_master_row(results_dir, batch_id, canonical)
                master_rows_written += 1
                continue

            closed_rets = rets[mask_closed]
            closed_entry_idxs = entry_idx[mask_closed].astype(np.int64)
            closed_exit_idxs = exit_idx[mask_closed].astype(np.int64)

            pnl_pct_arr, exit_times_ns, equity_arr, current_max_dd, breached = compute_equity_preview(
                closed_rets=closed_rets,
                closed_entry_idxs=closed_entry_idxs,
                closed_exit_idxs=closed_exit_idxs,
                main_close_arr=main_close_arr,
                main_time_ns_arr=main_time_ns_arr,
                main_spread_arr=main_spread_arr,
                main_funding_arr=main_funding_arr,
                sl_val=float(sl_val),
                tp_val=float(tp_val),
                run_cfg=run_cfg,
            )

            if breached:
                dd_fail += 1
                logger.debug(
                    "DD FAIL era=%s regime=%s side=%s SL=%.4f TP=%.4f max_dd=%.4f threshold=%.4f -> skip equity only",
                    era_label, regime_id, int(side_flag), float(sl_val), float(tp_val), float(current_max_dd), float(max_dd_threshold),
                )

                final_balance = float(equity_arr[-1]) if equity_arr.size else 100.0
                win_pos = int(np.count_nonzero(closed_rets > 0.0)) if closed_rets.size else 0

                master_row_raw = _make_master_row(
                    regime_id=regime_id,
                    era_int=era_int,
                    side_flag=side_flag,
                    sl_val=sl_val,
                    tp_val=tp_val,
                    total_pos=total_pos,
                    win_pos=win_pos,
                    balance=final_balance,
                    max_dd=current_max_dd,
                    regime_cfg=regime_cfg,
                )
                canonical = pl.DataFrame([master_row_raw]).pipe(enforce_schema, "master").to_dicts()[0]
                buffer_master_row(results_dir, batch_id, canonical)
                master_rows_written += 1
                continue

            dd_pass += 1

            gap_entry_a, gap_entry_b, gap_exit_a, gap_exit_b = get_ma_price_gaps_for_indices(
                df_main, closed_entry_idxs, closed_exit_idxs
            )

            final_balance, win_pos, max_dd = stage_equity_from_preview(
                pnl_pct_arr=pnl_pct_arr,
                exit_times_ns=exit_times_ns,
                equity_arr=equity_arr,
                closed_entry_idxs=closed_entry_idxs,
                closed_exit_idxs=closed_exit_idxs,
                sl_val=float(sl_val),
                tp_val=float(tp_val),
                regime_id=regime_id,
                era_int=era_int,
                side_flag=side_flag,
                stager=stager,
                max_dd=current_max_dd,
                ma_p_gap_a_entry=gap_entry_a,
                ma_p_gap_b_entry=gap_entry_b,
                ma_p_gap_a_exit=gap_exit_a,
                ma_p_gap_b_exit=gap_exit_b,
            )
            staged_equity += 1

            master_row_raw = _make_master_row(
                regime_id=regime_id,
                era_int=era_int,
                side_flag=side_flag,
                sl_val=sl_val,
                tp_val=tp_val,
                total_pos=total_pos,
                win_pos=win_pos,
                balance=final_balance,
                max_dd=max_dd,
                regime_cfg=regime_cfg,
            )
            canonical = pl.DataFrame([master_row_raw]).pipe(enforce_schema, "master").to_dicts()[0]
            buffer_master_row(results_dir, batch_id, canonical)
            master_rows_written += 1

    return {
        "dd_pass": dd_pass,
        "dd_fail": dd_fail,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "staged_equity": staged_equity,
        "master_rows_written": master_rows_written,
        "empty_rows_written": empty_rows_written,
    }
    
def process_era_combos(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    data_ctx: dict,
    results_dir: Path,
    batch_id: int,
    stager,
    max_dd_threshold: float,
    current_idx: int,
    total_in_batch: int
) -> Tuple[int, int]:
    """
    Strict per-era processing loop.

    Requirements (no legacy or fallback logic):
      - `data_ctx` MUST contain:
         * "df_main" (polars DataFrame)
         * "main_time_ns_arr" (numpy int64 array of timestamps in ns)
         * "feature_cols" (list of precomputed feature column names)
         * "lookback_map" (dict mapping era_label -> lookback_minutes)  <-- MANDATORY
      - `run_cfg` MUST contain "sl_tp_interval_months" (int) and other grid params.
      - No internal computation of lookback; an absent lookback_map entry is an error.
    """
    # validate data_ctx
    required_keys = ("df_main", "main_time_ns_arr", "feature_cols", "lookback_map")
    missing = [k for k in required_keys if k not in data_ctx]
    if missing:
        raise RuntimeError(f"data_ctx missing required keys: {missing}. This system enforces precomputed lookback_map and features.")

    df_main: pl.DataFrame = data_ctx["df_main"]
    main_time_ns_arr: np.ndarray = data_ctx["main_time_ns_arr"]
    lookback_map: dict = data_ctx["lookback_map"]  # strict: must exist and cover all eras
    feature_cols: list = data_ctx["feature_cols"]

    # grid params (strict read)
    months = int(run_cfg["sl_tp_interval_months"])  # KeyError if absent intentionally
    grid_start = pd.to_datetime(run_cfg["grid_start_date"]).to_pydatetime()
    grid_end = pd.to_datetime(run_cfg["grid_end_date"]).to_pydatetime()

    processed_inc = 0
    skipped_inc = 0

    sl_vals, tp_vals = _expand_sl_tp({**run_cfg, **regime_cfg})
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))

    windows = _split_period_windows_from_pl(df_main, months, min_dt=grid_start, max_dt=grid_end)
    logger.debug("process_era_combos: regime=%s windows=%d", regime_id, len(windows))

    empty_era_streak = 0
    max_empty_era_streak = int(run_cfg.get("MAX_EMPTY_ERA_STREAK", 2))

    # iterate strictly over precomputed windows (eras)
    for (start, end) in windows:
        if hasattr(start, "tzinfo") and start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if hasattr(end, "tzinfo") and end.tzinfo is not None:
            end = end.replace(tzinfo=None)

        era_label = start.strftime("%Y-%m")
        try:
            era_int = int(start.strftime("%Y%m%d"))
        except Exception:
            era_int = 0

        # require lookback entry for this era (no fallback allowed)
        if era_label not in lookback_map:
            raise RuntimeError(f"Missing precomputed lookback for era '{era_label}' in data_ctx['lookback_map']. Aborting; no fallback allowed.")
        lookback_hours = int(lookback_map[era_label])

        lookback_delta = datetime.timedelta(hours=lookback_hours)
        input_start = start - lookback_delta

        # convert to ns and clamp to df_main range
        start_ns = np.datetime64(input_start).astype("datetime64[ns]").astype(np.int64)
        end_ns = np.datetime64(end).astype("datetime64[ns]").astype(np.int64)

        min_ns = int(main_time_ns_arr[0])
        max_ns = int(main_time_ns_arr[-1])
        if start_ns < min_ns:
            start_ns = min_ns
        if end_ns > max_ns:
            end_ns = max_ns

        # LOGICAL CHECK: Is this era even inside our data?
        if end_ns < min_ns or start_ns > max_ns:
            logger.warning(f"SKIP Era {era_label}: Entirely outside data range. "
                        f"Data: {pd.to_datetime(min_ns)} to {pd.to_datetime(max_ns)}")
            continue

        start_idx = int(np.searchsorted(main_time_ns_arr, start_ns, side="left"))
        end_idx = int(np.searchsorted(main_time_ns_arr, end_ns, side="left"))

        if start_idx == end_idx:
            # This happens if the timestamps are so close they fall into the same candle 
            # OR if the range is in a gap in your data.
            logger.error(f"⚠️ INDEX COLLISION for Era {era_label}: start_idx and end_idx are both {start_idx}. "
                        f"Search range: {input_start} to {end}. Slice will be empty!")

        logger.debug("ERA %s start_idx=%d end_idx=%d lookback=%d", era_label, start_idx, end_idx, lookback_hours)

        # generate or load signals (slicing done inside)
        df_signals = _get_or_generate_signals(
            era_label=era_label,
            regime_id=regime_id,
            regime_cfg=regime_cfg,
            run_cfg=run_cfg,
            data_ctx=data_ctx,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # run backtests (this function reads arrays and flags from data_ctx / run_cfg)
        summary = _run_backtest_grid(
            regime_id=regime_id,
            regime_cfg=regime_cfg,
            run_cfg=run_cfg,
            data_ctx=data_ctx,
            df_signals=df_signals,
            months=months,
            era_label=era_label,
            era_int=era_int,
            results_dir=results_dir,
            batch_id=batch_id,
            stager=stager,
            max_dd_threshold=max_dd_threshold,
            current_idx=current_idx,
            total_in_batch=total_in_batch,
            combos=combos,
        )

        current_regime_display = current_idx + 1

        logger.info(
            f"📊 [Batch {batch_id}] Regime {current_regime_display}/{total_in_batch} (ID:{regime_id}) | "
            f"Era: {era_label} | PASS: {summary['dd_pass']} | FAIL: {summary['dd_fail']} | "
            f"Streak: {empty_era_streak}/{max_empty_era_streak}"
        )

        if summary["master_rows_written"] == 0:
            logger.warning(
                f"⚠️ Skipping Era {era_label} for regime={regime_id}: "
                f"All {summary['dd_fail']} combos failed DD threshold {max_dd_threshold}."
            )
            empty_era_streak += 1  # NOW your streak logic actually works!
            
            if empty_era_streak >= max_empty_era_streak:
                error_msg = f"❌ CRITICAL FAILURE: Regime {regime_id} killed. Hit max empty streak of {max_empty_era_streak}."
                logger.error(error_msg)
                
                # This stops EVERYTHING and tells Airflow the task failed
                raise RuntimeError(error_msg)
            
            continue # Go to the next era in the loop

        processed_inc += 1

    return processed_inc, skipped_inc

def compute_config_and_save(batch_path: str, session_dir: str, compute_backtest: bool = True) -> Dict:
    try:
        _HAS_PSUTIL = True
        _PROC = psutil.Process()
    except Exception:
        _HAS_PSUTIL = False
        _PROC = None

    try:
        logger.info("🔥 Worker-local Numba warmup starting...")
        warmup_numba_kernels()
        logger.info("🔥 Worker-local Numba warmup complete.")
    except Exception as e:
        logger.warning("Numba warmup failed: %s", e)

    batch_path = Path(batch_path)
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise AirflowFailException(f"FATAL: session_dir NOT FOUND at {session_dir}")

    session_snapshot = session_dir / "run_config.json"
    if not session_snapshot.exists():
        # Fallback for safety, though it should exist from init_session_task
        logger.warning("⚠️ Session run_config.json missing, falling back to global loader.")
        run_cfg = _load_run_config()
    else:
        with open(session_snapshot, "r", encoding="utf8") as f:
            run_cfg = json.load(f)
        logger.info("✅ Loaded session-specific run_config from snapshot.")

    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_path, "r", encoding="utf8") as f:
        batch_data = json.load(f)

    regimes = batch_data["regimes"]
    batch_id = int(batch_data.get("batch_id", 0))
    logger.info("🧵 Worker started: %s | Regimes to process: %d (batch_id=%d)", batch_path.name, len(regimes), batch_id)

    prepared = prepare_worker_data(session_dir, run_cfg)
    data_ctx = prepared  # keep the entire prepared dict intact; pass around as context

    if "lookback_map" not in data_ctx or not isinstance(data_ctx["lookback_map"], dict):
        raise RuntimeError("prepare_worker_data did not return a valid 'lookback_map'. Ensure precompute produced it.")

    max_dd_threshold = float(run_cfg.get("max_dd_threshold", 0.20))
    flush_rows = int(run_cfg.get("PARQUET_FLUSH_ROWS", 100_000))
    partition_by = str(run_cfg.get("EQUITY_PARTITION_BY", "era_int"))

    # Use session-local tmp dir so coordinator can find parts (session_dir/equity_partitioned/_tmp)
    equity_base = session_dir / "equity_partitioned"
    equity_tmp = equity_base / "_tmp"
    stager = EquityStager(
        equity_part_base=equity_base,
        batch_id=batch_id,
        flush_rows=flush_rows,
        partition_by=partition_by,
        tmp_dir=equity_tmp,
    )

    processed_count = 0
    skipped_count = 0

    try:
        total_in_batch = len(regimes)
        for idx, regime_cfg in enumerate(regimes):
            if _HAS_PSUTIL and _PROC:
                mem_info = psutil.virtual_memory()
                if mem_info.percent > 92.0:
                    logger.error("OOM RISK: System Memory at %s%%", mem_info.percent)
                    raise AirflowFailException(f"System memory threshold exceeded ({mem_info.percent}%).")

            gc.collect()

            if idx % 10 == 0:
                heartbeat_log(f"batch{batch_id}_progress", {"idx": idx, "regime_id": str(regime_cfg.get("regime_id"))})

            # normalize regime types
            try:
                regime_cfg["regime_id"] = int(regime_cfg.get("regime_id"))
            except Exception:
                logger.debug("regime_id not int-convertible: %s", regime_cfg.get("regime_id"))
            regime_cfg["ma_int"] = int(regime_cfg.get("ma_int", 0) or 0)
            regime_cfg["ma_reversion"] = bool(regime_cfg.get("ma_reversion", False))
            regime_cfg["entry_lookback_units"] = int(regime_cfg.get("entry_lookback_units", 0) or 0)
            regime_cfg["exit_window_h"] = int(regime_cfg.get("exit_window_h", 0) or 0)
            regime_cfg["use_stochastic"] = bool(regime_cfg.get("use_stochastic", False))
            regime_cfg["bbw_periods"] = int(regime_cfg.get("bbw_periods", 0) or 0)
            regime_cfg["bbw_std"] = float(regime_cfg.get("bbw_std", 0.0) or 0.0)
            regime_cfg["use_bbw"] = bool(regime_cfg.get("use_bbw", False))
            regime_cfg["bbw_thresholds"] = int(regime_cfg.get("bbw_thresholds", 0) or 0)

            regime_id = regime_cfg.get("regime_id")
            summary_path = results_dir / f"cfg_{regime_id}_summary.json"
            if summary_path.exists():
                skipped_count += 1
                continue

            if processed_count % 5 == 0 or processed_count == 0:
                logger.info("📑 [Batch %d] Progress: %d/%d (%.1f%%) ...", batch_id, idx, total_in_batch, (idx / total_in_batch) * 100.0)

            p_inc, s_inc = process_era_combos(
                regime_id=regime_id,
                regime_cfg=regime_cfg,
                run_cfg=run_cfg,
                data_ctx=data_ctx,
                results_dir=results_dir,
                batch_id=batch_id,
                stager=stager,
                max_dd_threshold=max_dd_threshold,
                current_idx=idx,
                total_in_batch=total_in_batch
            )
            processed_count += p_inc
            skipped_count += s_inc

            final_output = {"regime_id": regime_id, "regime_params": regime_cfg, "combos_tested": 0, "results": []}
            with open(results_dir / f"cfg_{regime_id}_summary.json", "w", encoding="utf8") as fh:
                json.dump(final_output, fh, indent=2)

    except Exception as e:
        logger.error("Error processing batch %s: %s", batch_path.name, e)
        raise
    finally:
        try:
            # spill staged equity parts to tmp files (workers do not merge final files)
            stager.flush_all()
            _flush_master_rows_buffer(results_dir, batch_id)
        except Exception as e:
            logger.debug("Flush local stage failed: %s", e)

        try:
            flush_all_buffers()
        except Exception:
            logger.debug("flush_all_buffers() failed")

        # Merge master metrics parts into batch master (unchanged)
        parts_dir = results_dir / "master_parts" / f"batch_{batch_id:04d}"
        parts = sorted(parts_dir.glob("*.parquet")) if parts_dir.exists() else []
        if parts:
            try:
                merge_parquet_files_fast(parts, results_dir / f"batch_{batch_id:04d}_master_metrics.parquet")
            except Exception as e:
                logger.exception("Failed to merge master parts for batch %s: %s", batch_id, e)
                raise
        else:
            try:
                empty_df = pl.DataFrame([]).pipe(enforce_schema, "master")
                empty_out = results_dir / f"batch_{batch_id:04d}_master_metrics.parquet"
                empty_df.write_parquet(str(empty_out), compression="snappy")
            except Exception as e:
                logger.exception("Failed to write empty master parquet for batch %s: %s", batch_id, e)
                raise

    logger.info("🏁 Batch %s finished. Processed: %d, Skipped: %d", batch_path.name, processed_count, skipped_count)
    return {"batch": batch_path.name, "processed": processed_count, "skipped": skipped_count}
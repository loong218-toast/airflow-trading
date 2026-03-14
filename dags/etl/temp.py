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
)
from etl.db import get_engine

from etl.backtest import (
    backtest_signals_sl_tp_rets,
    fast_compound_equity,
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

from etl.schema import enforce_schema, get_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # change to DEBUG when troubleshooting; can be raised at runtime

# DEFAULT_RUN_CONFIG omitted for brevity - keep same as before
DEFAULT_RUN_CONFIG = {
    "pair": "XXBTZUSD",
    "BASE_MINUTES": 5,
    "ma_timeframe": "1h",
    "ma_periods": [50, 600],
    "sl_tp_in_pct": True,
    "min_rr": 3.0,
    "sl_range": {"min": 0.1, "max": 2.0, "step": 0.2},
    "tp_range": {"min": 2.0, "max": 8.0, "step": 0.2},
    "exit_windows": [1, 4, 12, 24, 48, 72, 168],
    "entry_lookback_h": [0, 1, 4, 8, 12, 16, 20, 24, 48, 72, 168],
    "ma_reversion": [False, True],
    "use_stochastic": False,
    "BTC_SETTINGS": {"spread": 0.0002},
    "risk_pct": 0.005,
    "funding_period_hours": 8,
    "funding_rate_unit": "per_period",
    "conservative_sl_first": True,
    "treat_no_hit_as_loss": True,
    "BATCH_SIZE": 80,
    "COMBINE_BATCH_SIZE": 400,
    "grid_start_date": "2024-09-01T00:00:00Z",
    "grid_end_date": "2025-09-15T23:59:59Z",
    "max_dd_threshold": 0.20,
    "PARQUET_FLUSH_ROWS": 50000,
    "EQUITY_PARTITION_BY": "era_int",
    "force_rebuild_cache": False,
    "cache_lookback_minutes": None,
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

def _load_run_config() -> Dict:
    raw = None
    try:
        from airflow.sdk import Variable
        raw = Variable.get("run_config", default=None)
    except Exception:
        pass
    if not raw:
        f = Path("/opt/airflow/airflow-trading/run_config.json")
        if f.exists():
            raw = f.read_text(encoding="utf8")
    user_cfg = json.loads(raw) if raw else {}
    merged = DEFAULT_RUN_CONFIG.copy()
    if isinstance(user_cfg, dict):
        merged.update(user_cfg)
    if "sl" not in merged and "sl_range" in merged:
        merged["sl"] = None
    if "tp" not in merged and "tp_range" in merged:
        merged["tp"] = None
    return merged

def log_mem(label: str):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    sys_mem = psutil.virtual_memory().percent
    print(f"DEBUG_MEM | {sys_mem}% Sys | {mem_mb:.2f} MB Proc | {label}", flush=True)

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

class EquityStager:
    def __init__(self, equity_part_base: Path, batch_id: int, flush_rows: int, partition_by: str,
                 max_total_rows: int = 200_000, max_partitions: int = 128):
        self.base = Path(equity_part_base)
        self.batch_id = batch_id
        self.flush_rows = int(flush_rows)
        self.partition_by = partition_by
        self._staging = {}
        self._staging_rows = {}
        self._total_rows = 0
        self.max_total_rows = int(max_total_rows)
        self.max_partitions = int(max_partitions)

    def stage(self, part_key: str, df_small: pl.DataFrame):
        try:
            df_small = enforce_schema(df_small, "equity")
        except Exception:
            logger.debug("EquityStager: enforce_schema failed for partition %s; staging without enforcement", part_key)

        lst = self._staging.get(part_key)
        if lst is None:
            lst = []
            self._staging[part_key] = lst
            self._staging_rows[part_key] = 0
        lst.append(df_small)
        self._staging_rows[part_key] += int(df_small.height)
        self._total_rows += int(df_small.height)

        if self._staging_rows[part_key] >= self.flush_rows:
            self.flush(part_key)
            return

        if self._total_rows >= self.max_total_rows or len(self._staging) > self.max_partitions:
            self.flush(part_key)

    def flush(self, part_key: str):
        lst = self._staging.get(part_key)
        if not lst:
            self._staging_rows[part_key] = 0
            return
        to_write = pl.concat(lst, how='vertical') if len(lst) > 1 else lst[0]
        try:
            to_write = enforce_schema(to_write, "equity")
        except Exception:
            logger.debug("EquityStager.flush: enforce_schema failed; continuing with original DF")
        part_dir = self.base / f"{self.partition_by}={part_key}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_file = part_dir / f"batch_{self.batch_id:04d}_{part_key}_{int(time.time()*1000)}.parquet"
        _atomic_write_parquet(to_write, out_file)
        removed_rows = self._staging_rows.get(part_key, 0)
        self._total_rows = max(0, self._total_rows - removed_rows)
        self._staging[part_key] = []
        self._staging_rows[part_key] = 0
        try:
            gc.collect()
        except Exception:
            pass

    def flush_all(self):
        for k in list(self._staging.keys()):
            try:
                self.flush(k)
            except Exception as e:
                logger.exception("EquityStager.flush_all: failed for %s: %s", k, e)

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
    combos = []
    for s in sl_vals:
        for t in tp_vals:
            if t >= (min_rr * s):
                combos.append((s, t))
    return combos

def _partition_list(items: List, chunk_size: int) -> List[List]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def generate_configs(session_dir: Path) -> List[Path]:
    run_cfg = _load_run_config()
    for key in ("CACHE_USE_STREAMING_MERGE",
                "CACHE_FLUSH_ROWS",
                "CACHE_MAX_INMEM_ROWS",
                "CACHE_TMP_DIR",
                "CACHE_MERGE_CHUNK_SIZE",
                "DATA_LAKE_ROOT"):
        if key in run_cfg and run_cfg.get(key) is not None:
            os.environ[key] = str(run_cfg.get(key))

    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    exit_windows = run_cfg.get("exit_windows", [24])
    lookbacks = run_cfg.get("entry_lookback_h", [24])
    sl_vals, tp_vals = _expand_sl_tp(run_cfg)
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))
    ma_periods = run_cfg.get("ma_periods", []) or []
    if not isinstance(ma_periods, list):
        ma_periods = list(ma_periods) if ma_periods else []
    n_bits = max(1, len(ma_periods))
    max_ma_int = (1 << n_bits)
    all_regime_configs = []
    idx = 0
    ma_reversion_list = run_cfg.get("ma_reversion", [False, True])
    for ma_rev in ma_reversion_list:
        for ma_int in range(0, max_ma_int):
            for use_stoch in [False, True]:
                for lb_h in lookbacks:
                    for exit_h in exit_windows:
                        regime = {
                            "regime_id": f"{idx:05d}",
                            "ma_int": ma_int,
                            "ma_reversion": ma_rev,
                            "use_stochastic": use_stoch,
                            "use_entry_lookback": bool(lb_h > 0),
                            "entry_lookback_h": int(lb_h),
                            "exit_window_h": int(exit_h),
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
    norm_exprs.append(pl.col("spread").fill_null(0.0).cast(pl.Float32).alias("spread") if "spread" in df_main.columns else pl.lit(0.0).cast(pl.Float32).alias("spread"))
    norm_exprs.append(pl.col("funding_rate").fill_null(0.0).cast(pl.Float32).alias("funding_rate") if "funding_rate" in df_main.columns else pl.lit(0.0).cast(pl.Float32).alias("funding_rate"))

    df_main = df_main.with_columns(norm_exprs).with_row_count("idx").with_columns(pl.col("idx").cast(pl.Int64))

    # Precompute features (safe, ordered)
    df_main = precompute_all_possible_features(df_main, run_cfg)
    logger.debug("prepare_worker_data: after precompute cols=%s", df_main.columns)

    # Downcast any Float64 to Float32
    float64_cols = [c for c, dt in df_main.schema.items() if dt == pl.Float64]
    if float64_cols:
        df_main = df_main.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols])

    try:
        df_main = enforce_schema(df_main, "df_main")
    except Exception:
        logger.debug("prepare_worker_data: enforce_schema(df_main) failed - continuing")

    logger.info("prepare_worker_data: df_main rows=%d cols=%d sample_cols=%s", int(df_main.height), len(df_main.columns), df_main.columns[:20])

    return {
        "df_main": df_main,
        "main_close_arr": _to_numpy_ensure(df_main["close"], np.float32),
        "main_time_ns_arr": _to_numpy_ensure(df_main["time_ns"], np.int64),
        "main_spread_arr": _to_numpy_ensure(df_main["spread"], np.float32),
        "main_funding_arr": _to_numpy_ensure(df_main["funding_rate"], np.float32),
    }

def _compute_era_gap_stats(df_input_slice: pl.DataFrame, df_main: pl.DataFrame, regime_cfg: dict, era_label: str):
    """
    Returns tuple: (ma_price_gap, ma_price_gap_a, ma_price_gap_b, ma_price_gap_c)
    """
    if df_input_slice is None or df_input_slice.height == 0:
        logger.debug("ERA %s: input slice empty for gap stats", era_label)
        return (None, None, None, None)

    ma_int_val = int(regime_cfg.get("ma_int", 0) or 0)
    sel_gap_col = selected_gap_col_for_ma_int(ma_int_val)

    exprs = []
    for cname in ("ma_price_gap_a", "ma_price_gap_b", "ma_price_gap_c"):
        if cname in df_input_slice.columns:
            exprs.append(pl.col(cname).median().alias(cname))
        else:
            exprs.append(pl.lit(None).cast(pl.Float32).alias(cname))

    if sel_gap_col in df_input_slice.columns:
        exprs.append(pl.col(sel_gap_col).median().alias("ma_price_gap_sel"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float32).alias("ma_price_gap_sel"))

    df_med = df_input_slice.select(exprs)
    if df_med.height == 0:
        a = b = c = sel = None
    else:
        row = df_med.row(0)
        a = None if row[0] is None else float(row[0])
        b = None if row[1] is None else float(row[1])
        c = None if row[2] is None else float(row[2])
        sel = None if row[3] is None else float(row[3])

    # Debug counts/samples for investigation
    try:
        counts = {}
        samples = {}
        cols_to_check = [c for c in ("ma_price_gap_a", "ma_price_gap_b", "ma_price_gap_c") if c in df_input_slice.columns] + ([sel_gap_col] if sel_gap_col in df_input_slice.columns else [])
        for col in cols_to_check:
            non_null_count = int(df_input_slice.select(pl.col(col).drop_nulls().count()).row(0)[0])
            counts[col] = non_null_count
            samples[col] = None
            if non_null_count > 0:
                samples[col] = df_input_slice.filter(pl.col(col).is_not_null()).select(pl.col(col)).row(0)[0]
        logger.debug("ERA %s gap counts: %s sample_first: %s", era_label, counts, samples)
    except Exception:
        logger.debug("ERA %s: gap debug info failed", era_label)

    return (sel, a, b, c)

def _get_or_generate_signals(months: int, era_label: str, regime_id: int, regime_cfg: dict, run_cfg: dict, df_input_slice: pl.DataFrame, df_main: pl.DataFrame, base_minutes: int, force_rebuild_cache: bool):
    """
    Returns canonical signals DF (or empty DF with schema).
    """
    df_signals = None
    if not force_rebuild_cache:
        try:
            df_signals = load_signals_cached(months, era_label, str(regime_id))
        except Exception as e:
            logger.debug("signals cache read failed for cfg=%s era=%s: %s", regime_id, era_label, e)
            df_signals = None

    if df_signals is not None and not df_signals.is_empty():
        df_signals = normalize_signals_times(df_signals, df_main=df_main)
    else:
        if df_input_slice is None or df_input_slice.height == 0:
            return pl.DataFrame([], schema=get_schema("signals"))
        df_signals = generate_filtered_signals(df_input_slice, {**run_cfg, **regime_cfg}, df_main=df_main)
        if df_signals is None or df_signals.is_empty():
            df_signals = pl.DataFrame([], schema=get_schema("signals"))

    # apply lookback filter (preserve shift behavior)
    lookback_h = int(regime_cfg.get("entry_lookback_h", 0) or 0)
    if regime_cfg.get("use_entry_lookback", False) and lookback_h > 0:
        lookback_rows = int((lookback_h * 60) // base_minutes)
        df_signals = df_signals.with_columns(pl.col("close").shift(lookback_rows).alias("price_lookback"))
        df_signals = df_signals.filter(
            ((pl.col("side") == 1) & (pl.col("close") > pl.col("price_lookback"))) |
            ((pl.col("side") == -1) & (pl.col("close") < pl.col("price_lookback")))
        ).drop("price_lookback")

    # ensure idx exists
    if "idx" not in df_signals.columns and "time_ns" in df_signals.columns:
        try:
            sig_times = df_signals["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
            main_times = df_main["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
            idxs = np.searchsorted(main_times, sig_times, side="right").astype(np.int64)
            idxs = np.clip(idxs, 0, max(0, int(df_main.height) - 1)).astype(np.int64)
            df_signals = df_signals.with_columns(pl.Series("idx", idxs))
        except Exception:
            logger.debug("Failed to map signals.time_ns to idx for cfg=%s era=%s", regime_id, era_label)

    df_signals = df_signals.with_columns(pl.lit(regime_id).alias("regime_id"))
    stage_for_flush("signals", months, era_label, str(regime_id), df_signals)

    logger.info("SIGS cfg=%s era=%s -> rows=%d cols=%s", regime_id, era_label, (df_signals.height if df_signals is not None else 0), (df_signals.columns if df_signals is not None else []))
    return df_signals

def _run_backtest_grid(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    df_main: pl.DataFrame,
    main_close_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    main_spread_arr: np.ndarray,
    main_funding_arr: np.ndarray,
    df_signals: pl.DataFrame,
    months: int,
    era_label: str,
    era_int: int,
    ma_price_gap_vals: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    results_dir: Path,
    batch_id: int,
    stager,
    base_minutes: int,
    max_dd_threshold: float,
    force_rebuild_cache: bool,
    combos: List[Tuple[float, float]],
):
    (ma_price_gap, ma_price_gap_a, ma_price_gap_b, ma_price_gap_c) = ma_price_gap_vals

    buys = df_signals.filter(pl.col("side") == 1) if (df_signals is not None and not df_signals.is_empty()) else pl.DataFrame([], schema=get_schema("signals"))
    sells = df_signals.filter(pl.col("side") == -1) if (df_signals is not None and not df_signals.is_empty()) else pl.DataFrame([], schema=get_schema("signals"))

    bucket_df = None
    if not force_rebuild_cache:
        try:
            bucket_df = load_backtest_cached(months, era_label, str(regime_id))
            if bucket_df is not None:
                bucket_df = enforce_schema(bucket_df, "backtest")
        except Exception as e:
            logger.debug("load_backtest_cached failed for cfg=%s era=%s: %s", regime_id, era_label, e)
            bucket_df = None

    for sl_val, tp_val in combos:
        for side_df_pl, side_flag in ((buys, 1), (sells, -1)):
            total_pos = 0
            if side_df_pl is not None and not side_df_pl.is_empty():
                total_pos = int(side_df_pl.height)

            if total_pos == 0:
                master_row_raw = {
                    # 1. Identifiers & Temporal Context
                    "regime_id": int(regime_id),
                    "era_int": int(era_int),
                    "side": int(side_flag),

                    # 2. Strategy Hyperparameters (The Input Logic)
                    "ma_int": int(regime_cfg.get("ma_int", 0)),
                    "ma_reversion": bool(regime_cfg.get("ma_reversion", False)),
                    "entry_lookback_h": int(regime_cfg.get("entry_lookback_h", 0)),
                    "exit_window_h": int(regime_cfg.get("exit_window_h", 0)),
                    "SL": float(sl_val),
                    "TP": float(tp_val),

                    # 3. Market State / Features (The Gaps)
                    "ma_price_gap": ma_price_gap,
                    "ma_price_gap_a": ma_price_gap_a,
                    "ma_price_gap_b": ma_price_gap_b,
                    "ma_price_gap_c": ma_price_gap_c,

                    # 4. Performance Metrics (The Output Results)
                    "total_pos": int(total_pos),
                    "win_pos": int(win_pos),
                    "balance": final_balance,
                    "max_drawdown": max_dd,
                }

                # The single source of truth: 
                # Polars will now interpret 'None' as a null of the type defined in MASTER_SCHEMA.
                canonical = pl.DataFrame([master_row_raw]).pipe(enforce_schema, "master").to_dicts()[0]
                buffer_master_row(results_dir, batch_id, canonical)

                continue

            sig_idxs = side_df_pl["idx"].to_numpy(allow_copy=True).astype(np.int64)
            sig_times_ns = side_df_pl["time_ns"].to_numpy(allow_copy=True).astype(np.int64)
            sig_n = int(sig_idxs.size)
            sig_min_ns = int(sig_times_ns.min()) if sig_n > 0 else -1
            sig_max_ns = int(sig_times_ns.max()) if sig_n > 0 else -1

            entry_idx = None; exit_idx = None; rets = None
            if bucket_df is not None and not bucket_df.is_empty():
                try:
                    hit = bucket_df.filter(
                        (pl.col("sig_n") == sig_n) &
                        (pl.col("sig_min_ns") == sig_min_ns) &
                        (pl.col("sig_max_ns") == sig_max_ns) &
                        (pl.col("SL") == float(sl_val)) &
                        (pl.col("TP") == float(tp_val)) &
                        (pl.col("side") == int(side_flag)) &
                        (pl.col("exit_window_h") == int(regime_cfg.get("exit_window_h", 0)))
                    )
                    if hit.height > 0:
                        entry_idx = np.asarray(hit["entry_idx"][0], dtype=np.int64)
                        exit_idx = np.asarray(hit["exit_idx"][0], dtype=np.int64)
                        rets = np.asarray(hit["ret"][0], dtype=np.float64)
                except Exception:
                    entry_idx = None
                    exit_idx = None
                    rets = None

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
                        side_flag=side_flag
                    )
                    entry_idx = np.asarray(res.get("entry_idx", []), dtype=np.int64)
                    exit_idx = np.asarray(res.get("exit_idx", []), dtype=np.int64)
                    rets = np.asarray(res.get("rets", res.get("ret", [])), dtype=np.float64)

                    res_df = pl.DataFrame({
                        "sig_n": [sig_n],
                        "sig_min_ns": [sig_min_ns],
                        "sig_max_ns": [sig_max_ns],
                        "SL": [float(sl_val)],
                        "TP": [float(tp_val)],
                        "side": [int(side_flag)],
                        "exit_window_h": [int(regime_cfg.get("exit_window_h", 0))],
                        "entry_idx": [entry_idx.astype(np.int64)],
                        "exit_idx": [exit_idx.astype(np.int64)],
                        "ret": [rets.astype(np.float64)]
                    })
                    try:
                        res_df = enforce_schema(res_df, "backtest", strict=True)
                    except Exception:
                        logger.debug("res_df: enforce_schema(backtest) failed")
                    stage_for_flush("backtest", months, era_label, str(regime_id), res_df)
                except Exception as e:
                    logger.error("Backtest kernel error cfg=%s era=%s: %s", regime_id, era_label, e)
                    continue

            mask_closed = np.asarray([]) if exit_idx is None else (exit_idx >= 0)
            if mask_closed.size and mask_closed.any():
                closed_rets = rets[mask_closed]
                closed_entry_idxs = entry_idx[mask_closed].astype(np.int64)
                closed_exit_idxs = exit_idx[mask_closed].astype(np.int64)

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
                    spread_is_percent=bool(run_cfg.get("spread_is_percent", True))
                )
                equity_arr, max_dd = fast_compound_equity(pnl_pct_arr, 100.0)
            else:
                closed_rets = np.array([], dtype=np.float64)
                exit_times_ns = np.array([], dtype=np.int64)
                equity_arr = np.array([], dtype=np.float64)
                max_dd = 0.0

            if isinstance(max_dd, float) and max_dd > max_dd_threshold:
                logger.debug("Skipping cfg=%s SL=%s TP=%s era=%s due to max_dd %.3f", regime_id, sl_val, tp_val, era_label, max_dd)
                continue

            part_key = str(regime_id) if getattr(stager, "partition_by", "") == "regime_id" else str(era_int)

            if closed_rets.size == 0:
                final_balance = None
                win_pos = 0
            else:
                equity_data = {
                    # Foreign Keys (To link back to Master)
                    "regime_id": int(regime_id),
                    "era_int": int(era_int),
                    "side": int(side_flag),
                    
                    # Time-Series Data (The actual chart info)
                    "time_ns": exit_times_ns.astype(np.int64),
                    "pnl_pct": pnl_pct_arr.astype(np.float32),
                    "equity": equity_arr.astype(np.float32),
                }

                # Create the lean DataFrame
                equity_df = pl.DataFrame(
                    equity_data, 
                    schema=get_schema("equity") # Ensure schema.py matches these 6 columns
                ).select(get_schema("equity").keys())
                try:
                    equity_df = enforce_schema(equity_df, "equity")
                except Exception:
                    logger.debug("equity_df: enforce_schema(equity) failed; staging original")
                if equity_df.height > 0:
                    stager.stage(part_key, equity_df)

                final_balance = float(np.float32(equity_arr[-1])) if equity_arr.size else None
                win_pos = int(np.count_nonzero(closed_rets > 0.0))

            logger.debug("MASTER_ROW -> cfg=%s era=%s ma_int=%s ma_rev=%s side=%s SL=%s TP=%s total_pos=%d win_pos=%d balance=%s max_dd=%s",
                        regime_id, era_label, regime_cfg.get("ma_int"), regime_cfg.get("ma_reversion"),
                        side_flag, sl_val, tp_val, total_pos, win_pos,
                        (str(final_balance) if final_balance is not None else "None"),
                        (f"{max_dd:.3f}" if isinstance(max_dd, float) else str(max_dd))
                        )

            master_row_raw = {
                "balance": None if final_balance is None else float(final_balance),
                "SL": float(sl_val),
                "TP": float(tp_val),
                "win_pos": int(win_pos),
                "total_pos": int(total_pos),
                "side": int(side_flag),
                "exit_window_h": int(regime_cfg.get("exit_window_h", 0)),
                "era_int": int(era_int),
                "regime_id": int(regime_id),
                "ma_int": int(regime_cfg.get("ma_int", 0)),
                "ma_price_gap": None if ma_price_gap is None else float(ma_price_gap),
                "ma_price_gap_a": None if ma_price_gap_a is None else float(ma_price_gap_a),
                "ma_price_gap_b": None if ma_price_gap_b is None else float(ma_price_gap_b),
                "ma_price_gap_c": None if ma_price_gap_c is None else float(ma_price_gap_c),
                "ma_reversion": bool(regime_cfg.get("ma_reversion", False)),
                "entry_lookback_h": int(regime_cfg.get("entry_lookback_h", 0)),
                "max_drawdown": None if (not isinstance(max_dd, float) or math.isnan(max_dd)) else float(max_dd)
            }

            canonical = pl.DataFrame([master_row_raw]).pipe(enforce_schema, "master").to_dicts()[0]
            buffer_master_row(results_dir, batch_id, canonical)

def process_era_combos(
    regime_id: int,
    regime_cfg: dict,
    run_cfg: dict,
    df_main: pl.DataFrame,
    main_close_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    main_spread_arr: np.ndarray,
    main_funding_arr: np.ndarray,
    results_dir: Path,
    batch_id: int,
    stager,
    force_rebuild_cache: bool,
    cache_lookback_minutes_cfg,
    base_minutes: int,
    max_dd_threshold: float,
) -> Tuple[int, int]:
    processed_inc = 0
    skipped_inc = 0

    sl_vals, tp_vals = _expand_sl_tp({**run_cfg, **regime_cfg})
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))

    months = int(run_cfg.get("sl_tp_interval_months", 6))
    grid_start = pd.to_datetime(run_cfg.get("grid_start_date")).to_pydatetime()
    grid_end = pd.to_datetime(run_cfg.get("grid_end_date")).to_pydatetime()

    windows = _split_period_windows_from_pl(df_main, months, min_dt=grid_start, max_dt=grid_end)
    logger.debug("process_era_combos: config=%s windows=%d", regime_id, len(windows))

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

        # compute lookback
        if cache_lookback_minutes_cfg is not None:
            lookback_minutes = int(cache_lookback_minutes_cfg)
        elif (full_cfg_lookback := (regime_cfg.get("lookback_minutes") or run_cfg.get("lookback_minutes"))):
            lookback_minutes = int(full_cfg_lookback)
        else:
            ma_periods = run_cfg.get("ma_periods", []) or []
            max_ma_period = int(max(ma_periods)) if ma_periods else 0
            entry_lookbacks = run_cfg.get("entry_lookback_h", []) or []
            max_entry_lb_minutes = int(max(entry_lookbacks) * 60) if entry_lookbacks else 0
            default_lookback_minutes = max(max_ma_period * base_minutes, max_entry_lb_minutes) + (24 * 60)
            lookback_minutes = int(default_lookback_minutes)

        lookback_delta = datetime.timedelta(minutes=lookback_minutes)
        input_start = start - lookback_delta
        start_ns = np.datetime64(input_start).astype("datetime64[ns]").astype(np.int64)
        end_ns = np.datetime64(end).astype("datetime64[ns]").astype(np.int64)

        min_ns = int(main_time_ns_arr[0])
        max_ns = int(main_time_ns_arr[-1])
        if start_ns < min_ns:
            start_ns = min_ns
        if end_ns > max_ns:
            end_ns = max_ns

        start_idx = int(np.searchsorted(main_time_ns_arr, start_ns, side="left"))
        end_idx = int(np.searchsorted(main_time_ns_arr, end_ns, side="left"))
        df_input_slice = df_main.slice(start_idx, max(0, end_idx - start_idx))

        logger.debug("ERA %s slice rows=%d start_idx=%d end_idx=%d", era_label, (df_input_slice.height if df_input_slice is not None else 0), start_idx, end_idx)

        # compute gap stats
        ma_price_gap_vals = _compute_era_gap_stats(df_input_slice, df_main, regime_cfg, era_label)
        logger.debug("ERA %s gap stats -> selected=%s a=%s b=%s c=%s", era_label, ma_price_gap_vals[0], ma_price_gap_vals[1], ma_price_gap_vals[2], ma_price_gap_vals[3])

        # get or generate signals
        df_signals = _get_or_generate_signals(months, era_label, regime_id, regime_cfg, run_cfg, df_input_slice, df_main, base_minutes, force_rebuild_cache)

        # run backtests
        _run_backtest_grid(
            regime_id=regime_id,
            regime_cfg=regime_cfg,
            run_cfg=run_cfg,
            df_main=df_main,
            main_close_arr=main_close_arr,
            main_time_ns_arr=main_time_ns_arr,
            main_spread_arr=main_spread_arr,
            main_funding_arr=main_funding_arr,
            df_signals=df_signals,
            months=months,
            era_label=era_label,
            era_int=era_int,
            ma_price_gap_vals=ma_price_gap_vals,
            results_dir=results_dir,
            batch_id=batch_id,
            stager=stager,
            base_minutes=base_minutes,
            max_dd_threshold=max_dd_threshold,
            force_rebuild_cache=force_rebuild_cache,
            combos=combos,
        )

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

    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_path, "r", encoding="utf8") as f:
        batch_data = json.load(f)

    run_cfg = _load_run_config()
    regimes = batch_data["regimes"]
    batch_id = int(batch_data.get("batch_id", 0))
    logger.info("🧵 Worker started: %s | Regimes to process: %d (batch_id=%d)", batch_path.name, len(regimes), batch_id)

    prepared = prepare_worker_data(session_dir, run_cfg)
    df_main = prepared["df_main"]
    main_close_arr = prepared["main_close_arr"]
    main_time_ns_arr = prepared["main_time_ns_arr"]
    main_spread_arr = prepared["main_spread_arr"]
    main_funding_arr = prepared["main_funding_arr"]

    max_dd_threshold = float(run_cfg.get("max_dd_threshold", 0.20))
    flush_rows = int(run_cfg.get("PARQUET_FLUSH_ROWS", 100_000))
    partition_by = str(run_cfg.get("EQUITY_PARTITION_BY", "era_int"))
    stager = EquityStager(session_dir / "equity_partitioned", batch_id, flush_rows, partition_by)

    processed_count = 0
    skipped_count = 0
    force_rebuild_cache = bool(run_cfg.get("force_rebuild_cache", False))
    cache_lookback_minutes_cfg = run_cfg.get("cache_lookback_minutes", None)
    base_minutes = int(run_cfg.get("BASE_MINUTES", 5))

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
            regime_cfg["entry_lookback_h"] = int(regime_cfg.get("entry_lookback_h", 0) or 0)
            regime_cfg["exit_window_h"] = int(regime_cfg.get("exit_window_h", 0) or 0)
            regime_cfg["use_stochastic"] = bool(regime_cfg.get("use_stochastic", False))
            regime_cfg["use_entry_lookback"] = bool(regime_cfg.get("use_entry_lookback", False))

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
                df_main=df_main,
                main_close_arr=main_close_arr,
                main_time_ns_arr=main_time_ns_arr,
                main_spread_arr=main_spread_arr,
                main_funding_arr=main_funding_arr,
                results_dir=results_dir,
                batch_id=batch_id,
                stager=stager,
                force_rebuild_cache=force_rebuild_cache,
                cache_lookback_minutes_cfg=cache_lookback_minutes_cfg,
                base_minutes=base_minutes,
                max_dd_threshold=max_dd_threshold,
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
            stager.flush_all()
            _flush_master_rows_buffer(results_dir, batch_id)
        except Exception as e:
            logger.debug("Flush local stage failed: %s", e)

        try:
            flush_all_buffers()
        except Exception:
            logger.debug("flush_all_buffers() failed")

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
# feature_prep.py
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from etl.schema import enforce_schema

from etl.feature_helpers import (
    _unwrap_singleton,
    _as_list,
    _as_bool,
    _as_int,
    _as_float,
    _as_str,
    _as_int_list,
    _as_float_list,
    _as_threshold_pairs,
    _as_ma_type_list,
)

FEATURE_CACHE_DIR = Path(os.getenv(
    "FEATURE_CACHE_DIR",
    "/opt/airflow/airflow-trading/data_lake/cache",
))

FEATURE_CACHE_VERSION = 1

def _ordered_unique_ints(values: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in _as_int_list(values):
        if x > 0 and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _hash_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf8")
    return hashlib.sha1(payload).hexdigest()


def _file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _compute_ma_np(values: np.ndarray, ma_type: str, period: int) -> np.ndarray:
    ma_type = (ma_type or "sma").strip().lower()
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]

    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period <= 0:
        return out

    if ma_type == "ema":
        alpha = 2.0 / (period + 1.0)
        out[0] = arr[0]
        for i in range(1, n):
            v = arr[i]
            if np.isnan(v):
                out[i] = out[i - 1]
            else:
                prev = out[i - 1]
                out[i] = v if np.isnan(prev) else (alpha * v + (1.0 - alpha) * prev)
        return out

    if ma_type == "kama":
        fast = 2
        slow = 30
        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)

        if n < period:
            out[:] = arr
            return out

        seed = np.nanmean(arr[:period])
        if np.isnan(seed):
            seed = arr[period - 1]
        out[period - 1] = seed

        for i in range(period, n):
            current = arr[i]
            if np.isnan(current):
                out[i] = out[i - 1]
                continue

            prev_seed = arr[i - period]
            if np.isnan(prev_seed):
                out[i] = out[i - 1]
                continue

            window = arr[i - period : i + 1]
            if np.isnan(window).any():
                out[i] = out[i - 1]
                continue

            change = abs(current - prev_seed)
            volatility = float(np.sum(np.abs(np.diff(window))))
            er = 0.0 if volatility == 0.0 else (change / volatility)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            prev = out[i - 1]
            out[i] = current if np.isnan(prev) else (prev + sc * (current - prev))

        return out

    # default = sma
    if period == 1:
        return arr.copy()

    csum = np.cumsum(np.insert(arr, 0, 0.0))
    rolling = (csum[period:] - csum[:-period]) / float(period)
    out[period - 1:] = rolling
    return out

def _build_ma_specs(run_cfg: dict, modifier: int) -> list[dict]:
    periods = [int(x) for x in _as_int_list(run_cfg.get("ma_periods", [])) if int(x) > 0]
    if not periods:
        return []

    types = _as_ma_type_list(run_cfg.get("ma_types", None), len(periods))
    specs: list[dict] = []

    for idx, (period, ma_type) in enumerate(zip(periods, types)):
        ma_type = (ma_type or "sma").strip().lower()
        eff = max(1, int(period) * int(modifier))
        specs.append({
            "slot": int(idx),
            "type": ma_type,
            "period": int(period),
            "effective_window": int(eff),
            "col": f"ma_{idx:02d}_{ma_type}_{eff}",
        })

    return specs

def build_feature_manifest(run_cfg: dict) -> dict:
    base_min = _as_int(run_cfg.get("BASE_MINUTES", 5), 5)
    modifier = max(1, _as_int(run_cfg.get("signal_timeframe_modifier", 3), 3))

    ma_specs = _build_ma_specs(run_cfg, modifier)

    stoch_specs = []
    if _as_bool(run_cfg.get("use_stochastic", False), False):
        ks = _ordered_unique_ints(run_cfg.get("stoch_k", [12])) or [12]
        ds = _ordered_unique_ints(run_cfg.get("stoch_d", [3])) or [3]
        ss = _ordered_unique_ints(run_cfg.get("stoch_s", [3])) or [3]
        for k in ks:
            for d in ds:
                for s in ss:
                    stoch_specs.append({
                        "k": int(k),
                        "d": int(d),
                        "s": int(s),
                        "col": f"stoch_k{k}_d{d}_s{s}",
                    })

    lookback_units = _ordered_unique_ints(run_cfg.get("entry_lookback_units", []))

    bbw_specs = []
    if _as_bool(run_cfg.get("use_bbw", False), False):
        periods = _ordered_unique_ints(run_cfg.get("bbw_periods", [])) or [96]
        stds = _as_float_list(run_cfg.get("bbw_std", [2.5])) or [2.5]
        for p in periods:
            for s in stds:
                bbw_specs.append({
                    "period": int(p),
                    "std": float(s),
                    "col": f"bbw_p{p}_s{s}",
                })

    manifest = {
        "feature_version": FEATURE_CACHE_VERSION,
        "BASE_MINUTES": base_min,
        "signal_timeframe_modifier": modifier,
        "ma_specs": ma_specs,
        "stoch_specs": stoch_specs,
        "lookback_units": lookback_units,
        "bbw_specs": bbw_specs,
        "ma_cols": [x["col"] for x in ma_specs],
        "stoch_cols": [x["col"] for x in stoch_specs],
        "bbw_cols": [x["col"] for x in bbw_specs],
    }
    manifest["cache_key"] = _hash_json(manifest)
    return manifest


def _build_lookback_map(run_cfg: dict, manifest: dict) -> dict[str, int]:
    base_min = int(manifest["BASE_MINUTES"])
    modifier = int(manifest["signal_timeframe_modifier"])

    max_ma = max([x["effective_window"] for x in manifest["ma_specs"]], default=0)
    max_stoch = max([
        max(int(x["k"]), int(x["d"]), int(x["s"])) * modifier
        for x in manifest["stoch_specs"]
    ], default=0)
    max_lb = max([int(x) for x in manifest["lookback_units"]], default=0) * modifier
    max_bbw = max([int(x["period"]) * modifier for x in manifest["bbw_specs"]], default=0)

    absolute_max_units = max(max_ma, max_stoch, max_lb, max_bbw)
    required_hours = int(np.ceil((absolute_max_units * base_min) / 60.0)) + 2

    start_str = _as_str(run_cfg.get("grid_start_date", ""))
    end_str = _as_str(run_cfg.get("grid_end_date", ""))
    if not start_str or not end_str:
        return {}

    interval = f"{_as_int(run_cfg.get('sl_tp_interval_months', 6), 6)}mo"
    start_dt = pl.select(pl.lit(start_str).str.to_datetime(time_zone="UTC")).item()
    end_dt = pl.select(pl.lit(end_str).str.to_datetime(time_zone="UTC")).item()

    era_series = pl.datetime_range(
        start=start_dt,
        end=end_dt,
        interval=interval,
        eager=True,
    ).dt.truncate("1mo")

    return {dt.strftime("%Y-%m"): int(required_hours) for dt in era_series}


def precompute_all_possible_features(
    df: pl.DataFrame,
    run_cfg: dict,
    manifest: Optional[dict] = None,
) -> Tuple[pl.DataFrame, dict]:
    if df is None or df.height == 0:
        run_cfg["lookback_map"] = {}
        run_cfg["ma_cols"] = []
        run_cfg["stoch_cols"] = []
        run_cfg["bbw_cols"] = []
        return df, run_cfg

    if manifest is None:
        manifest = build_feature_manifest(run_cfg)

    df = df.clone()

    close_arr = (
        df["close"]
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )

    # 1) Moving averages
    for spec in manifest["ma_specs"]:
        arr = _compute_ma_np(close_arr, spec["type"], spec["effective_window"])
        df = df.with_columns(pl.Series(spec["col"], arr.astype(np.float32, copy=False)))
        
# 2) Stochastic (Exponential Version)
    if manifest["stoch_specs"]:
        for spec in manifest["stoch_specs"]:
            k = spec["k"]
            d = spec["d"]
            s = spec["s"]
            col_name = spec["col"]
            
            # Apply timeframe modifier
            modifier = manifest["signal_timeframe_modifier"]
            k_mod = max(1, int(k) * modifier)
            s_mod = max(1, int(s) * modifier)
            d_mod = max(1, int(d) * modifier)

            s_min = pl.col("close").rolling_min(k_mod, min_periods=1)
            s_max = pl.col("close").rolling_max(k_mod, min_periods=1)

            df = df.with_columns([
                # Step 1 & 2: Calculate Raw %K and smooth it by 'S'
                (100.0 * (pl.col("close") - s_min) / (s_max - s_min))
                .fill_nan(50.0)
                .fill_null(50.0)
                .ewm_mean(span=s_mod, adjust=False) 
                .alias("_smoothed_k")
            ]).with_columns([
                # Step 3: Smooth the 'Smoothed K' by 'D' to get the final %D line
                pl.col("_smoothed_k")
                .ewm_mean(span=d_mod, adjust=False)
                .alias(col_name)
            ]).drop(["_smoothed_k"])

    # 3) Lookback breakout columns
    for lb_units in manifest["lookback_units"]:
        if lb_units < 0:
            continue

        periods = max(1, int(lb_units) * manifest["signal_timeframe_modifier"])
        hi = pl.col("high").rolling_max(periods).shift(1)
        lo = pl.col("low").rolling_min(periods).shift(1)

        df = df.with_columns([
            ((pl.col("close") - lo) / (hi - lo))
            .fill_nan(0.0)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(f"breakout_{lb_units}u")
        ])

        if lb_units == 0:
            df = df.with_columns([
                pl.lit(1.0).cast(pl.Float32).alias("breakout_0u")
            ])

    # 4) BBW columns
    for spec in manifest["bbw_specs"]:
        n = max(1, int(spec["period"]) * manifest["signal_timeframe_modifier"])
        s = float(spec["std"])
        col_name = spec["col"]

        rolling_mean = pl.col("close").rolling_mean(n)
        rolling_std = pl.col("close").rolling_std(n, ddof=0)
        raw_bbw = (2 * s * rolling_std) / rolling_mean

        b_min = raw_bbw.rolling_min(window_size=500)
        b_max = raw_bbw.rolling_max(window_size=500)

        df = df.with_columns([
            (100 * (raw_bbw - b_min) / (b_max - b_min))
            .fill_nan(50)
            .fill_null(50)
            .clip(0, 100)
            .cast(pl.Float32)
            .alias(col_name)
        ])

    # 5) Regime amplitude columns
    base_min = int(manifest["BASE_MINUTES"])
    modifier = int(manifest["signal_timeframe_modifier"])
    effective_min = max(1, base_min * modifier)

    amp_windows = {
        "regime_amp_index_24h": int(round((24 * 60) / effective_min)),
        "regime_amp_index_72h": int(round((72 * 60) / effective_min)),
        "regime_amp_index_1w": int(round((7 * 24 * 60) / effective_min)),
        "regime_amp_index_1m": int(round((30 * 24 * 60) / effective_min)),
    }

    exprs = []
    for new_name, win in amp_windows.items():
        win = max(1, int(win))
        rolling_min = pl.col("close").rolling_min(window_size=win, min_periods=win)
        rolling_max = pl.col("close").rolling_max(window_size=win, min_periods=win)
        raw = ((rolling_max - rolling_min) / rolling_min).fill_nan(0.0).fill_null(0.0).cast(pl.Float32)
        exprs.append(raw.alias(new_name))

    df = df.with_columns(exprs)
    df = df.with_columns([
        pl.col("regime_amp_index_24h").alias("rng_24h"),
        pl.col("regime_amp_index_72h").alias("rng_72h"),
        pl.col("regime_amp_index_1w").alias("rng_1w"),
        pl.col("regime_amp_index_1m").alias("rng_1m"),
    ])

    # 6) Loadable runtime metadata
    run_cfg["feature_manifest"] = manifest
    run_cfg["feature_cache_key"] = manifest["cache_key"]
    run_cfg["ma_cols"] = [x["col"] for x in manifest["ma_specs"]]
    run_cfg["stoch_cols"] = [x["col"] for x in manifest["stoch_specs"]]
    run_cfg["bbw_cols"] = [x["col"] for x in manifest["bbw_specs"]]
    run_cfg["lookback_map"] = _build_lookback_map(run_cfg, manifest)

    return df, run_cfg


def _cache_paths(base_path: Path, manifest: dict) -> Tuple[Path, Path]:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_fp = _file_fingerprint(base_path)
    key = _hash_json({
        "feature_version": FEATURE_CACHE_VERSION,
        "base_fp": base_fp,
        "manifest_key": manifest["cache_key"],
    })[:24]

    parquet_path = FEATURE_CACHE_DIR / f"prepared_main_{key}.parquet"
    ref_path = FEATURE_CACHE_DIR / f"prepared_main_{key}.json"
    return parquet_path, ref_path


def prepare_feature_cache(
    base_path: Path,
    run_cfg: dict,
    force_rebuild: bool = False,
) -> Tuple[Path, dict]:
    base_path = Path(base_path)
    if not base_path.exists():
        raise RuntimeError(f"Base data file not found at {base_path}")

    manifest = build_feature_manifest(run_cfg)
    out_path, ref_path = _cache_paths(base_path, manifest)

    if out_path.exists() and ref_path.exists() and not force_rebuild:
        try:
            cached_ref = json.loads(ref_path.read_text(encoding="utf8"))
            if cached_ref.get("manifest_key") == manifest["cache_key"]:
                run_cfg["feature_manifest"] = manifest
                run_cfg["feature_cache_key"] = manifest["cache_key"]
                run_cfg["ma_cols"] = [x["col"] for x in manifest["ma_specs"]]
                run_cfg["stoch_cols"] = [x["col"] for x in manifest["stoch_specs"]]
                run_cfg["bbw_cols"] = [x["col"] for x in manifest["bbw_specs"]]
                run_cfg["lookback_map"] = cached_ref.get("lookback_map", {})
                return out_path, manifest
        except Exception:
            pass

    df = pl.read_parquet(str(base_path))
    df = df.sort("time")

    norm_exprs = []
    if "time" in df.columns:
        norm_exprs.append(pl.col("time").dt.replace_time_zone(None).alias("time"))
    if "time_ns" in df.columns:
        norm_exprs.append(pl.col("time_ns").cast(pl.Int64).alias("time_ns"))
    else:
        norm_exprs.append(pl.col("time").cast(pl.Datetime("ns")).cast(pl.Int64).alias("time_ns"))

    if "close" not in df.columns:
        raise RuntimeError("df_main missing required 'close' column")
    if "high" not in df.columns or "low" not in df.columns:
        raise RuntimeError("df_main missing required 'high'/'low' columns")

    norm_exprs += [
        pl.col("open").cast(pl.Float32).alias("open") if "open" in df.columns else pl.lit(None).cast(pl.Float32).alias("open"),
        pl.col("high").cast(pl.Float32).alias("high"),
        pl.col("low").cast(pl.Float32).alias("low"),
        pl.col("close").cast(pl.Float32).alias("close"),
        pl.col("volume").fill_null(0.0).cast(pl.Float32).alias("volume") if "volume" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("volume"),
        pl.col("spread").fill_null(0.0).cast(pl.Float32).alias("spread") if "spread" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("spread"),
        pl.col("funding_rate").fill_null(0.0).cast(pl.Float32).alias("funding_rate") if "funding_rate" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("funding_rate"),
    ]

    df = df.with_columns(norm_exprs).with_row_count("idx").with_columns(pl.col("idx").cast(pl.Int64))

    df, run_cfg = precompute_all_possible_features(df, run_cfg, manifest=manifest)

    # Keep only a single normalization step
    df = enforce_schema(df, "df_main", strict=False)
    df = df.with_columns([
        pl.col(c).cast(pl.Float32)
        for c in df.columns
        if df.schema[c] in (pl.Float64, pl.Float32)
    ])

    df.write_parquet(str(out_path), compression="snappy")

    ref_payload = {
        "feature_version": FEATURE_CACHE_VERSION,
        "manifest_key": manifest["cache_key"],
        "parquet_path": str(out_path),
        "lookback_map": run_cfg.get("lookback_map", {}),
        "ma_cols": run_cfg.get("ma_cols", []),
        "stoch_cols": run_cfg.get("stoch_cols", []),
        "bbw_cols": run_cfg.get("bbw_cols", []),
    }
    ref_path.write_text(json.dumps(ref_payload, indent=2), encoding="utf8")

    return out_path, manifest


def load_prepared_feature_ref(session_dir: Path) -> dict:
    session_dir = Path(session_dir)
    ref_path = session_dir / "prepared_main_ref.json"
    if not ref_path.exists():
        raise RuntimeError(f"prepared_main_ref.json not found at {ref_path}")
    return json.loads(ref_path.read_text(encoding="utf8"))
# common/feature_prep.py

"""Feature cache builder for multi-timeframe signal families.

This module precomputes reusable signal columns on the base price series and on
higher timeframe series, then aligns them back to the base rows.

Design goals:
- stable column names
- backward compatible config parsing
- each signal family can use its own timeframe list
- one cache can support many CCD cycles
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl

from common.timeframes import timeframe_bars, timeframe_minutes
from common.feature_helpers import build_signal_manifest, timeframe_to_polars_every, timeframe_to_minutes
from common.schema import enforce_schema

FEATURE_CACHE_DIR = Path(
    os.getenv(
        "FEATURE_CACHE_DIR",
        "/opt/airflow/airflow-trading/data_lake/cache",
    )
)

FEATURE_CACHE_VERSION = 1


def _hash_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf8")
    return hashlib.sha1(payload).hexdigest()


def _file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _float_token(value: float) -> str:
    s = f"{float(value):.10g}"
    if "e" in s or "E" in s:
        return s.replace("+", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _compute_ma_np(values: np.ndarray, ma_type: str, period: int) -> np.ndarray:
    ma_type = (ma_type or "sma").strip().lower()
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]

    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period <= 0:
        return out

    if ma_type == "ema":
        alpha = 2.0 / (period + 1.0)

        seed_idx = None
        for i in range(n):
            if not np.isnan(arr[i]):
                seed_idx = i
                break

        if seed_idx is None:
            return out

        out[seed_idx] = arr[seed_idx]
        for i in range(seed_idx + 1, n):
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

    if period == 1:
        return arr.copy()

    csum = np.cumsum(np.insert(arr, 0, 0.0))
    rolling = (csum[period:] - csum[:-period]) / float(period)
    out[period - 1 :] = rolling
    return out


def _dedupe_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else None
    if pdf is not None:
        pdf = pdf.loc[:, ~pdf.columns.duplicated()].copy()
        return pl.from_pandas(pdf)
    return df


def _resample_ohlcv(df: pl.DataFrame, every: str) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()
    if "time" not in df.columns:
        raise RuntimeError("df_main missing required 'time' column")

    agg_exprs = []
    if "open" in df.columns:
        agg_exprs.append(pl.first("open").alias("open"))
    if "high" in df.columns:
        agg_exprs.append(pl.max("high").alias("high"))
    if "low" in df.columns:
        agg_exprs.append(pl.min("low").alias("low"))
    if "close" in df.columns:
        agg_exprs.append(pl.last("close").alias("close"))
    if "volume" in df.columns:
        agg_exprs.append(pl.sum("volume").alias("volume"))
    if "spread" in df.columns:
        agg_exprs.append(pl.mean("spread").alias("spread"))
    if "funding_rate" in df.columns:
        agg_exprs.append(pl.mean("funding_rate").alias("funding_rate"))
    if "time_ns" in df.columns:
        agg_exprs.append(pl.last("time_ns").alias("time_ns"))

    return (
        df.sort("time")
        .group_by_dynamic(
            index_column="time",
            every=every,
            period=every,
            closed="right",
            label="right",
        )
        .agg(agg_exprs)
        .sort("time")
    )


def _join_features(base_df: pl.DataFrame, tf_df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    if base_df is None or base_df.is_empty() or tf_df is None or tf_df.is_empty():
        return base_df
    if not feature_cols:
        return base_df
    join_df = tf_df.select(["time"] + [c for c in feature_cols if c in tf_df.columns]).sort("time")
    return base_df.join_asof(join_df, on="time", strategy="backward")


def _add_ma_features(tf_df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    if tf_df is None or tf_df.is_empty() or not specs:
        return tf_df
    if "close" not in tf_df.columns:
        return tf_df

    close_arr = (
        tf_df["close"]
        .fill_nan(np.nan)
        .fill_null(np.nan)
        .to_numpy()
        .astype(np.float64, copy=False)
    )

    for spec in specs:
        col = spec["col"]
        arr = _compute_ma_np(close_arr, spec["type"], int(spec["period"]))
        tf_df = tf_df.with_columns(pl.Series(col, arr.astype(np.float32, copy=False)))
    return tf_df


def _add_stoch_features(tf_df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    if tf_df is None or tf_df.is_empty() or not specs:
        return tf_df
    if "close" not in tf_df.columns:
        return tf_df

    close = pl.col("close")

    for spec in specs:
        k = max(1, int(spec["k"]))
        d = max(1, int(spec["d"]))
        s = max(1, int(spec["s"]))
        col_name = spec["col"]

        low_k = close.rolling_min(k, min_periods=1)
        high_k = close.rolling_max(k, min_periods=1)
        denom = high_k - low_k

        tf_df = (
            tf_df.with_columns(
                (
                    pl.when(denom != 0)
                    .then(100.0 * (close - low_k) / denom)
                    .otherwise(50.0)
                    .fill_nan(50.0)
                    .fill_null(50.0)
                    .ewm_mean(span=s, adjust=False)
                    .alias("_stoch_k_smooth")
                )
            )
            .with_columns(
                pl.col("_stoch_k_smooth")
                .ewm_mean(span=d, adjust=False)
                .clip(0, 100)
                .alias(col_name)
            )
            .drop(["_stoch_k_smooth"])
        )

    return tf_df


def _add_lookback_features(tf_df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    if tf_df is None or tf_df.is_empty() or not specs:
        return tf_df
    if not all(c in tf_df.columns for c in ("high", "low", "close")):
        return tf_df

    close = pl.col("close")
    high = pl.col("high")
    low = pl.col("low")

    for spec in specs:
        units = max(1, int(spec["units"]))
        col_name = spec["col"]

        prior_high = high.rolling_max(units).shift(1)
        prior_low = low.rolling_min(units).shift(1)

        tf_df = tf_df.with_columns(
            pl.when((close > prior_high) & prior_high.is_not_null())
            .then(1.0)
            .when((close < prior_low) & prior_low.is_not_null())
            .then(0.0)
            .otherwise(0.5)
            .cast(pl.Float32)
            .alias(col_name)
        )
    return tf_df


def _add_bbw_features(tf_df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    if tf_df is None or tf_df.is_empty() or not specs:
        return tf_df
    if "close" not in tf_df.columns:
        return tf_df

    close = pl.col("close")

    for spec in specs:
        period = max(1, int(spec["period"]))
        std = float(spec["std"])
        col_name = spec["col"]

        rolling_mean = close.rolling_mean(period)
        rolling_std = close.rolling_std(period, ddof=0)
        raw_bbw = pl.when(rolling_mean != 0).then((2 * std * rolling_std) / rolling_mean).otherwise(0.0)

        norm_window = max(50, period * 3)
        b_min = raw_bbw.rolling_min(window_size=norm_window)
        b_max = raw_bbw.rolling_max(window_size=norm_window)
        den = b_max - b_min

        tf_df = tf_df.with_columns(
            pl.when(den != 0)
            .then(100 * (raw_bbw - b_min) / den)
            .otherwise(50.0)
            .fill_nan(50)
            .fill_null(50)
            .clip(0, 100)
            .cast(pl.Float32)
            .alias(col_name)
        )
    return tf_df

def _build_lookback_map(run_cfg: dict, manifest: dict) -> dict[str, int]:
    """
    Runtime metadata only.

    This is not a surrogate feature. It tells the worker roughly how much
    historical warmup is needed for each era, using the same timeframe logic
    used everywhere else.
    """
    base_min = int(manifest["BASE_MINUTES"])
    max_bars = 0

    def _tf(spec: dict) -> str:
        return str(spec.get("tf") or spec.get("timeframe") or "5m")

    for spec in manifest.get("ma_specs", []):
        bars = timeframe_bars(_tf(spec), base_min)
        max_bars = max(max_bars, bars * max(1, int(spec.get("period", 1))))

    for spec in manifest.get("stoch_specs", []):
        bars = timeframe_bars(_tf(spec), base_min)
        max_bars = max(max_bars, bars * max(1, int(spec.get("k", 1)), int(spec.get("d", 1)), int(spec.get("s", 1))))

    for spec in manifest.get("lookback_specs", []):
        bars = timeframe_bars(_tf(spec), base_min)
        max_bars = max(max_bars, bars * max(1, int(spec.get("units", 1))))

    for spec in manifest.get("bbw_specs", []):
        bars = timeframe_bars(_tf(spec), base_min)
        max_bars = max(max_bars, bars * max(1, int(spec.get("period", 1))))

    required_minutes = max_bars * base_min
    required_hours = int(np.ceil(required_minutes / 60.0)) + 2

    start_str = str(run_cfg.get("grid_start_date", "") or "")
    end_str = str(run_cfg.get("grid_end_date", "") or "")
    if not start_str or not end_str:
        return {}

    try:
        interval = f"{int(run_cfg.get('sl_tp_interval_months', 6) or 6)}mo"
        start_dt = pl.select(pl.lit(start_str).str.to_datetime(time_zone="UTC")).item()
        end_dt = pl.select(pl.lit(end_str).str.to_datetime(time_zone="UTC")).item()
    except Exception:
        return {}

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
        run_cfg["lookback_cols"] = []
        run_cfg["bbw_cols"] = []
        return df, run_cfg

    if manifest is None:
        manifest = build_signal_manifest(run_cfg)

    # --- FIX: inject base timeframe invariant ---
    if "BASE_MINUTES" not in manifest:
        base_tf = run_cfg.get("base_timeframe", "5m")
        manifest["BASE_MINUTES"] = timeframe_to_minutes(base_tf)

    df = df.clone().sort("time")

    if "time" not in df.columns:
        raise RuntimeError("df_main missing required 'time' column")
    if "close" not in df.columns:
        raise RuntimeError("df_main missing required 'close' column")
    if "high" not in df.columns or "low" not in df.columns:
        raise RuntimeError("df_main missing required 'high'/'low' columns")

    if "time_ns" not in df.columns:
        df = df.with_columns(pl.col("time").cast(pl.Datetime("ns")).cast(pl.Int64).alias("time_ns"))
    else:
        df = df.with_columns(pl.col("time_ns").cast(pl.Int64))

    df = df.with_columns(
        [
            pl.col("open").cast(pl.Float32).alias("open") if "open" in df.columns else pl.lit(None).cast(pl.Float32).alias("open"),
            pl.col("high").cast(pl.Float32).alias("high"),
            pl.col("low").cast(pl.Float32).alias("low"),
            pl.col("close").cast(pl.Float32).alias("close"),
            pl.col("volume").fill_null(0.0).cast(pl.Float32).alias("volume") if "volume" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("volume"),
            pl.col("spread").fill_null(0.0).cast(pl.Float32).alias("spread") if "spread" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("spread"),
            pl.col("funding_rate").fill_null(0.0).cast(pl.Float32).alias("funding_rate") if "funding_rate" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("funding_rate"),
        ]
    )

    for tf in manifest["signal_timeframes"]:
        every = timeframe_to_polars_every(tf)
        tf_df = _resample_ohlcv(df, every)
        if tf_df is None or tf_df.is_empty():
            continue

        tf_ma_specs = [x for x in manifest["ma_specs"] if x["tf"] == tf]
        tf_stoch_specs = [x for x in manifest["stoch_specs"] if x["tf"] == tf]
        tf_lookback_specs = [x for x in manifest["lookback_specs"] if x["tf"] == tf]
        tf_bbw_specs = [x for x in manifest["bbw_specs"] if x["tf"] == tf]

        tf_df = _add_ma_features(tf_df, tf_ma_specs)
        tf_df = _add_stoch_features(tf_df, tf_stoch_specs)
        tf_df = _add_lookback_features(tf_df, tf_lookback_specs)
        tf_df = _add_bbw_features(tf_df, tf_bbw_specs)

        feature_cols = [
            c
            for c in tf_df.columns
            if c not in {"time", "open", "high", "low", "close", "volume", "spread", "funding_rate", "time_ns"}
        ]
        df = _join_features(df, tf_df, feature_cols)

    run_cfg["feature_manifest"] = manifest
    run_cfg["feature_cache_key"] = manifest["cache_key"]
    run_cfg["ma_cols"] = list(manifest.get("ma_cols", []))
    run_cfg["stoch_cols"] = list(manifest.get("stoch_cols", []))
    run_cfg["lookback_cols"] = list(manifest.get("lookback_cols", []))
    run_cfg["bbw_cols"] = list(manifest.get("bbw_cols", []))
    run_cfg["lookback_map"] = _build_lookback_map(run_cfg, manifest)

    return df, run_cfg


def _cache_paths(base_path: Path, manifest: dict) -> Tuple[Path, Path]:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_fp = _file_fingerprint(base_path)
    key = _hash_json(
        {
            "feature_version": FEATURE_CACHE_VERSION,
            "base_fp": base_fp,
            "manifest_key": manifest["cache_key"],
        }
    )[:24]

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

    manifest = build_signal_manifest(run_cfg)
    out_path, ref_path = _cache_paths(base_path, manifest)

    if out_path.exists() and ref_path.exists() and not force_rebuild:
        try:
            cached_ref = json.loads(ref_path.read_text(encoding="utf8"))
            if cached_ref.get("manifest_key") == manifest["cache_key"] and cached_ref.get("feature_version") == FEATURE_CACHE_VERSION:
                run_cfg["feature_manifest"] = manifest
                run_cfg["feature_cache_key"] = manifest["cache_key"]
                run_cfg["ma_cols"] = list(manifest.get("ma_cols", []))
                run_cfg["stoch_cols"] = list(manifest.get("stoch_cols", []))
                run_cfg["lookback_cols"] = list(manifest.get("lookback_cols", []))
                run_cfg["bbw_cols"] = list(manifest.get("bbw_cols", []))
                run_cfg["lookback_map"] = cached_ref.get("lookback_map", {})
                return out_path, manifest
        except Exception:
            pass

    df = pl.read_parquet(str(base_path)).sort("time")

    norm_exprs = []
    if "time" in df.columns:
        norm_exprs.append(pl.col("time").dt.replace_time_zone(None).alias("time"))
    if "time_ns" in df.columns:
        norm_exprs.append(pl.col("time_ns").cast(pl.Int64).alias("time_ns"))
    else:
        norm_exprs.append(pl.col("time").cast(pl.Datetime("ns")).cast(pl.Int64).alias("time_ns"))

    norm_exprs += [
        pl.col("open").cast(pl.Float32).alias("open") if "open" in df.columns else pl.lit(None).cast(pl.Float32).alias("open"),
        pl.col("high").cast(pl.Float32).alias("high"),
        pl.col("low").cast(pl.Float32).alias("low"),
        pl.col("close").cast(pl.Float32).alias("close"),
        pl.col("volume").fill_null(0.0).cast(pl.Float32).alias("volume") if "volume" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("volume"),
        pl.col("spread").fill_null(0.0).cast(pl.Float32).alias("spread") if "spread" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("spread"),
        pl.col("funding_rate").fill_null(0.0).cast(pl.Float32).alias("funding_rate") if "funding_rate" in df.columns else pl.lit(0.0).cast(pl.Float32).alias("funding_rate"),
    ]

    df = df.with_columns(norm_exprs).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))

    df, run_cfg = precompute_all_possible_features(df, run_cfg, manifest=manifest)

    df = enforce_schema(df, "df_main", strict=False)
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in df.columns if df.schema[c] in (pl.Float32, pl.Float64)])

    df.write_parquet(str(out_path), compression="snappy")

    ref_payload = {
        "feature_version": FEATURE_CACHE_VERSION,
        "manifest_key": manifest["cache_key"],
        "parquet_path": str(out_path),
        "lookback_map": run_cfg.get("lookback_map", {}),
        "ma_cols": run_cfg.get("ma_cols", []),
        "stoch_cols": run_cfg.get("stoch_cols", []),
        "lookback_cols": run_cfg.get("lookback_cols", []),
        "bbw_cols": run_cfg.get("bbw_cols", []),
    }
    ref_path.write_text(json.dumps(ref_payload, indent=2), encoding="utf8")

    return out_path, manifest


def load_prepared_feature_ref(session_dir: Path) -> dict:
    session_dir = Path(session_dir)

    preferred = session_dir / "prepared_main_ref.json"
    if preferred.exists():
        return json.loads(preferred.read_text(encoding="utf8"))

    candidates = sorted(
        session_dir.glob("prepared_main_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return json.loads(candidates[0].read_text(encoding="utf8"))

    raise RuntimeError(f"prepared_main_ref.json not found under {session_dir}")


__all__ = [
    "FEATURE_CACHE_VERSION",
    "build_signal_manifest",
    "precompute_all_possible_features",
    "prepare_feature_cache",
    "load_prepared_feature_ref",
]
# common/feature_helpers.py
"""Signal helpers for nested by_timeframe configs and runtime signal filtering."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl

from common.schema import get_schema
from common.timeframes import (
    normalize_timeframe,
    normalize_timeframe_list,
    timeframe_bars,
    timeframe_minutes,
)


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_bool(value: Any, default: bool = False) -> bool:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _as_int(value: Any, default: int = 0) -> int:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        return int(float(s))
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        return float(s)
    return default


def _as_str(value: Any, default: str = "") -> str:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    return str(value)


def _as_int_list(value: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in _as_list(value):
        v = _as_int(x, 0)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _as_float_list(value: Any) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for x in _as_list(value):
        v = float(_as_float(x, 0.0))
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _positive_ints(values: Any) -> list[int]:
    return [v for v in _ordered_unique_ints(values) if v > 0]

def _as_threshold_pairs(value: Any) -> list[list[float]]:
    value = _unwrap_singleton(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append([_as_float(item[0]), _as_float(item[1])])
        return out
    if isinstance(value, (list, tuple)):
        vals = [_as_float(x) for x in value]
        return [vals] if vals else []
    return [[_as_float(value)]]


def _ordered_unique_ints(values: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in _as_int_list(values):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ordered_unique_floats(values: Any) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for x in _as_float_list(values):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ordered_unique_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in _as_list(values):
        try:
            s = normalize_timeframe(x)
        except Exception:
            s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _as_ma_type_list(value: Any, count: int) -> list[str]:
    if count <= 0:
        return []
    if value is None:
        return ["sma"] * count
    if isinstance(value, str):
        v = value.strip().lower() or "sma"
        return [v] * count
    items = _as_list(value)
    if not items:
        return ["sma"] * count
    if len(items) == 1:
        v = _as_str(items[0], "sma").strip().lower() or "sma"
        return [v] * count
    out = [_as_str(x, "sma").strip().lower() or "sma" for x in items]
    if len(out) < count:
        out.extend(["sma"] * (count - len(out)))
    return out[:count]


def get_signal_structure(cfg: dict) -> dict:
    block = cfg.get("signal_structure")
    return block if isinstance(block, dict) else {}


def get_family_cfg(cfg: dict, family: str) -> dict:
    fam = get_signal_structure(cfg).get(family, {})
    return fam if isinstance(fam, dict) else {}


def family_enabled(cfg: dict, family: str, default: bool = False) -> bool:
    fam = get_family_cfg(cfg, family)
    if "enabled" in fam:
        return _as_bool(fam.get("enabled"), default)
    return default


def family_combine_mode(cfg: dict, family: str, default: str = "all") -> str:
    fam = get_family_cfg(cfg, family)
    mode = str(fam.get("combine", default)).strip().lower()
    return mode if mode in {"all", "any"} else default


def family_timeframes(cfg: dict, family: str, default: Optional[list[str]] = None) -> list[str]:
    fam = get_family_cfg(cfg, family)
    by_tf = fam.get("by_timeframe", {})

    if not isinstance(by_tf, dict) or not by_tf:
        return normalize_timeframe_list(default or [])

    return normalize_timeframe_list(by_tf.keys())


def family_tf_cfg(cfg: dict, family: str, tf: str) -> dict:
    fam = get_family_cfg(cfg, family)
    merged = {k: v for k, v in fam.items() if k != "by_timeframe"}
    by_tf = fam.get("by_timeframe", {})
    if not isinstance(by_tf, dict):
        return merged
    key = normalize_timeframe(tf)
    tf_cfg = by_tf.get(key, {})
    if isinstance(tf_cfg, dict):
        merged.update(tf_cfg)
    return merged


def signal_timeframes_from_cfg(values: Any, default: Optional[list[str]] = None) -> list[str]:
    out = normalize_timeframe_list(values if values is not None else default or ["5m"])
    return out or list(default or ["5m"])


def timeframe_to_minutes(value: Any) -> int:
    return timeframe_minutes(value)


def timeframe_to_polars_every(value: Any) -> str:
    tf = normalize_timeframe(value)
    m = re.match(r"^(\d+)([mhdwM])$", tf)
    if not m:
        raise ValueError(f"Invalid timeframe: {value!r}")

    n = int(m.group(1))
    u = m.group(2)

    if u == "m":
        return f"{n}m"
    if u == "h":
        return f"{n}h"
    if u == "D":
        return f"{n}d"
    if u == "W":
        return f"{n}w"
    if u == "M":
        return f"{n}mo"
    raise ValueError(f"Unsupported timeframe unit: {u!r}")


def _resolve_threshold_pair(value: Any, default_low: float = 30.0, default_high: float = 70.0) -> Tuple[float, float]:
    pairs = _as_threshold_pairs(value)
    if pairs and len(pairs[0]) >= 2:
        return float(pairs[0][0]), float(pairs[0][1])
    return float(default_low), float(default_high)


def _resolve_scalar(value: Any, default: float) -> float:
    value = _unwrap_singleton(value)
    if isinstance(value, dict):
        if "threshold" in value:
            return _as_float(value.get("threshold"), default)
        if "value" in value:
            return _as_float(value.get("value"), default)
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return _as_float(value[0], default)
    return _as_float(value, default)


def build_signal_manifest(run_cfg: dict) -> dict:
    signal_block = get_signal_structure(run_cfg)
    if not signal_block:
        raise ValueError("signal_structure is required")

    families = {}
    all_timeframes: list[str] = []

    for family_name in ("ma", "stochastic", "lookback", "bbw"):
        fam = get_family_cfg(run_cfg, family_name)
        if not fam:
            continue

        tfs = family_timeframes(run_cfg, family_name)
        if not tfs:
            continue

        specs: list[dict] = []
        for tf in tfs:
            tf_cfg = family_tf_cfg(run_cfg, family_name, tf)

            if family_name == "ma":
                periods = _positive_ints(tf_cfg.get("periods", [])) or [8]
                types = _as_ma_type_list(tf_cfg.get("types", None), len(periods))
                for period, ma_type in zip(periods, types):
                    specs.append(
                        {
                            "family": "ma",
                            "tf": tf,
                            "period": int(period),
                            "type": str(ma_type).strip().lower() or "sma",
                            "col": f"ma_{tf}_p{int(period)}_{str(ma_type).strip().lower() or 'sma'}",
                        }
                    )

            elif family_name == "stochastic":
                ks = _positive_ints(tf_cfg.get("k", [])) or [12]
                ds = _positive_ints(tf_cfg.get("d", [])) or [3]
                ss = _positive_ints(tf_cfg.get("s", [])) or [3]
                for k in ks:
                    for d in ds:
                        for s in ss:
                            specs.append(
                                {
                                    "family": "stochastic",
                                    "tf": tf,
                                    "k": int(k),
                                    "d": int(d),
                                    "s": int(s),
                                    "col": f"stoch_{tf}_k{int(k)}_d{int(d)}_s{int(s)}",
                                }
                            )

            elif family_name == "lookback":
                units = _ordered_unique_ints(tf_cfg.get("entry_lookback_units", [])) or [1]
                for units_v in units:
                    if int(units_v) <= 0:
                        continue
                    specs.append(
                        {
                            "family": "lookback",
                            "tf": tf,
                            "units": int(units_v),
                            "col": f"lookback_{tf}_{int(units_v)}u",
                        }
                    )

            elif family_name == "bbw":
                periods = _positive_ints(tf_cfg.get("periods", [])) or [96]
                stds = _ordered_unique_floats(tf_cfg.get("std", [])) or [2.5]
                for period in periods:
                    for std in stds:
                        specs.append(
                            {
                                "family": "bbw",
                                "tf": tf,
                                "period": int(period),
                                "std": float(std),
                                "col": f"bbw_{tf}_p{int(period)}_s{float(std):g}",
                            }
                        )

        families[family_name] = specs
        all_timeframes.extend(tfs)

    signal_timeframes = _ordered_unique_strings(all_timeframes)

    manifest = {
        "feature_version": 1,
        "signal_structure": signal_block,
        "signal_timeframes": signal_timeframes,
        "ma_specs": families.get("ma", []),
        "stoch_specs": families.get("stochastic", []),
        "lookback_specs": families.get("lookback", []),
        "bbw_specs": families.get("bbw", []),
        "ma_cols": [x["col"] for x in families.get("ma", [])],
        "stoch_cols": [x["col"] for x in families.get("stochastic", [])],
        "lookback_cols": [x["col"] for x in families.get("lookback", [])],
        "bbw_cols": [x["col"] for x in families.get("bbw", [])],
    }
    manifest["signal_cols"] = (
        manifest["ma_cols"]
        + manifest["stoch_cols"]
        + manifest["lookback_cols"]
        + manifest["bbw_cols"]
    )
    manifest["cache_key"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return manifest


def normalize_signals_times(df_signals: pl.DataFrame, df_context: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    if df_signals is None or df_signals.height == 0:
        return df_signals

    df = df_signals.with_columns(
        pl.col("time").cast(pl.Datetime("ns")).dt.replace_time_zone(None)
    )
    df = df.with_columns(pl.col("time").cast(pl.Int64).alias("time_ns"))

    if df_context is not None and df_context.height > 0:
        main_times = df_context["time_ns"].to_numpy()
        sig_times = df["time_ns"].to_numpy()
        idxs = np.searchsorted(main_times, sig_times, side="left")
        df = df.with_columns(
            pl.Series("idx", idxs).clip(0, df_context.height - 1).cast(pl.Int64)
        )

    return df


def _stoch_state_masks(
    stoch_values: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(stoch_values, dtype=np.float64)
    n = vals.shape[0]
    buy_mask = np.zeros(n, dtype=bool)
    sell_mask = np.zeros(n, dtype=bool)

    if n == 0:
        return buy_mask, sell_mask

    tolerance = float(max(0.0, tolerance))
    buy_reset_level = min(float(upper), float(lower) + tolerance)
    sell_reset_level = max(float(lower), float(upper) - tolerance)

    buy_active = False
    sell_active = False

    for i in range(n):
        v = vals[i]
        if np.isnan(v):
            continue

        if buy_active:
            if v > buy_reset_level:
                buy_active = False
            else:
                buy_mask[i] = True
        else:
            if v <= lower:
                buy_active = True
                buy_mask[i] = True

        if sell_active:
            if v < sell_reset_level:
                sell_active = False
            else:
                sell_mask[i] = True
        else:
            if v >= upper:
                sell_active = True
                sell_mask[i] = True

    return buy_mask, sell_mask


def _combine_masks(masks: list[np.ndarray], mode: str, n: int) -> np.ndarray:
    if not masks:
        return np.zeros(n, dtype=bool)
    out = np.ones(n, dtype=bool) if mode == "all" else np.zeros(n, dtype=bool)
    for m in masks:
        if mode == "all":
            out &= m
        else:
            out |= m
    return out


def _resolve_stoch_bounds(cfg: dict, tf: Optional[str] = None) -> Tuple[float, float]:
    fam = get_family_cfg(cfg, "stochastic")
    tf_cfg = family_tf_cfg(cfg, "stochastic", tf) if tf is not None else fam

    if "thresholds" in tf_cfg:
        return _resolve_threshold_pair(tf_cfg.get("thresholds"), 30.0, 70.0)

    if "stoch_lower" in tf_cfg or "stoch_upper" in tf_cfg:
        return _as_float(tf_cfg.get("stoch_lower", 30.0), 30.0), _as_float(tf_cfg.get("stoch_upper", 70.0), 70.0)

    if "thresholds" in fam:
        return _resolve_threshold_pair(fam.get("thresholds"), 30.0, 70.0)

    return 30.0, 70.0


def _resolve_bbw_threshold(cfg: dict, tf: Optional[str] = None) -> float:
    fam = get_family_cfg(cfg, "bbw")
    tf_cfg = family_tf_cfg(cfg, "bbw", tf) if tf is not None else fam

    if "thresholds" in tf_cfg:
        return _resolve_scalar(tf_cfg.get("thresholds"), 50.0)
    if "threshold" in tf_cfg:
        return _as_float(tf_cfg.get("threshold"), 50.0)

    if "thresholds" in fam:
        return _resolve_scalar(fam.get("thresholds"), 50.0)
    if "threshold" in fam:
        return _as_float(fam.get("threshold"), 50.0)

    return 50.0


def generate_filtered_signals(
    df_slice: pl.DataFrame,
    cfg: dict,
    df_context: Optional[pl.DataFrame] = None,
    return_stats: bool = False,
):
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        empty = pl.DataFrame([], schema=get_schema("signals"))
        if return_stats:
            return empty, {
                "input_rows": 0,
                "final_buy_signals": 0,
                "final_sell_signals": 0,
                "final_total_signals": 0,
                "buy_filtered_out": 0,
                "sell_filtered_out": 0,
            }
        return empty

    manifest = cfg.get("feature_manifest")
    if not isinstance(manifest, dict):
        manifest = build_signal_manifest(cfg)

    df_slice = normalize_signals_times(df_slice, df_context=df_context)
    if "idx" in df_slice.columns:
        df_slice = df_slice.sort(["idx", "time_ns"])
    else:
        df_slice = df_slice.sort(["time_ns"])

    n = df_slice.height
    final_buy = np.ones(n, dtype=bool)
    final_sell = np.ones(n, dtype=bool)

    stats = {
        "input_rows": int(n),
        "ma_enabled": False,
        "stochastic_enabled": False,
        "lookback_enabled": False,
        "bbw_enabled": False,
        "after_ma_buy": int(n),
        "after_ma_sell": int(n),
        "after_stochastic_buy": int(n),
        "after_stochastic_sell": int(n),
        "after_lookback_buy": int(n),
        "after_lookback_sell": int(n),
        "after_bbw_buy": int(n),
        "after_bbw_sell": int(n),
        "final_buy_signals": 0,
        "final_sell_signals": 0,
        "final_total_signals": 0,
        "buy_filtered_out": 0,
        "sell_filtered_out": 0,
    }

    ma_specs = list(manifest.get("ma_specs", []))
    stoch_specs = list(manifest.get("stoch_specs", []))
    lookback_specs = list(manifest.get("lookback_specs", []))
    bbw_specs = list(manifest.get("bbw_specs", []))

    if family_enabled(cfg, "ma", default=False) and ma_specs:
        stats["ma_enabled"] = True
        mode = family_combine_mode(cfg, "ma", default="all")

        close_arr = df_slice["close"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)
        ma_buy_masks = []
        ma_sell_masks = []

        for spec in ma_specs:
            col = str(spec.get("col", "")).strip()
            if not col or col not in df_slice.columns:
                continue

            tf = str(spec.get("tf", "")).strip()
            tf_cfg = family_tf_cfg(cfg, "ma", tf)
            reversion = _as_bool(tf_cfg.get("reversion", False), False)

            ma_arr = df_slice[col].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)

            if not reversion:
                ma_buy_masks.append(close_arr > ma_arr)
                ma_sell_masks.append(close_arr < ma_arr)
            else:
                ma_buy_masks.append(close_arr < ma_arr)
                ma_sell_masks.append(close_arr > ma_arr)

        if ma_buy_masks:
            ma_buy = _combine_masks(ma_buy_masks, mode, n)
            ma_sell = _combine_masks(ma_sell_masks, mode, n)
            final_buy &= ma_buy
            final_sell &= ma_sell
            stats["after_ma_buy"] = int(ma_buy.sum())
            stats["after_ma_sell"] = int(ma_sell.sum())

    if family_enabled(cfg, "stochastic", default=False) and stoch_specs:
        stats["stochastic_enabled"] = True
        mode = family_combine_mode(cfg, "stochastic", default="all")
        stoch_buy_masks = []
        stoch_sell_masks = []

        for spec in stoch_specs:
            col = str(spec.get("col", "")).strip()
            if not col or col not in df_slice.columns:
                continue
            tf = str(spec.get("tf", "")).strip()
            lower, upper = _resolve_stoch_bounds(cfg, tf=tf)
            tf_cfg = family_tf_cfg(cfg, "stochastic", tf)
            tolerance = _as_float(tf_cfg.get("threshold_tolerance", 10.0), 10.0)

            arr = df_slice[col].fill_nan(np.nan).fill_null(np.nan).to_numpy().astype(np.float64, copy=False)
            b, s = _stoch_state_masks(arr, lower, upper, tolerance)
            stoch_buy_masks.append(b)
            stoch_sell_masks.append(s)

        if stoch_buy_masks:
            stoch_buy = _combine_masks(stoch_buy_masks, mode, n)
            stoch_sell = _combine_masks(stoch_sell_masks, mode, n)
            final_buy &= stoch_buy
            final_sell &= stoch_sell
            stats["after_stochastic_buy"] = int(stoch_buy.sum())
            stats["after_stochastic_sell"] = int(stoch_sell.sum())

    if family_enabled(cfg, "lookback", default=False) and lookback_specs:
        stats["lookback_enabled"] = True
        mode = family_combine_mode(cfg, "lookback", default="all")
        lb_buy_masks = []
        lb_sell_masks = []

        for spec in lookback_specs:
            col = str(spec.get("col", "")).strip()
            if not col or col not in df_slice.columns:
                continue
            arr = df_slice[col].fill_nan(np.nan).fill_null(np.nan).to_numpy().astype(np.float64, copy=False)
            lb_buy_masks.append(np.isfinite(arr) & (arr >= 1.0))
            lb_sell_masks.append(np.isfinite(arr) & (arr <= 0.0))

        if lb_buy_masks:
            lb_buy = _combine_masks(lb_buy_masks, mode, n)
            lb_sell = _combine_masks(lb_sell_masks, mode, n)
            final_buy &= lb_buy
            final_sell &= lb_sell
            stats["after_lookback_buy"] = int(lb_buy.sum())
            stats["after_lookback_sell"] = int(lb_sell.sum())

    if family_enabled(cfg, "bbw", default=False) and bbw_specs:
        stats["bbw_enabled"] = True
        mode = family_combine_mode(cfg, "bbw", default="all")
        bbw_masks = []

        for spec in bbw_specs:
            col = str(spec.get("col", "")).strip()
            if not col or col not in df_slice.columns:
                continue
            tf = str(spec.get("tf", "")).strip()
            threshold = _resolve_bbw_threshold(cfg, tf=tf)
            arr = df_slice[col].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)
            bbw_masks.append(arr <= threshold)

        if bbw_masks:
            bbw_mask = _combine_masks(bbw_masks, mode, n)
            final_buy &= bbw_mask
            final_sell &= bbw_mask
            stats["after_bbw_buy"] = int(bbw_mask.sum())
            stats["after_bbw_sell"] = int(bbw_mask.sum())

    regime_id = _as_int(cfg.get("regime_id", 0), 0)

    buy_df = df_slice.filter(pl.Series(final_buy)).select(
        [
            pl.col("idx"),
            pl.col("time_ns"),
            pl.lit(1, dtype=pl.Int8).alias("side"),
            pl.lit(regime_id, dtype=pl.Int32).alias("regime_id"),
        ]
    )

    sell_df = df_slice.filter(pl.Series(final_sell)).select(
        [
            pl.col("idx"),
            pl.col("time_ns"),
            pl.lit(-1, dtype=pl.Int8).alias("side"),
            pl.lit(regime_id, dtype=pl.Int32).alias("regime_id"),
        ]
    )

    out = pl.concat([buy_df, sell_df], how="vertical").sort(["idx", "side"])

    stats["final_buy_signals"] = int(final_buy.sum())
    stats["final_sell_signals"] = int(final_sell.sum())
    stats["final_total_signals"] = int(final_buy.sum() + final_sell.sum())
    stats["buy_filtered_out"] = int(n - final_buy.sum())
    stats["sell_filtered_out"] = int(n - final_sell.sum())

    if return_stats:
        return out, stats
    return out


__all__ = [
    "build_signal_manifest",
    "normalize_signals_times",
    "generate_filtered_signals",
    "get_signal_structure",
    "get_family_cfg",
    "family_enabled",
    "family_combine_mode",
    "family_timeframes",
    "family_tf_cfg",
    "signal_timeframes_from_cfg",
    "timeframe_to_minutes",
    "timeframe_to_polars_every",
    "_as_bool",
    "_as_float",
    "_as_int",
    "_as_int_list",
    "_as_threshold_pairs",
    "_as_list",
    "_as_str",
    "_ordered_unique_ints",
    "_ordered_unique_floats",
    "_ordered_unique_strings",
    "_as_ma_type_list",
]
# feature_helpers.py

"""Signal helpers for nested by_timeframe configs and runtime signal filtering."""
from __future__ import annotations

import calendar
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations, product
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl

from common.casting import (
    _unwrap_singleton,
    _as_bool,
    _as_int,
    _as_float,
    _as_list,
    _as_bool_list,
    _as_int_list,
    _as_float_list,
    _as_threshold_pairs,
    _ordered_unique_ints,
    _ordered_unique_floats,
    _ordered_unique_strings,
    _as_str,
    _positive_ints,
    _as_positive_int_list,
)
from common.schema import get_schema
from common.timeframes import (
    normalize_timeframe,
    normalize_timeframe_list,
    timeframe_bars,
    timeframe_minutes,
)

# -----------------------------------------------------------------------------
# Small coercion helpers
# -----------------------------------------------------------------------------

def _normalize_signal_scope_id(cfg: dict) -> str:
    scope_id = str(cfg.get("signal_scope_id") or "").strip().lower()
    if scope_id in {"all_buy", "all_sell"}:
        return f"baseline:{scope_id}"
    return scope_id

def _fmt_num(v: Any) -> str:
    v = _unwrap_singleton(v)
    if v is None:
        return "na"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else f"{fv:g}"
    return str(v)

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


# -----------------------------------------------------------------------------
# Signal structure access
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Scope helpers
# -----------------------------------------------------------------------------

def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

def _concrete_scope_options(family_name: str, tf_cfg: dict) -> list[dict[str, Any]]:
    """
    Expand one family/timeframe config into concrete single-choice configs.

    This is the key fix:
    - [1, 2] becomes two separate lookback choices
    - [4, 6, 12] x [4, 6, 8] x [4, 6, 8] becomes all concrete stochastic combos
    - each returned item is one concrete slot choice
    """
    if not isinstance(tf_cfg, dict):
        return []

    tf_cfg = deepcopy(tf_cfg)

    if family_name == "ma":
        periods = _as_int_list(tf_cfg.get("periods", [])) or [96]
        types = _ordered_unique_strings(tf_cfg.get("types", [])) or ["sma"]
        reversions = _as_bool_list(tf_cfg.get("reversion", False), False)

        out: list[dict[str, Any]] = []
        for period, ma_type, reversion in product(periods, types, reversions):
            out.append(
                {
                    "periods": [int(period)],
                    "types": [str(ma_type).strip().lower() or "sma"],
                    "reversion": [bool(reversion)],
                }
            )
        return out

    if family_name == "stochastic":
        ks = _as_int_list(tf_cfg.get("k", [])) or [12]
        ds = _as_int_list(tf_cfg.get("d", [])) or [3]
        ss = _as_int_list(tf_cfg.get("s", [])) or [3]
        thresholds = _as_threshold_pairs(tf_cfg.get("thresholds", [[30, 70]])) or [[30, 70]]
        tolerances = _as_float_list(tf_cfg.get("threshold_tolerance", 10.0)) or [10.0]

        out: list[dict[str, Any]] = []
        for k, d, s, th, tol in product(ks, ds, ss, thresholds, tolerances):
            low, high = float(th[0]), float(th[1])
            out.append(
                {
                    "k": [int(k)],
                    "d": [int(d)],
                    "s": [int(s)],
                    "thresholds": [[low, high]],
                    "threshold_tolerance": float(tol),
                }
            )
        return out

    if family_name == "lookback":
        units = _as_int_list(tf_cfg.get("entry_lookback_units", []))
        if not units:
            return []

        out: list[dict[str, Any]] = []
        for u in units:
            if int(u) <= 0:
                continue
            out.append({"entry_lookback_units": [int(u)]})
        return out

    if family_name == "bbw":
        periods = _as_int_list(tf_cfg.get("periods", [])) or [96]
        stds = _as_float_list(tf_cfg.get("std", [])) or [2.5]
        thresholds = _as_float_list(tf_cfg.get("thresholds", [])) or [50.0]

        out: list[dict[str, Any]] = []
        for p, s, t in product(periods, stds, thresholds):
            out.append(
                {
                    "periods": [int(p)],
                    "std": [float(s)],
                    "thresholds": [float(t)],
                }
            )
        return out

    return [deepcopy(tf_cfg)]


def scope_feature_specs(signal_structure: dict) -> list[dict[str, Any]]:
    """
    Flatten one concrete signal_structure into concrete feature specs.

    Each returned spec is a single concrete choice, not a list-valued bundle.
    """
    if not isinstance(signal_structure, dict) or not signal_structure:
        return []

    specs: list[dict[str, Any]] = []

    for family_name in ("ma", "stochastic", "lookback", "bbw"):
        family_cfg = signal_structure.get(family_name)
        if not isinstance(family_cfg, dict):
            continue
        if not _as_bool(family_cfg.get("enabled", False), False):
            continue

        by_tf = family_cfg.get("by_timeframe", {})
        if not isinstance(by_tf, dict) or not by_tf:
            continue

        for tf in sorted(by_tf.keys(), key=lambda x: normalize_timeframe(x)):
            tf_cfg = by_tf.get(tf, {})
            if not isinstance(tf_cfg, dict):
                continue

            tf_key = normalize_timeframe(tf)
            for concrete_cfg in _concrete_scope_options(family_name, tf_cfg):
                specs.append(
                    {
                        "family": family_name,
                        "tf": tf_key,
                        "cfg": deepcopy(concrete_cfg),
                    }
                )

    return specs

def scope_feature_groups(signal_structure: dict) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    Group candidates by (family, timeframe).

    This is the key rule:
    - alternatives within the same slot are OR'd
    - different slots are AND'd
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for spec in scope_feature_specs(signal_structure):
        key = (str(spec["family"]), str(spec["tf"]))
        groups.setdefault(key, []).append(spec)
    return groups


def scope_token(spec: dict[str, Any]) -> str:
    family_name = str(spec.get("family", "")).strip().lower()
    tf_key = normalize_timeframe(spec.get("tf", ""))
    tf_cfg = spec.get("cfg", {}) if isinstance(spec.get("cfg", {}), dict) else {}

    if family_name == "ma":
        p = _unwrap_singleton(tf_cfg.get("periods", 96))
        t = str(_unwrap_singleton(tf_cfg.get("types", "sma"))).strip().lower() or "sma"
        r = bool(_as_bool(_unwrap_singleton(tf_cfg.get("reversion", False)), False))
        return f"ma__{tf_key}__p{_fmt_num(p)}_{t}_rev{int(r)}"

    if family_name == "stochastic":
        k = _unwrap_singleton(tf_cfg.get("k", 12))
        d = _unwrap_singleton(tf_cfg.get("d", 3))
        s = _unwrap_singleton(tf_cfg.get("s", 3))
        thr = _unwrap_singleton(tf_cfg.get("thresholds", [[30, 70]]))
        tol = _unwrap_singleton(tf_cfg.get("threshold_tolerance", 10))

        low, high = 30, 70
        if isinstance(thr, (list, tuple)) and thr:
            first = thr[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                low, high = first[0], first[1]
            elif len(thr) >= 2 and not any(isinstance(i, (list, tuple, dict)) for i in thr[:2]):
                low, high = thr[0], thr[1]

        return f"stoch__{tf_key}__k{_fmt_num(k)}_d{_fmt_num(d)}_s{_fmt_num(s)}_l{_fmt_num(low)}_u{_fmt_num(high)}_tol{_fmt_num(tol)}"

    if family_name == "lookback":
        u = _unwrap_singleton(tf_cfg.get("entry_lookback_units", 1))
        return f"lookback__{tf_key}__u{_fmt_num(u)}"

    if family_name == "bbw":
        p = _unwrap_singleton(tf_cfg.get("periods", 96))
        std = _unwrap_singleton(tf_cfg.get("std", 2.5))
        thr = _unwrap_singleton(tf_cfg.get("thresholds", 50))
        return f"bbw__{tf_key}__p{_fmt_num(p)}_s{_fmt_num(std)}_t{_fmt_num(thr)}"

    return f"{family_name}__{tf_key}"


def subset_signal_structure(signal_structure: dict, selected_specs: tuple[dict[str, Any], ...]) -> dict:
    """
    Build a new signal_structure that keeps only the selected feature tokens.
    """
    out: dict[str, Any] = {}

    for spec in selected_specs:
        family_name = str(spec.get("family", "")).strip().lower()
        tf_key = normalize_timeframe(spec.get("tf", ""))
        tf_cfg = deepcopy(spec.get("cfg", {}))

        if family_name not in out:
            out[family_name] = {"enabled": True, "by_timeframe": {}}

        tf_cfg["timeframe"] = tf_key
        out[family_name]["enabled"] = True
        out[family_name]["by_timeframe"][tf_key] = tf_cfg

    return out


def signal_scope_text(signal_structure: dict) -> str:
    specs = scope_feature_specs(signal_structure)
    return "|".join(scope_token(spec) for spec in specs)


def _ordered_scope_groups(signal_structure: dict) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    """
    Return scope groups in a stable order.

    Each group is one (family, timeframe) slot.
    We never mix different values inside the same slot.
    """
    grouped = scope_feature_groups(signal_structure)
    family_order = {
        "ma": 0,
        "stochastic": 1,
        "lookback": 2,
        "bbw": 3,
    }

    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            family_order.get(str(item[0][0]).strip().lower(), 99),
            normalize_timeframe(item[0][1]),
        ),
    )
    return [(key, specs) for key, specs in ordered if specs]


def signal_scope_variants(signal_structure: dict) -> list[tuple[int, dict, str]]:
    """
    Build every valid scope combination.

    Layer meaning:
    - layer 1 = 1 selected feature slot
    - layer 2 = 2 selected feature slots
    - layer 3 = 3 selected feature slots
    - ...

    Rules:
    - one choice per (family, timeframe) slot
    - same family + same timeframe never mixes different values
    - different families and/or different timeframes may combine

    Baseline is handled outside this function.
    """
    groups = _ordered_scope_groups(signal_structure)
    if not groups:
        return []

    out: list[tuple[int, dict, str]] = []

    for size in range(1, len(groups) + 1):
        layer = size

        for group_idx_combo in combinations(range(len(groups)), size):
            chosen_groups = [groups[i][1] for i in group_idx_combo]

            for selected_specs in product(*chosen_groups):
                subset = subset_signal_structure(signal_structure, tuple(selected_specs))
                scope_id = signal_scope_text(subset)
                if not scope_id:
                    continue
                out.append((layer, subset, scope_id))

    seen: set[str] = set()
    unique: list[tuple[int, dict, str]] = []
    for layer, subset, scope_id in out:
        if scope_id in seen:
            continue
        seen.add(scope_id)
        unique.append((layer, subset, scope_id))

    unique.sort(key=lambda x: (x[0], x[2]))
    return unique


def signal_layer_id(signal_structure: dict) -> int:
    """
    Number of selected slots.

    0 is baseline / no-signal.
    1 is one selected feature slot.
    2 is two selected feature slots.
    """
    return max(len(scope_feature_groups(signal_structure)), 0)


def build_signal_json(regime_cfg: dict, run_cfg: Optional[dict] = None) -> dict:
    """
    Compact nested payload used by master rows and analysis.
    """
    signal_block = get_signal_structure(regime_cfg)
    out: dict[str, Any] = {
        "version": 3,
        "signals": {},
    }

    if not isinstance(signal_block, dict):
        return out

    for signal_name, signal_cfg in signal_block.items():
        if not isinstance(signal_cfg, dict):
            continue

        enabled = _as_bool(_unwrap_singleton(signal_cfg.get("enabled", True)), True)
        if not enabled:
            continue

        tf_map = signal_cfg.get("by_timeframe", {})
        if not isinstance(tf_map, dict):
            continue

        signal_out: dict[str, Any] = {}
        for tf, tf_cfg in tf_map.items():
            if not isinstance(tf_cfg, dict):
                continue
            cfg = deepcopy(tf_cfg)
            cfg["timeframe"] = normalize_timeframe(tf)
            signal_out[cfg["timeframe"]] = cfg

        if signal_out:
            out["signals"][signal_name] = signal_out

    return out

# -----------------------------------------------------------------------------
# Feature manifest and feature precompute helpers
# -----------------------------------------------------------------------------

def build_signal_manifest(run_cfg: dict) -> dict:
    """
    Build the concrete feature manifest used by feature_prep.py and signal filtering.

    Key behavior:
    - Each timeframe is a separate slot.
    - Alternatives within the same (family, timeframe) slot are OR'd at runtime.
    - Different slots are AND'd at runtime.
    """
    signal_block = get_signal_structure(run_cfg)
    if not signal_block:
        raise ValueError("signal_structure is required")

    base_minutes = int(run_cfg.get("BASE_MINUTES", 5) or 5)

    families: dict[str, list[dict]] = {}
    all_timeframes: list[str] = []
    signal_layers: list[dict] = []

    for family_name in ("ma", "stochastic", "lookback", "bbw"):
        if not family_enabled(run_cfg, family_name, default=False):
            continue

        fam = get_family_cfg(run_cfg, family_name)
        if not fam:
            continue

        tfs = family_timeframes(run_cfg, family_name)
        if not tfs:
            continue

        specs: list[dict] = []

        for tf in tfs:
            tf_cfg = family_tf_cfg(run_cfg, family_name, tf)
            tf_key = normalize_timeframe(tf)

            if family_name == "ma":
                periods = _positive_ints(tf_cfg.get("periods", [])) or [8]
                types = _as_ma_type_list(tf_cfg.get("types", None), len(periods))
                reversions = _as_bool_list(tf_cfg.get("reversion", False), False)
                if len(reversions) == 1 and len(periods) > 1:
                    reversions = reversions * len(periods)

                for period, ma_type, reversion in zip(periods, types, reversions):
                    layer_id = f"ma__{tf_key}__p{int(period)}_{str(ma_type).strip().lower() or 'sma'}_rev{int(bool(reversion))}"
                    spec = {
                        "family": "ma",
                        "tf": tf_key,
                        "period": int(period),
                        "type": str(ma_type).strip().lower() or "sma",
                        "reversion": bool(reversion),
                        "col": f"ma_{tf_key}_p{int(period)}_{str(ma_type).strip().lower() or 'sma'}",
                        "layer_id": layer_id,
                        "slot_key": ("ma", tf_key),
                        "enabled": True,
                    }
                    specs.append(spec)
                    signal_layers.append(spec)

            elif family_name == "stochastic":
                ks = _positive_ints(tf_cfg.get("k", [])) or [12]
                ds = _positive_ints(tf_cfg.get("d", [])) or [3]
                ss = _positive_ints(tf_cfg.get("s", [])) or [3]
                thresholds = _as_threshold_pairs(tf_cfg.get("thresholds", [[30, 70]])) or [[30, 70]]
                tolerances = _as_float_list(tf_cfg.get("threshold_tolerance", 10.0)) or [10.0]

                for k in ks:
                    for d in ds:
                        for s in ss:
                            for th in thresholds:
                                for tol in tolerances:
                                    low, high = float(th[0]), float(th[1])
                                    layer_id = f"stoch__{tf_key}__k{int(k)}_d{int(d)}_s{int(s)}_l{_fmt_num(low)}_u{_fmt_num(high)}_tol{_fmt_num(tol)}"
                                    spec = {
                                        "family": "stochastic",
                                        "tf": tf_key,
                                        "k": int(k),
                                        "d": int(d),
                                        "s": int(s),
                                        "thresholds": [[low, high]],
                                        "threshold_tolerance": float(tol),
                                        "col": f"stoch_{tf_key}_k{int(k)}_d{int(d)}_s{int(s)}",
                                        "layer_id": layer_id,
                                        "slot_key": ("stochastic", tf_key),
                                        "enabled": True,
                                    }
                                    specs.append(spec)
                                    signal_layers.append(spec)

            elif family_name == "lookback":
                units_raw = _positive_ints(tf_cfg.get("entry_lookback_units", []))
                if not units_raw:
                    units_raw = [1]

                for units_v in units_raw:
                    units_v = int(units_v)
                    if units_v <= 0:
                        raise ValueError(
                            f"Invalid lookback units for {family_name}/{tf_key}: {units_raw!r}. "
                            "Use positive integers only."
                        )

                    layer_id = f"lookback__{tf_key}__u{units_v}"
                    spec = {
                        "family": "lookback",
                        "tf": tf_key,
                        "units": units_v,
                        "col": f"lookback_{tf_key}_{units_v}u",
                        "layer_id": layer_id,
                        "slot_key": ("lookback", tf_key),
                        "enabled": True,
                    }
                    specs.append(spec)
                    signal_layers.append(spec)

            elif family_name == "bbw":
                periods = _positive_ints(tf_cfg.get("periods", [])) or [96]
                stds = _ordered_unique_floats(tf_cfg.get("std", [])) or [2.5]
                thresholds = _ordered_unique_floats(tf_cfg.get("thresholds", [])) or [50.0]

                for period in periods:
                    for std in stds:
                        for threshold in thresholds:
                            layer_id = f"bbw__{tf_key}__p{int(period)}_s{_fmt_num(std)}_t{_fmt_num(threshold)}"
                            spec = {
                                "family": "bbw",
                                "tf": tf_key,
                                "period": int(period),
                                "std": float(std),
                                "threshold": float(threshold),
                                "col": f"bbw_{tf_key}_p{int(period)}_s{float(std):g}",
                                "layer_id": layer_id,
                                "slot_key": ("bbw", tf_key),
                                "enabled": True,
                            }
                            specs.append(spec)
                            signal_layers.append(spec)

        families[family_name] = specs
        all_timeframes.extend(tfs)

    signal_timeframes = _ordered_unique_strings(all_timeframes)
    active_layers = [x for x in signal_layers if x.get("enabled", True) and x.get("col")]

    # Layer 0 is the no-signal control: one regime that buys every bar and one
    # regime that sells every bar. These are used as the baseline scopes.
    if not active_layers:
        scope_id = _normalize_signal_scope_id(run_cfg)
        baseline_side = "all_sell" if scope_id == "baseline:all_sell" else "all_buy"
        baseline_spec = {
            "family": "baseline",
            "tf": "base",
            "side": baseline_side,
            "col": f"baseline_{baseline_side}",
            "layer_id": baseline_side,
            "slot_key": ("baseline", "base"),
            "enabled": True,
        }
        signal_layers = [baseline_spec]
        active_layers = [baseline_spec]

    manifest = {
        "feature_version": 2,
        "BASE_MINUTES": base_minutes,
        "signal_structure": signal_block,
        "signal_timeframes": signal_timeframes,
        "signal_layers": signal_layers,
        "active_signal_layers": active_layers,
        "ma_specs": families.get("ma", []),
        "stoch_specs": families.get("stochastic", []),
        "lookback_specs": families.get("lookback", []),
        "bbw_specs": families.get("bbw", []),
        "ma_cols": [x["col"] for x in families.get("ma", []) if x.get("col")],
        "stoch_cols": [x["col"] for x in families.get("stochastic", []) if x.get("col")],
        "lookback_cols": [x["col"] for x in families.get("lookback", []) if x.get("col")],
        "bbw_cols": [x["col"] for x in families.get("bbw", []) if x.get("col")],
    }
    manifest["signal_cols"] = (
        manifest["ma_cols"]
        + manifest["stoch_cols"]
        + manifest["lookback_cols"]
        + manifest["bbw_cols"]
    )
    manifest["active_signal_cols"] = [x["col"] for x in active_layers if x.get("col")]
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


def _combine_masks(masks: list[np.ndarray], n: int) -> np.ndarray:
    if not masks:
        return np.zeros(n, dtype=bool)
    out = np.ones(n, dtype=bool)
    for m in masks:
        out &= m
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


def _spec_mask_pair(
    df_slice: pl.DataFrame,
    cfg: dict,
    spec: dict[str, Any],
    close_arr: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    family = str(spec.get("family", "")).strip().lower()
    tf = str(spec.get("tf", "")).strip()
    col = str(spec.get("col", "")).strip()

    if not col or col not in df_slice.columns:
        return None

    if family == "ma":
        if close_arr is None:
            close_arr = df_slice["close"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)

        tf_cfg = family_tf_cfg(cfg, "ma", tf)
        reversion = _as_bool(tf_cfg.get("reversion", False), False)
        ma_arr = df_slice[col].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)

        if not reversion:
            return close_arr > ma_arr, close_arr < ma_arr
        return close_arr < ma_arr, close_arr > ma_arr

    if family == "stochastic":
        lower, upper = _resolve_stoch_bounds(cfg, tf=tf)
        tf_cfg = family_tf_cfg(cfg, "stochastic", tf)
        tolerance = _as_float(tf_cfg.get("threshold_tolerance", 10.0), 10.0)
        arr = df_slice[col].fill_nan(np.nan).fill_null(np.nan).to_numpy().astype(np.float64, copy=False)
        return _stoch_state_masks(arr, lower, upper, tolerance)

    if family == "lookback":
        arr = df_slice[col].fill_nan(np.nan).fill_null(np.nan).to_numpy().astype(np.float64, copy=False)
        buy = np.isfinite(arr) & (arr >= 1.0)
        sell = np.isfinite(arr) & (arr <= 0.0)
        return buy, sell

    if family == "bbw":
        threshold = _resolve_bbw_threshold(cfg, tf=tf)
        arr = df_slice[col].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)
        mask = arr <= threshold
        return mask, mask

    if family == "baseline":
        side = str(spec.get("side", "all_buy")).strip().lower()
        if side == "all_sell":
            return np.zeros(df_slice.height, dtype=bool), np.ones(df_slice.height, dtype=bool)
        return np.ones(df_slice.height, dtype=bool), np.zeros(df_slice.height, dtype=bool)

    return None


def generate_filtered_signals(
    df_slice: pl.DataFrame,
    cfg: dict,
    df_context: Optional[pl.DataFrame] = None,
    return_stats: bool = False,
):
    """
    Build runtime signals from the prepared feature columns.

    Rule:
    - OR within the same (family, timeframe) slot
    - AND across different slots
    """
    if df_slice is None or not isinstance(df_slice, pl.DataFrame) or df_slice.height == 0:
        empty = pl.DataFrame([], schema=get_schema("signals"))
        if return_stats:
            return empty, {
                "input_rows": 0,
                "ma_enabled": False,
                "stochastic_enabled": False,
                "lookback_enabled": False,
                "bbw_enabled": False,
                "final_buy_signals": 0,
                "final_sell_signals": 0,
                "final_total_signals": 0,
                "buy_filtered_out": 0,
                "sell_filtered_out": 0,
            }
        return empty

    scope_id = _normalize_signal_scope_id(cfg)

    df_slice = normalize_signals_times(df_slice, df_context=df_context)
    if "idx" in df_slice.columns:
        df_slice = df_slice.sort(["idx", "time_ns"])
    else:
        df_slice = df_slice.sort(["time_ns"])

    # Baseline control path: no signal filtering at all.
    if scope_id.startswith("baseline:"):
        baseline_side = scope_id.split(":", 1)[1]
        if baseline_side not in {"all_buy", "all_sell"}:
            baseline_side = "all_buy"

        n = df_slice.height
        regime_id = _as_int(cfg.get("regime_id", 0), 0)

        final_buy = np.ones(n, dtype=bool) if baseline_side == "all_buy" else np.zeros(n, dtype=bool)
        final_sell = np.ones(n, dtype=bool) if baseline_side == "all_sell" else np.zeros(n, dtype=bool)

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
            "final_buy_signals": int(final_buy.sum()),
            "final_sell_signals": int(final_sell.sum()),
            "final_total_signals": int(final_buy.sum() + final_sell.sum()),
            "buy_filtered_out": int(n - final_buy.sum()),
            "sell_filtered_out": int(n - final_sell.sum()),
        }

        if return_stats:
            return out, stats
        return out

    manifest = cfg.get("feature_manifest")
    if not isinstance(manifest, dict):
        manifest = build_signal_manifest(cfg)

    all_specs = [x for x in manifest.get("signal_layers", []) if isinstance(x, dict) and x.get("enabled", True)]
    if not all_specs:
        empty = pl.DataFrame([], schema=get_schema("signals"))
        if return_stats:
            return empty, {
                "input_rows": int(df_slice.height),
                "ma_enabled": False,
                "stochastic_enabled": False,
                "lookback_enabled": False,
                "bbw_enabled": False,
                "final_buy_signals": 0,
                "final_sell_signals": 0,
                "final_total_signals": 0,
                "buy_filtered_out": int(df_slice.height),
                "sell_filtered_out": int(df_slice.height),
            }
        return empty

    families_present = {str(s.get("family", "")).strip().lower() for s in all_specs}
    stats = {
        "input_rows": int(df_slice.height),
        "ma_enabled": "ma" in families_present,
        "stochastic_enabled": "stochastic" in families_present,
        "lookback_enabled": "lookback" in families_present,
        "bbw_enabled": "bbw" in families_present,
        "after_ma_buy": int(df_slice.height),
        "after_ma_sell": int(df_slice.height),
        "after_stochastic_buy": int(df_slice.height),
        "after_stochastic_sell": int(df_slice.height),
        "after_lookback_buy": int(df_slice.height),
        "after_lookback_sell": int(df_slice.height),
        "after_bbw_buy": int(df_slice.height),
        "after_bbw_sell": int(df_slice.height),
        "final_buy_signals": 0,
        "final_sell_signals": 0,
        "final_total_signals": 0,
        "buy_filtered_out": 0,
        "sell_filtered_out": 0,
    }

    slot_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for spec in all_specs:
        key = (str(spec.get("family", "")).strip().lower(), normalize_timeframe(spec.get("tf", "")))
        slot_groups.setdefault(key, []).append(spec)

    close_arr = None
    if any(str(s.get("family", "")).strip().lower() == "ma" for s in all_specs):
        if "close" in df_slice.columns:
            close_arr = df_slice["close"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64, copy=False)

    final_buy = np.ones(df_slice.height, dtype=bool)
    final_sell = np.ones(df_slice.height, dtype=bool)

    slot_buy_masks: list[np.ndarray] = []
    slot_sell_masks: list[np.ndarray] = []

    for (_family, _tf), specs in slot_groups.items():
        group_buy = None
        group_sell = None

        for spec in specs:
            pair = _spec_mask_pair(df_slice, cfg, spec, close_arr=close_arr)
            if pair is None:
                continue

            buy_mask, sell_mask = pair
            group_buy = buy_mask if group_buy is None else (group_buy | buy_mask)
            group_sell = sell_mask if group_sell is None else (group_sell | sell_mask)

        if group_buy is None or group_sell is None:
            continue

        slot_buy_masks.append(group_buy)
        slot_sell_masks.append(group_sell)

    if not slot_buy_masks:
        empty = pl.DataFrame([], schema=get_schema("signals"))
        if return_stats:
            return empty, stats
        return empty

    for m in slot_buy_masks:
        final_buy &= m
    for m in slot_sell_masks:
        final_sell &= m

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
    stats["buy_filtered_out"] = int(df_slice.height - final_buy.sum())
    stats["sell_filtered_out"] = int(df_slice.height - final_sell.sum())

    if return_stats:
        return out, stats
    return out


__all__ = [
    "build_signal_manifest",
    "build_signal_json",
    "compact_json",
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
    "scope_feature_specs",
    "scope_feature_groups",
    "scope_token",
    "subset_signal_structure",
    "signal_scope_text",
    "signal_scope_variants",
    "signal_layer_id",
    "_signal_layer_id",
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
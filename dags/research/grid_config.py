# grid_config.py

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.casting import (
    _as_bool,
    _as_bool_list,
    _as_int,
    _as_float,
    _as_list,
    _as_int_list,
    _as_float_list,
    _as_threshold_pairs,
    _ordered_unique_ints,
    _ordered_unique_floats,
    _ordered_unique_strings,
    _ordered_unique_nonneg_ints,
    _as_str,
)

from common.feature_helpers import (
    family_enabled,
    family_tf_cfg,
    family_timeframes,
    get_signal_structure,
    normalize_timeframe,
    normalize_timeframe_list,
)
from research.grid import _expand_sl_tp, _prune_by_min_rr

logger = logging.getLogger(__name__)

def _normalize_signal_structure(run_cfg: dict) -> dict:
    """
    Strict signal-structure loader.

    New grid runs should provide nested signal_structure only.
    Legacy flat config fallback is intentionally disabled so bad configs fail
    early instead of silently changing the feature set.
    """
    signal_block = get_signal_structure(run_cfg)
    if isinstance(signal_block, dict) and signal_block:
        return deepcopy(signal_block)

    flat_legacy_keys = (
        "ma_periods",
        "ma_types",
        "ma_reversion",
        "ma_timeframe",
        "use_stochastic",
        "stoch_k",
        "stoch_d",
        "stoch_s",
        "stoch_thresholds",
        "stoch_threshold_tolerance",
        "stoch_timeframe",
        "entry_lookback_units",
        "lookback_timeframe",
        "use_bbw",
        "bbw_periods",
        "bbw_std",
        "bbw_thresholds",
        "bbw_timeframe",
    )
    if any(k in run_cfg for k in flat_legacy_keys):
        raise ValueError(
            "Legacy flat signal keys detected, but nested signal_structure is missing. "
            "Please migrate the config to signal_structure."
        )

    return {}


def _concrete_tf_options(family_name: str, tf_cfg: dict) -> list[dict]:
    """
    Convert one timeframe config into concrete single-choice configs.

    The output stays compatible with feature_prep.py and live signal logic:
    single selected values are still represented in the same nested structure.
    """
    if not isinstance(tf_cfg, dict):
        return []

    tf_cfg = deepcopy(tf_cfg)

    if family_name == "ma":
        periods = _as_int_list(tf_cfg.get("periods", [])) or [96]
        types = _ordered_unique_strings(tf_cfg.get("types", [])) or ["sma"]
        reversions = _as_bool_list(tf_cfg.get("reversion", False), False)

        out: list[dict] = []
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

        out: list[dict] = []
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
        units = _ordered_unique_nonneg_ints(tf_cfg.get("entry_lookback_units", []))
        if not units:
            units = [0]

        return [{"entry_lookback_units": [int(u)]} for u in units if int(u) >= 0]

    if family_name == "bbw":
        periods = _as_int_list(tf_cfg.get("periods", [])) or [96]
        stds = _as_float_list(tf_cfg.get("std", [])) or [2.5]
        thresholds = _as_float_list(tf_cfg.get("thresholds", [])) or [50.0]

        out: list[dict] = []
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


def _concrete_family_variants(family_name: str, family_cfg: dict) -> list[dict]:
    """
    Expand a family across all of its timeframes.

    Each returned family variant is one concrete, grid-ready selection.
    """
    if not isinstance(family_cfg, dict):
        return []

    enabled = _as_bool(family_cfg.get("enabled", False), False)
    combine = str(family_cfg.get("combine", "all")).strip().lower()
    by_tf = family_cfg.get("by_timeframe", {})
    if not isinstance(by_tf, dict):
        by_tf = {}

    # Disabled family stays as-is: it does not participate in the grid.
    if not enabled:
        return [
            {
                "enabled": False,
                "combine": combine if combine in {"all", "any"} else "all",
                "by_timeframe": deepcopy(by_tf),
            }
        ]

    tfs = family_timeframes({"signal_structure": {family_name: family_cfg}}, family_name, default=list(by_tf.keys()))
    if not tfs:
        return [
            {
                "enabled": True,
                "combine": combine if combine in {"all", "any"} else "all",
                "by_timeframe": deepcopy(by_tf),
            }
        ]

    tf_options_by_tf: list[tuple[str, list[dict]]] = []
    for tf in tfs:
        raw_tf_cfg = by_tf.get(tf, {})
        options = _concrete_tf_options(family_name, raw_tf_cfg)
        if not options:
            options = [deepcopy(raw_tf_cfg)]
        tf_options_by_tf.append((tf, options))

    out: list[dict] = []
    for choice_combo in product(*(opts for _, opts in tf_options_by_tf)):
        concrete_by_tf: dict = {}
        single_tf_only = len(tf_options_by_tf) == 1

        for (tf, _), concrete_cfg in zip(tf_options_by_tf, choice_combo):
            tf_key = normalize_timeframe(tf)
            concrete_by_tf[tf_key] = dict(concrete_cfg)

            # Only keep timeframe field if it actually varies
            if not single_tf_only:
                concrete_by_tf[tf_key]["timeframe"] = tf_key

        out.append(
            {
                "enabled": True,
                "combine": combine if combine in {"all", "any"} else "all",
                "by_timeframe": concrete_by_tf,
            }
        )

    return out


def _explicit_signal_variants(run_cfg: dict) -> list[dict]:
    signal_structure = _normalize_signal_structure(run_cfg)
    if not signal_structure:
        return [{}]

    family_order = ["ma", "stochastic", "lookback", "bbw"]
    variants = [deepcopy(signal_structure)]

    for family_name in family_order:
        family_cfg = signal_structure.get(family_name)
        if not isinstance(family_cfg, dict):
            continue

        family_variants = _concrete_family_variants(family_name, family_cfg)
        next_variants = []

        for base in variants:
            for fv in family_variants:
                merged = deepcopy(base)
                merged[family_name] = deepcopy(fv)
                next_variants.append(merged)

        variants = next_variants or variants

    unique = []
    seen = set()
    for v in variants:
        key = json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)
        if key not in seen:
            seen.add(key)
            unique.append(v)

    return unique


def _top_level_axes(run_cfg: dict) -> Dict[str, list]:
    """
    Grid axes outside signal_structure.

    These are kept explicit so the grid remains easy to inspect and easy to
    prune later.
    """
    axes: Dict[str, list] = {}

    axes["SL"] = _as_float_list(run_cfg.get("SL", run_cfg.get("sl_range", {}).get("min", 0.2))) or [0.2]
    axes["TP"] = _as_float_list(run_cfg.get("TP", run_cfg.get("tp_range", {}).get("max", 6.0))) or [6.0]

    axes["use_trailing_sl"] = _as_bool_list(run_cfg.get("use_trailing_sl", False), False)
    axes["trailing_sl_pct"] = _as_float_list(run_cfg.get("trailing_sl_pct", 0.0)) or [0.0]
    axes["trailing_sl_interval"] = _as_int_list(run_cfg.get("trailing_sl_interval", 0)) or [0]
    axes["trailing_sl_stop_at_pos"] = _as_bool_list(run_cfg.get("trailing_sl_stop_at_pos", True), True)

    axes["use_limit_entry"] = _as_bool_list(run_cfg.get("use_limit_entry", True), True)
    axes["limit_order_expiry_bars"] = _as_int_list(run_cfg.get("limit_order_expiry_bars", 0)) or [0]
    axes["trade_window_interval"] = _as_int_list(run_cfg.get("trade_window_interval", 0)) or [0]

    axes["exit_window_h"] = _as_int_list(run_cfg.get("exit_windows_h", [24])) or [24]

    return axes


def _safe_int_regime_id(value: int) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _candidate_count_from_axes(axis_map: Dict[str, list]) -> int:
    total = 1
    for vals in axis_map.values():
        total *= max(1, len(vals))
    return int(total)


def _write_batch(path: Path, batch_id: int, rows: list[dict]) -> None:
    payload = {
        "batch_id": int(batch_id),
        "search_mode": "grid_search",
        "regimes": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")


def generate_configs(session_dir: Path, run_cfg: dict) -> list[Path]:
    """
    Build grid-search batches.

    Important:
    - No CCD state.
    - No surrogate ranking.
    - No random sampling.
    - Every regime is a concrete configuration.
    """
    session_dir = Path(session_dir)
    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    sl_vals, tp_vals = _expand_sl_tp(run_cfg)
    combos = _prune_by_min_rr(sl_vals, tp_vals, float(run_cfg.get("min_rr", 3.0)))

    signal_variants = _explicit_signal_variants(run_cfg)
    top_axes = _top_level_axes(run_cfg)

    max_regimes = int(run_cfg.get("grid_max_regimes", 0) or 0)
    saved_batch_paths: list[Path] = []
    all_regimes: list[dict] = []

    regime_idx = 0
    for signal_structure in signal_variants:
        signal_cfg = deepcopy(run_cfg)
        signal_cfg["signal_structure"] = deepcopy(signal_structure)

        for sl_val, tp_val in combos:
            for use_trailing_sl in top_axes["use_trailing_sl"]:
                for trailing_sl_pct in top_axes["trailing_sl_pct"]:
                    for trailing_sl_interval in top_axes["trailing_sl_interval"]:
                        for trailing_sl_stop_at_pos in top_axes["trailing_sl_stop_at_pos"]:
                            for use_limit_entry in top_axes["use_limit_entry"]:
                                for limit_order_expiry_bars in top_axes["limit_order_expiry_bars"]:
                                    for trade_window_interval in top_axes["trade_window_interval"]:
                                        for exit_window_h in top_axes["exit_window_h"]:
                                            regime = deepcopy(signal_cfg)

                                            regime["regime_id"] = _safe_int_regime_id(regime_idx)
                                            regime["SL"] = float(sl_val)
                                            regime["TP"] = float(tp_val)

                                            regime["use_trailing_sl"] = bool(use_trailing_sl)
                                            regime["trailing_sl_pct"] = float(trailing_sl_pct)
                                            regime["trailing_sl_interval"] = int(trailing_sl_interval)
                                            regime["trailing_sl_stop_at_pos"] = bool(trailing_sl_stop_at_pos)

                                            regime["use_limit_entry"] = bool(use_limit_entry)
                                            regime["limit_order_expiry_bars"] = int(limit_order_expiry_bars)
                                            regime["trade_window_interval"] = int(trade_window_interval)
                                            regime["exit_window_h"] = int(exit_window_h)

                                            regime["search_mode"] = "grid_search"

                                            all_regimes.append(regime)
                                            regime_idx += 1

                                            if max_regimes > 0 and len(all_regimes) >= max_regimes:
                                                break
                                        if max_regimes > 0 and len(all_regimes) >= max_regimes:
                                            break
                                    if max_regimes > 0 and len(all_regimes) >= max_regimes:
                                        break
                                if max_regimes > 0 and len(all_regimes) >= max_regimes:
                                    break
                            if max_regimes > 0 and len(all_regimes) >= max_regimes:
                                break
                        if max_regimes > 0 and len(all_regimes) >= max_regimes:
                            break
                    if max_regimes > 0 and len(all_regimes) >= max_regimes:
                        break
                if max_regimes > 0 and len(all_regimes) >= max_regimes:
                    break
            if max_regimes > 0 and len(all_regimes) >= max_regimes:
                break
        if max_regimes > 0 and len(all_regimes) >= max_regimes:
            break

    total_regimes = len(all_regimes)
    batch_size = int(run_cfg.get("BATCH_SIZE", 150) or 150)

    logger.info("=" * 60)
    logger.info("🚀 GRID CONFIG GENERATION")
    logger.info("📈 Total Regimes: %d", total_regimes)
    logger.info("📦 Batch Size: %d | Expected Tasks: %d", batch_size, max(1, math.ceil(total_regimes / batch_size)))
    logger.info("🧪 SL/TP combos per regime group: %d", len(combos))
    logger.info("🧩 Signal variants: %d", len(signal_variants))
    logger.info("=" * 60)

    for i in range(0, total_regimes, batch_size):
        batch_slice = all_regimes[i : i + batch_size]
        batch_num = i // batch_size
        batch_filename = cfg_dir / f"batch_{batch_num:04d}.json"
        _write_batch(batch_filename, batch_num, batch_slice)
        saved_batch_paths.append(batch_filename)

        if len(saved_batch_paths) % 20 == 0 or i + batch_size >= total_regimes:
            logger.info("📝 Written %d batch files...", len(saved_batch_paths))

    return saved_batch_paths


def list_pending_config_paths(session_dir: Path) -> List[str]:
    cfg_dir = session_dir / "configs"
    results_dir = session_dir / "results"
    batch_files = sorted(cfg_dir.glob("batch_*.json"))
    pending_batches: list[str] = []

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
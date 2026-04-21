# grid_config.py

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
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
    _ordered_unique_floats,
    _ordered_unique_strings,
)

from common.feature_helpers import (
    build_signal_json,
    get_signal_structure,
    signal_scope_variants,
)
from research.grid import _expand_sl_tp, _prune_by_min_rr

logger = logging.getLogger(__name__)


def _normalize_signal_structure(run_cfg: dict) -> dict:
    """
    Strict signal-structure loader.

    New grid runs must provide nested signal_structure only.
    """
    signal_block = get_signal_structure(run_cfg)
    if isinstance(signal_block, dict) and signal_block:
        return deepcopy(signal_block)

    raise ValueError(
        "signal_structure is required for grid search. "
        "Legacy flat signal keys are no longer supported."
    )

def _explicit_signal_variants(run_cfg: dict) -> list[tuple[int, dict, str]]:
    signal_structure = _normalize_signal_structure(run_cfg)

    variants: list[tuple[int, dict, str]] = []
    seen = set()

    # Real signal variants only.
    for layer, scoped_signal_structure, scope in signal_scope_variants(signal_structure):
        scope_id = str(scope or "").strip().lower()

        # Drop the old empty baseline variants coming from signal_scope_variants().
        if scope_id in {"all_buy", "all_sell"}:
            continue

        key = (
            int(layer),
            scope_id,
            json.dumps(scoped_signal_structure, sort_keys=True, separators=(",", ":"), default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        variants.append((int(layer), deepcopy(scoped_signal_structure), scope_id))

    # Baseline is its own control case, and it keeps the full signal_structure.
    for scope_id in ("baseline:all_buy", "baseline:all_sell"):
        variants.append((0, deepcopy(signal_structure), scope_id))

    variants.sort(key=lambda x: (x[0], x[2]))
    return variants


def _top_level_axes(run_cfg: dict) -> Dict[str, list]:
    """
    Grid axes outside signal_structure.

    These are kept explicit so the grid remains easy to inspect and easy to
    prune later.
    """
    axes: Dict[str, list] = {}

    axes["SL"] = _as_float_list(run_cfg.get("SL", run_cfg.get("sl_range", {}).get("min", 0.2))) or [0.2]
    axes["TP"] = _as_float_list(run_cfg.get("TP", run_cfg.get("tp_range", {}).get("max", 6.0))) or [6.0]

    axes["trade_overlap"] = _as_bool_list(run_cfg.get("trade_overlap", True), True)
    axes["trade_flip_on_entry"] = _as_bool_list(run_cfg.get("trade_flip_on_entry", False), False)
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


def _write_batch(path: Path, batch_id: int, rows: list[dict]) -> None:
    payload = {
        "batch_id": int(batch_id),
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
    for signal_layer, signal_structure, signal_scope_id in signal_variants:
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
                                        for trade_overlap in top_axes["trade_overlap"]:
                                            for trade_flip_on_entry in top_axes["trade_flip_on_entry"]:
                                                for exit_window_h in top_axes["exit_window_h"]:
                                                    regime = deepcopy(signal_cfg)

                                                    regime["regime_id"] = _safe_int_regime_id(regime_idx)
                                                    regime["signal_layer"] = int(signal_layer)
                                                    regime["signal_scope_id"] = str(signal_scope_id)
                                                    regime.pop("signal_scope", None)

                                                    regime["SL"] = float(sl_val)
                                                    regime["TP"] = float(tp_val)

                                                    regime["use_trailing_sl"] = bool(use_trailing_sl)
                                                    regime["trailing_sl_pct"] = float(trailing_sl_pct)
                                                    regime["trailing_sl_interval"] = int(trailing_sl_interval)
                                                    regime["trailing_sl_stop_at_pos"] = bool(trailing_sl_stop_at_pos)

                                                    regime["use_limit_entry"] = bool(use_limit_entry)
                                                    regime["limit_order_expiry_bars"] = int(limit_order_expiry_bars)
                                                    regime["trade_window_interval"] = int(trade_window_interval)
                                                    regime["trade_overlap"] = bool(trade_overlap)
                                                    regime["trade_flip_on_entry"] = bool(trade_flip_on_entry)
                                                    regime["exit_window_h"] = int(exit_window_h)

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
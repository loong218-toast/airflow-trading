# research/grid_row_builders.py

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional

import numpy as np

from common.timeframes import normalize_timeframe
from research.ccd_config import _scalarize


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


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


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


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
        v = _as_float(x, 0.0)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _as_threshold_pairs(value: Any) -> list[list[float]]:
    value = _unwrap_singleton(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        out = []
        for pair in value:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                out.append([_as_float(pair[0]), _as_float(pair[1])])
        return out
    if isinstance(value, (list, tuple)):
        vals = [_as_float(x) for x in value]
        return [vals] if vals else []
    return [[_as_float(value)]]


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _build_signal_json(regime_cfg: dict, run_cfg: Optional[dict] = None) -> dict:
    """
    Compact, canonical snapshot of the nested signal tree.

    This is the only signal signature the master row needs.
    CCD can later rehydrate this back into signal_structure with no indicator-specific
    special case, so adding a new family only needs the family to exist in
    signal_structure.
    """
    signal_block = regime_cfg.get("signal_structure", {})
    out: dict = {
        "version": 2,
        "signals": {},
    }

    if not isinstance(signal_block, dict):
        return out

    for signal_name, signal_cfg in signal_block.items():
        if not isinstance(signal_cfg, dict):
            continue

        enabled = signal_cfg.get("enabled", True)
        enabled = _unwrap_singleton(enabled)
        if isinstance(enabled, (list, tuple)):
            enabled = enabled[0] if enabled else True
        enabled = _as_bool(enabled, True)
        if not enabled:
            continue

        tf_map = signal_cfg.get("by_timeframe", {})
        if not isinstance(tf_map, dict):
            continue

        signal_out: dict = {}
        for tf, tf_cfg in tf_map.items():
            if not isinstance(tf_cfg, dict):
                continue
            cfg = deepcopy(tf_cfg)
            cfg["timeframe"] = normalize_timeframe(tf)
            signal_out[cfg["timeframe"]] = cfg

        if signal_out:
            out["signals"][signal_name] = signal_out

    return out


def _make_empty_master_row(
    regime_id: int,
    era_int: int,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    regime_cfg: dict,
    run_cfg: Optional[dict] = None,
) -> dict:
    """
    Canonical empty master row.

    This keeps schema alignment stable even when a regime has no closed trades.
    """
    signal_json = _build_signal_json(regime_cfg, run_cfg=run_cfg)

    return {
        "regime_id": int(regime_id),
        "era_int": int(era_int),
        "side": int(side_flag),
        "exit_window_h": int(_scalarize(regime_cfg.get("exit_window_h", 0)) or 0),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "SL_hit": None,
        "TP_hit": None,
        "use_trailing_sl": bool(_as_bool(_scalarize(regime_cfg.get("use_trailing_sl", False)), False)),
        "trailing_sl_pct": float(_scalarize(regime_cfg.get("trailing_sl_pct", 0.0)) or 0.0),
        "trailing_sl_interval": int(_scalarize(regime_cfg.get("trailing_sl_interval", 0)) or 0),
        "trailing_sl_stop_at_pos": bool(_as_bool(_scalarize(regime_cfg.get("trailing_sl_stop_at_pos", True)), True)),
        "use_limit_entry": bool(_as_bool(_scalarize(regime_cfg.get("use_limit_entry", True)), True)),
        "limit_order_expiry_bars": int(_scalarize(regime_cfg.get("limit_order_expiry_bars", 0)) or 0),
        "trade_window_interval": int(_scalarize(regime_cfg.get("trade_window_interval", 0)) or 0),
        "total_pos": 0,
        "win_pos": 0,
        "balance": 100.0,
        "max_drawdown": 0.0,
        "max_consecutive_losses": 0,
        "signal_json": _json_compact(signal_json),
    }


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
    max_consecutive_losses: int,
    regime_cfg: dict,
    run_cfg: Optional[dict] = None,
    sl_hit: float = np.nan,
    tp_hit: float = np.nan,
) -> dict:
    """
    Canonical populated master row.

    This is the row that becomes df_master and later feeds:
    - notebook analysis
    - CCD scoring
    - surrogate history
    """
    row = _make_empty_master_row(
        regime_id=regime_id,
        era_int=era_int,
        side_flag=side_flag,
        sl_val=sl_val,
        tp_val=tp_val,
        regime_cfg=regime_cfg,
        run_cfg=run_cfg,
    )

    row.update(
        {
            "total_pos": int(total_pos),
            "win_pos": int(win_pos),
            "balance": float(balance),
            "max_drawdown": float(max_dd),
            "max_consecutive_losses": int(max_consecutive_losses),
            "SL_hit": float(sl_hit) if np.isfinite(sl_hit) else None,
            "TP_hit": float(tp_hit) if np.isfinite(tp_hit) else None,
        }
    )
    return row
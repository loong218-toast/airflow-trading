# casting.py

from __future__ import annotations

from typing import Any

import numpy as np

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

def _as_bool_list(value: Any, default: bool = False) -> list[bool]:
    out: list[bool] = []
    seen: set[bool] = set()

    for x in _as_list(value):
        v = _as_bool(x, default)
        if v not in seen:
            seen.add(v)
            out.append(v)

    # If nothing came out, fall back to default
    if not out:
        out.append(bool(default))

    return out

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
        if v > 0 and v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _as_nonneg_int_list(value: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    for x in _as_list(value):
        v = _as_int(x, 0)
        if v >= 0 and v not in seen:
            seen.add(v)
            out.append(v)

    return out


def _ordered_unique_nonneg_ints(values: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    for x in _as_nonneg_int_list(values):
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out

def _positive_ints(value: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    for x in _as_list(value):
        v = _as_int(x, 0)
        if v > 0 and v not in seen:
            seen.add(v)
            out.append(v)

    return out
    
def _as_str(value: Any, default: str = "") -> str:
    value = _unwrap_singleton(value)
    if value is None:
        return default
    return str(value)

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
        s = str(x).strip()
        if not s:
            continue
        try:
            s = normalize_timeframe(s)
        except Exception:
            pass
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

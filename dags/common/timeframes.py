# timeframes.py

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, List, Optional

import numpy as np


_TIMEFRAME_RE = re.compile(r"^(?P<n>\d+)(?P<u>[mhdwM])$")


@dataclass(frozen=True)
class TimeframeSpec:
    raw: str
    value: int
    unit: str
    approx_minutes: int
    
    @property
    def bars_at_5m(self) -> int:
        return max(1, int(round(self.approx_minutes / 5.0)))

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "value": int(self.value),
            "unit": self.unit,
            "approx_minutes": int(self.approx_minutes),
            "bars_at_5m": int(self.bars_at_5m),
        }


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalize_timeframe(value: Any) -> str:
    """
    Normalize a timeframe token into canonical text like:
    5m, 15m, 1h, 4h, 1D, 3D, 1W, 3W, 1M

    Accepts:
      - "5m", "1h", "1D", ...
      - 5 -> "5m"
      - ["5m"] -> "5m"
    """
    value = _unwrap_singleton(value)

    if value is None:
        raise ValueError("timeframe is None")

    if isinstance(value, (int, np.integer)):
        n = int(value)
        if n <= 0:
            raise ValueError(f"Invalid timeframe integer: {value}")
        return f"{n}m"

    s = str(value).strip()
    if not s:
        raise ValueError("timeframe is empty")

    s = s.replace(" ", "")
    s = s.replace("min", "m").replace("mins", "m").replace("minute", "m").replace("minutes", "m")
    s = s.replace("hr", "h").replace("hrs", "h").replace("hour", "h").replace("hours", "h")
    s = s.replace("day", "D").replace("days", "D")
    s = s.replace("week", "W").replace("weeks", "W")
    s = s.replace("month", "M").replace("months", "M")

    m = _TIMEFRAME_RE.match(s)
    if not m:
        raise ValueError(f"Invalid timeframe format: {value!r}")

    n = int(m.group("n"))
    u = m.group("u")
    if n <= 0:
        raise ValueError(f"Invalid timeframe value: {value!r}")

    return f"{n}{u}"


def parse_timeframe(value: Any) -> TimeframeSpec:
    """
    Parse a timeframe token into a spec.

    Months are treated as an approximate 30-day window for bar-count purposes.
    If you need exact calendar-month resampling, keep the raw label and resample
    separately.
    """
    raw = normalize_timeframe(value)
    m = _TIMEFRAME_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid canonical timeframe: {raw!r}")

    n = int(m.group("n"))
    u = m.group("u")

    if u == "m":
        approx_minutes = n
    elif u == "h":
        approx_minutes = n * 60
    elif u == "D":
        approx_minutes = n * 60 * 24
    elif u == "W":
        approx_minutes = n * 60 * 24 * 7
    elif u == "M":
        approx_minutes = n * 60 * 24 * 30
    else:
        raise ValueError(f"Unsupported timeframe unit: {u!r}")

    return TimeframeSpec(raw=raw, value=n, unit=u, approx_minutes=int(approx_minutes))


def normalize_timeframe_list(values: Any) -> List[str]:
    """
    Return a deduplicated ordered list of canonical timeframe labels.
    """
    if values is None:
        return []

    if isinstance(values, (str, int, np.integer)):
        return [normalize_timeframe(values)]

    if not isinstance(values, Iterable):
        return [normalize_timeframe(values)]

    out: List[str] = []
    seen = set()
    for item in values:
        tf = normalize_timeframe(item)
        if tf not in seen:
            seen.add(tf)
            out.append(tf)
    return out


def timeframe_minutes(value: Any) -> int:
    return parse_timeframe(value).approx_minutes


def timeframe_bars(value: Any, base_minutes: int = 5) -> int:
    base = max(1, int(base_minutes))
    mins = timeframe_minutes(value)
    return max(1, int(round(mins / base)))


def build_timeframe_specs(values: Any, base_minutes: int = 5) -> List[dict]:
    out = []
    for tf in normalize_timeframe_list(values):
        spec = parse_timeframe(tf)
        out.append(
            {
                "raw": spec.raw,
                "value": spec.value,
                "unit": spec.unit,
                "approx_minutes": spec.approx_minutes,
                "bars_at_base_minutes": max(1, int(round(spec.approx_minutes / max(1, int(base_minutes))))),
            }
        )
    return out
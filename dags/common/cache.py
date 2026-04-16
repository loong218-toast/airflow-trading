# research/cache.py
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import logging
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from common.schema import get_schema, enforce_schema

try:
    from common.timeframe import (
        build_timeframe_specs,
        normalize_timeframe_list,
    )
except Exception:  # pragma: no cover - migration fallback
    from common.timeframes import (  # type: ignore
        build_timeframe_specs,
        normalize_timeframe_list,
    )

_LOG = logging.getLogger(__name__)

GLOBAL_SIGNAL_ERA = "__global__"


def load_global_signals_cached(months: int, regime_id: str) -> Optional[pl.DataFrame]:
    return load_cached("signals", months, GLOBAL_SIGNAL_ERA, regime_id)


def stage_global_signals(months: int, regime_id: str, df_new: pl.DataFrame) -> None:
    stage_for_flush("signals", months, GLOBAL_SIGNAL_ERA, regime_id, df_new)


@lru_cache(maxsize=1)
def _cache_settings() -> Dict[str, Any]:
    data_lake_root = os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake")
    default_cache_root = Path(data_lake_root) / "cache"

    def _int_env(k: str, default: int) -> int:
        v = os.getenv(k)
        try:
            return int(v) if v is not None else int(default)
        except Exception:
            return int(default)

    def _bool_env(k: str, default: int) -> bool:
        v = os.getenv(k)
        try:
            return bool(int(v)) if v is not None else bool(int(default))
        except Exception:
            return bool(int(default))

    def _str_env(k: str, default: str) -> str:
        return os.getenv(k, default)

    return {
        "DEFAULT_CACHE_ROOT": default_cache_root,
        "CACHE_FLUSH_ROWS": _int_env("CACHE_FLUSH_ROWS", 50000),
        "CACHE_MAX_INMEM_ROWS": _int_env("CACHE_MAX_INMEM_ROWS", 20000),
        "CACHE_USE_STREAMING_MERGE": _bool_env("CACHE_USE_STREAMING_MERGE", 1),
        "TMP_DIR": Path(_str_env("CACHE_TMP_DIR", str(default_cache_root / "tmp_cache"))),
        "PARTS_FLUSH_THRESHOLD": _int_env("CACHE_PARTS_FLUSH_THRESHOLD", 8),
        "GRID_RESUME_IF_POSSIBLE": _str_env("GRID_RESUME_IF_POSSIBLE", "true").lower() not in ("0", "false", "no"),
    }


def _get_cache_root() -> Path:
    return _cache_settings()["DEFAULT_CACHE_ROOT"]


def _get_tmp_dir() -> Path:
    p = _cache_settings()["TMP_DIR"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_cache_flush_rows() -> int:
    return _cache_settings()["CACHE_FLUSH_ROWS"]


def _get_cache_max_inmem_rows() -> int:
    return _cache_settings()["CACHE_MAX_INMEM_ROWS"]


def _get_cache_use_streaming_merge() -> bool:
    return bool(_cache_settings()["CACHE_USE_STREAMING_MERGE"])


def _get_parts_flush_threshold() -> int:
    return int(_cache_settings()["PARTS_FLUSH_THRESHOLD"])


def _resume_enabled() -> bool:
    return bool(_cache_settings()["GRID_RESUME_IF_POSSIBLE"])


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def signal_signature(signal_payload: Dict[str, Any]) -> str:
    return hashlib.sha1(_stable_json(signal_payload).encode("utf8")).hexdigest()


def _legacy_signal_structure(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a backward-compatible signal structure from flat config fields.

    This keeps old configs working while the nested block format is adopted.
    """
    stoch_lower = cfg.get("stoch_lower", None)
    stoch_upper = cfg.get("stoch_upper", None)
    thresholds = cfg.get("stoch_thresholds", None)

    if (stoch_lower is None or stoch_upper is None) and isinstance(thresholds, (list, tuple)) and len(thresholds) == 2:
        stoch_lower, stoch_upper = thresholds[0], thresholds[1]

    return {
        "ma": {
            "int": int(_unwrap_singleton(cfg.get("ma_int", 0)) or 0),
            "reversion": bool(_unwrap_singleton(cfg.get("ma_reversion", False))),
        },
        "lookback": {
            "units": _unwrap_singleton(cfg.get("entry_lookback_units", [])),
        },
        "stochastic": {
            "enabled": bool(_unwrap_singleton(cfg.get("use_stochastic", cfg.get("enabled", False)))),
            "k": _unwrap_singleton(cfg.get("stoch_k", 12)),
            "d": _unwrap_singleton(cfg.get("stoch_d", 3)),
            "s": _unwrap_singleton(cfg.get("stoch_s", 3)),
            "lower": _unwrap_singleton(stoch_lower if stoch_lower is not None else 30.0),
            "upper": _unwrap_singleton(stoch_upper if stoch_upper is not None else 70.0),
            "key": _unwrap_singleton(cfg.get("stoch_key", "")),
        },
        "bbw": {
            "enabled": bool(_unwrap_singleton(cfg.get("use_bbw", cfg.get("enabled", False)))),
            "periods": _unwrap_singleton(cfg.get("bbw_periods", 0)),
            "std": _unwrap_singleton(cfg.get("bbw_std", 0.0)),
            "thresholds": _unwrap_singleton(cfg.get("bbw_thresholds", 0)),
        },
    }


def build_signal_payload(regime_cfg: Dict[str, Any], base_minutes: int = 5) -> Dict[str, Any]:
    """
    Canonical signal payload.

    The payload is the source of truth for signal construction and can be stored
    as JSON in master rows for later reconstruction or flattening.
    """
    cfg = deepcopy(regime_cfg or {})

    raw_timeframes = cfg.get("signal_timeframes", cfg.get("signal_timeframe", []))
    if raw_timeframes is None:
        raw_timeframes = []
    if not isinstance(raw_timeframes, (list, tuple)):
        raw_timeframes = [raw_timeframes]

    signal_timeframes = normalize_timeframe_list(raw_timeframes)
    if not signal_timeframes:
        signal_timeframes = [f"{int(base_minutes)}m"]

    signal_structure = cfg.get("signal_structure")
    if not isinstance(signal_structure, dict) or not signal_structure:
        signal_structure = _legacy_signal_structure(cfg)

    payload: Dict[str, Any] = {
        "base_minutes": int(base_minutes),
        "signal_timeframes": signal_timeframes,
        "signal_timeframe_primary": signal_timeframes[0],
        "signal_timeframe_count": len(signal_timeframes),
        "signal_timeframe_specs": build_timeframe_specs(signal_timeframes, base_minutes=base_minutes),
        "signal_structure": _jsonable(signal_structure),
    }

    payload["signal_sig"] = signal_signature(payload)
    return payload


def signal_payload_to_json(signal_payload: Dict[str, Any]) -> str:
    return _stable_json(signal_payload)


def flatten_dict(value: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested dict into flat columns.

    Lists are stored as JSON strings so the column count stays stable.
    """
    out: Dict[str, Any] = {}

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
            if isinstance(v, dict):
                out.update(flatten_dict(v, key))
            elif isinstance(v, (list, tuple)):
                out[key] = _stable_json(v)
            else:
                out[key] = v
        return out

    out[prefix or "value"] = value
    return out


def signal_json_to_columns(signal_json: Any, prefix: str = "signal__") -> Dict[str, Any]:
    if signal_json is None:
        return {}

    if isinstance(signal_json, str):
        try:
            payload = json.loads(signal_json)
        except Exception:
            return {}
    elif isinstance(signal_json, dict):
        payload = signal_json
    else:
        return {}

    return flatten_dict(payload, prefix=prefix.rstrip("_"))


def add_signal_columns(df: pl.DataFrame, signal_json_col: str = "signal_json", prefix: str = "signal__") -> pl.DataFrame:
    """
    Expand signal_json into columns on demand.

    Useful for analysis, debugging, or surrogate training without forcing the
    master schema to carry every derived field permanently.
    """
    if df is None or df.is_empty() or signal_json_col not in df.columns:
        return df

    rows = df.to_dicts()
    expanded = []
    for row in rows:
        expanded.append(signal_json_to_columns(row.get(signal_json_col), prefix=prefix))

    flat_df = pl.DataFrame(expanded) if expanded else pl.DataFrame()
    if flat_df.is_empty():
        return df

    return pl.concat([df, flat_df], how="horizontal")


def _era_dir_base(months: int, kind: str, era_label: str) -> Path:
    base = _get_cache_root() / f"{int(months)}mo" / kind / f"era_{era_label}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _target_path(kind: str, months: int, era_label: str, regime_id: str) -> Path:
    base = _era_dir_base(months, kind, era_label)
    if kind == "signals":
        return base / f"config_{regime_id}.parquet"
    return base / f"config_{regime_id}_combos.parquet"


def _worker_part_path(kind: str, months: int, era_label: str, regime_id: str, worker_id: str) -> Path:
    tmp = _get_tmp_dir()
    return tmp / f"{kind}_config_{regime_id}_era_{era_label}_batch_{worker_id}.parquet"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _union_columns_and_cast(dfs: List[pl.DataFrame], kind: str) -> pl.DataFrame:
    if not dfs:
        return enforce_schema(None, kind)
    combined = pl.concat(dfs, how="diagonal")
    return enforce_schema(combined, kind, strict=True)


def _atomic_merge_files(inputs: List[str], out_path: Path) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    if not inputs:
        raise ValueError("No input files provided for merge")

    tmp_out = out_path.with_suffix(".tmp.parquet")
    try:
        pl.scan_parquet(inputs).sink_parquet(str(tmp_out), compression="snappy")
        os.replace(str(tmp_out), str(out_path))
    except Exception as e:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        _LOG.error("Streaming merge failed for %s. Error: %s", out_path, e)
        raise


def load_cached(kind: str, months: int, era_label: str, regime_id: str) -> Optional[pl.DataFrame]:
    if kind not in ("signals", "backtest"):
        raise ValueError(f"Unknown kind: {kind}")

    p = _target_path(kind, months, era_label, regime_id)
    if not p.exists():
        return None

    try:
        file_schema = pl.read_parquet_schema(str(p))
        target_schema = get_schema(kind)

        required_cols = set(target_schema.keys())
        existing_cols = set(file_schema.keys())
        missing_from_file = required_cols - existing_cols

        if missing_from_file:
            _LOG.info(
                "Cache invalidation for %s: missing columns %s. Re-generating.",
                p.name,
                sorted(missing_from_file),
            )
            p.unlink(missing_ok=True)
            return None

        df = pl.read_parquet(str(p))
        df = enforce_schema(df, kind, strict=True)
        return df

    except Exception as exc:
        _LOG.warning("Cache read error for %s: %s", p, exc)
        p.unlink(missing_ok=True)
        return None


def stage_for_flush(kind: str, months: int, era_label: str, regime_id: str, df_new: pl.DataFrame) -> None:
    if df_new is None or df_new.height == 0:
        return

    worker_id = str(os.getenv("AIRFLOW_MAP_INDEX", "0"))
    worker_part = _worker_part_path(kind, months, era_label, regime_id, worker_id)

    df_to_write = enforce_schema(df_new, kind, strict=True)

    tmp_dir = _get_tmp_dir()
    fd, tmp_name = tempfile.mkstemp(prefix=worker_part.name + ".", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        df_to_write.write_parquet(str(tmp_path), compression="snappy")
        os.replace(str(tmp_path), str(worker_part))
        _LOG.debug(
            "Worker %s staged %d rows for kind=%s cfg=%s era=%s -> %s",
            worker_id,
            df_to_write.height,
            kind,
            regime_id,
            era_label,
            worker_part.name,
        )
    except Exception as e:
        _LOG.error(
            "Worker %s failed to stage cache for kind=%s cfg=%s era=%s: %s",
            worker_id,
            kind,
            regime_id,
            era_label,
            e,
        )
        raise
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def flush_all_buffers() -> None:
    _LOG.debug("flush_all_buffers called (no-op in worker-only design).")


def inspect_cache_root() -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    base = _get_cache_root()

    for cat in sorted(base.glob("*mo")):
        cat_name = cat.name
        out[cat_name] = {}
        for kind in ("signals", "backtest"):
            base_k = cat / kind
            if not base_k.exists():
                out[cat_name][kind] = 0
                continue
            count = sum(1 for _ in base_k.rglob("*.parquet"))
            out[cat_name][kind] = count

    try:
        tmp = _get_tmp_dir()
        out["tmp_parts"] = {
            "signals": len(list(tmp.glob("signals_config_*_era_*_batch_*.parquet"))),
            "backtest": len(list(tmp.glob("backtest_config_*_era_*_batch_*.parquet"))),
        }
    except Exception:
        pass

    return out


def load_signals_cached(months: int, era_label: str, regime_id: str) -> Optional[pl.DataFrame]:
    return load_cached("signals", months, era_label, regime_id)


def load_backtest_cached(months: int, era_label: str, regime_id: str) -> Optional[pl.DataFrame]:
    return load_cached("backtest", months, era_label, regime_id)


__all__ = [
    "build_signal_payload",
    "signal_signature",
    "signal_payload_to_json",
    "signal_json_to_columns",
    "add_signal_columns",
    "load_cached",
    "load_signals_cached",
    "load_backtest_cached",
    "load_global_signals_cached",
    "stage_for_flush",
    "stage_global_signals",
    "inspect_cache_root",
]
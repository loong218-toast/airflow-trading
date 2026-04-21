# research/cache.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from common.schema import enforce_schema, get_schema

_LOG = logging.getLogger(__name__)

GLOBAL_SIGNAL_ERA = "__global__"
CACHE_FORMAT_VERSION = 2


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
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
        "CACHE_FLUSH_ROWS": _int_env("CACHE_FLUSH_ROWS", 50_000),
        "CACHE_MAX_INMEM_ROWS": _int_env("CACHE_MAX_INMEM_ROWS", 20_000),
        "CACHE_USE_STREAMING_MERGE": _bool_env("CACHE_USE_STREAMING_MERGE", 1),
        "TMP_DIR": Path(_str_env("CACHE_TMP_DIR", str(default_cache_root / "tmp_cache"))),
        "PARTS_FLUSH_THRESHOLD": _int_env("CACHE_PARTS_FLUSH_THRESHOLD", 8),
    }


def _get_cache_root() -> Path:
    p = _cache_settings()["DEFAULT_CACHE_ROOT"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_tmp_dir() -> Path:
    p = _cache_settings()["TMP_DIR"]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------
def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _safe_token(value: Any, max_len: int = 72) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "all"
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf8")).hexdigest()[:12]
    return f"{text[:max_len]}__{digest}"


def build_scope_cache_id(regime_id: int, signal_layer: int, signal_scope: str) -> str:
    """
    Stable cache id for one concrete scope.

    This is the key that prevents collisions across:
    - single-feature scopes
    - 2-feature scopes
    - 3+ feature scopes
    """
    scope = _safe_token(signal_scope or "all")
    return f"{int(regime_id)}__l{int(signal_layer)}__{scope}"


def signal_signature(signal_payload: Dict[str, Any]) -> str:
    return hashlib.sha1(_stable_json(signal_payload).encode("utf8")).hexdigest()


# ---------------------------------------------------------------------
# File layout
# ---------------------------------------------------------------------
def _era_dir_base(months: int, kind: str, era_label: str) -> Path:
    base = _get_cache_root() / f"{int(months)}mo" / kind / f"era_{_safe_token(era_label)}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _target_path(kind: str, months: int, era_label: str, cache_id: str) -> Path:
    base = _era_dir_base(months, kind, era_label)
    token = _safe_token(cache_id)
    if kind == "signals":
        return base / f"config_{token}.parquet"
    if kind == "backtest":
        return base / f"config_{token}_combos.parquet"
    raise ValueError(f"Unknown kind: {kind}")


def _part_path(kind: str, months: int, era_label: str, cache_id: str, worker_id: str) -> Path:
    tmp = _get_tmp_dir()
    token = _safe_token(cache_id)
    return tmp / (
        f"{kind}_config_{token}_era_{_safe_token(era_label)}"
        f"_batch_{_safe_token(worker_id)}.parquet"
    )


# ---------------------------------------------------------------------
# Read / stage
# ---------------------------------------------------------------------
def load_cached(kind: str, months: int, era_label: str, cache_id: str) -> Optional[pl.DataFrame]:
    if kind not in ("signals", "backtest"):
        raise ValueError(f"Unknown kind: {kind}")

    p = _target_path(kind, months, era_label, cache_id)
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


def stage_for_flush(kind: str, months: int, era_label: str, cache_id: str, df_new: pl.DataFrame) -> None:
    """
    Write one worker-local shard only.

    Final merging happens elsewhere, in a single-worker combine task.
    """
    if df_new is None or df_new.height == 0:
        return

    worker_id = str(os.getenv("AIRFLOW_MAP_INDEX", "0"))
    part = _part_path(kind, months, era_label, cache_id, worker_id)

    df_to_write = enforce_schema(df_new, kind, strict=True)
    tmp_dir = _get_tmp_dir()

    fd, tmp_name = tempfile.mkstemp(prefix=part.name + ".", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        df_to_write.write_parquet(str(tmp_path), compression="snappy")
        os.replace(str(tmp_path), str(part))
        _LOG.debug(
            "Worker %s staged %d rows for kind=%s id=%s era=%s -> %s",
            worker_id,
            df_to_write.height,
            kind,
            cache_id,
            era_label,
            part.name,
        )
    except Exception as e:
        _LOG.error(
            "Worker %s failed to stage cache for kind=%s id=%s era=%s: %s",
            worker_id,
            kind,
            cache_id,
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


def stage_global_signals(months: int, cache_id: str, df_new: pl.DataFrame) -> None:
    stage_for_flush("signals", months, GLOBAL_SIGNAL_ERA, cache_id, df_new)


def load_global_signals_cached(months: int, cache_id: str) -> Optional[pl.DataFrame]:
    return load_cached("signals", months, GLOBAL_SIGNAL_ERA, cache_id)


def load_signals_cached(months: int, era_label: str, cache_id: str) -> Optional[pl.DataFrame]:
    return load_cached("signals", months, era_label, cache_id)


def load_backtest_cached(months: int, era_label: str, cache_id: str) -> Optional[pl.DataFrame]:
    return load_cached("backtest", months, era_label, cache_id)


def stage_backtest_cached(months: int, era_label: str, cache_id: str, df_new: pl.DataFrame) -> None:
    stage_for_flush("backtest", months, era_label, cache_id, df_new)


def flush_all_buffers() -> None:
    """
    Compatibility no-op.

    Final merges are handled by the dedicated combine tasks, not here.
    """
    return


# ---------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------
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
            out[cat_name][kind] = sum(1 for _ in base_k.rglob("*.parquet"))

    try:
        tmp = _get_tmp_dir()
        out["tmp_parts"] = {
            "signals": len(list(tmp.glob("signals_config_*_era_*_batch_*.parquet"))),
            "backtest": len(list(tmp.glob("backtest_config_*_era_*_batch_*.parquet"))),
        }
    except Exception:
        pass

    return out


__all__ = [
    "CACHE_FORMAT_VERSION",
    "GLOBAL_SIGNAL_ERA",
    "build_scope_cache_id",
    "signal_signature",
    "load_cached",
    "load_signals_cached",
    "load_backtest_cached",
    "load_global_signals_cached",
    "stage_for_flush",
    "stage_backtest_cached",
    "stage_global_signals",
    "flush_all_buffers",
    "inspect_cache_root",
]
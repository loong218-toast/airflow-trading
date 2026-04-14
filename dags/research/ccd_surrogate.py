from __future__ import annotations

import json
import os
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from common.schema import enforce_schema, get_schema

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from lightgbm import LGBMRegressor  # optional fallback, not required
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

STATE_FILE_NAME = "coord_descent_state.json"

# The archive lives outside the session directory so future runs can reuse it.
# This is deliberate: the surrogate should remember past backtests across runs,
# while the current session keeps only the control state.
CCD_DATA_LAKE_ROOT = Path(os.getenv("DATA_LAKE_ROOT", "/opt/airflow/airflow-trading/data_lake"))
CCD_EVAL_ARCHIVE_ROOT = CCD_DATA_LAKE_ROOT / "cache" / "ccd_eval_parts"

# Keep the surrogate bounded. This is the main safety guard against "hangs"
# caused by huge historical parquet scans and very large pandas training frames.
SURROGATE_HISTORY_LIMIT = 5_000
SURROGATE_HISTORY_PART_LIMIT = 200
SURROGATE_MIN_TRAIN_ROWS = 32
SURROGATE_BOOTSTRAP_RANDOM = 4

WINNER_METRICS = [
    "recency_weighted_era_consistency_score",
    "era_consistency_score",
    "dominance_score",
    "elite_median_alpha",
]

DEFAULT_WEIGHTS = {
    "recency_weighted_era_consistency_score": 0.35,
    "era_consistency_score": 0.25,
    "dominance_score": 0.30,
    "elite_median_alpha": 0.10,
}

DEFAULT_METRIC_BOUNDS = {
    "recency_weighted_era_consistency_score": (0.0, 1.0),
    "era_consistency_score": (0.0, 1.0),
    "dominance_score": (0.0, 1.0),
    "elite_median_alpha": (-1.0, 10.0),
}


@dataclass(frozen=True)
class ParamSpec:
    path: tuple[str, ...]
    kind: str  # "float", "int", "bool", "choice", "fixed"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _now_iso_ns() -> datetime:
    return datetime.now(timezone.utc)


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


def _path_get(cfg: dict, path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _path_set(cfg: dict, path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _iter_leaf_paths(node: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_leaf_paths(v, prefix + (str(k),))
    else:
        yield prefix, node


def _is_num_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def flatten_cfg(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
        if isinstance(v, dict):
            out.update(flatten_cfg(v, key))
        elif isinstance(v, (list, tuple)):
            if len(v) == 1 and not isinstance(v[0], (list, dict, tuple)):
                out[key] = v[0]
            else:
                out[key] = _stable_json(v)
        else:
            out[key] = v
    return out


def infer_metric_bounds(
    df: Optional[pl.DataFrame],
    metrics: Sequence[str] = WINNER_METRICS,
    fallback: Dict[str, Tuple[float, float]] = DEFAULT_METRIC_BOUNDS,
) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = dict(fallback)

    if df is None or df.is_empty():
        return bounds

    for m in metrics:
        if m not in df.columns:
            continue
        try:
            arr = df[m].drop_nulls().to_numpy().astype(np.float64)
        except Exception:
            continue
        if arr.size == 0:
            continue

        if arr.size >= 8:
            lo = float(np.nanpercentile(arr, 5))
            hi = float(np.nanpercentile(arr, 95))
        else:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))

        if not np.isfinite(lo):
            lo = fallback.get(m, (0.0, 1.0))[0]
        if not np.isfinite(hi):
            hi = fallback.get(m, (0.0, 1.0))[1]
        if hi <= lo:
            hi = lo + 1.0

        bounds[m] = (float(lo), float(hi))

    return bounds


def scalarize_winner_score(
    score: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    metric_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """
    Convert the real CCD winner metrics into one scalar loss.

    Lower is better.
    This scalar is only used for surrogate training and ranking.
    The actual incumbent decision still uses the full backtest-derived metrics.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    metric_bounds = dict(metric_bounds or DEFAULT_METRIC_BOUNDS)

    total = 0.0
    for metric, weight in weights.items():
        raw = score.get(metric, np.nan)
        try:
            val = float(raw)
        except Exception:
            val = np.nan

        if not np.isfinite(val):
            continue

        lo, hi = metric_bounds.get(metric, (0.0, 1.0))
        if hi <= lo:
            continue

        clipped = float(np.clip(val, lo, hi))
        norm = (clipped - lo) / (hi - lo)
        total -= float(weight) * float(norm)

    return float(total)


def _load_pair_from_session(session_dir: Path) -> str:
    """
    The archive path is namespaced by pair so multiple markets can share one cache root.
    """
    session_dir = Path(session_dir)
    run_cfg_path = session_dir / "run_config.json"
    if not run_cfg_path.exists():
        return "unknown_pair"
    try:
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf8"))
        pair = str(run_cfg.get("pair", "unknown_pair")).strip()
        return pair or "unknown_pair"
    except Exception:
        return "unknown_pair"


def _eval_archive_dir(session_dir: Path, batch_id: int) -> Path:
    session_dir = Path(session_dir)
    pair = _load_pair_from_session(session_dir)
    return CCD_EVAL_ARCHIVE_ROOT / pair / session_dir.name / f"batch_{int(batch_id):04d}"


def make_ccd_eval_row(
    *,
    regime_id: int,
    era_int: int,
    side: int,
    block_name: str,
    profile_name: str,
    candidate_sig: str,
    candidate_rank: int,
    accepted: bool,
    selected: bool,
    total_pos: int,
    win_pos: int,
    balance: float,
    max_drawdown: float,
    max_consecutive_losses: int,
    sl_val: float,
    tp_val: float,
    sl_hit: float,
    tp_hit: float,
    regime_cfg: Dict[str, Any],
    winner_score: Dict[str, Any],
    created_at: Optional[datetime] = None,
    weights: Optional[Dict[str, float]] = None,
    metric_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Build one compact CCD evaluation row.

    This is the bridge between backtest truth and surrogate learning:
    - `regime_cfg` stores the exact candidate configuration
    - `winner_score` stores the real backtest-derived ranking metrics
    - `score_loss` is the scalar target used by the surrogate later
    """
    cfg_json = _stable_json(regime_cfg or {})

    return {
        "regime_id": int(regime_id),
        "era_int": int(era_int),
        "side": int(side),
        "block_name": str(block_name),
        "profile_name": str(profile_name),
        "candidate_sig": str(candidate_sig),
        "candidate_rank": int(candidate_rank),
        "accepted": bool(accepted),
        "selected": bool(selected),
        "total_pos": int(total_pos),
        "win_pos": int(win_pos),
        "balance": float(balance),
        "max_drawdown": float(max_drawdown),
        "max_consecutive_losses": int(max_consecutive_losses),
        "SL": float(sl_val),
        "TP": float(tp_val),
        "SL_hit": float(sl_hit) if np.isfinite(sl_hit) else None,
        "TP_hit": float(tp_hit) if np.isfinite(tp_hit) else None,
        "use_trailing_sl": bool(regime_cfg.get("use_trailing_sl", False)),
        "trailing_sl_pct": float(regime_cfg.get("trailing_sl_pct", 0.0) or 0.0),
        "trailing_sl_interval": int(regime_cfg.get("trailing_sl_interval", 0) or 0),
        "trailing_sl_stop_at_pos": bool(regime_cfg.get("trailing_sl_stop_at_pos", True)),
        "use_limit_entry": bool(regime_cfg.get("use_limit_entry", True)),
        "limit_order_expiry_bars": int(regime_cfg.get("limit_order_expiry_bars", 0) or 0),
        "trade_window_interval": int(regime_cfg.get("trade_window_interval", 0) or 0),
        "recency_weighted_era_consistency_score": float(winner_score.get("recency_weighted_era_consistency_score", 0.0)),
        "era_consistency_score": float(winner_score.get("era_consistency_score", 0.0)),
        "dominance_score": float(winner_score.get("dominance_score", 0.0)),
        "elite_median_alpha": float(winner_score.get("elite_median_alpha", 0.0)),
        "score_loss": scalarize_winner_score(
            winner_score,
            weights=weights,
            metric_bounds=metric_bounds,
        ),
        "cfg_json": cfg_json,
        "created_at": created_at or _now_iso_ns(),
    }


def stage_ccd_eval_rows(results_dir: str | Path, batch_id: int, rows: pl.DataFrame) -> Optional[Path]:
    """
    Persist one compact evaluation fragment for later surrogate training.

    Important: this writes into the shared archive root, not just session-local
    results, so future runs can reuse the history even after a new DAG run starts.
    """
    if rows is None or rows.is_empty():
        return None

    results_dir = Path(results_dir)
    session_dir = results_dir.parent  # expected to be session_dir / "results"

    out_dir = _eval_archive_dir(session_dir, batch_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = enforce_schema(rows, "ccd_eval", strict=True)
    worker_id = str(__import__("os").getenv("AIRFLOW_MAP_INDEX", "0"))
    stamp = int(time.time() * 1000)
    out_path = out_dir / f"ccd_eval_w{worker_id}_{stamp}_{uuid.uuid4().hex}.parquet"
    df.write_parquet(str(out_path), compression="snappy")
    return out_path


def load_ccd_eval_history(session_dir: str | Path) -> pl.DataFrame:
    """
    Load history from both:
    - the archive cache, for persistent learning across runs
    - the legacy session-local folder, for backward compatibility

    This function is intentionally bounded so the surrogate does not get stuck
    scanning an ever-growing history corpus.
    """
    session_dir = Path(session_dir)
    pair = _load_pair_from_session(session_dir)

    archive_root = CCD_EVAL_ARCHIVE_ROOT / pair / session_dir.name
    legacy_root = session_dir / "results" / "ccd_eval_parts"

    parts: list[Path] = []

    if archive_root.exists():
        parts.extend(sorted(archive_root.rglob("*.parquet")))

    if legacy_root.exists():
        parts.extend(sorted(legacy_root.rglob("*.parquet")))

    if not parts:
        return pl.DataFrame([], schema=get_schema("ccd_eval"))

    # Keep only the newest fragments. Older history is still useful, but not at
    # the cost of blocking the entire CCD cycle.
    try:
        parts = sorted(parts, key=lambda p: p.stat().st_mtime, reverse=True)[:SURROGATE_HISTORY_PART_LIMIT]
    except Exception:
        parts = parts[:SURROGATE_HISTORY_PART_LIMIT]

    try:
        lf = pl.scan_parquet([str(p) for p in parts])
        df = lf.collect(streaming=True)
    except Exception:
        df = pl.concat([pl.read_parquet(str(p)) for p in parts], how="diagonal")

    return enforce_schema(df, "ccd_eval", strict=False)


def _expand_json_columns(pdf: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    extra_frames = []
    for c in cols:
        if c not in pdf.columns:
            continue

        rows = []
        for raw in pdf[c].tolist():
            try:
                payload = json.loads(raw) if isinstance(raw, str) and raw else (raw if isinstance(raw, dict) else {})
            except Exception:
                payload = {}
            rows.append(flatten_cfg(payload) if isinstance(payload, dict) else {})

        if rows:
            extra_frames.append(pd.DataFrame(rows).add_prefix(f"{c}__"))

    if not extra_frames:
        return pdf

    merged = pd.concat([pdf.reset_index(drop=True)] + [f.reset_index(drop=True) for f in extra_frames], axis=1)
    return _dedupe_columns(merged)


def build_training_frame(
    eval_df: pl.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    metric_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Turn CCD eval fragments into a surrogate-ready pandas frame.

    The surrogate only needs:
    - the flattened candidate config
    - the scalar score_loss target
    Everything else is kept as metadata.
    """
    if eval_df is None or eval_df.is_empty():
        return pd.DataFrame()

    pdf = eval_df.to_pandas()

    for c in pdf.columns:
        if pd.api.types.is_bool_dtype(pdf[c]):
            pdf[c] = pdf[c].astype("int8")
        elif pd.api.types.is_object_dtype(pdf[c]):
            pdf[c] = pdf[c].astype("string")

    json_cols = [c for c in ("cfg_json", "signal_json") if c in pdf.columns]
    if json_cols:
        pdf = _expand_json_columns(pdf, json_cols)

    if metric_bounds is None:
        metric_bounds = infer_metric_bounds(eval_df)

    if "score_loss" not in pdf.columns:
        losses = []
        for _, row in pdf.iterrows():
            winner_score = {m: row.get(m, np.nan) for m in WINNER_METRICS}
            losses.append(
                scalarize_winner_score(
                    winner_score,
                    weights=weights,
                    metric_bounds=metric_bounds,
                )
            )
        pdf["score_loss"] = losses

    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    pdf = pdf.dropna(subset=["score_loss"])
    pdf = _dedupe_columns(pdf)
    return pdf


def _candidate_signature(cfg: Dict[str, Any]) -> str:
    return _stable_json(flatten_cfg(cfg))


def _explicit_bool_values(value: Any) -> List[bool]:
    if isinstance(value, (list, tuple)):
        out: List[bool] = []
        for x in value:
            b = bool(x)
            if b not in out:
                out.append(b)
        return out
    if value is None:
        return []
    if isinstance(value, bool):
        return [value]
    return [bool(value)]


def _normalize_incumbent_for_search(run_cfg: dict, incumbent_cfg: dict) -> dict:
    return deepcopy(incumbent_cfg or {})


def _range_to_values(node: Any) -> list[Any]:
    if not isinstance(node, dict):
        return []

    if "values" in node and isinstance(node["values"], (list, tuple)):
        return list(node["values"])

    lo = node.get("min", node.get("low", None))
    hi = node.get("max", node.get("high", None))
    step = node.get("step", None)

    if lo is None or hi is None:
        return []

    try:
        lo_f = float(lo)
        hi_f = float(hi)
        step_f = float(step) if step is not None else None
    except Exception:
        return []

    if step_f is None or step_f <= 0:
        return [lo_f, hi_f] if lo_f != hi_f else [lo_f]

    vals: list[float] = []
    cur = lo_f
    eps = abs(step_f) * 1e-9 + 1e-12
    while cur <= hi_f + eps:
        vals.append(round(cur, 12))
        cur += step_f
    return vals


def _trade_param_range(run_cfg: dict, key: str) -> list[Any]:
    """
    Map block keys to explicit range values.

    This is what lets you define:
      limit_order_expiry_bars: [3, 36]
    or:
      limit_order_expiry_bars: {min: 3, max: 36, step: 3}
    and have the surrogate treat it as a real search dimension.
    """
    range_map = {
        "SL": "sl_range",
        "TP": "tp_range",
        "limit_order_expiry_bars": "limit_order_expiry_bars",
        "trade_window_interval": "trade_window_interval",
        "exit_window_h": "exit_windows_h",
        "trailing_sl_pct": "trailing_sl_pct",
    }

    src = range_map.get(key)
    if src is None:
        return []

    node = run_cfg.get(src, None)

    if isinstance(node, dict):
        return _range_to_values(node)

    if isinstance(node, (list, tuple)):
        out = []
        for x in node:
            if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
                out.append(x.item() if hasattr(x, "item") else x)
            elif isinstance(x, bool):
                out.append(bool(x))
        return out

    return []


def _value_options_for_path(run_cfg: dict, incumbent_cfg: dict, path: tuple[str, ...]) -> list[Any]:
    run_v = _path_get(run_cfg, path, None)
    inc_v = _path_get(incumbent_cfg, path, None)

    if path and path[-1] == "enabled":
        vals = _explicit_bool_values(run_v if run_v is not None else inc_v)
        return vals if vals else [False, True]

    if path and path[-1] == "combine":
        if isinstance(run_v, (list, tuple)):
            return [str(x).strip().lower() for x in run_v if str(x).strip()]
        if isinstance(inc_v, (list, tuple)):
            return [str(x).strip().lower() for x in inc_v if str(x).strip()]
        return [run_v if run_v is not None else inc_v] if (run_v is not None or inc_v is not None) else []

    if path:
        ranged = _trade_param_range(run_cfg, str(path[-1]))
        if ranged:
            return ranged

    if isinstance(run_v, (list, tuple)):
        return list(run_v)
    if isinstance(inc_v, (list, tuple)):
        return list(inc_v)

    if isinstance(run_v, bool) or isinstance(inc_v, bool):
        return _explicit_bool_values(run_v if run_v is not None else inc_v)

    if _is_num_scalar(run_v) or _is_num_scalar(inc_v):
        return [run_v if run_v is not None else inc_v]

    if run_v is not None:
        return [run_v]
    if inc_v is not None:
        return [inc_v]
    return []


def build_param_specs(run_cfg: dict, incumbent_cfg: dict, active_block: str) -> List[ParamSpec]:
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    blocks = dict(coord_cfg.get("blocks") or {})
    active_keys = list(blocks.get(active_block, []))

    specs: List[ParamSpec] = []

    if active_block == "signal_structure":
        signal_block = _path_get(run_cfg, ("signal_structure",), {})
        if not isinstance(signal_block, dict):
            signal_block = {}

        for family_name, family_node in signal_block.items():
            if family_name not in active_keys and active_keys != ["signal_json"]:
                continue
            if not isinstance(family_node, dict):
                continue

            # Only the leaf settings matter to the surrogate. This keeps the search
            # aligned with how the signal builder actually consumes the config.
            for leaf_path, leaf_value in _iter_leaf_paths(family_node, prefix=("signal_structure", family_name)):
                if leaf_path[-1] == "timeframes":
                    continue

                vals = _value_options_for_path(run_cfg, incumbent_cfg, leaf_path)
                if isinstance(leaf_value, dict) or not vals:
                    continue

                cur = _path_get(incumbent_cfg, leaf_path, None)
                if isinstance(cur, bool) or any(isinstance(v, bool) for v in vals):
                    kind = "choice" if len(vals) > 1 else "fixed"
                    specs.append(ParamSpec(path=leaf_path, kind=kind, choices=vals))
                    continue

                if all(_is_num_scalar(v) for v in vals):
                    kind = "choice" if len(vals) > 1 else "fixed"
                    specs.append(ParamSpec(path=leaf_path, kind=kind, choices=vals))
                    continue

                specs.append(ParamSpec(path=leaf_path, kind="choice", choices=vals))

        return specs

    for key in active_keys:
        path = (key,)
        vals = _value_options_for_path(run_cfg, incumbent_cfg, path)
        if not vals:
            continue

        if all(isinstance(v, bool) for v in vals):
            kind = "choice" if len(vals) > 1 else "fixed"
            specs.append(ParamSpec(path=path, kind=kind, choices=vals))
        elif all(_is_num_scalar(v) for v in vals):
            kind = "choice" if len(vals) > 1 else "fixed"
            specs.append(ParamSpec(path=path, kind=kind, choices=vals))
        else:
            specs.append(ParamSpec(path=path, kind="choice", choices=vals))

    return specs


def _probe_values(spec: ParamSpec, current: Any, search_scale: float = 1.0) -> List[Any]:
    if spec.kind == "fixed":
        return [current]

    if spec.kind == "choice":
        vals = list(spec.choices or [])
        if not vals:
            return [current]
        if current is not None and current in vals:
            ordered = [current] + [v for v in vals if v != current]
        else:
            ordered = vals[:]
        if search_scale <= 0.75 and len(ordered) > 2:
            return ordered[:2]
        if search_scale <= 1.25 and len(ordered) > 4:
            return ordered[:4]
        return ordered

    return [current]


def _sample_value(spec: ParamSpec, rng: np.random.Generator, current: Any, search_scale: float = 1.0) -> Any:
    if spec.kind == "fixed":
        return current

    if spec.kind == "choice":
        vals = list(spec.choices or [])
        if not vals:
            return current
        if current is not None and current in vals and len(vals) > 1:
            idx = vals.index(current)
            step = int(rng.integers(1, min(len(vals), 1 + int(round(search_scale * 2)))))
            return vals[(idx + step) % len(vals)]
        return vals[int(rng.integers(0, len(vals)))]

    return current


def _bootstrap_probe_candidates(
    base: Dict[str, Any],
    specs: List[ParamSpec],
    active_block: str,
    seed: Optional[int] = None,
    n_random: int = SURROGATE_BOOTSTRAP_RANDOM,
) -> List[Dict[str, Any]]:
    """
    Bootstrap phase = very small, controlled diversity before the surrogate has
    enough history to be meaningful.

    This intentionally does not explore too aggressively. It creates just enough
    variation to avoid getting locked to the seed.
    """
    rng = np.random.default_rng(seed if seed is not None else 12345)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_cfg(cfg: Dict[str, Any]) -> None:
        sig = _candidate_signature(cfg)
        if sig in seen:
            return
        seen.add(sig)
        cfg = deepcopy(cfg)
        cfg["_candidate_sig"] = sig
        cfg["_active_block"] = active_block
        out.append(cfg)

    add_cfg(base)

    for spec in specs[: min(len(specs), 6)]:
        vals = list(spec.choices or [])
        for v in vals[:2]:
            cfg = deepcopy(base)
            _path_set(cfg, spec.path, v)
            add_cfg(cfg)

    for _ in range(max(0, int(n_random))):
        cfg = deepcopy(base)
        moved = 0
        for spec in specs:
            if moved >= 2 and rng.random() < 0.75:
                continue
            cur = _path_get(cfg, spec.path, None)
            new_val = _sample_value(spec, rng, cur, search_scale=0.85)
            if new_val != cur:
                _path_set(cfg, spec.path, new_val)
                moved += 1
        add_cfg(cfg)

    return out


def generate_probe_candidates(
    run_cfg: dict,
    incumbent_cfg: dict,
    active_block: str,
    n_candidates: int = 40,
    seed: Optional[int] = None,
    search_scale: Optional[float] = None,
    joint_probe_max_moves: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed if seed is not None else 12345)
    specs = build_param_specs(run_cfg, incumbent_cfg, active_block)

    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    scale = float(
        search_scale
        if search_scale is not None
        else coord_cfg.get("search_scale_start", 1.0)
        or 1.0
    )
    scale = float(max(
        float(coord_cfg.get("search_scale_min", 0.35) or 0.35),
        min(scale, float(coord_cfg.get("search_scale_max", 2.0) or 2.0)),
    ))

    base = _normalize_incumbent_for_search(run_cfg, incumbent_cfg)
    candidates: List[Dict[str, Any]] = []
    seen = set()

    def add_cfg(cfg: Dict[str, Any]) -> None:
        sig = _candidate_signature(cfg)
        if sig in seen:
            return
        seen.add(sig)
        out = deepcopy(cfg)
        out["_candidate_sig"] = sig
        out["_active_block"] = active_block
        out["_search_scale"] = float(scale)
        candidates.append(out)

    add_cfg(base)

    for spec in specs:
        current = _path_get(base, spec.path, None)
        for value in _probe_values(spec, current, search_scale=scale):
            cfg = deepcopy(base)
            _path_set(cfg, spec.path, value)
            add_cfg(cfg)

    max_needed = max(int(n_candidates), 1)
    safety = max_needed * 12
    tries = 0
    while len(candidates) < max_needed and tries < safety:
        tries += 1
        cfg = deepcopy(base)
        moved = 0

        for spec in specs:
            if moved >= 3 and rng.random() < 0.70:
                continue
            cur = _path_get(cfg, spec.path, None)
            new_val = _sample_value(spec, rng, cur, search_scale=scale)
            if new_val != cur:
                _path_set(cfg, spec.path, new_val)
                moved += 1

        add_cfg(cfg)

    return candidates[:max_needed]


def _compact_history_for_block(
    history_df: pl.DataFrame,
    active_block: str,
    limit: int = SURROGATE_HISTORY_LIMIT,
) -> pl.DataFrame:
    if history_df is None or history_df.is_empty():
        return pl.DataFrame([], schema=get_schema("ccd_eval"))

    df = history_df

    if "block_name" in df.columns:
        try:
            df = df.filter(pl.col("block_name") == str(active_block))
        except Exception:
            pass

    if df.is_empty():
        return pl.DataFrame([], schema=get_schema("ccd_eval"))

    sort_cols = [c for c in ("created_at", "era_int", "candidate_rank") if c in df.columns]
    if sort_cols:
        try:
            df = df.sort(sort_cols)
        except Exception:
            pass

    if "candidate_sig" in df.columns:
        try:
            df = df.unique(subset=["candidate_sig"], keep="last")
        except Exception:
            pass

    if df.height > int(limit):
        df = df.tail(int(limit))

    return df


def rank_candidates_with_surrogate(
    history_df: pl.DataFrame,
    candidate_cfgs: List[Dict[str, Any]],
    active_block: str,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rank candidate configs using historical surrogate memory when possible.

    Decision flow:
    1) exact history match first
    2) if enough history exists, train a small surrogate
    3) otherwise keep a diverse, deterministic fallback order
    """
    if not candidate_cfgs:
        return []

    top_k = max(1, int(top_k or 1))
    candidates = [deepcopy(c) for c in candidate_cfgs]

    train_pdf = pd.DataFrame()
    if history_df is not None and not history_df.is_empty():
        hist = history_df
        if "block_name" in hist.columns:
            try:
                hist = hist.filter(pl.col("block_name") == str(active_block))
            except Exception:
                pass
        train_pdf = build_training_frame(hist, weights=weights)

    history_loss_by_sig: dict[str, float] = {}
    if not train_pdf.empty and "candidate_sig" in train_pdf.columns and "score_loss" in train_pdf.columns:
        tmp = train_pdf[["candidate_sig", "score_loss"]].copy()
        tmp = tmp.dropna(subset=["candidate_sig", "score_loss"])
        for sig, grp in tmp.groupby("candidate_sig"):
            try:
                best_loss = float(np.nanmin(grp["score_loss"].astype(float).to_numpy()))
                history_loss_by_sig[str(sig)] = best_loss
            except Exception:
                continue

    def _candidate_row(cfg: Dict[str, Any]) -> Dict[str, Any]:
        row = flatten_cfg(cfg)
        row["candidate_sig"] = _candidate_signature(cfg)
        return row

    cand_rows = [_candidate_row(cfg) for cfg in candidates]
    cand_pdf = pd.DataFrame(cand_rows)

    # Not enough history to train a meaningful model.
    if train_pdf.empty or "score_loss" not in train_pdf.columns or train_pdf.shape[0] < SURROGATE_MIN_TRAIN_ROWS:
        scored = []
        for idx, cfg in enumerate(candidates):
            row = cand_rows[idx]
            sig = str(row.get("candidate_sig", ""))
            loss = history_loss_by_sig.get(sig, float("inf"))
            new_cfg = deepcopy(cfg)
            new_cfg["_surrogate_score_loss"] = float(loss if np.isfinite(loss) else idx)
            new_cfg["_surrogate_score"] = float(-new_cfg["_surrogate_score_loss"])
            scored.append(new_cfg)

        scored.sort(key=lambda x: x.get("_surrogate_score_loss", float("inf")))
        return scored[:top_k]

    # Small model only. RandomForest is stable and works even when features are messy.
    try:
        train_X = train_pdf.drop(columns=["score_loss"], errors="ignore")
        train_y = train_pdf["score_loss"].astype(float)

        drop_cols = [c for c in ("created_at",) if c in train_X.columns]
        if drop_cols:
            train_X = train_X.drop(columns=drop_cols, errors="ignore")

        combined = pd.concat([train_X, cand_pdf], axis=0, ignore_index=True)
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = pd.get_dummies(combined, dummy_na=True)

        X_train = combined.iloc[: len(train_X)].fillna(0.0)
        X_cand = combined.iloc[len(train_X):].fillna(0.0)

        if HAS_SKLEARN:
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            )
            model.fit(X_train, train_y.to_numpy())
            preds = model.predict(X_cand)
        else:
            preds = np.full(len(candidates), float(train_y.median()), dtype=np.float64)

        out: List[Dict[str, Any]] = []
        for cfg, pred, row in zip(candidates, preds, cand_rows):
            sig = str(row.get("candidate_sig", ""))

            new_cfg = deepcopy(cfg)
            new_cfg["_surrogate_pred_loss"] = float(pred)
            new_cfg["_surrogate_exact_loss"] = float(history_loss_by_sig.get(sig, np.nan)) if sig in history_loss_by_sig else None
            out.append(new_cfg)

        out.sort(
            key=lambda x: (
                x.get("_surrogate_exact_loss", float("inf"))
                if x.get("_surrogate_exact_loss", None) is not None
                else float("inf"),
                x.get("_surrogate_pred_loss", float("inf")),
            )
        )
        return out[:top_k]

    except Exception:
        # Hard fallback: exact history first, then deterministic order.
        scored = []
        for idx, cfg in enumerate(candidates):
            row = cand_rows[idx]
            sig = str(row.get("candidate_sig", ""))
            loss = history_loss_by_sig.get(sig, float("inf"))
            new_cfg = deepcopy(cfg)
            new_cfg["_surrogate_score_loss"] = float(loss)
            new_cfg["_surrogate_score"] = float(-loss if np.isfinite(loss) else -1e18)
            scored.append(new_cfg)

        scored.sort(key=lambda x: x.get("_surrogate_score_loss", float("inf")))
        return scored[:top_k]


def suggest_next_candidates(
    session_dir: str | Path,
    run_cfg: dict,
    incumbent_cfg: dict,
    active_block: str,
    n_candidates: int = 40,
    top_k: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Public entrypoint used by CCD config generation.

    This is the only place that should decide whether the run is in bootstrap
    mode or in surrogate-guided mode.
    """
    session_dir = Path(session_dir)
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})

    top_k = max(1, min(int(top_k or 1), int(n_candidates or 1)))
    state = {}
    state_path = session_dir / STATE_FILE_NAME
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf8"))
        except Exception:
            state = {}

    search_scale = float(
        state.get("search_scale", coord_cfg.get("search_scale_start", 1.0) or 1.0)
        or 1.0
    )

    probes = generate_probe_candidates(
        run_cfg=run_cfg,
        incumbent_cfg=incumbent_cfg,
        active_block=active_block,
        n_candidates=n_candidates,
        seed=seed,
        search_scale=search_scale,
        joint_probe_max_moves=int(coord_cfg.get("joint_probe_max_moves", 3) or 3),
    )

    if not probes:
        return []

    history_df = load_ccd_eval_history(session_dir)
    if history_df is None or history_df.is_empty():
        return probes[:top_k]

    compact_hist = _compact_history_for_block(
        history_df=history_df,
        active_block=active_block,
        limit=int(coord_cfg.get("surrogate_history_limit", SURROGATE_HISTORY_LIMIT) or SURROGATE_HISTORY_LIMIT),
    )

    if compact_hist.is_empty():
        return probes[:top_k]

    try:
        train_pdf = build_training_frame(compact_hist, weights=None)
    except Exception:
        train_pdf = pd.DataFrame()

    # Bootstrap phase: when there is not enough usable history, return a small
    # diverse set quickly instead of trying to fit a weak model.
    if train_pdf.empty or train_pdf.shape[0] < SURROGATE_MIN_TRAIN_ROWS:
        scored = []
        for idx, cfg in enumerate(probes):
            new_cfg = deepcopy(cfg)
            new_cfg["_surrogate_score_loss"] = float(idx)
            new_cfg["_surrogate_score"] = float(-idx)
            scored.append(new_cfg)
        return scored[:top_k]

    return rank_candidates_with_surrogate(
        history_df=compact_hist,
        candidate_cfgs=probes,
        active_block=active_block,
        weights=None,
        top_k=top_k,
    )


__all__ = [
    "make_ccd_eval_row",
    "stage_ccd_eval_rows",
    "load_ccd_eval_history",
    "build_training_frame",
    "scalarize_winner_score",
    "suggest_next_candidates",
    "generate_probe_candidates",
    "rank_candidates_with_surrogate",
    "infer_metric_bounds",
    "flatten_cfg",
]
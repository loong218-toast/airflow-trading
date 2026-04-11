from __future__ import annotations

import json
import math
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
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

STATE_FILE_NAME = "coord_descent_state.json"

# These metrics are the CCD "winner score" features.
# They come from real backtest results, not from the surrogate.
# The surrogate learns to predict a loss built from these metrics.
WINNER_METRICS = [
    "recency_weighted_era_consistency_score",
    "era_consistency_score",
    "dominance_score",
    "elite_median_alpha",
]

# Default weights for turning the multi-metric winner score into one scalar loss.
# Lower score_loss is better.
DEFAULT_WEIGHTS = {
    "recency_weighted_era_consistency_score": 0.35,
    "era_consistency_score": 0.25,
    "dominance_score": 0.30,
    "elite_median_alpha": 0.10,
}

# These are only normalization bounds for the scalar loss target.
# They do NOT cap the actual backtest metrics.
DEFAULT_METRIC_BOUNDS = {
    "recency_weighted_era_consistency_score": (0.0, 1.0),
    "era_consistency_score": (0.0, 1.0),
    "dominance_score": (0.0, 1.0),
    "elite_median_alpha": (-1.0, 10.0),
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "float", "int", "bool", "choice", "fixed"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()

def _drop_all_null_cols(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().any()]

def _now_iso_ns() -> datetime:
    return datetime.now(timezone.utc)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


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
        "limit_order_expiry_h": int(regime_cfg.get("limit_order_expiry_h", 0) or 0),
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
    if rows is None or rows.is_empty():
        return None

    results_dir = Path(results_dir)
    out_dir = results_dir / "ccd_eval_parts" / f"batch_{int(batch_id):04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = enforce_schema(rows, "ccd_eval", strict=True)
    worker_id = str(__import__("os").getenv("AIRFLOW_MAP_INDEX", "0"))
    stamp = int(time.time() * 1000)
    out_path = out_dir / f"ccd_eval_w{worker_id}_{stamp}_{uuid.uuid4().hex}.parquet"
    df.write_parquet(str(out_path), compression="snappy")
    return out_path


def load_ccd_eval_history(session_dir: str | Path) -> pl.DataFrame:
    session_dir = Path(session_dir)
    parts_root = session_dir / "results" / "ccd_eval_parts"
    if not parts_root.exists():
        return pl.DataFrame([], schema=get_schema("ccd_eval"))

    parts = sorted(parts_root.rglob("*.parquet"))
    if not parts:
        return pl.DataFrame([], schema=get_schema("ccd_eval"))

    try:
        lf = pl.scan_parquet([str(p) for p in parts])
        df = lf.collect(streaming=True)
    except Exception:
        df = pl.concat([pl.read_parquet(str(p)) for p in parts], how="diagonal")

    return enforce_schema(df, "ccd_eval", strict=False)


def refresh_ccd_eval_snapshot(session_dir: str | Path) -> Path:
    session_dir = Path(session_dir)
    out_path = session_dir / "ccd_eval.parquet"
    df = load_ccd_eval_history(session_dir)
    df.write_parquet(str(out_path), compression="snappy")
    return out_path


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("int8")
        elif pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype("string")
    return df


def build_training_frame(
    eval_df: pl.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    metric_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    if eval_df is None or eval_df.is_empty():
        return pd.DataFrame()

    pdf = eval_df.to_pandas()
    pdf = _normalize_types(pdf)

    if "cfg_json" in pdf.columns:
        cfg_rows = []
        for raw in pdf["cfg_json"].tolist():
            try:
                cfg = json.loads(raw) if isinstance(raw, str) and raw else {}
            except Exception:
                cfg = {}
            cfg_rows.append(flatten_cfg(cfg))

        cfg_df = pd.DataFrame(cfg_rows)
        cfg_df = _dedupe_columns(cfg_df)

        pdf = pd.concat([pdf.reset_index(drop=True), cfg_df.reset_index(drop=True)], axis=1)
        pdf = _dedupe_columns(pdf)

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

def _stoch_options_from_run_cfg(run_cfg: dict, incumbent_cfg: dict) -> List[Dict[str, Any]]:
    ks = run_cfg.get("stoch_k", [12])
    ds = run_cfg.get("stoch_d", [3])
    ss = run_cfg.get("stoch_s", [3])
    ths = run_cfg.get("stoch_thresholds", [[30, 70]])

    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    if not isinstance(ds, (list, tuple)):
        ds = [ds]
    if not isinstance(ss, (list, tuple)):
        ss = [ss]
    if not isinstance(ths, (list, tuple)):
        ths = [ths]

    out = []
    for k in ks:
        for d in ds:
            for s in ss:
                for th in ths:
                    if not isinstance(th, (list, tuple)) or len(th) != 2:
                        continue
                    low = float(th[0])
                    high = float(th[1])
                    out.append({
                        "use_stochastic": True,
                        "stoch_k": int(k),
                        "stoch_d": int(d),
                        "stoch_s": int(s),
                        "stoch_thresholds": [low, high],
                        "stoch_key": f"k{int(k)}_d{int(d)}_s{int(s)}_l{low:g}_u{high:g}",
                        "stoch_col": f"stoch_k{int(k)}_d{int(d)}_s{int(s)}",
                        "stoch_lower": low,
                        "stoch_upper": high,
                    })

    off = {
        "use_stochastic": False,
        "stoch_key": "OFF",
        "stoch_col": None,
        "stoch_lower": None,
        "stoch_upper": None,
    }

    current_key = str(incumbent_cfg.get("stoch_key", "OFF") or "OFF")
    out = sorted(out, key=lambda x: 0 if str(x["stoch_key"]) == current_key else 1)
    return [off] + out

def build_param_specs(run_cfg: dict, incumbent_cfg: dict, active_block: str) -> List[ParamSpec]:
    """
    Convert the active CCD block into a search specification list.

    The spec list defines:
    - which parameters are searchable
    - whether each parameter is fixed, boolean, categorical, or numeric
    - which values are legal inside the current config snapshot
    """
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    blocks = dict(coord_cfg.get("blocks") or {})
    active_keys = list(blocks.get(active_block, []))

    use_stochastic_vals = _explicit_bool_values(run_cfg.get("use_stochastic", incumbent_cfg.get("use_stochastic", False)))
    use_bbw_vals = _explicit_bool_values(run_cfg.get("use_bbw", incumbent_cfg.get("use_bbw", False)))

    stochastic_forced_off = len(use_stochastic_vals) == 1 and use_stochastic_vals[0] is False
    bbw_forced_off = len(use_bbw_vals) == 1 and use_bbw_vals[0] is False

    specs: List[ParamSpec] = []

    for key in active_keys:
        if key == "ma_int":
            specs.append(ParamSpec(name="ma_int", kind="fixed"))
            continue

        if key == "stoch_key":
            if stochastic_forced_off:
                specs.append(ParamSpec(name="stoch_key", kind="fixed"))
            else:
                opts = _stoch_options_from_run_cfg(run_cfg, incumbent_cfg)
                choices = [o["stoch_key"] for o in opts if o["stoch_key"] != "OFF"]
                if "OFF" not in choices:
                    choices = ["OFF"] + choices
                specs.append(ParamSpec(name="stoch_key", kind="choice", choices=choices))
            continue

        if key in {"use_stochastic", "use_bbw", "use_trailing_sl", "trailing_sl_stop_at_pos", "use_limit_entry", "ma_reversion"}:
            raw = run_cfg.get(key, incumbent_cfg.get(key))
            vals = _explicit_bool_values(raw)
            if len(vals) <= 1:
                specs.append(ParamSpec(name=key, kind="fixed"))
            else:
                specs.append(ParamSpec(name=key, kind="choice", choices=vals))
            continue

        if key in {"bbw_periods", "bbw_std", "bbw_thresholds"} and bbw_forced_off:
            specs.append(ParamSpec(name=key, kind="fixed"))
            continue

        if key == "SL":
            r = run_cfg.get("sl_range", {})
            if isinstance(r, dict) and "min" in r and "max" in r:
                specs.append(
                    ParamSpec(
                        name=key,
                        kind="float",
                        low=float(r["min"]),
                        high=float(r["max"]),
                        step=float(r.get("step", 0.0) or 0.0),
                    )
                )
            else:
                specs.append(ParamSpec(name=key, kind="fixed"))
            continue

        if key == "TP":
            r = run_cfg.get("tp_range", {})
            if isinstance(r, dict) and "min" in r and "max" in r:
                specs.append(
                    ParamSpec(
                        name=key,
                        kind="float",
                        low=float(r["min"]),
                        high=float(r["max"]),
                        step=float(r.get("step", 0.0) or 0.0),
                    )
                )
            else:
                specs.append(ParamSpec(name=key, kind="fixed"))
            continue

        raw = run_cfg.get(key, incumbent_cfg.get(key))

        if isinstance(raw, dict) and "min" in raw and "max" in raw:
            is_int = all(isinstance(raw.get(x), int) for x in ("min", "max")) and not any(
                isinstance(raw.get(x), float) for x in ("min", "max")
            )
            specs.append(
                ParamSpec(
                    name=key,
                    kind="int" if is_int else "float",
                    low=float(raw["min"]),
                    high=float(raw["max"]),
                    step=float(raw.get("step", 0.0) or 0.0),
                )
            )
            continue

        if isinstance(raw, (list, tuple)):
            flat = list(raw)
            if len(flat) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in flat):
                kind = "int" if all(isinstance(x, int) for x in flat) else "float"
                specs.append(ParamSpec(name=key, kind=kind, low=float(flat[0]), high=float(flat[1]), step=0.0))
            else:
                specs.append(ParamSpec(name=key, kind="choice", choices=flat))
            continue

        if isinstance(raw, bool):
            specs.append(ParamSpec(name=key, kind="fixed"))
            continue

        if isinstance(raw, (int, float)):
            specs.append(ParamSpec(name=key, kind="fixed"))
            continue

        specs.append(ParamSpec(name=key, kind="fixed"))

    return specs

def _load_coord_state(session_dir: str | Path) -> dict:
    path = Path(session_dir) / STATE_FILE_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_search_scale(run_cfg: dict, state: dict) -> float:
    """
    Search scale is the local/global width of the surrogate probing step.

    Smaller scale:
      - tighter local moves around the incumbent
      - more exploitation

    Larger scale:
      - wider moves around the incumbent
      - more exploration
    """
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    start = float(coord_cfg.get("search_scale_start", 1.0) or 1.0)
    raw = state.get("search_scale", start)
    try:
        scale = float(raw)
    except Exception:
        scale = start
    return float(max(0.10, min(10.0, scale)))


def _explicit_bool_values(value: Any) -> List[bool]:
    """
    Normalize config inputs into an explicit boolean option list.

    Important CCD rule:
    - if the user config only allows false, then the search should NOT invent true
    - if the user config explicitly allows both [false, true], then the search may probe both
    """
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
    """
    Make the incumbent safe for candidate generation.

    This is where we prevent the generator from inventing options that are not
    allowed by the config snapshot.

    Examples:
    - ma_periods empty / [0] => ma_int must stay 0
    - use_stochastic=[false] => never generate true candidates
    - use_bbw=[false] => never generate true candidates
    """
    base = deepcopy(incumbent_cfg or {})

    ma_periods = run_cfg.get("ma_periods", None)
    has_ma = False
    if isinstance(ma_periods, (list, tuple)):
        for x in ma_periods:
            try:
                if float(x) > 0:
                    has_ma = True
                    break
            except Exception:
                continue
    elif ma_periods is not None:
        try:
            has_ma = float(ma_periods) > 0
        except Exception:
            has_ma = False

    if not has_ma:
        base["ma_int"] = 0
        base["ma_reversion"] = False

    use_stoch_vals = _explicit_bool_values(run_cfg.get("use_stochastic", base.get("use_stochastic", False)))
    if len(use_stoch_vals) == 1 and use_stoch_vals[0] is False:
        base["use_stochastic"] = False
        base["stoch_key"] = "OFF"
        base["stoch_col"] = None
        base["stoch_lower"] = None
        base["stoch_upper"] = None

    use_bbw_vals = _explicit_bool_values(run_cfg.get("use_bbw", base.get("use_bbw", False)))
    if len(use_bbw_vals) == 1 and use_bbw_vals[0] is False:
        base["use_bbw"] = False
        base["bbw_periods"] = 0
        base["bbw_std"] = 0.0
        base["bbw_thresholds"] = 0

    return base


def _probe_values(spec: ParamSpec, current: Any, search_scale: float = 1.0) -> List[Any]:
    """
    Produce a compact ordered probe set for one parameter.

    Probes are created before any new backtest is run.
    The surrogate only ranks these probes; it does not produce the true score.
    """
    scale = float(max(0.10, min(float(search_scale), 10.0)))

    if spec.kind == "fixed":
        return [current]

    if spec.kind == "bool":
        vals = [bool(current)] if current is not None else []
        vals.extend([False, True])
        out = []
        seen = set()
        for v in vals:
            b = bool(v)
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

    if spec.kind == "choice":
        vals = list(spec.choices or [])
        if not vals:
            return [current]

        if current is not None and current in vals:
            ordered = [current] + [v for v in vals if v != current]
        else:
            ordered = vals[:]

        if scale <= 0.75 and len(ordered) > 2:
            return ordered[:2]
        if scale <= 1.25 and len(ordered) > 4:
            return ordered[:4]
        return ordered

    if spec.kind == "int" and spec.low is not None and spec.high is not None:
        lo = int(math.floor(spec.low))
        hi = int(math.ceil(spec.high))
        if hi < lo:
            return [current if current is not None else lo]

        vals = list(range(lo, hi + 1))
        if current is not None and current in vals:
            vals.remove(current)
            return [current] + vals
        return vals

    if spec.kind == "float" and spec.low is not None and spec.high is not None:
        lo = float(spec.low)
        hi = float(spec.high)
        if hi <= lo:
            return [current if current is not None else lo]

        span = hi - lo
        mid = (lo + hi) / 2.0
        local = 0.12 * span * scale

        raw_vals = [
            current,
            mid,
            lo,
            hi,
            mid - local,
            mid + local,
        ]

        out = []
        seen = set()
        for v in raw_vals:
            v = float(np.clip(v, lo, hi))
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return [current]


def _sample_value(spec: ParamSpec, rng: np.random.Generator, current: Any, search_scale: float = 1.0) -> Any:
    """
    Random local sampling around the incumbent.

    The returned value is still only a candidate. The real evaluation comes from
    the backtest stage.
    """
    scale = float(max(0.10, min(float(search_scale), 10.0)))

    if spec.kind == "fixed":
        return current

    if spec.kind == "bool":
        return bool(rng.integers(0, 2))

    if spec.kind == "choice":
        vals = list(spec.choices or [])
        if not vals:
            return current
        if current is not None and current in vals and len(vals) > 1:
            idx = vals.index(current)
            step = int(rng.integers(1, min(len(vals), 1 + int(round(scale * 2)))))
            return vals[(idx + step) % len(vals)]
        return vals[int(rng.integers(0, len(vals)))]

    if spec.kind == "int" and spec.low is not None and spec.high is not None:
        lo = int(math.floor(spec.low))
        hi = int(math.ceil(spec.high))
        if hi < lo:
            return int(current) if current is not None else lo

        if current is None:
            return int(rng.integers(lo, hi + 1))

        cur = int(current)
        span = max(1, hi - lo)
        sigma = max(1.0, span * 0.15 * scale)
        v = int(round(rng.normal(cur, sigma)))
        return int(np.clip(v, lo, hi))

    if spec.kind == "float" and spec.low is not None and spec.high is not None:
        lo = float(spec.low)
        hi = float(spec.high)
        if hi <= lo:
            return float(current) if current is not None else lo

        if current is None:
            if spec.log and lo > 0 and hi > 0:
                return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            return float(rng.uniform(lo, hi))

        cur = float(current)
        span = hi - lo
        sigma = max(1e-9, span * 0.15 * scale)
        v = float(rng.normal(cur, sigma))
        return float(np.clip(v, lo, hi))

    return current


def _joint_probe_candidates(
    specs: List[ParamSpec],
    base: dict,
    search_scale: float,
    n_candidates: int,
    rng: np.random.Generator,
    max_joint_moves: int = 3,
) -> List[dict]:
    """
    Generate coordinated probes that move multiple parameters together.

    This is the piece that makes the search more than a plain one-at-a-time grid.
    It tries to discover simple "slope-like" directions:
    for example, if changing SL together with trailing settings looks promising,
    we want to test that as a combined move instead of only isolated moves.
    """
    varying_specs = [s for s in specs if s.kind != "fixed"]
    if len(varying_specs) < 2:
        return []

    out: List[dict] = []
    seen = set()

    def add_cfg(cfg: dict) -> None:
        sig = _candidate_signature(cfg)
        if sig in seen:
            return
        seen.add(sig)
        cfg = deepcopy(cfg)
        cfg["_candidate_sig"] = sig
        out.append(cfg)

    max_joint_moves = max(2, int(max_joint_moves))
    joint_budget = max(12, int(n_candidates) // 2)

    for _ in range(joint_budget):
        cfg = deepcopy(base)

        k = int(rng.integers(2, min(max_joint_moves, len(varying_specs)) + 1))
        chosen_idx = rng.choice(len(varying_specs), size=k, replace=False)

        for idx in np.atleast_1d(chosen_idx):
            spec = varying_specs[int(idx)]
            current = cfg.get(spec.name)
            vals = _probe_values(spec, current, search_scale)
            if not vals:
                continue

            if spec.kind in {"float", "int"}:
                near = [v for v in vals if v != current]
                if near:
                    cfg[spec.name] = near[int(rng.integers(0, len(near)))]
                else:
                    cfg[spec.name] = vals[int(rng.integers(0, len(vals)))]
            else:
                cfg[spec.name] = vals[int(rng.integers(0, len(vals)))]

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
    """
    Build the candidate pool for the current active block.

    Important mental model:
    - This happens before backtest.
    - It does NOT know the real balance/drawdown/win rate yet.
    - It only builds a search set around the incumbent.
    - The surrogate ranks these candidates based on history.
    - The backtest returns the real score_loss used to update CCD state.

    n_candidates is per active block call.
    So yes, if the active block is trade_management, these are trade_management candidates.
    If the active block is execution, the candidate pool is for execution.
    """
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

    max_moves = int(
        joint_probe_max_moves
        if joint_probe_max_moves is not None
        else coord_cfg.get("joint_probe_max_moves", 3)
        or 3
    )
    max_moves = max(1, min(max_moves, max(1, len(specs))))

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

    # Always include the incumbent first.
    # This gives the surrogate and the backtest a stable baseline.
    add_cfg(base)

    # One-parameter probes: isolate the effect of each parameter.
    for spec in specs:
        current = base.get(spec.name)
        for value in _probe_values(spec, current, search_scale=scale):
            cfg = deepcopy(base)
            cfg[spec.name] = value
            add_cfg(cfg)

    # Joint probes: move multiple parameters together.
    # This is useful when the useful direction is not axis-aligned.
    movable_specs = [s for s in specs if s.kind != "fixed"]
    if movable_specs:
        priority = {"float": 0, "int": 1, "choice": 2, "bool": 3}
        movable_specs = sorted(movable_specs, key=lambda s: (priority.get(s.kind, 9), s.name))
        joint_specs = movable_specs[:max_moves]

        def directional_value(spec: ParamSpec, current: Any, direction: int) -> Any:
            if spec.kind in {"float", "int"} and spec.low is not None and spec.high is not None:
                lo = float(spec.low)
                hi = float(spec.high)
                span = max(hi - lo, 1e-9)
                try:
                    cur = float(current)
                except Exception:
                    cur = (lo + hi) / 2.0

                frac = 0.18 * scale
                step = span * frac
                if direction < 0:
                    v = cur - step
                else:
                    v = cur + step

                v = float(np.clip(v, lo, hi))
                if spec.kind == "int":
                    v = int(round(v))
                return v

            if spec.kind == "choice":
                vals = list(spec.choices or [])
                if not vals:
                    return current
                if current in vals:
                    idx = vals.index(current)
                else:
                    idx = 0
                if direction < 0:
                    return vals[max(0, idx - 1)]
                return vals[min(len(vals) - 1, idx + 1)]

            if spec.kind == "bool":
                return not bool(current)

            return current

        for move_count in range(2, max_moves + 1):
            chosen_specs = joint_specs[:move_count]
            if len(chosen_specs) < 2:
                break

            cfg_plus = deepcopy(base)
            cfg_minus = deepcopy(base)

            for spec in chosen_specs:
                cur = base.get(spec.name)
                v_plus = directional_value(spec, cur, +1)
                v_minus = directional_value(spec, cur, -1)

                if v_plus != cur:
                    cfg_plus[spec.name] = v_plus
                if v_minus != cur:
                    cfg_minus[spec.name] = v_minus

            add_cfg(cfg_plus)
            add_cfg(cfg_minus)

    # Random local joint probes until the pool is big enough.
    max_needed = max(int(n_candidates), 1)
    safety = max_needed * 12
    tries = 0

    while len(candidates) < max_needed and tries < safety:
        tries += 1
        cfg = deepcopy(base)
        moved = 0

        for spec in specs:
            if spec.kind == "fixed":
                continue

            if moved >= max_moves and rng.random() < 0.70:
                continue

            cur = cfg.get(spec.name)
            new_val = _sample_value(spec, rng, cur, search_scale=scale)
            if new_val != cur:
                cfg[spec.name] = new_val
                moved += 1

        add_cfg(cfg)

    return candidates[:max_needed]

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
    Full local-search pipeline:
    1) load prior CCD evaluation history
    2) build a block-specific candidate pool
    3) train surrogate on real historical backtest outcomes
    4) rank the new candidates by predicted score_loss
    5) return only the top_k to backtest

    - surrogate = candidate filter / ranker
    - backtest = truth source
    """
    session_dir = Path(session_dir)
    history_df = load_ccd_eval_history(session_dir)

    state = load_coord_descent_state(session_dir)
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    search_scale = float(
        state.get("search_scale", coord_cfg.get("search_scale_start", 1.0) or 1.0)
        or 1.0
    )
    joint_probe_max_moves = int(coord_cfg.get("joint_probe_max_moves", 3) or 3)

    probes = generate_probe_candidates(
        run_cfg=run_cfg,
        incumbent_cfg=incumbent_cfg,
        active_block=active_block,
        n_candidates=n_candidates,
        seed=seed,
        search_scale=search_scale,
        joint_probe_max_moves=joint_probe_max_moves,
    )

    return rank_candidates_with_surrogate(
        history_df=history_df,
        candidate_cfgs=probes,
        active_block=active_block,
        weights=None,
        top_k=top_k,
    )


def _candidate_frame(candidate_cfgs: Sequence[Dict[str, Any]], active_block: str) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(candidate_cfgs):
        flat = flatten_cfg(cfg)
        row = dict(flat)
        row["_candidate_idx"] = int(i)
        row["_active_block"] = active_block
        row["_candidate_sig"] = cfg.get("_candidate_sig", _candidate_signature(cfg))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _normalize_types(df)
    return df.replace([np.inf, -np.inf], np.nan)


def fit_auto_surrogate(
    history_df: pl.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    train_df = build_training_frame(history_df, weights=weights)
    if train_df.empty or len(train_df) < 12:
        return None

    target_col = "score_loss"
    feature_cols = [c for c in train_df.columns if c != target_col]

    numeric_cols = []
    categorical_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]) or pd.api.types.is_bool_dtype(train_df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    X = train_df[feature_cols].copy()
    numeric_cols = _drop_all_null_cols(X, numeric_cols)
    categorical_cols = _drop_all_null_cols(X, categorical_cols)

    y = train_df[target_col].astype(np.float64).to_numpy()

    if not np.isfinite(y).all():
        mask = np.isfinite(y)
        X = X.loc[mask].copy()
        y = y[mask]

    if y.size < 12 or np.nanstd(y) < 1e-12:
        return None

    n_rows = int(X.shape[0])
    n_features = int(X.shape[1])
    has_cat = len(categorical_cols) > 0

    if HAS_SKLEARN and n_rows <= 250 and n_features <= 12 and not has_cat:
        preprocess = ColumnTransformer(
            [
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), numeric_cols),
            ],
            remainder="drop",
        )
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
            normalize_y=True,
            random_state=42,
            n_restarts_optimizer=2,
        )
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X[numeric_cols], y)

        return {
            "kind": "gp",
            "model": pipe,
            "feature_cols": numeric_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": [],
        }

    if HAS_LGBM:
        X_lgbm = X.copy()
        X_lgbm = _dedupe_columns(X_lgbm)

        feature_cols = [c for c in feature_cols if c in X_lgbm.columns]
        numeric_cols = [c for c in numeric_cols if c in X_lgbm.columns]
        categorical_cols = [c for c in categorical_cols if c in X_lgbm.columns]

        for c in categorical_cols:
            X_lgbm[c] = X_lgbm[c].astype("string").fillna("__NA__").astype("category")
        for c in numeric_cols:
            X_lgbm[c] = pd.to_numeric(X_lgbm[c], errors="coerce")

        if categorical_cols:
            for c in categorical_cols:
                if str(X_lgbm[c].dtype) != "category":
                    X_lgbm[c] = X_lgbm[c].astype("category")

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
        )
        model.fit(
            X_lgbm[feature_cols],
            y,
            categorical_feature=categorical_cols if categorical_cols else "auto",
        )
        return {
            "kind": "lgbm",
            "model": model,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

    if HAS_SKLEARN:
        preprocess = ColumnTransformer(
            [
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]), numeric_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", _one_hot_encoder()),
                ]), categorical_cols),
            ],
            remainder="drop",
        )
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2,
        )
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X[feature_cols], y)
        return {
            "kind": "rf",
            "model": pipe,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

    return None


def _predict_bundle(bundle: Dict[str, Any], cand_df: pd.DataFrame) -> np.ndarray:
    if bundle is None or cand_df.empty:
        return np.zeros((0,), dtype=np.float64)

    feature_cols = list(bundle["feature_cols"])
    kind = str(bundle["kind"])
    model = bundle["model"]

    X = cand_df.copy()
    X = _dedupe_columns(X)

    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_cols].copy()
    X = _normalize_types(X)

    if kind == "gp":
        pred = model.predict(X[bundle["numeric_cols"]])
        return np.asarray(pred, dtype=np.float64)

    if kind == "lgbm":
        for c in bundle["categorical_cols"]:
            if c in X.columns:
                X[c] = X[c].astype("string").fillna("__NA__").astype("category")
        pred = model.predict(X[feature_cols])
        return np.asarray(pred, dtype=np.float64)

    pred = model.predict(X[feature_cols])
    return np.asarray(pred, dtype=np.float64)


def rank_candidates_with_surrogate(
    history_df: pl.DataFrame,
    candidate_cfgs: Sequence[Dict[str, Any]],
    active_block: str,
    weights: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not candidate_cfgs:
        return []

    cand_df = _candidate_frame(candidate_cfgs, active_block=active_block)
    if cand_df.empty:
        return list(candidate_cfgs)

    bundle = fit_auto_surrogate(history_df, weights=weights)

    if bundle is None:
        # simple fallback: keep existing order
        ranked = cand_df.copy()
        ranked["_pred_loss"] = 0.0
    else:
        try:
            ranked = cand_df.copy()
            ranked["_pred_loss"] = _predict_bundle(bundle, ranked)
        except Exception:
            ranked = cand_df.copy()
            ranked["_pred_loss"] = 0.0

    ranked = ranked.sort_values(["_pred_loss", "_candidate_idx"], ascending=[True, True])

    if top_k is None:
        top_k = len(candidate_cfgs)
    top_k = max(1, int(top_k))

    out: List[Dict[str, Any]] = []
    for _, row in ranked.head(top_k).iterrows():
        idx = int(row["_candidate_idx"])
        cfg = deepcopy(candidate_cfgs[idx])
        cfg["_surrogate_pred_loss"] = float(row["_pred_loss"])
        cfg["_surrogate_rank"] = len(out) + 1
        out.append(cfg)

    return out


def suggest_next_candidates(
    session_dir: str | Path,
    run_cfg: dict,
    incumbent_cfg: dict,
    active_block: str,
    n_candidates: int = 40,
    top_k: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    session_dir = Path(session_dir)
    history_df = load_ccd_eval_history(session_dir)
    probes = generate_probe_candidates(
        run_cfg=run_cfg,
        incumbent_cfg=incumbent_cfg,
        active_block=active_block,
        n_candidates=n_candidates,
        seed=seed,
    )
    return rank_candidates_with_surrogate(
        history_df=history_df,
        candidate_cfgs=probes,
        active_block=active_block,
        weights=None,
        top_k=top_k,
    )
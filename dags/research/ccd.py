# research/ccd.py

from __future__ import annotations

import math
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from common.schema import get_schema, enforce_schema
from research.ccd_surrogate import make_ccd_eval_row, stage_ccd_eval_rows, scalarize_winner_score

import logging

logger = logging.getLogger(__name__)


STATE_FILE_NAME = "coord_descent_state.json"

DEFAULT_BLOCK_ORDER = [
    "signal_structure",
    "trade_management",
    "execution",
]

DEFAULT_BLOCKS = {
    "signal_structure": [
        "ma",
        "stochastic",
        "lookback",
        "bbw",
    ],
    "trade_management": [
        "SL",
        "TP",
        "use_trailing_sl",
        "trailing_sl_pct",
        "trailing_sl_interval",
        "trailing_sl_stop_at_pos",
    ],
    "execution": [
        "use_limit_entry",
        "limit_order_expiry_bars",
        "trade_window_interval",
    ],
}

DEFAULT_PROFILES = [
    {
        "name": "strict",
        "top_pct": 0.02,
        "trade_floor": 80,
        "min_return_era": 0.20,
    }
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def state_path(session_dir: Path) -> Path:
    return Path(session_dir) / STATE_FILE_NAME


def _scalarize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _compose_stoch_key(cfg: dict) -> str:
    k = _scalarize(cfg.get("stoch_k"))
    d = _scalarize(cfg.get("stoch_d"))
    s = _scalarize(cfg.get("stoch_s"))
    th = _scalarize(cfg.get("stoch_thresholds"))

    try:
        k = int(k)
    except Exception:
        k = 12
    try:
        d = int(d)
    except Exception:
        d = 3
    try:
        s = int(s)
    except Exception:
        s = 3

    low = 30.0
    high = 70.0
    if isinstance(th, (list, tuple)) and len(th) == 2:
        try:
            low = float(th[0])
            high = float(th[1])
        except Exception:
            pass

    return f"k{k}_d{d}_s{s}_l{low:g}_u{high:g}"


def _seeded_regime_cfg(run_cfg: dict) -> dict:
    """
    Build the initial incumbent configuration for CCD.

    This is the starting point for every CCD session. It keeps two goals in balance:
    1) preserve compatibility with older flat configs,
    2) normalize values so later candidate generation can mutate them safely.

    Important:
    - This function does not decide the winner.
    - It only prepares a clean baseline config for the first search round.
    - Nested signal_structure is preserved because signal_json / CCD history
      rely on that structure being available later.
    """
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    seed = dict(coord_cfg.get("seed", {}) or {})

    # Start from the current run config, but remove CCD-only control fields.
    # The result should be a plain regime config that can be written into state.
    out: dict = {}
    for k, v in run_cfg.items():
        if k in {"coord_descent", "search_mode"}:
            continue
        out[k] = _scalarize(v)

    # Manual seed overrides always win over flat defaults.
    # This lets a session start from a carefully chosen configuration.
    for k, v in seed.items():
        out[k] = _scalarize(v)

    # Keep the search mode visible in the incumbent snapshot.
    out["search_mode"] = str(run_cfg.get("search_mode", "cyclic_coordinate_descent"))

    # -------------------------
    # Signal block normalization
    # -------------------------
    # Legacy MA gating used ma_int as a bitmask/selector.
    # If you are fully migrating away from that model, keep this only as a
    # compatibility field until all downstream MA logic stops reading it.
    ma_periods = run_cfg.get("ma_periods", None)
    has_ma_periods = False
    if isinstance(ma_periods, (list, tuple)):
        for x in ma_periods:
            try:
                if float(_scalarize(x) or 0) > 0:
                    has_ma_periods = True
                    break
            except Exception:
                continue
    elif ma_periods is not None:
        try:
            has_ma_periods = float(_scalarize(ma_periods) or 0) > 0
        except Exception:
            has_ma_periods = False

    out["ma_int"] = 0 if not has_ma_periods else int(out.get("ma_int", 0) or 0)
    out["ma_reversion"] = bool(out.get("ma_reversion", False))
    out["entry_lookback_units"] = int(_scalarize(out.get("entry_lookback_units")) or 0)

    # Stochastic parameters need to be normalized so signal_json can round-trip.
    out["use_stochastic"] = bool(out.get("use_stochastic", False))
    if not out["use_stochastic"]:
        out["stoch_key"] = "OFF"
        out["stoch_col"] = None
        out["stoch_lower"] = None
        out["stoch_upper"] = None
    else:
        if not out.get("stoch_key"):
            out["stoch_key"] = _compose_stoch_key(out)
        out["stoch_col"] = str(out.get("stoch_col") or f"stoch_{out['stoch_key'].split('_l')[0]}")
        out["stoch_lower"] = float(out.get("stoch_lower", 30.0) or 30.0)
        out["stoch_upper"] = float(out.get("stoch_upper", 70.0) or 70.0)

    out["use_bbw"] = bool(out.get("use_bbw", False))
    if not out["use_bbw"]:
        out["bbw_periods"] = 0
        out["bbw_std"] = 0.0
        out["bbw_thresholds"] = 0
    else:
        out["bbw_periods"] = int(out.get("bbw_periods", 0) or 0)
        out["bbw_std"] = float(out.get("bbw_std", 0.0) or 0.0)
        out["bbw_thresholds"] = int(out.get("bbw_thresholds", 0) or 0)

    # -------------------------
    # Trade management
    # -------------------------
    out["SL"] = float(out.get("SL", _scalarize((run_cfg.get("sl_range") or {}).get("min", 0.2))) or 0.2)
    out["TP"] = float(out.get("TP", _scalarize((run_cfg.get("tp_range") or {}).get("max", 6.0))) or 6.0)
    out["use_trailing_sl"] = bool(out.get("use_trailing_sl", False))
    out["trailing_sl_pct"] = float(out.get("trailing_sl_pct", 0.0) or 0.0)
    out["trailing_sl_interval"] = int(out.get("trailing_sl_interval", 0) or 0)
    out["trailing_sl_stop_at_pos"] = bool(out.get("trailing_sl_stop_at_pos", True))

    # -------------------------
    # Execution
    # -------------------------
    out["use_limit_entry"] = bool(out.get("use_limit_entry", True))
    out["limit_order_expiry_bars"] = int(out.get("limit_order_expiry_bars", 0) or 0)
    out["trade_window_interval"] = int(out.get("trade_window_interval", 0) or 0)

    # -------------------------
    # Time / era defaults
    # -------------------------
    # This keeps the incumbent aligned with the current session’s time horizon.
    exit_windows = run_cfg.get("exit_windows_h", [24])
    out["exit_window_h"] = int(_scalarize(out.get("exit_window_h", _scalarize(exit_windows))) or 24)

    return out

def _merge_signal_json_into_structure(regime_cfg: dict, signal_json: Any) -> dict:
    """
    Rehydrate the nested signal_structure tree from the compact master-side signal_json.

    This keeps CCD aligned:
    - master rows store compact signal_json
    - accepted CCD state stores nested signal_structure again
    - the next candidate generation pass can mutate the real nested fields
    """
    out = deepcopy(regime_cfg or {})

    if not signal_json:
        return out

    try:
        payload = json.loads(signal_json) if isinstance(signal_json, str) else dict(signal_json)
    except Exception:
        return out

    signals = payload.get("signals", {})
    if not isinstance(signals, dict):
        return out

    signal_structure = dict(out.get("signal_structure") or {})

    for family_name, family_payload in signals.items():
        if not isinstance(family_payload, dict):
            continue

        family_cfg = dict(signal_structure.get(family_name) or {})
        family_cfg["enabled"] = True

        tf_map = dict(family_cfg.get("by_timeframe") or {})
        for tf, tf_cfg in family_payload.items():
            if not isinstance(tf_cfg, dict):
                continue
            clean_tf_cfg = dict(tf_cfg)
            clean_tf_cfg.pop("timeframe", None)
            tf_map[str(tf)] = clean_tf_cfg

        family_cfg["by_timeframe"] = tf_map
        signal_structure[family_name] = family_cfg

    out["signal_structure"] = signal_structure
    out["signal_json"] = payload
    return out


def default_coord_descent_state(session_dir: Path, run_cfg: dict) -> dict:
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    profiles = coord_cfg.get("search_profiles") or coord_cfg.get("profiles") or DEFAULT_PROFILES
    block_order = coord_cfg.get("block_order") or DEFAULT_BLOCK_ORDER
    blocks = coord_cfg.get("blocks") or DEFAULT_BLOCKS
    seed_cfg = _seeded_regime_cfg(run_cfg)

    block_refine_rounds = int(coord_cfg.get("block_refine_rounds", 1) or 1)
    search_scale_start = float(coord_cfg.get("search_scale_start", 1.0) or 1.0)
    search_scale_min = float(coord_cfg.get("search_scale_min", 0.35) or 0.35)
    search_scale_max = float(coord_cfg.get("search_scale_max", 2.0) or 2.0)
    search_scale_shrink = float(coord_cfg.get("search_scale_shrink", 0.75) or 0.75)
    search_scale_expand = float(coord_cfg.get("search_scale_expand", 1.20) or 1.20)

    return {
        "version": 3,
        "updated_at": _now_utc_iso(),
        "updated_at_epoch_s": int(datetime.now(timezone.utc).timestamp()),
        "search_mode": str(run_cfg.get("search_mode", "cyclic_coordinate_descent")),
        "session_dir": str(Path(session_dir)),
        "cycle_idx": int(coord_cfg.get("cycle_idx", 0) or 0),
        "block_cycle_idx": int(coord_cfg.get("block_cycle_idx", 0) or 0),
        "block_idx": int(coord_cfg.get("block_idx", 0) or 0),
        "block_refine_idx": 0,
        "block_refine_rounds": int(max(1, block_refine_rounds)),
        "search_scale_start": float(search_scale_start),
        "search_scale": float(search_scale_start),
        "search_scale_min": float(search_scale_min),
        "search_scale_max": float(search_scale_max),
        "search_scale_shrink": float(search_scale_shrink),
        "search_scale_expand": float(search_scale_expand),
        "block_order": list(block_order),
        "blocks": dict(blocks),
        "profiles": list(profiles),
        "seed": dict(seed_cfg),
        "selection_rules": {
            "alpha_min_margin": float(coord_cfg.get("alpha_min_margin", 0.016) or 0.016),
            "rank_order": [
                "recency_weighted_era_consistency_score",
                "era_consistency_score",
                "dominance_score",
                "elite_median_alpha",
            ],
            "use_trade_floor": True,
            "min_return_name": "min_return_era",
            "rank_top_n": int(coord_cfg.get("rank_top_n", 30) or 30),
        },
        "incumbent": {
            "regime_id": None,
            "regime_cfg": dict(seed_cfg),
            "score": {},
            "summary_path": None,
            "updated_at": _now_iso(),
        },
        "last_generated": {
            "batch_id": None,
            "active_block": block_order[0] if block_order else None,
            "best_seen_loss": None,
            "progress": {},
            "profile_name": None,
            "regime_count": 0,
            "updated_at": None,
        },
        "last_rejected": None,
        "last_ranked": [],
        "history": [],
        "stagnant_blocks": 0,
    }


def load_coord_descent_state(session_dir: Path) -> dict:
    path = state_path(session_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_coord_descent_state(session_dir: Path, state: dict) -> Path:
    path = state_path(session_dir)

    # Stamp the state right before saving so the JSON always shows the last write time.
    state = dict(state or {})
    state["updated_at"] = _now_utc_iso()
    state["updated_at_epoch_s"] = int(datetime.now(timezone.utc).timestamp())

    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf8")
    return path


def ensure_coord_descent_state(session_dir: Path, run_cfg: dict) -> dict:
    state = load_coord_descent_state(session_dir)
    if not state:
        state = default_coord_descent_state(session_dir, run_cfg)
        save_coord_descent_state(session_dir, state)
        return state

    changed = False
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})

    if "version" not in state:
        state["version"] = 3
        changed = True

    if "active_block" not in state:
        state["active_block"] = (
            state.get("last_generated", {}).get("active_block")
            or (state.get("block_order") or DEFAULT_BLOCK_ORDER)[int(state.get("block_idx", 0) or 0) % len(state.get("block_order") or DEFAULT_BLOCK_ORDER)]
        )
        changed = True

    if "best_seen_loss" not in state:
        state["best_seen_loss"] = None
        changed = True

    if "progress" not in state:
        state["progress"] = {}
        changed = True

    if "block_order" not in state:
        state["block_order"] = list(coord_cfg.get("block_order") or DEFAULT_BLOCK_ORDER)
        changed = True

    if "blocks" not in state:
        state["blocks"] = dict(coord_cfg.get("blocks") or DEFAULT_BLOCKS)
        changed = True

    if "block_cycle_idx" not in state:
        state["block_cycle_idx"] = int(state.get("cycle_idx", 0) or 0)
        changed = True

    if "profiles" not in state:
        state["profiles"] = list(coord_cfg.get("search_profiles") or coord_cfg.get("profiles") or DEFAULT_PROFILES)
        changed = True

    if "selection_rules" not in state:
        state["selection_rules"] = {
            "alpha_min_margin": float(coord_cfg.get("alpha_min_margin", 0.016) or 0.016),
            "rank_order": [
                "recency_weighted_era_consistency_score",
                "era_consistency_score",
                "dominance_score",
                "elite_median_alpha",
            ],
            "use_trade_floor": True,
            "min_return_name": "min_return_era",
            "rank_top_n": int(coord_cfg.get("rank_top_n", 30) or 30),
        }
        changed = True

    if "seed" not in state:
        state["seed"] = _seeded_regime_cfg(run_cfg)
        changed = True

    if "incumbent" not in state:
        state["incumbent"] = {
            "regime_id": None,
            "regime_cfg": dict(state.get("seed") or _seeded_regime_cfg(run_cfg)),
            "score": {},
            "summary_path": None,
            "updated_at": _now_iso(),
        }
        changed = True

    if "last_generated" not in state:
        state["last_generated"] = {
            "batch_id": None,
            "active_block": None,
            "profile_name": None,
            "regime_count": 0,
            "updated_at": None,
        }
        changed = True

    if "block_refine_idx" not in state:
        state["block_refine_idx"] = 0
        changed = True

    if "block_refine_rounds" not in state:
        state["block_refine_rounds"] = int(coord_cfg.get("block_refine_rounds", 1) or 1)
        changed = True

    if "search_scale_start" not in state:
        state["search_scale_start"] = float(coord_cfg.get("search_scale_start", 1.0) or 1.0)
        changed = True

    if "search_scale" not in state:
        state["search_scale"] = float(coord_cfg.get("search_scale_start", 1.0) or 1.0)
        changed = True

    if "search_scale_min" not in state:
        state["search_scale_min"] = float(coord_cfg.get("search_scale_min", 0.35) or 0.35)
        changed = True

    if "search_scale_max" not in state:
        state["search_scale_max"] = float(coord_cfg.get("search_scale_max", 2.0) or 2.0)
        changed = True

    if "search_scale_shrink" not in state:
        state["search_scale_shrink"] = float(coord_cfg.get("search_scale_shrink", 0.75) or 0.75)
        changed = True

    if "search_scale_expand" not in state:
        state["search_scale_expand"] = float(coord_cfg.get("search_scale_expand", 1.20) or 1.20)
        changed = True

    if "stagnant_blocks" not in state:
        state["stagnant_blocks"] = 0
        changed = True

    if "last_rejected" not in state:
        state["last_rejected"] = None
        changed = True

    if "last_ranked" not in state:
        state["last_ranked"] = []
        changed = True

    if changed:
        save_coord_descent_state(session_dir, state)

    return state


def load_compact_master_metrics(
    session_dir: str | Path,
    master_path: str | Path | None = None,
    columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Load only the compact merged master metrics needed by CCD scoring.

    Compatible with:
      - session_dir/master_metrics.parquet
      - results/batch_*_master_metrics.parquet
      - a directory containing parquet parts
    """
    session_dir = Path(session_dir)

    if master_path is None:
        candidates = [
            session_dir / "master_metrics.parquet",
            session_dir / "results" / "master_metrics.parquet",
            session_dir / "results" / "batch_master_metrics.parquet",
        ]
        master_path = next((p for p in candidates if p.exists()), None)

        if master_path is None:
            # fallback: any merged batch master files in results/
            batch_files = sorted((session_dir / "results").glob("batch_*_master_metrics.parquet"))
            if batch_files:
                master_path = batch_files[-1]

    if master_path is None:
        return pl.DataFrame([], schema=get_schema("master"))

    master_path = Path(master_path)

    if master_path.is_dir():
        files = sorted(master_path.glob("*.parquet"))
        if not files:
            return pl.DataFrame([], schema=get_schema("master"))
        scan_src = [str(p) for p in files]
        schema_probe = files[0]
    else:
        if not master_path.exists():
            return pl.DataFrame([], schema=get_schema("master"))
        scan_src = str(master_path)
        schema_probe = master_path

    try:
        available_cols = list(pq.ParquetFile(str(schema_probe)).schema_arrow.names)
    except Exception:
        available_cols = list(get_schema("master").keys())

    master_cols = list(get_schema("master").keys())

    if columns is None:
        wanted = [c for c in master_cols if c in available_cols]
    else:
        wanted = [c for c in columns if c in available_cols]

    if not wanted:
        return pl.DataFrame([], schema=get_schema("master"))

    lf = pl.scan_parquet(scan_src).select([pl.col(c) for c in wanted])

    try:
        df = lf.collect(streaming=True)
    except Exception:
        df = lf.collect()

    return enforce_schema(df, "master", strict=False)

def _winner_loss_from_score(score: dict, run_cfg: dict) -> float:
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    return scalarize_winner_score(
        score or {},
        weights=coord_cfg.get("loss_weights"),
        metric_bounds=coord_cfg.get("metric_bounds"),
    ) 

def pick_next_incumbent(
    session_dir: Path,
    run_cfg: dict,
    master_df: pl.DataFrame,
) -> Dict:
    """
    CCD decision step.

    Big picture:
    1) A batch of candidates for the current active block has already been backtested.
    2) The combined master metrics table contains the real results.
    3) This function ranks those real results, chooses the winner, and updates CCD state.
    4) The updated state controls the next candidate generation pass:
       - stay on the same block for several refine rounds
       - then rotate to the next block
       - shrink/expand search_scale so the surrogate probes tighter or wider neighborhoods

    Important:
    - The surrogate never decides the true winner.
    - The backtest result is the source of truth.
    - The surrogate only ranks candidate configs before backtest.
    """
    session_dir = Path(session_dir)
    state = ensure_coord_descent_state(session_dir, run_cfg)

    if master_df is None or master_df.is_empty():
        return {
            "selected": False,
            "reason": "empty_master_df",
            "state": state,
        }

    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    blocks = dict(state.get("blocks") or {})
    block_order = list(state.get("block_order") or [])
    profiles = list(state.get("profiles") or [])
    selection_rules = dict(state.get("selection_rules") or {})

    if not block_order:
        block_order = list(DEFAULT_BLOCK_ORDER)
    if not blocks:
        blocks = dict(DEFAULT_BLOCKS)

    # These values control the outer CCD loop behavior.
    # block_refine_rounds:
    #   how many times we keep refining the same block before we rotate to the next one.
    # search_scale:
    #   how wide the surrogate candidate generator should probe around the incumbent.
    #   smaller => local refinement
    #   larger  => broader exploration
    search_scale_min = float(coord_cfg.get("search_scale_min", 0.35) or 0.35)
    search_scale_max = float(coord_cfg.get("search_scale_max", 2.0) or 2.0)
    search_scale_start = float(coord_cfg.get("search_scale_start", 1.0) or 1.0)
    search_scale_shrink = float(coord_cfg.get("search_scale_shrink", 0.75) or 0.75)
    search_scale_expand = float(coord_cfg.get("search_scale_expand", 1.20) or 1.20)
    block_refine_rounds = int(coord_cfg.get("block_refine_rounds", 3) or 3)
    block_refine_rounds = max(1, block_refine_rounds)

    if "search_scale" not in state:
        state["search_scale"] = float(np.clip(search_scale_start, search_scale_min, search_scale_max))
    if "block_refine_idx" not in state:
        state["block_refine_idx"] = 0
    if "block_refine_rounds" not in state:
        state["block_refine_rounds"] = block_refine_rounds
    if "reject_streak" not in state:
        state["reject_streak"] = 0

    active_block = state.get("active_block") or state.get("last_generated", {}).get("active_block")
    if not active_block or active_block not in blocks:
        block_idx = int(state.get("block_idx", 0) or 0)
        if block_order:
            active_block = block_order[block_idx % len(block_order)]
        else:
            active_block = next(iter(blocks.keys()), "trade_management")

    state["active_block"] = active_block

    candidate_keys = [k for k in blocks.get(active_block, []) if k in master_df.columns]

    if active_block == "signal_structure":
        if "signal_json" in master_df.columns:
            candidate_keys = ["signal_json"]
        else:
            candidate_keys = []

    alpha_min_margin = float(selection_rules.get("alpha_min_margin", 0.016) or 0.016)
    rank_order = list(
        selection_rules.get(
            "rank_order",
            [
                "recency_weighted_era_consistency_score",
                "era_consistency_score",
                "dominance_score",
                "elite_median_alpha",
            ],
        )
    )
    rank_top_n = int(selection_rules.get("rank_top_n", 30) or 30)
    rank_top_n = max(1, rank_top_n)

    def _era_weight_map(eras: List[int], decay_days: float = 90.0) -> dict[int, float]:
        # Later eras matter more, but we still keep older eras in the score.
        eras_clean = []
        for e in eras:
            try:
                eras_clean.append(int(e))
            except Exception:
                continue
        eras_clean = sorted(set(eras_clean))
        if not eras_clean:
            return {}

        latest_era = eras_clean[-1]
        latest_dt = datetime.strptime(str(latest_era), "%Y%m%d").date()

        weights = {}
        for era in eras_clean:
            era_dt = datetime.strptime(str(era), "%Y%m%d").date()
            age_days = max((latest_dt - era_dt).days, 0)
            weight = 1.0 / (1.0 + math.log1p(age_days / decay_days))
            weights[int(era)] = float(weight)
        return weights

    def _row_tuple(row: dict) -> tuple:
        # Ranking tuple used to compare candidates.
        # Higher is better for all entries in rank_order.
        return tuple(float(row.get(k, 0.0) or 0.0) for k in rank_order)

    def _score_profile(
        df_in: pl.DataFrame,
        top_pct: float,
        trade_floor: int,
        min_return_era: float,
    ) -> pl.DataFrame:
        """
        Convert raw master rows into profile-level ranked candidates.

        This is not the surrogate. This is the real backtest result scoring:
        - filter out weak trade counts
        - build a candidate signature from the block keys
        - score elite rows per era
        - measure consistency across eras
        - measure dominance inside the elite slice
        """
        if df_in.is_empty():
            return pl.DataFrame()

        df_filtered = df_in.filter(pl.col("total_pos") >= int(trade_floor))
        if df_filtered.is_empty():
            return pl.DataFrame()

        sig_exprs = [pl.col(k).cast(pl.Utf8) for k in candidate_keys if k in df_filtered.columns]
        if not sig_exprs:
            return pl.DataFrame()

        df_eff = df_filtered.with_columns([
            pl.concat_str(sig_exprs, separator="|").alias("_cand_sig"),
            (((pl.col("balance") - 100.0) / 100.0) / (pl.col("total_pos") + 1e-9)).alias("return_per_trade"),
            (
                (
                    ((pl.col("balance") - 100.0) / 100.0) / (pl.col("total_pos") + 1e-9)
                )
                * (
                    (((pl.col("balance") - 100.0) / 100.0) / (pl.col("total_pos") + 1e-9)).exp()
                )
                / (pl.col("max_drawdown") + 1e-9)
            ).alias("alpha_per_trade"),
        ])

        eras = sorted(df_eff["era_int"].unique().to_list()) if "era_int" in df_eff.columns else []
        era_weight_map = _era_weight_map(
            eras,
            decay_days=float(run_cfg.get("CCD_RECENCY_DECAY_DAYS", 90.0) or 90.0),
        )
        total_era_weight = sum(era_weight_map.values()) if era_weight_map else 0.0

        global_stats = (
            df_eff.group_by("_cand_sig")
            .agg([
                pl.col("alpha_per_trade").median().alias("global_median_alpha"),
                pl.col("return_per_trade").median().alias("global_median_return"),
            ])
        )

        top_rows = []
        for era in eras:
            era_df = df_eff.filter(pl.col("era_int") == era)
            if era_df.is_empty():
                continue
            cutoff = max(1, int(len(era_df) * float(top_pct)))
            top_rows.append(era_df.sort("alpha_per_trade", descending=True).head(cutoff))

        if not top_rows:
            return pl.DataFrame()

        top_elite_per_era = pl.concat(top_rows)
        total_pool_size = max(1, top_elite_per_era.height)

        elite_quality = (
            top_elite_per_era.group_by("_cand_sig")
            .agg([
                (pl.len().cast(pl.Float64) / float(total_pool_size)).alias("dominance_score"),
                pl.col("alpha_per_trade").median().alias("elite_median_alpha"),
                pl.col("return_per_trade").median().alias("elite_median_return"),
            ])
        )

        consistency_presence = (
            df_eff.filter(((pl.col("balance") - 100.0) / 100.0) >= float(min_return_era))
            .select(["era_int", "_cand_sig"])
            .unique()
            .with_columns(
                pl.col("era_int")
                .map_elements(lambda x: float(era_weight_map.get(int(x), 1.0)), return_dtype=pl.Float64)
                .alias("era_recency_weight")
            )
        )

        elite_consistency = (
            consistency_presence.group_by("_cand_sig")
            .agg([
                pl.col("era_recency_weight").sum().alias("weighted_era_hits"),
                pl.col("era_int").n_unique().alias("_eras_found"),
            ])
            .with_columns([
                (pl.col("weighted_era_hits") / (total_era_weight + 1e-9)).alias("recency_weighted_era_consistency_score"),
                (pl.col("_eras_found") / max(1, len(eras))).alias("era_consistency_score"),
            ])
        )

        scored = (
            elite_quality
            .join(elite_consistency, on="_cand_sig", how="left")
            .join(global_stats, on="_cand_sig", how="left")
            .with_columns([
                pl.col("weighted_era_hits").fill_null(0.0),
                pl.col("recency_weighted_era_consistency_score").fill_null(0.0),
                pl.col("era_consistency_score").fill_null(0.0),
                pl.col("elite_median_alpha").fill_null(0.0),
                pl.col("elite_median_return").fill_null(0.0),
                pl.col("global_median_alpha").fill_null(0.0),
                pl.col("global_median_return").fill_null(0.0),
                (pl.col("elite_median_alpha") - pl.col("global_median_alpha")).alias("alpha_lift"),
                (pl.col("elite_median_return") - pl.col("global_median_return")).alias("return_lift"),
            ])
            .sort(rank_order, descending=True)
        )

        return scored

    profile_ranked_rows = []

    if profiles:
        # Multiple profiles let the same block be judged under different strictness levels.
        # If only one profile is needed, keep just one entry in run_config.
        for p in profiles:
            if not isinstance(p, dict):
                continue

            top_pct = float(p.get("top_pct", run_cfg.get("TOP_PCT", 0.05)) or 0.05)
            trade_floor = int(p.get("trade_floor", run_cfg.get("TRADE_FLOOR", 40)) or 40)
            min_return_era = float(
                p.get(
                    "min_return_era",
                    p.get("min_return", run_cfg.get("MIN_RETURN_ERA", 0.10)),
                ) or 0.10
            )

            scored = _score_profile(master_df, top_pct, trade_floor, min_return_era)
            if scored.is_empty():
                continue

            rows = scored.head(rank_top_n).to_dicts()
            profile_name = str(p.get("name", "profile"))

            for r in rows:
                r["_profile_name"] = profile_name

            profile_ranked_rows.extend(rows)
    else:
        top_pct = float(run_cfg.get("TOP_PCT", 0.05) or 0.05)
        trade_floor = int(run_cfg.get("TRADE_FLOOR", 40) or 40)
        min_return_era = float(run_cfg.get("MIN_RETURN_ERA", 0.10) or 0.10)

        scored = _score_profile(master_df, top_pct, trade_floor, min_return_era)
        if scored.is_empty():
            return {
                "selected": False,
                "reason": "no_scored_rows",
                "state": state,
            }

        rows = scored.head(rank_top_n).to_dicts()
        for r in rows:
            r["_profile_name"] = "default"

        profile_ranked_rows.extend(rows)

    if not profile_ranked_rows:
        return {
            "selected": False,
            "reason": "no_profile_winner",
            "state": state,
        }

    # If the same candidate appears in several profiles, aggregate it so
    # the final decision prefers candidates that stay strong across settings.
    agg = {}
    for row in profile_ranked_rows:
        sig = row.get("_cand_sig")
        if not sig:
            continue
        if sig not in agg:
            agg[sig] = {"count": 0, "rows": []}
        agg[sig]["count"] += 1
        agg[sig]["rows"].append(row)

    if not agg:
        return {
            "selected": False,
            "reason": "no_aggregated_candidates",
            "state": state,
        }

    def _agg_key(item):
        sig, data = item
        rows = data["rows"]
        best = max(rows, key=_row_tuple)
        return (
            data["count"],
            best.get("recency_weighted_era_consistency_score", 0.0),
            best.get("era_consistency_score", 0.0),
            best.get("dominance_score", 0.0),
            best.get("elite_median_alpha", 0.0),
        )

    sorted_sigs = [sig for sig, _ in sorted(agg.items(), key=_agg_key, reverse=True)]

    ranked = []
    for rank_i, sig in enumerate(sorted_sigs[:rank_top_n], start=1):
        rows = agg[sig]["rows"]
        best = max(rows, key=_row_tuple).copy()
        best["_rank"] = rank_i
        best["_profile_hit_count"] = int(agg[sig]["count"])
        ranked.append(best)

    best_profile_row = ranked[0]
    best_sig = str(best_profile_row.get("_cand_sig", ""))

    sig_exprs = [pl.col(k).cast(pl.Utf8) for k in candidate_keys if k in master_df.columns]
    candidate_df = master_df.with_columns(
        pl.concat_str(sig_exprs, separator="|").alias("_cand_sig")
    )

    chosen_candidate_rows = candidate_df.filter(pl.col("_cand_sig") == best_sig)
    if chosen_candidate_rows.is_empty():
        return {
            "selected": False,
            "reason": "winner_not_found_in_master_df",
            "state": state,
            "ranked": ranked,
        }

    chosen = chosen_candidate_rows.head(1).to_dicts()[0]

    incumbent_state = dict(state.get("incumbent") or {})
    incumbent_score = dict(incumbent_state.get("score") or {})
    incumbent_exists = bool(incumbent_score)

    cand_tuple = _row_tuple(best_profile_row)
    cand_alpha = float(best_profile_row.get("elite_median_alpha", 0.0) or 0.0)

    candidate_score = {
        "recency_weighted_era_consistency_score": float(best_profile_row.get("recency_weighted_era_consistency_score", 0.0)),
        "era_consistency_score": float(best_profile_row.get("era_consistency_score", 0.0)),
        "dominance_score": float(best_profile_row.get("dominance_score", 0.0)),
        "elite_median_alpha": float(best_profile_row.get("elite_median_alpha", 0.0)),
    }
    candidate_loss = _winner_loss_from_score(candidate_score, run_cfg)

    incumbent_loss = None
    if incumbent_exists:
        incumbent_loss = _winner_loss_from_score(incumbent_score, run_cfg)

    best_seen_prev = state.get("best_seen_loss", None)
    try:
        best_seen_prev = float(best_seen_prev)
    except Exception:
        best_seen_prev = float("inf")

    best_seen_loss = float(min(best_seen_prev, candidate_loss))
    loss_delta = None if incumbent_loss is None else float(incumbent_loss - candidate_loss)

    # Real acceptance test.
    # The surrogate may rank the candidate highly, but the incumbent only changes
    # if the real backtest metrics are better enough to justify it.
    #
    # This keeps era consistency as a hard gate by default, while still allowing
    # the score itself to drive improvement decisions.
    era_consistency_floor = float(coord_cfg.get("accept_era_consistency_floor", 1.0) or 1.0)
    accept_loss_margin = float(coord_cfg.get("accept_loss_margin", 0.0) or 0.0)

    accepted = False
    if float(candidate_score.get("era_consistency_score", 0.0) or 0.0) >= era_consistency_floor:
        if not incumbent_exists:
            accepted = True
        else:
            # Primary decision: lower loss is better.
            if incumbent_loss is not None and candidate_loss < (incumbent_loss - accept_loss_margin):
                accepted = True
            else:
                # Secondary tie-break: only use alpha margin when rank actually improves.
                inc_tuple = _row_tuple(incumbent_score)
                inc_alpha = float(incumbent_score.get("elite_median_alpha", -1e30) or -1e30)

                if cand_tuple > inc_tuple and (cand_alpha - inc_alpha) >= alpha_min_margin:
                    accepted = True

    if accepted:
        state["reject_streak"] = 0
        incumbent = dict(incumbent_state.get("regime_cfg") or {})

        # Copy normal scalar keys from the chosen row.
        for k in candidate_keys:
            if k in chosen:
                incumbent[k] = chosen[k]

        # IMPORTANT:
        # signal_json is compact master output; rebuild signal_structure from it
        # so the next search cycle can actually mutate signal params.
        if active_block == "signal_structure" and "signal_json" in chosen:
            incumbent = _merge_signal_json_into_structure(incumbent, chosen["signal_json"])

        state["incumbent"] = {
            "regime_id": int(chosen.get("regime_id", 0) or 0),
            "regime_cfg": incumbent,
            "score": {
                "recency_weighted_era_consistency_score": float(best_profile_row.get("recency_weighted_era_consistency_score", 0.0)),
                "era_consistency_score": float(best_profile_row.get("era_consistency_score", 0.0)),
                "dominance_score": float(best_profile_row.get("dominance_score", 0.0)),
                "elite_median_alpha": float(best_profile_row.get("elite_median_alpha", 0.0)),
                "alpha_lift": float(best_profile_row.get("alpha_lift", 0.0)),
                "return_lift": float(best_profile_row.get("return_lift", 0.0)),
                "profile_name": str(best_profile_row.get("_profile_name", "default")),
                "candidate_sig": str(best_sig),
            },
            "summary_path": incumbent_state.get("summary_path"),
            "updated_at": _now_iso(),
        }
    else:
        reject_streak = int(state.get("reject_streak", 0) or 0) + 1
        state["reject_streak"] = reject_streak

        # After several failed rounds, the next generation pass should widen the
        # search so the block does not keep probing too narrowly around the same point.
        if reject_streak >= int(coord_cfg.get("force_explore_after", 3) or 3):
            state["search_scale"] = float(search_scale_max)

        state["last_rejected"] = {
            "candidate_sig": str(best_sig),
            "candidate_score": {
                "recency_weighted_era_consistency_score": float(best_profile_row.get("recency_weighted_era_consistency_score", 0.0)),
                "era_consistency_score": float(best_profile_row.get("era_consistency_score", 0.0)),
                "dominance_score": float(best_profile_row.get("dominance_score", 0.0)),
                "elite_median_alpha": float(best_profile_row.get("elite_median_alpha", 0.0)),
            },
            "reason": "alpha_min_margin_not_met_or_rank_not_better",
            "updated_at": _now_iso(),
        }

    state["best_seen_loss"] = best_seen_loss
    state["progress"] = {
        "candidate_loss": candidate_loss,
        "incumbent_loss_before": incumbent_loss,
        "loss_delta": loss_delta,
        "best_seen_loss": best_seen_loss,
        "accepted": bool(accepted),
    }

    # Build the eval row that will train the surrogate later.
    # This is the bridge between backtest truth and surrogate learning.
    regime_cfg_for_eval = dict(incumbent_state.get("regime_cfg") or state.get("seed") or {})
    for k in candidate_keys:
        if k in chosen:
            regime_cfg_for_eval[k] = chosen[k]

    eval_row = make_ccd_eval_row(
        regime_id=int(chosen.get("regime_id", 0) or 0),
        era_int=int(best_profile_row.get("era_int", 0) or 0),
        side=int(best_profile_row.get("side", 0) or 0),
        block_name=str(active_block),
        profile_name=str(best_profile_row.get("_profile_name", "default")),
        candidate_sig=str(best_sig),
        candidate_rank=int(best_profile_row.get("_rank", 1) or 1),
        accepted=bool(accepted),
        selected=bool(accepted),
        total_pos=int(best_profile_row.get("total_pos", 0) or 0),
        win_pos=int(best_profile_row.get("win_pos", 0) or 0),
        balance=float(best_profile_row.get("balance", 100.0) or 100.0),
        max_drawdown=float(best_profile_row.get("max_drawdown", 0.0) or 0.0),
        max_consecutive_losses=int(best_profile_row.get("max_consecutive_losses", 0) or 0),
        sl_val=float(best_profile_row.get("SL", 0.0) or 0.0),
        tp_val=float(best_profile_row.get("TP", 0.0) or 0.0),
        sl_hit=float(best_profile_row.get("SL_hit", np.nan)) if np.isfinite(best_profile_row.get("SL_hit", np.nan)) else np.nan,
        tp_hit=float(best_profile_row.get("TP_hit", np.nan)) if np.isfinite(best_profile_row.get("TP_hit", np.nan)) else np.nan,
        regime_cfg=regime_cfg_for_eval,
        winner_score={
            "recency_weighted_era_consistency_score": float(best_profile_row.get("recency_weighted_era_consistency_score", 0.0)),
            "era_consistency_score": float(best_profile_row.get("era_consistency_score", 0.0)),
            "dominance_score": float(best_profile_row.get("dominance_score", 0.0)),
            "elite_median_alpha": float(best_profile_row.get("elite_median_alpha", 0.0)),
        },
    )

    # This row becomes training data for the surrogate on future runs.
    stage_ccd_eval_rows(
        session_dir / "results",
        0,
        pl.DataFrame([eval_row]),
    )

    # Decide what block to search next.
    # block_cycle_idx is the internal CCD loop counter.
    # cycle_idx is reserved for the outer Airflow self-trigger loop.
    current_block_cycle = int(state.get("block_cycle_idx", 0) or 0)

    current_block_idx = 0
    if block_order:
        try:
            current_block_idx = block_order.index(active_block)
        except Exception:
            current_block_idx = int(state.get("block_idx", 0) or 0) % len(block_order)
    else:
        current_block_idx = int(state.get("block_idx", 0) or 0)

    # Internal state only tracks block refinement/rotation.
    # The DAG run_id / dag conf cycle number should stay separate.
    refine_idx = int(state.get("block_refine_idx", 0) or 0)
    refine_rounds = int(state.get("block_refine_rounds", block_refine_rounds) or block_refine_rounds)
    refine_rounds = max(1, refine_rounds)

    cur_scale = float(state.get("search_scale", search_scale_start) or search_scale_start)
    if accepted:
        next_scale = max(search_scale_min, cur_scale * search_scale_shrink)
    else:
        next_scale = min(search_scale_max, cur_scale * search_scale_expand)
    state["search_scale"] = float(np.clip(next_scale, search_scale_min, search_scale_max))

    if refine_idx + 1 < refine_rounds:
        state["block_refine_idx"] = refine_idx + 1
    else:
        state["block_refine_idx"] = 0
        if block_order:
            current_block_idx = 0
            try:
                current_block_idx = block_order.index(active_block)
            except Exception:
                current_block_idx = int(state.get("block_idx", 0) or 0) % len(block_order)

            next_block_idx = current_block_idx + 1
            if next_block_idx >= len(block_order):
                next_block_idx = 0
                state["cycle_idx"] = int(state.get("cycle_idx", 0) or 0) + 1

            state["block_idx"] = next_block_idx
            state["active_block"] = block_order[next_block_idx]
        else:
            state["cycle_idx"] = int(state.get("cycle_idx", 0) or 0) + 1
            state["block_idx"] = 0
            state["active_block"] = active_block

        state["block_cycle_idx"] = current_block_cycle

    state["last_generated"] = {
        "batch_id": None,
        "active_block": state["active_block"],
        "profile_name": str(best_profile_row.get("_profile_name", "default")),
        "regime_count": len(ranked),
        "updated_at": _now_iso(),
    }

    state["last_ranked"] = {
        "top_n": rank_top_n,
        "rows": ranked,
        "updated_at": _now_iso(),
    }

    history = list(state.get("history", []))[-100:]
    history.append({
        "ts": _now_iso(),
        "active_block": active_block,
        "winner_sig": best_sig,
        "accepted": accepted,
        "search_scale": float(state["search_scale"]),
        "block_refine_idx": int(state["block_refine_idx"]),
        "block_cycle_idx": int(state.get("block_cycle_idx", 0) or 0),
        "block_refine_rounds": int(refine_rounds),
        "loss": {
            "candidate": candidate_loss,
            "incumbent_before": incumbent_loss,
            "delta": loss_delta,
            "best_seen": best_seen_loss,
        },
        "winner_score": {
            "recency_weighted_era_consistency_score": float(best_profile_row.get("recency_weighted_era_consistency_score", 0.0)),
            "era_consistency_score": float(best_profile_row.get("era_consistency_score", 0.0)),
            "dominance_score": float(best_profile_row.get("dominance_score", 0.0)),
            "elite_median_alpha": float(best_profile_row.get("elite_median_alpha", 0.0)),
        },
    })
    state["history"] = history

    save_coord_descent_state(session_dir, state)

    return {
        "selected": accepted,
        "winner_sig": best_sig,
        "winner": chosen,
        "ranked": ranked,
        "score": state.get("incumbent", {}).get("score", best_profile_row),
        "state": state,
    }
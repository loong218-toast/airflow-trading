# ccd_config.py
from __future__ import annotations

import time
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from research.ccd import ensure_coord_descent_state, save_coord_descent_state, state_path
from research.ccd_surrogate import suggest_next_candidates

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scalarize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _write_batch(path: Path, batch_id: int, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump({"batch_id": batch_id, "regimes": rows}, f, indent=2, default=str)


def _base_cfg(run_cfg: dict) -> dict:
    out = {}
    for k, v in run_cfg.items():
        if k in {"coord_descent", "search_mode"}:
            continue
        out[k] = _scalarize(v)
    return out


def _seeded_regime_cfg(run_cfg: dict) -> dict:
    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    seed = dict(coord_cfg.get("seed", {}) or {})

    out = _base_cfg(run_cfg)
    for k, v in seed.items():
        out[k] = deepcopy(_scalarize(v))

    out["search_mode"] = str(run_cfg.get("search_mode", "cyclic_coordinate_descent"))
    out["signal_structure"] = deepcopy(seed.get("signal_structure", run_cfg.get("signal_structure", {})) or {})

    out["SL"] = float(out.get("SL", _scalarize((run_cfg.get("sl_range") or {}).get("min", 0.2)) or 0.2))
    out["TP"] = float(out.get("TP", _scalarize((run_cfg.get("tp_range") or {}).get("max", 6.0)) or 6.0))
    out["use_trailing_sl"] = bool(out.get("use_trailing_sl", False))
    out["trailing_sl_pct"] = float(out.get("trailing_sl_pct", 0.0) or 0.0)
    out["trailing_sl_interval"] = int(out.get("trailing_sl_interval", 0) or 0)
    out["trailing_sl_stop_at_pos"] = bool(out.get("trailing_sl_stop_at_pos", True))
    out["use_limit_entry"] = bool(out.get("use_limit_entry", True))
    out["limit_order_expiry_bars"] = int(out.get("limit_order_expiry_bars", 0) or 0)
    out["trade_window_interval"] = int(out.get("trade_window_interval", 0) or 0)
    out["exit_window_h"] = int(out.get("exit_window_h", 24) or 24)

    return out


def generate_configs(session_dir: Path, run_cfg: dict) -> list[Path]:
    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    state = ensure_coord_descent_state(session_dir, run_cfg)

    block_order = list(
        state.get("block_order")
        or coord_cfg.get("block_order")
        or DEFAULT_BLOCK_ORDER
    )
    blocks = dict(state.get("blocks") or coord_cfg.get("blocks") or DEFAULT_BLOCKS)

    incumbent = dict((state.get("incumbent") or {}).get("regime_cfg") or state.get("seed") or {})
    if not incumbent:
        incumbent = _seeded_regime_cfg(run_cfg)

    active_block_idx = int(state.get("block_idx", 0) or 0) % max(1, len(block_order))
    active_block = str(
        state.get("active_block")
        or (block_order[active_block_idx] if block_order else "trade_management")
    )

    t0 = time.monotonic()
    logger.info("CCD generate_configs | start active_block=%s", active_block)

    all_regime_configs = suggest_next_candidates(
        session_dir=session_dir,
        run_cfg=run_cfg,
        incumbent_cfg=incumbent,
        active_block=active_block,
        n_candidates=int(coord_cfg.get("surrogate_n_candidates", 60) or 60),
        top_k=int(coord_cfg.get("surrogate_top_k", 30) or 30),
        seed=run_cfg.get("CCD_SEED", 42),
    )

    start = time.monotonic()
    all_regime_configs = suggest_next_candidates(
        session_dir=session_dir,
        run_cfg=run_cfg,
        incumbent_cfg=incumbent,
        active_block=active_block,
        n_candidates=int(coord_cfg.get("surrogate_n_candidates", 60) or 60),
        top_k=int(coord_cfg.get("surrogate_top_k", 30) or 30),
        seed=run_cfg.get("CCD_SEED", 42),
    )
    logger.info("CCD generate_configs | surrogate returned %d candidates in %.2fs", len(all_regime_configs), time.monotonic() - start)

    if not all_regime_configs:
        fallback = deepcopy(incumbent)
        fallback["_candidate_sig"] = ""
        fallback["_active_block"] = active_block
        all_regime_configs = [fallback]

    batch_size = int(coord_cfg.get("BATCH_SIZE", run_cfg.get("BATCH_SIZE", 150)) or 150)
    saved_batch_paths: list[Path] = []

    logger.info("=" * 60)
    logger.info("SURROGATE-RANKED CCD CONFIG GENERATION")
    logger.info("Active block: %s", active_block)
    logger.info("Total candidates: %d", len(all_regime_configs))
    logger.info("Batch size: %d", batch_size)
    logger.info("State: %s", state_path(session_dir))
    logger.info("Block keys: %s", blocks.get(active_block, []))
    logger.info("=" * 60)

    for i in range(0, len(all_regime_configs), batch_size):
        batch_slice = all_regime_configs[i : i + batch_size]
        batch_num = i // batch_size

        for j, regime in enumerate(batch_slice):
            regime["regime_id"] = f"{i + j:05d}"
            regime["coord_search_mode"] = "cyclic_coordinate_descent"
            regime["coord_block"] = active_block
            regime["coord_block_idx"] = int(active_block_idx)
            regime["coord_cycle_idx"] = int(state.get("cycle_idx", 0) or 0)
            regime["coord_profile_name"] = str(regime.get("_profile_name", "surrogate"))
            regime["coord_state_path"] = str(state_path(session_dir))

        batch_payload = {
            "batch_id": batch_num,
            "search_mode": "cyclic_coordinate_descent",
            "coord_descent": {
                "cycle_idx": int(state.get("cycle_idx", 0) or 0),
                "block_idx": int(active_block_idx),
                "active_block": active_block,
                "profile_count": 1,
                "selection_rules": state.get("selection_rules", {}),
                "state_path": str(state_path(session_dir)),
            },
            "regimes": batch_slice,
        }

        batch_filename = cfg_dir / f"batch_{batch_num:04d}.json"
        with open(batch_filename, "w", encoding="utf8") as f:
            json.dump(batch_payload, f, indent=2, default=str)

        saved_batch_paths.append(batch_filename)

        if len(saved_batch_paths) % 20 == 0 or i + batch_size >= len(all_regime_configs):
            logger.info("Written %d batch files...", len(saved_batch_paths))

    state["last_generated"] = {
        "batch_id": int(saved_batch_paths[0].stem.split("_")[-1]) if saved_batch_paths else None,
        "active_block": active_block,
        "profile_name": "surrogate",
        "regime_count": len(all_regime_configs),
        "updated_at": _now_iso(),
    }
    save_coord_descent_state(session_dir, state)

    return saved_batch_paths


def list_pending_config_paths(session_dir: str | Path) -> List[str]:
    session_dir = Path(session_dir)
    cfg_dir = session_dir / "configs"
    results_dir = session_dir / "results"

    if not cfg_dir.exists():
        return []

    pending: List[str] = []

    for cfg_path in sorted(cfg_dir.glob("batch_*.json")):
        if not cfg_path.is_file():
            continue

        try:
            batch_num = int(cfg_path.stem.split("_")[-1])
        except Exception:
            continue

        out_path = results_dir / f"batch_{batch_num:04d}_master_metrics.parquet"
        if out_path.exists():
            continue

        pending.append(str(cfg_path))

    return pending
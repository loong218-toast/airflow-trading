# ccd_config.py

import json
import math
import logging
from pathlib import Path
from typing import List, Union, Any, Optional
from copy import deepcopy
from datetime import datetime

from research.grid import _expand_sl_tp, _prune_by_min_rr
from research.ccd import (
    ensure_coord_descent_state, 
    save_coord_descent_state, 
    state_path
)

from research.ccd_surrogate import suggest_next_candidates

logger = logging.getLogger(__name__)


def _list(x, default):
    if x is None:
        return list(default)
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]

def _write_batch(path: Path, batch_id: int, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump({"batch_id": batch_id, "regimes": rows}, f, indent=2)

def generate_configs(session_dir: Path, run_cfg: dict) -> list[Path]:
    """
    CCD batch generator.

    Role in the pipeline:
    - read the current CCD state
    - determine the active block
    - ask the surrogate for a small ranked candidate pool
    - write those candidates into batch files for the backtest worker

    The batch files are only proposals.
    The worker backtest remains the truth source.
    """
    cfg_dir = session_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    coord_cfg = dict(run_cfg.get("coord_descent", {}) or {})
    state = ensure_coord_descent_state(session_dir, run_cfg)

    block_order = list(
        state.get("block_order")
        or coord_cfg.get("block_order")
        or [
            "signal_structure",
            "trade_management",
            "execution",
        ]
    )

    blocks = dict(state.get("blocks") or coord_cfg.get("blocks") or {})

    # Incumbent is the current best known regime configuration.
    # All probes begin from this base unless the active block allows another move.
    incumbent = dict((state.get("incumbent") or {}).get("regime_cfg") or state.get("seed") or {})
    if not incumbent:
        incumbent = dict(state.get("seed") or {})

    active_block_idx = int(state.get("block_idx", 0) or 0) % max(1, len(block_order))
    active_block = str(
        state.get("active_block")
        or (block_order[active_block_idx] if block_order else "trade_management")
    )

    # Candidate selection is local to the active block.
    # Only the knobs inside the active block are intended to move in the next batch.
    all_regime_configs = suggest_next_candidates(
        session_dir=session_dir,
        run_cfg=run_cfg,
        incumbent_cfg=incumbent,
        active_block=active_block,
        n_candidates=int(coord_cfg.get("surrogate_n_candidates", 60) or 60),
        top_k=int(coord_cfg.get("surrogate_top_k", 30) or 30),
        seed=run_cfg.get("CCD_SEED", 42),
    )

    # Safety fallback: preserve one candidate if the surrogate returns nothing.
    # This prevents the DAG from stalling on an empty batch.
    if not all_regime_configs:
        fallback = deepcopy(incumbent)
        fallback["_candidate_sig"] = ""
        fallback["_active_block"] = active_block
        all_regime_configs = [fallback]

    batch_size = int(coord_cfg.get("BATCH_SIZE", run_cfg.get("BATCH_SIZE", 150)) or 150)
    saved_batch_paths: list[Path] = []

    logger.info("=" * 60)
    logger.info("🧭 SURROGATE-RANKED CCD CONFIG GENERATION")
    logger.info("🔁 Active block: %s", active_block)
    logger.info("📈 Total candidates: %d", len(all_regime_configs))
    logger.info("📦 Batch size: %d", batch_size)
    logger.info("🧩 State: %s", state_path(session_dir))
    logger.info("🧪 Block keys: %s", blocks.get(active_block, []))
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
            logger.info("📝 Written %d batch files...", len(saved_batch_paths))

    state["last_generated"] = {
        "batch_id": int(saved_batch_paths[0].stem.split("_")[-1]) if saved_batch_paths else None,
        "active_block": active_block,
        "profile_name": "surrogate",
        "regime_count": len(all_regime_configs),
        "updated_at": datetime.utcnow().isoformat(),
    }
    save_coord_descent_state(session_dir, state)

    return saved_batch_paths

def list_pending_config_paths(session_dir: str | Path) -> List[str]:
    """
    Return config batch paths that still need processing.

    A batch is considered pending if:
    - configs/batch_XXXX.json exists
    - results/batch_XXXX_master_metrics.parquet does not exist yet
    """
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
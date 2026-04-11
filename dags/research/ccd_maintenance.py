from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_BATCH_MASTER_RE = re.compile(r"^batch_\d{4}_master_metrics\.parquet$")
_BATCH_SUMMARY_RE = re.compile(r"^batch_\d{4}_summary\.json$")
_CFG_SUMMARY_RE = re.compile(r"^cfg_\d+_summary\.json$")


def _safe_remove(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            return True
        if path.exists():
            path.unlink()
            return True
    except Exception as e:
        logger.debug("Failed to remove %s: %s", path, e)
    return False


def cleanup_old_ccd_batch_masters(
    session_dir: str | Path,
    *,
    remove_batch_summaries: bool = True,
    remove_cfg_summaries: bool = False,
    remove_master_parts: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Remove stale CCD batch-level master artifacts before the next CCD cycle.

    Safe to call after:
      - combine_results_to_master()
      - pick_next_incumbent()

    Removes:
      - results/batch_XXXX_master_metrics.parquet
      - optional batch summaries
      - optional cfg summaries
      - optional batch master_parts folders
    """
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"

    report = {
        "session_dir": str(session_dir),
        "results_dir": str(results_dir),
        "deleted": [],
        "kept": [],
        "dry_run": bool(dry_run),
        "removed_count": 0,
    }

    if not results_dir.exists():
        return report

    candidates: List[Path] = []

    for p in results_dir.glob("batch_*_master_metrics.parquet"):
        if _BATCH_MASTER_RE.match(p.name):
            candidates.append(p)

    if remove_batch_summaries:
        for p in results_dir.glob("batch_*_summary.json"):
            if _BATCH_SUMMARY_RE.match(p.name):
                candidates.append(p)

    if remove_cfg_summaries:
        for p in results_dir.glob("cfg_*_summary.json"):
            if _CFG_SUMMARY_RE.match(p.name):
                candidates.append(p)

    if remove_master_parts:
        master_parts_root = results_dir / "master_parts"
        if master_parts_root.exists():
            for p in master_parts_root.glob("batch_*"):
                candidates.append(p)

    for path in candidates:
        if dry_run:
            report["deleted"].append(str(path))
            continue

        if _safe_remove(path):
            report["deleted"].append(str(path))
        else:
            report["kept"].append(str(path))

    report["removed_count"] = len(report["deleted"])
    logger.info(
        "CCD master cleanup finished: removed=%d kept=%d dry_run=%s",
        report["removed_count"],
        len(report["kept"]),
        dry_run,
    )
    return report


def cleanup_ccd_equity_search_artifacts(
    session_dir: str | Path,
    *,
    dry_run: bool = False,
    remove_search_root: bool = True,
) -> Dict[str, Any]:
    """
    Remove equity artifacts generated during CCD search.

    This assumes CCD search equity is written into:
      session_dir/equity_search_partitioned/

    That folder should be separate from the final equity output folder:
      session_dir/equity_partitioned/

    If remove_search_root=True, the whole CCD search equity tree is deleted.
    """
    session_dir = Path(session_dir)
    search_root = session_dir / "equity_search_partitioned"

    report = {
        "session_dir": str(session_dir),
        "search_root": str(search_root),
        "deleted": [],
        "kept": [],
        "dry_run": bool(dry_run),
        "removed_count": 0,
    }

    if not search_root.exists():
        return report

    if dry_run:
        report["deleted"].append(str(search_root))
        report["removed_count"] = 1
        return report

    if remove_search_root:
        if _safe_remove(search_root):
            report["deleted"].append(str(search_root))
        else:
            report["kept"].append(str(search_root))
    else:
        tmp_dir = search_root / "_tmp"
        if tmp_dir.exists() and _safe_remove(tmp_dir):
            report["deleted"].append(str(tmp_dir))

        for p in search_root.glob("era_int=*"):
            if _safe_remove(p):
                report["deleted"].append(str(p))
            else:
                report["kept"].append(str(p))

    report["removed_count"] = len(report["deleted"])
    logger.info(
        "CCD equity cleanup finished: removed=%d dry_run=%s",
        report["removed_count"],
        dry_run,
    )
    return report


def cleanup_ccd_cycle_artifacts(session_dir: str | Path, *, dry_run: bool = False) -> Dict[str, Any]:
    """
    Convenience cleanup for a completed CCD cycle.
    """
    master_report = cleanup_old_ccd_batch_masters(session_dir, dry_run=dry_run)
    equity_report = cleanup_ccd_equity_search_artifacts(session_dir, dry_run=dry_run)

    return {
        "master": master_report,
        "equity": equity_report,
    }
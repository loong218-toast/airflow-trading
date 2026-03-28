from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from etl.cache import _get_cache_root as _io_get_cache_root
from etl.schema import enforce_schema, get_schema

_LOG = logging.getLogger(__name__)

_EQUITY_SHARD_RE = re.compile(
    r"^equity_era_int=(?P<era>\d+)_batch=(?P<batch>\d+)_(?:worker|task)=(?P<unit>\d+)\.parquet$"
)


# ---------------------------------------------------------------------
# small filesystem helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _tmp_dir() -> Path:
    tmp = os.getenv("CACHE_TMP_DIR", None)
    if tmp:
        p = Path(tmp)
    else:
        p = Path(_io_get_cache_root()) / "tmp_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _era_dir_base(months: int, kind: str, era_label: str) -> Path:
    base = Path(_io_get_cache_root()) / f"{int(months)}mo" / kind / f"era_{era_label}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _target_path(kind: str, months: int, era_label: str, regime_id: str) -> Path:
    base = _era_dir_base(months, kind, era_label)
    if kind == "signals":
        return base / f"config_{regime_id}.parquet"
    return base / f"config_{regime_id}_combos.parquet"


def _worker_parts_for_config(kind: str, months: int, era_label: str, regime_id: str) -> List[Path]:
    tmp = _tmp_dir()
    pattern = f"{kind}_config_{regime_id}_era_{era_label}_batch_*.parquet"
    return sorted(tmp.glob(pattern))


def _equity_tmp_dir(session_dir: Path) -> Path:
    tmp_dir = session_dir / "equity_partitioned" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _cleanup_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _cleanup_stage_dir(stage_dir: Path) -> None:
    try:
        if stage_dir.exists():
            for p in stage_dir.glob("*"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                stage_dir.rmdir()
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------
# parquet validation
# ---------------------------------------------------------------------
def _open_parquet(path: Path) -> pq.ParquetFile:
    """
    Open a parquet file. If it is corrupt/truncated, raise with the file path.
    """
    try:
        return pq.ParquetFile(str(path))
    except Exception as e:
        raise RuntimeError(f"Invalid parquet file: {path}") from e


# ---------------------------------------------------------------------
# direct streaming merge with skip-bad behavior
# ---------------------------------------------------------------------
def _merge_parquet_files_skip_bad(
    inputs: List[str],
    out_path: Path,
    kind: str,
    batch_size: int = 65_536,
) -> Dict[str, Any]:
    """
    Merge parquet files one by one.
    - bad/corrupt shards are logged and skipped
    - good shards are streamed into one output file
    - schema stays based on etl.schema.get_schema(kind)

    Returns:
      {"merged_files": int, "skipped_files": int, "skipped": [paths...]}
    """
    if not inputs:
        raise ValueError(f"No inputs for {kind} merge")

    expected_schema = get_schema(kind)
    expected_cols = list(expected_schema.keys())

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    tmp_out = out_path.with_suffix(".tmp.parquet")
    _cleanup_path(tmp_out)

    writer: Optional[pq.ParquetWriter] = None
    merged_files = 0
    skipped: List[str] = []

    try:
        for input_path in inputs:
            p = Path(input_path)

            try:
                pf = _open_parquet(p)
            except Exception as e:
                _LOG.error("Skipping corrupt parquet shard: %s | reason=%s", p, e)
                skipped.append(str(p))
                continue

            try:
                file_cols = list(pf.schema.names)
                if file_cols != expected_cols:
                    raise RuntimeError(
                        f"schema mismatch: expected {expected_cols}, got {file_cols}"
                    )

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(tmp_out),
                        pf.schema_arrow,
                        compression="snappy",
                    )

                for batch in pf.iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([batch], schema=pf.schema_arrow)
                    writer.write_table(table)

                merged_files += 1

            except Exception as e:
                _LOG.error("Skipping bad parquet shard: %s | reason=%s", p, e)
                skipped.append(str(p))
                continue

        if writer is None:
            raise RuntimeError(f"No valid parquet shards found for {kind}")

        writer.close()
        writer = None

        if merged_files == 0:
            raise RuntimeError(f"All parquet shards were invalid for {kind}")

        os.replace(str(tmp_out), str(out_path))
        _LOG.info(
            "Merged %d valid files into %s (kind=%s); skipped %d bad files",
            merged_files,
            out_path.name,
            kind,
            len(skipped),
        )

        return {
            "merged_files": merged_files,
            "skipped_files": len(skipped),
            "skipped": skipped,
        }

    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _cleanup_path(tmp_out)
        raise


# ---------------------------------------------------------------------
# Public: combine_cache
# ---------------------------------------------------------------------
def combine_cache(kind: str, months: int, era_label: str, regime_id: str) -> Optional[Path]:
    """
    Merge all worker temp parts for one (kind, months, era_label, regime_id)
    into the final per-era cache file.
    """
    if kind not in ("signals", "backtest"):
        raise ValueError("kind must be 'signals' or 'backtest'")

    parts = _worker_parts_for_config(kind, months, era_label, regime_id)
    if not parts:
        _LOG.debug(
            "combine_cache: no worker parts found for kind=%s regime=%s era=%s",
            kind, regime_id, era_label
        )
        return None

    target = _target_path(kind, months, era_label, regime_id)
    inputs = [str(p) for p in parts]

    _LOG.info(
        "combine_cache: merging %d worker parts for kind=%s cfg=%s era=%s -> %s",
        len(inputs), kind, regime_id, era_label, target
    )

    result = _merge_parquet_files_skip_bad(inputs, target, kind=kind)

    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            _LOG.debug("combine_cache: failed to remove worker part %s (ignored)", p)

    _LOG.info(
        "combine_cache: finished kind=%s cfg=%s era=%s merged=%d skipped=%d",
        kind, regime_id, era_label, result["merged_files"], result["skipped_files"]
    )
    return target


# ---------------------------------------------------------------------
# Equity merge
# ---------------------------------------------------------------------
def _equity_parts_for_partition(session_dir: Path, partition_key: str, batch_id: Optional[int] = None) -> List[Path]:
    tmp_dir = _equity_tmp_dir(session_dir)
    if batch_id is None:
        candidates = sorted(tmp_dir.glob(f"equity_era_int={partition_key}_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"equity_era_int={partition_key}_batch={int(batch_id)}_*.parquet"))

    parts: List[Path] = []
    for p in candidates:
        if _EQUITY_SHARD_RE.match(p.name):
            parts.append(p)
    return parts


def _append_parquet_file(writer: pq.ParquetWriter, path: Path) -> bool:
    try:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches():
            writer.write_table(batch.to_table())
        return True
    except Exception as e:
        _LOG.error("Skipping corrupt parquet shard: %s | %s", path, e)
        return False


def combine_equity_parts(session_dir: Path, partition_key: str, batch_id: Optional[int] = None) -> Optional[Path]:
    parts = _equity_parts_for_partition(session_dir, partition_key, batch_id=batch_id)
    if not parts:
        return None

    total_parts = len(parts)
    _LOG.info("Starting merge of %d shards for era %s", total_parts, partition_key)

    final_dir = session_dir / "equity_partitioned" / f"era_int={partition_key}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"equity_era_int={partition_key}.parquet"
    tmp_path = final_path.with_suffix(".tmp.parquet")

    _cleanup_path(final_path)
    _cleanup_path(tmp_path)
    writer = None
    merged = 0

    try:
        for idx, p in enumerate(parts):
            if idx % 25 == 0 and idx > 0:
                _LOG.info("[%s] Merged %d/%d shards...", partition_key, idx, total_parts)
            try:
                pf = pq.ParquetFile(str(p))
            except Exception as e:
                _LOG.error("Skipping corrupt parquet shard: %s | %s", p, e)
                continue

            if writer is None:
                writer = pq.ParquetWriter(str(final_path.with_suffix(".tmp.parquet")), pf.schema_arrow, compression="snappy")

            try:
                for batch in pf.iter_batches():
                    # Wrap the single batch in a list to create a Table
                    table = pa.Table.from_batches([batch]) 
                    writer.write_table(table)
                merged += 1
            except Exception as e:
                _LOG.error("Skipping bad parquet shard during merge: %s | %s", p, e)
                continue

        if writer is None or merged == 0:
            return None

        writer.close()
        os.replace(str(final_path.with_suffix(".tmp.parquet")), str(final_path))

        manifest = {
            "parts": [str(p) for p in parts],
            "final": str(final_path),
            "merged_files": merged,
        }
        final_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        return final_path

    except Exception:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        _cleanup_path(final_path.with_suffix(".tmp.parquet"))
        raise


def combine_all_equity_parts(session_dir: str | Path, batch_id: Optional[int] = None) -> List[Path]:
    session_dir = Path(session_dir)
    tmp_dir = _equity_tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob("equity_era_int=*_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"equity_era_int=*_batch={int(batch_id)}_*.parquet"))

    eras: Dict[str, List[Path]] = defaultdict(list)
    for p in candidates:
        m = _EQUITY_SHARD_RE.match(p.name)
        if not m:
            continue
        if batch_id is not None and int(m.group("batch")) != int(batch_id):
            continue
        eras[m.group("era")].append(p)

    outputs: List[Path] = []
    for era in sorted(eras.keys()):
        out = combine_equity_parts(session_dir=session_dir, partition_key=era, batch_id=batch_id)
        if out is not None:
            outputs.append(out)

    return outputs


# ---------------------------------------------------------------------
# Master merge
# ---------------------------------------------------------------------

def merge_batch_master_parts(session_dir: str | Path, batch_id: int) -> Dict[str, Any]:
    """
    Merge worker-local master parts into:
      results/batch_{batch_id:04d}_master_metrics.parquet
    """
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    parts_dir = results_dir / "master_parts" / f"batch_{batch_id:04d}"
    out_path = results_dir / f"batch_{batch_id:04d}_master_metrics.parquet"

    parts = sorted(parts_dir.glob("*.parquet")) if parts_dir.exists() else []

    if not parts:
        df_master = pl.DataFrame([], schema=get_schema("master"))
        df_master.write_parquet(str(out_path), compression="snappy")
        return {
            "status": "empty_batch_master",
            "path": str(out_path),
            "rows": 0,
            "merged_files": 0,
            "skipped_files": 0,
            "skipped": [],
        }

    result = _merge_parquet_files_skip_bad([str(p) for p in parts], out_path, kind="master")

    return {
        "status": "batch_master_merged",
        "path": str(out_path),
        "rows": None,
        "merged_files": result["merged_files"],
        "skipped_files": result["skipped_files"],
        "skipped": result["skipped"],
    }
    
def combine_results_to_master(session_dir: str) -> Dict[str, Any]:
    """
    Streaming combine for final master_metrics.parquet.
    Corrupt batch files are skipped and logged.
    """
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    master_metrics_path = session_dir / "master_metrics.parquet"

    batch_files = sorted(results_dir.glob("batch_*_master_metrics.parquet")) if results_dir.exists() else []
    nonempty: List[Path] = []

    for p in batch_files:
        try:
            meta = pq.ParquetFile(str(p)).metadata
            if meta.num_rows > 0:
                nonempty.append(p)
        except Exception as e:
            _LOG.error("Skipping corrupted master batch file %s: %s", p.name, e)

    if not nonempty:
        df_master = pl.DataFrame([], schema=get_schema("master"))
        df_master.write_parquet(master_metrics_path)
        _LOG.info("⚠️ No batch data found. Created empty master_metrics.")
        return {"status": "complete_no_batch_data", "path": str(master_metrics_path), "rows": 0}

    _LOG.info("Merging %d batch masters...", len(nonempty))
    tmp_master = master_metrics_path.with_suffix(".tmp.parquet")
    _cleanup_path(tmp_master)

    try:
        result = _merge_parquet_files_skip_bad([str(p) for p in nonempty], master_metrics_path, kind="master")
    except Exception:
        _cleanup_path(tmp_master)
        raise

    df_master = pl.read_parquet(master_metrics_path)
    df_master = enforce_schema(df_master, "master", strict=True)

    _LOG.info(
        "✅ Successfully merged master_metrics: merged=%d skipped=%d",
        result["merged_files"],
        result["skipped_files"],
    )
    return {
        "status": "complete_master_only",
        "path": str(master_metrics_path),
        "rows": df_master.height,
        "merged_files": result["merged_files"],
        "skipped_files": result["skipped_files"],
        "skipped": result["skipped"],
    }
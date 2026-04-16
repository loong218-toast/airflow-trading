from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from common.schema import enforce_schema, get_schema
from research.grid_row_builders import build_trade_ml_rows_from_backtest

logger = logging.getLogger(__name__)

_TRADE_ML_SHARD_RE = re.compile(
    r"^trade_ml_era_int=(?P<era>\d+)_batch=(?P<batch>\d+)_(?:worker|task)=(?P<unit>\d+)(?:_.*)?\.parquet$"
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _tmp_dir(session_dir: Path) -> Path:
    p = Path(session_dir) / "trade_ml_partitioned" / "_tmp"
    _ensure_dir(p)
    return p


def _cleanup_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _worker_id() -> str:
    return os.getenv("AIRFLOW_MAP_INDEX", os.getenv("MAP_INDEX", "0"))


def _atomic_write_parquet(df: pl.DataFrame, path: Path, schema_type: str = "trade_ml") -> None:
    path = Path(path)
    _ensure_dir(path.parent)

    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    tmp_p = Path(tmp)

    try:
        df = enforce_schema(df, schema_type, strict=True)
        df.write_parquet(str(tmp_p), compression="snappy")
        os.replace(str(tmp_p), str(path))
    finally:
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except Exception:
                pass


def stage_trade_ml_part(
    session_dir: str | Path,
    batch_id: int,
    era_int: int,
    df: pl.DataFrame,
) -> Optional[Path]:
    """
    Write one trade_ml shard for a given era/batch.

    This writes into the worker temp area first.
    The final merge step will combine all shards into one parquet per era.
    """
    if df is None or df.height == 0:
        return None

    session_dir = Path(session_dir)
    tmp_dir = _tmp_dir(session_dir)

    wid = _worker_id()
    stamp = int(time.time() * 1000)
    out_path = tmp_dir / (
        f"trade_ml_era_int={int(era_int)}"
        f"_batch={int(batch_id):04d}"
        f"_worker={wid}_{stamp}_{uuid.uuid4().hex}.parquet"
    )

    _atomic_write_parquet(df, out_path, "trade_ml")
    return out_path

def _stage_trade_ml_era_rows(
    df_main: pl.DataFrame,
    main_close_arr: np.ndarray,
    main_high_arr: np.ndarray,
    main_low_arr: np.ndarray,
    main_time_ns_arr: np.ndarray,
    side_sig_idxs_all: np.ndarray,
    side_sig_times_all: np.ndarray,
    start_ns: int,
    end_ns: int,
    side_flag: int,
    sl_val: float,
    tp_val: float,
    use_limit_entry: bool,
    limit_order_expiry_bars: int,
    trade_window_interval: int,
    regime_id: int,
    era_int: int,
    backtest_res: Dict,
    regime_cfg: dict,
    run_cfg: dict,
    session_dir: Path,
    batch_id: int,
) -> int:
    """
    Build and stage trade-level rows for one era.

    This is where SL_hit / TP_hit belong.
    It keeps those fields out of equity while still writing durable analysis rows.
    """
    era_mask = (side_sig_times_all >= start_ns) & (side_sig_times_all < end_ns)
    era_sig_idxs = side_sig_idxs_all[era_mask]

    if era_sig_idxs.size == 0:
        return 0

    trade_ml_df = build_trade_ml_rows_from_backtest(
        df_main=df_main,
        main_close_arr=main_close_arr,
        main_high_arr=main_high_arr,
        main_low_arr=main_low_arr,
        main_time_ns_arr=main_time_ns_arr,
        sig_idxs=era_sig_idxs,
        side_flag=side_flag,
        sl_val=float(sl_val),
        tp_val=float(tp_val),
        use_limit_entry=bool(use_limit_entry),
        limit_order_expiry_bars=int(limit_order_expiry_bars),
        trade_window_interval=int(trade_window_interval),
        regime_id=int(regime_id),
        era_int=int(era_int),
        backtest_res=backtest_res,
        regime_cfg=regime_cfg,
        run_cfg=run_cfg,
    )

    if trade_ml_df is None or trade_ml_df.height == 0:
        return 0

    try:
        stage_trade_ml_part(
            session_dir=session_dir,
            batch_id=int(batch_id),
            era_int=int(era_int),
            df=trade_ml_df,
        )
    except Exception as e:
        logger.error(
            "stage_trade_ml_part failed cfg=%s era=%s side=%s SL=%.4f TP=%.4f: %s",
            regime_id,
            era_int,
            side_flag,
            float(sl_val),
            float(tp_val),
            e,
        )
        return 0

    return int(trade_ml_df.height)

def _trade_ml_parts_for_partition(
    session_dir: Path,
    partition_key: str,
    batch_id: Optional[int] = None,
) -> List[Path]:
    tmp_dir = _tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob(f"trade_ml_era_int={partition_key}_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"trade_ml_era_int={partition_key}_batch={int(batch_id):04d}_*.parquet"))

    parts: List[Path] = []
    for p in candidates:
        if _TRADE_ML_SHARD_RE.match(p.name):
            parts.append(p)

    return parts


def _open_parquet(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(str(path))
    except Exception as e:
        raise RuntimeError(f"Invalid parquet file: {path}") from e


def _merge_parquet_files_skip_bad(
    inputs: List[str],
    out_path: Path,
    kind: str,
    batch_size: int = 65_536,
) -> Dict[str, Any]:
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
                logger.error("Skipping corrupt parquet shard: %s | reason=%s", p, e)
                skipped.append(str(p))
                continue

            try:
                file_cols = list(pf.schema.names)
                if file_cols != expected_cols:
                    raise RuntimeError(f"schema mismatch: expected {expected_cols}, got {file_cols}")

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
                logger.error("Skipping bad parquet shard: %s | reason=%s", p, e)
                skipped.append(str(p))
                continue

        if writer is None:
            raise RuntimeError(f"No valid parquet shards found for {kind}")

        writer.close()
        writer = None

        if merged_files == 0:
            raise RuntimeError(f"All parquet shards were invalid for {kind}")

        os.replace(str(tmp_out), str(out_path))
        logger.info(
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


def combine_trade_ml_parts(
    session_dir: str | Path,
    partition_key: str,
    batch_id: Optional[int] = None,
) -> Optional[Path]:
    """
    Merge worker trade_ml parts for one era into one final parquet file.

    Output layout:
      trade_ml_partitioned/era_int=<era>/
        trade_ml_era_int=<era>.parquet
        trade_ml_era_int=<era>.manifest.json
    """
    session_dir = Path(session_dir)
    parts = _trade_ml_parts_for_partition(session_dir, partition_key, batch_id=batch_id)
    if not parts:
        return None

    final_dir = session_dir / "trade_ml_partitioned" / f"era_int={partition_key}"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_path = final_dir / f"trade_ml_era_int={partition_key}.parquet"
    tmp_path = final_path.with_suffix(".tmp.parquet")

    _cleanup_path(final_path)
    _cleanup_path(tmp_path)

    result = _merge_parquet_files_skip_bad([str(p) for p in parts], tmp_path, kind="trade_ml")

    os.replace(str(tmp_path), str(final_path))

    manifest = {
        "parts": [str(p) for p in parts],
        "final": str(final_path),
        "merged_files": int(result["merged_files"]),
        "skipped_files": int(result["skipped_files"]),
        "skipped": result["skipped"],
    }
    manifest_path = final_dir / f"trade_ml_era_int={partition_key}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")

    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    logger.info(
        "combine_trade_ml_parts: era=%s merged=%d skipped=%d",
        partition_key,
        result["merged_files"],
        result["skipped_files"],
    )
    return final_path


def combine_all_trade_ml_parts(session_dir: str | Path, batch_id: Optional[int] = None) -> List[Path]:
    """
    Merge every era partition for trade_ml.
    """
    session_dir = Path(session_dir)
    tmp_dir = _tmp_dir(session_dir)

    if batch_id is None:
        candidates = sorted(tmp_dir.glob("trade_ml_era_int=*_batch=*.parquet"))
    else:
        candidates = sorted(tmp_dir.glob(f"trade_ml_era_int=*_batch={int(batch_id):04d}_*.parquet"))

    eras: Dict[str, List[Path]] = {}
    for p in candidates:
        m = _TRADE_ML_SHARD_RE.match(p.name)
        if not m:
            logger.debug("Skipping non-matching trade_ml shard name: %s", p.name)
            continue
        if batch_id is not None and int(m.group("batch")) != int(batch_id):
            continue
        era = m.group("era")
        eras.setdefault(era, []).append(p)

    outputs: List[Path] = []
    for era in sorted(eras.keys()):
        out = combine_trade_ml_parts(session_dir=session_dir, partition_key=era, batch_id=batch_id)
        if out is not None:
            outputs.append(out)

    return outputs


__all__ = [
    "stage_trade_ml_part",
    "combine_trade_ml_parts",
    "combine_all_trade_ml_parts",
]
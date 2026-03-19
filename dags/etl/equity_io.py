# equity_io.py
from __future__ import annotations

from pathlib import Path
import os
import time
import secrets
import json
from typing import Optional

import polars as pl

from etl.schema import get_schema, enforce_schema
from etl.merge_utils import merge_parquet_files_streaming
from etl.io_utils import _atomic_write_parquet

def combine_equity_parts(session_dir: Path, partition_key: str, batch_id: Optional[int] = None) -> Optional[Path]:
    """
    Merge worker files for one era into a single final equity parquet.
    Expects worker files like:
      equity_era_int=20230901_batch=2_worker=0.parquet
    """
    tmp_dir = session_dir / "equity_partitioned" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if batch_id is None:
        pattern = f"equity_era_int={partition_key}_batch=*_worker=*.parquet"
    else:
        pattern = f"equity_era_int={partition_key}_batch={int(batch_id)}_worker=*.parquet"

    parts = sorted(tmp_dir.glob(pattern))
    if not parts:
        return None

    final_dir = session_dir / "equity_partitioned" / f"era_int={partition_key}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"equity_era_int={partition_key}.parquet"

    merge_parquet_files_streaming([str(p) for p in parts], final_path, kind="equity")

    merged_schema = pl.read_parquet_schema(str(final_path))
    missing = set(get_schema("equity").keys()) - set(merged_schema.keys())
    if missing:
        raise RuntimeError(f"Post-merge: missing columns in final equity {missing}")

    manifest = {"parts": [str(p) for p in parts], "final": str(final_path)}
    final_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    for p in parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    return final_path
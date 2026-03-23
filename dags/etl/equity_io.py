from __future__ import annotations

from pathlib import Path
from typing import Optional

from etl.merge_utils import combine_equity_parts as _combine_equity_parts


def combine_equity_parts(session_dir: Path, partition_key: str, batch_id: Optional[int] = None):
    return _combine_equity_parts(session_dir=session_dir, partition_key=partition_key, batch_id=batch_id)
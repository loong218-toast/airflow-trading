# compute_code_version.py
from pathlib import Path
import hashlib
import json
import sys

def _sha256_of_str(s: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(s.encode("utf8"))
    return h.hexdigest()

def code_version_from_files(file_paths):
    parts=[]
    for p in file_paths:
        try:
            st = Path(p).stat()
            parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
        except Exception:
            parts.append(f"{p}:missing")
    return _sha256_of_str("|".join(parts))

files = [
    "etl/grid.py",
    "etl/feature_helpers.py",
    "etl/backtest.py"
]
files_abs = [str(Path(f).resolve()) for f in files]
cv = code_version_from_files(files_abs)
print("Computed code_version:", cv)
print("Files used:", json.dumps(files_abs, indent=2))
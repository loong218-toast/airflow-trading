# /tmp/debug_mem.py
import os, sys, time, gc
import psutil
import polars as pl
from collections import Counter

proc = psutil.Process(os.getpid())
print("PID", proc.pid, "cmdline:", proc.cmdline())
print("RSS MB:", proc.memory_info().rss / 1024**2)
print("VMS MB:", proc.memory_info().vms / 1024**2)
print("System mem %:", psutil.virtual_memory().percent)

# show Polars file cache (if present)
try:
    import pyarrow as pa
    print("pyarrow version", pa.__version__)
except Exception:
    pass

# Python-level object summary (counts by type)
gc.collect()
objs = gc.get_objects()
types = Counter(type(o).__name__ for o in objs)
for t, cnt in types.most_common(30):
    print(f"{t:30s} {cnt}")

# list tmp_cache small summary
tmpdir = "/opt/airflow/airflow-trading/data_lake/cache/tmp_cache"
if os.path.isdir(tmpdir):
    files = os.listdir(tmpdir)
    print("tmp_cache files:", len(files))
    sizes = []
    for f in files[:200]:
        try:
            p = os.path.join(tmpdir, f)
            sizes.append((f, os.path.getsize(p)))
        except Exception:
            pass
    print("first 20 tmp parts (name,size):")
    for n,s in sizes[:20]:
        print(n, s)
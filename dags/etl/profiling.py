# profiling.py
import cProfile
import os
import time
from contextlib import contextmanager
from pathlib import Path
from airflow.sdk import Variable

def profile_enabled() -> bool:
    val = Variable.get("enable_cprofile", default="0").lower()
    return val in ("1", "true", "yes")

@contextmanager
def maybe_profile(section_name: str, out_dir: str = "/opt/airflow/profiles"):
    if not profile_enabled():
        yield None
        return

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        filename = f"{section_name}_{ts}_pid{pid}.prof"
        profiler.dump_stats(str(out_path / filename))
# profiling.py
import cProfile
import os
import time
from contextlib import contextmanager
from pathlib import Path
from airflow.sdk import Variable

def profile_enabled() -> bool:
    val = Variable.get("grid_enable_cprofile", default="0").lower()
    return val in ("1", "true", "yes")

@contextmanager
def maybe_profile(section_name: str, out_dir: str = "/opt/airflow/airflow-trading/profiles"):
    if not profile_enabled():
        yield None
        return

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        try:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            ts = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{section_name}_{ts}_pid{os.getpid()}.prof"
            full_path = out_path / filename
            
            profiler.dump_stats(str(full_path))
            print(f"✅ Profiler dumped to: {full_path}")
        except Exception as e:
            # This will show up in your Airflow worker logs if it fails!
            print(f"❌ PROFILER ERROR: Could not write to {out_dir}. Error: {e}")
# inspect_session.py
import sys
import pathlib
import json
import pyarrow.parquet as pq
import pandas as pd

# UPDATE this to your session path if different
session = pathlib.Path(r"C:\Users\Owner\airflow-trading\data_lake\Opt_Session_20260309_161208_01")

def count_json_summaries(results_dir):
    return len(list(results_dir.glob("cfg_*_summary.json")))

def list_master_parquets(results_dir):
    out = []
    for p in sorted(results_dir.glob("batch_*_master_metrics.parquet")):
        try:
            size_mb = p.stat().st_size / (1024*1024)
            rows = pq.read_metadata(str(p)).num_rows if p.exists() and p.stat().st_size>0 else 0
        except Exception as e:
            rows = f"ERR:{e}"
        out.append((p.name, round(size_mb,2), rows))
    return out

def sum_master_rows(results_dir):
    total = 0
    for p in results_dir.glob("batch_*_master_metrics.parquet"):
        try:
            if p.exists() and p.stat().st_size>0:
                total += int(pq.read_metadata(str(p)).num_rows)
        except Exception:
            pass
    return total

def count_equity_parts(equity_dir):
    files = list(equity_dir.rglob("*.parquet")) if equity_dir.exists() else []
    total_rows = 0
    for f in files:
        try:
            if f.stat().st_size>0:
                total_rows += int(pq.read_metadata(str(f)).num_rows)
        except Exception:
            pass
    return len(files), total_rows

def count_backtest_cache(cache_backtest_dir):
    files = list(cache_backtest_dir.rglob("**/*.parquet")) if cache_backtest_dir.exists() else []
    return len(files)

def sample_signals(signals_dir, n=5):
    files = sorted(signals_dir.glob("*.parquet")) if signals_dir.exists() else []
    if not files:
        return "no signal parquet files"
    # read first non-empty file with pyarrow -> pandas
    for f in files[:n]:
        try:
            if f.stat().st_size > 0:
                df = pq.read_table(str(f)).to_pandas()
                return {"file": str(f.name), "rows": len(df), "columns": list(df.columns)[:50]}
        except Exception:
            continue
    return "no readable signal parquet found"

results_dir = session / "results"
equity_dir = session / "equity_partitioned"
signals_dir = pathlib.Path(r"C:\Users\Owner\airflow-trading\data_lake\cache\signals")
cache_backtest = pathlib.Path(r"C:\Users\Owner\airflow-trading\data_lake\cache\backtest")

print("session:", session)
print("cfg summaries:", count_json_summaries(results_dir))
print("master parquet files (name, size_MB, rows):")
for info in list_master_parquets(results_dir):
    print("  ", info)
print("sum master rows:", sum_master_rows(results_dir))
parts_count, parts_rows = count_equity_parts(equity_dir)
print("equity partition: files=", parts_count, "rows(total)=", parts_rows)
print("backtest cache parquet files count:", count_backtest_cache(cache_backtest))
print("signals cache sample:", sample_signals(signals_dir))
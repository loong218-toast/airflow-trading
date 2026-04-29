from pathlib import Path
import pyarrow.parquet as pq

root = Path("C:\\Users\\Owner\\airflow-trading\\data_lake\\Opt_Session_20260423_010923_01\\trade_ml_partitioned\\_tmp")

files = sorted(root.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True)

for p in files[:20]:
    try:
        meta = pq.ParquetFile(str(p)).metadata
        print(f"{p.stat().st_size / 1024 / 1024:.1f} MB | rows={meta.num_rows} | {p.name}")
    except Exception as e:
        print(f"BAD | {p.name} | {e}")
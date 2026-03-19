import pyarrow.parquet as pq
import random
import pandas as pd

file_path = r"C:\Users\Owner\airflow-trading\data_lake\Opt_Session_20260317_114035_01\equity_partitioned\era_int=20230901\batch_0000_20230901_1773747688349.parquet"

def get_pyarrow_samples(path, n=5):
    try:
        # 1. Open the file metadata only (Zero RAM)
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        
        if total_rows == 0:
            print("File is empty.")
            return None

        # 2. Pick 5 random indices
        # We use a small range/slice to keep it fast
        random_start = random.randint(0, max(0, total_rows - n))
        
        # 3. Read only the specific slice
        # This is extremely fast because PyArrow uses memory mapping
        table = parquet_file.read_row_group(0) if parquet_file.num_row_groups == 1 else parquet_file.read()
        
        # Convert just the slice to a pandas/polars dataframe for viewing
        sample_df = table.slice(random_start, n).to_pandas()
        
        return sample_df

    except Exception as e:
        print(f"PyArrow Error: {e}")
        return None

# Execute
df = get_pyarrow_samples(file_path)

if df is not None:
    print(f"\n--- PyArrow Random 5 Samples (Offset: {random.getstate()[1][0]}) ---")
    print(df.to_string())
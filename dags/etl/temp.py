import pyarrow.fs as fs
import pyarrow.parquet as pq
import os

# Your target path
target_path = r"C:\Users\Owner\airflow-trading\data_lake\Opt_Session_20260324_114114_01\equity_partitioned\_tmp"

# 1. Initialize Local File System
local = fs.LocalFileSystem()

# 2. Get file info
selector = fs.FileSelector(target_path, recursive=False)
files_info = local.get_file_info(selector)

# Filter for files only
files = [f for f in files_info if f.type == fs.FileType.File]
file_count = len(files)

print(f"Total files in directory: {file_count}")

if file_count > 0:
    # --- FIX: Pick the first element from the list ---
    sample_file_path = files[0].path 
    
    print(f"\n--- Sample File Metadata ---")
    print(f"File: {os.path.basename(sample_file_path)}")
    
    try:
        # 3. Read Metadata & Schema
        parquet_file = pq.ParquetFile(sample_file_path)
        
        print(f"Row Count: {parquet_file.metadata.num_rows}")
        print("-" * 30)
        print("Schema:")
        print(parquet_file.schema.to_arrow_schema())
        
    except Exception as e:
        print(f"Could not read the file as Parquet: {e}")
else:
    print("No files found in the specified directory.")
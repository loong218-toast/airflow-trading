# df_main 2017 to 2025

import pandas as pd
import os

# 1. Load data
p = r"C:\Users\Owner\airflow-trading\analysis\Backtest_results\BTC\@Main\BTC_5m_processed_main.parquet"
df = pd.read_parquet(p)

# 2. Sanitize Column Names
# Postgres doesn't like '%' in column names unless they are double-quoted. 
# Renaming is much safer.
df = df.rename(columns={"%K":"pct_k","%D":"pct_d","%DSlow":"pct_dslow"})

# 3. Add pair and ensure Datetime format
df['pair'] = 'XXBTZUSD'
# Force the timestamp to ISO format so Postgres doesn't trip on the import
df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

# 4. Align Columns with your SQL Table Schema
cols = ['pair','time','time_ns','open','high','low','close','volume',
        'ma_hour','ma_day','ma_week','ma_5min',
        'price_ma_gap','ma_gap_a','ma_gap_b','ma_gap_c',
        'pct_k','pct_d','pct_dslow','era_int','era_day_int']

# Handle missing columns (if any)
for col in cols:
    if col not in df.columns:
        df[col] = None

df = df[cols]

# 5. Write TSV
target_dir = os.path.dirname(p) 
out = os.path.join(target_dir, "df_main_for_copy.tsv")
df.to_csv(out, sep="\t", index=False, na_rep='\\N')
print(f"✅ Success: {out} is ready for Postgres COPY.")
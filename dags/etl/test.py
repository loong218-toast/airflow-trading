import pyarrow.csv as pv
import polars as pl

path = r"C:\Users\Owner\airflow-trading\analysis\Backtest_results\BTC\@Main\BTCUSD_1m_Binance.csv"

reader = pv.open_csv(path)
schema = reader.schema

#print(schema)
#print(schema.names)   # column names only

import sys
print(sys.executable)
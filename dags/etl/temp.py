import pandas as pd
df = pd.DataFrame({"time_ns":})

print(type(df["time_ns"].iloc))     # Output: <class 'pandas.core.indexing._iLocIndexer'>
print(type(df["time_ns"].iloc))  # Output: <class 'numpy.int64'> (The actual number!)
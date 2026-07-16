# test_temp.py

import MetaTrader5 as mt5

print("INIT:", mt5.initialize())

symbols = mt5.symbols_get()

for s in symbols:
    if any(x in s.name.upper() for x in ["UK", "FTSE", "100"]):
        print("FOUND:", s.name)

print("\nALL SYMBOLS:")
for s in symbols:
    print(s.name)

mt5.shutdown()
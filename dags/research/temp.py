import MetaTrader5 as mt5

mt5.initialize()
symbols = mt5.symbols_get()
print([s.name for s in symbols if "UK" in s.name or "FTSE" in s.name])

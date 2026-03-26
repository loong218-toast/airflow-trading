# etl/kraken_api.py
import time
import requests
import pandas as pd

API_BASE = "https://api.kraken.com/0/public"

def get_asset_pairs():
    url = f"{API_BASE}/AssetPairs"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()["result"]

def validate_asset_exists(pair: str, market_type: str):
    """
    Aggressive validation: Raises ValueError if the pair is not tradeable.
    This prevents the DAG from proceeding with 'ghost' symbols.
    """
    if market_type == "future":
        valid_symbols = get_tradeable_futures_symbols()
        # Kraken Futures symbols are often like 'PI_XBTUSD' or 'PF_SOLUSD'
        if pair.upper() not in [s.upper() for s in valid_symbols]:
            raise ValueError(f"❌ CRITICAL: Future pair '{pair}' is NOT tradeable on Kraken Futures.")
    else:
        # For Spot, check against AssetPairs
        asset_pairs = get_asset_pairs()
        if pair.upper() not in [p.upper() for p in asset_pairs.keys()]:
            raise ValueError(f"❌ CRITICAL: Spot pair '{pair}' does not exist on Kraken.")
    
    print(f"✅ Validation Passed: {pair} ({market_type})")
    return True

def get_futures_resolution(minutes: int) -> str:
    """
    Translates integer minutes to Kraken Futures resolution strings.
    """
    mapping = {
        1: "1m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1h",
        240: "4h",    # 4 hours
        720: "12h",   # 12 hours
        1440: "1d",   # Daily
        10080: "1w"   # Weekly
    }
    # Default to 5m if the integer isn't in the list
    return mapping.get(minutes, "5m")

def get_tradeable_futures_symbols():
    """
    Fetches all active, tradeable symbols from Kraken Futures.
    Useful for validating symbols before making OHLC requests.
    """
    url = "https://futures.kraken.com/derivatives/api/v3/instruments"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    
    if data.get("result") != "success":
        return []

    # Filter for tradeable instruments only
    # We can also filter by 'tradfi': True if we only want things like Gold
    active_symbols = [
        inst["symbol"] for inst in data["instruments"] 
        if inst.get("tradeable") is True
    ]
    return active_symbols

def fetch_ohlc(pair: str, interval: int, since: int = None, market_type: str = "spot"):
    """
    pair: Kraken pair code like 'XXBTZUSD' or 'XBTUSD' depending on API mapping
    interval: minutes (1,5,15,30,60,240,1440,10080,21600 typical)
    since: unix epoch seconds (optional) — note Kraken returns up to 720 latest rows
    returns: (df, last) where df columns = [time, open, high, low, close, vwap, volume, count]
    """
    params = {"pair": pair, "interval": interval}
    if since:
        params["since"] = int(since)

    if market_type == "xstock":
        params["asset_class"] = "tokenized_asset"
        
    url = f"{API_BASE}/OHLC"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError("Kraken OHLC error: %s" % data["error"])

    result = data.get("result", {})
    available_keys = [k for k in result.keys() if k != "last"]
    if not available_keys:
        print(f"⚠️ No data keys found for {pair}")
        return pd.DataFrame(), 0

    pair_key = available_keys[0]
    rows = data["result"][pair_key]
    last = int(data["result"].get("last", 0))
    if not rows:
        return pd.DataFrame(), last
    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ns"] = df["time"].values.astype("int64")
    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df["close"] = df["close"].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    return df, last


# Add this to etl/kraken_api.py
def fetch_futures_ohlc(symbol: str, interval: int):
    # 1. Self-Correction / Validation
    # In a production scenario, you could cache get_tradeable_futures_symbols()
    # to avoid calling it every single time.
    valid_symbols = get_tradeable_futures_symbols()
    
    # Check if the symbol exists (case-insensitive check)
    match = next((s for s in valid_symbols if s.lower() == symbol.lower()), None)
    
    if not match:
        print(f"❌ {symbol} is not a valid tradeable instrument on Kraken Futures.")
        print(f"Available symbols (sample): {valid_symbols[:5]}")
        return pd.DataFrame(), 0

    # 2. Proceed with the correct symbol casing found in the instruments list
    res_str = get_futures_resolution(interval)
    url = f"https://futures.kraken.com/api/charts/v1/trade/{match.lower()}/{res_str}"
    
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    
    if "candles" not in data:
        return pd.DataFrame(), 0

    df = pd.DataFrame(data["candles"])
    # Futures API returns time in milliseconds
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["time_ns"] = df["time"].values.astype("int64")
    
    # Cast for DB compatibility
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float32")
        
    return df, int(df["time_ns"].max() if not df.empty else 0)


def fetch_futures_tickers(symbol: str = None):
    """
    Fetches ticker data (change24h, funding rate, spread) for Kraken Futures.
    If symbol is None, returns all tickers.
    """
    url = "https://futures.kraken.com/derivatives/api/v3/tickers"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("result") != "success":
        return pd.DataFrame()

    df = pd.DataFrame(data["tickers"])
    
    # 1. Calculate Spread
    # Spread = Ask - Bid
    df["spread"] = df["ask"] - df["bid"]
    
    # 2. Funding Rate (Relative %)
    # Note: 'fundingRate' in the API is the absolute rate. 
    # To get the % usually displayed: (absolute_funding / mark_price) * 100
    if "fundingRate" in df.columns:
        df["funding_rate_pct"] = (df["fundingRate"] / df["markPrice"]) * 100
    
    # 3. 24h Change
    # Field 'change24h' is already provided as a percentage by the API.
    
    if symbol:
        df = df[df["symbol"].str.upper() == symbol.upper()]
        
    # Return useful subset
    cols = ["symbol", "last", "change24h", "spread", "fundingRate", "funding_rate_pct", "markPrice"]
    return df[cols] if not df.empty else df

def fetch_historical_funding(symbol: str):
    import requests
    import pandas as pd
    
    url = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"
    params = {"symbol": symbol}
    
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    
    if data.get("result") != "success":
        return pd.DataFrame()

    df = pd.DataFrame(data["rates"])
    
    # 1. Standardize Timestamps
    df["time"] = pd.to_datetime(df["timestamp"], utc=True)
    df["time_ns"] = df["time"].values.astype("int64")
    
    # 2. Map Columns for DB
    df = df.rename(columns={
        "fundingRate": "funding_rate_abs",
        "relativeFundingRate": "funding_rate_rel"
    })
    
    return df[["time", "time_ns", "funding_rate_abs", "funding_rate_rel"]]
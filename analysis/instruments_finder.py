import requests
import pandas as pd

def get_kraken_instruments():
    url = "https://futures.kraken.com/derivatives/api/v3/instruments"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get("result") != "success":
            print("Error: API returned unsuccessful result.")
            return
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(data["instruments"])
        
        # --- 1. Filter for Perpetuals (The ones with Funding Rates) ---
        perps = df[df['type'] == 'flexible_futures']
        
        # --- 2. Filter for Indices (The benchmark price) ---
        indices = df[df['symbol'].str.startswith('PI_')]
        
        print(f"Total Instruments Found: {len(df)}")
        print(f"Tradeable Perpetuals: {len(perps)}")
        print("-" * 30)
        
        # Display the first few Perpetuals with key scaling info
        print("\nTOP PERPETUALS FOR YOUR PIPELINE:")
        cols_to_show = ['symbol', 'pair', 'tickSize', 'contractSize', 'tradeable']
        print(perps[cols_to_show].head(10).to_string(index=False))
        
        # --- 3. Identify Multi-Collateral Categories ---
        # This helps you group assets in your DB by Sector (DeFi, Layer 1, etc.)
        if 'category' in df.columns:
            print("\nASSET SECTORS (CATEGORIES):")
            print(perps['category'].value_counts())
            
        return perps
        
    except Exception as e:
        print(f"Connection failed: {e}")

def get_tradfi_instruments():
    url = "https://futures.kraken.com/derivatives/api/v3/instruments"
    data = requests.get(url).json()
    df = pd.DataFrame(data["instruments"])

    # 1. Get xStocks (Tokenized Equities/ETFs)
    # They are usually identified by 'xStocks' category or 'tradfi': True
    xstocks = df[df['category'] == 'xStocks'].copy()

    # 2. Get Forex (Currency Pairs like EUR, GBP)
    # These often have the 'Forex' category
    forex = df[df['category'] == 'Forex'].copy()

    print(f"--- Found {len(xstocks)} xStocks ---")
    print(xstocks[['symbol', 'pair', 'tickSize']].head(10))

    print(f"\n--- Found {len(forex)} Forex Pairs ---")
    print(forex[['symbol', 'pair', 'tickSize']].head())

    return xstocks, forex

def find_gold_assets():
    url = "https://futures.kraken.com/derivatives/api/v3/instruments"
    data = requests.get(url).json()
    df = pd.DataFrame(data["instruments"])

    # Search for anything containing 'GOLD', 'GLD', 'XAU', or 'PAXG'
    search_terms = ['GOLD', 'GLD', 'XAU', 'PAXG']
    pattern = '|'.join(search_terms)
    
    gold_df = df[df['symbol'].str.contains(pattern, case=False)].copy()

    print("\n--- GOLD ASSET DISCOVERY ---")
    if not gold_df.empty:
        # We categorize them based on the 'type' and 'symbol'
        print(gold_df[['symbol', 'pair', 'type', 'category', 'tradeable']])
    else:
        print("No specific Gold instruments found with those keywords.")

if __name__ == "__main__":
    find_gold_assets()

    # This runs the discovery logic
    perpetual_list = get_kraken_instruments()
    get_tradfi_instruments()
    
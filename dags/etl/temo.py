import requests
import pandas as pd

def check_kraken_inventory():
    url = "https://api.kraken.com/0/public/AssetPairs"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get("error"):
            print(f"API Error: {data['error']}")
            return

        pairs = data["result"]
        df = pd.DataFrame.from_dict(pairs, orient='index')

        # Total Count
        total_pairs = len(df)
        
        # Breakdown by Status (Online, Cancel Only, etc.)
        status_counts = df['status'].value_counts()
        
        # Separate "XStocks" (Tokenized Assets) from standard Currencies
        # Note: 'aclass_base' identifies the asset class
        xstocks = df[df['aclass_base'] == 'tokenized_asset']
        spot_pairs = df[df['aclass_base'] == 'currency']

        print("--- KRAKEN INVENTORY SUMMARY ---")
        print(f"Total Asset Pairs: {total_pairs}")
        print(f"Online & Tradable: {status_counts.get('online', 0)}")
        print(f"Tokenized Assets (XStocks): {len(xstocks)}")
        print(f"Standard Spot Pairs: {len(spot_pairs)}")
        print("-" * 32)
        
        if len(xstocks) > 0:
            print(f"Example XStocks: {', '.join(xstocks['wsname'].head(5).tolist())}")
        
        print(f"Example Spot: {', '.join(spot_pairs['wsname'].head(5).tolist())}")
        
        # Optional: Save to CSV for your own research on your IdeaPad
        # df.to_csv("kraken_pairs_full_list.csv")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    check_kraken_inventory()
import requests
import pandas as pd

def get_all_kraken_assets():
    base_url = "https://api.kraken.com/0/public/AssetPairs"
    
    # We fetch both classes to be sure
    classes = ["currency", "tokenized_asset"]
    all_data = []

    for aclass in classes:
        params = {'aclass_base': aclass}
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if not data.get("error"):
            # Convert the result dictionary to a list of dicts
            pairs = data["result"]
            for pair_id, info in pairs.items():
                info['pair_id'] = pair_id # Keep the API ID
                all_data.append(info)

    df = pd.DataFrame(all_data)
    
    # Filter for the ones you actually want to see
    xstocks = df[df['aclass_base'] == 'tokenized_asset']
    
    print(f"Total xStocks found: {len(xstocks)}")
    if not xstocks.empty:
        # Displaying 'wsname' as it's the most readable for your Docker ingestor
        print("First 10 xStocks Names:")
        print(xstocks[['wsname', 'altname']].head(10))
    
    return df

df_full = get_all_kraken_assets()
import requests

def check_kraken_pair(pair_name: str):
    url = "https://api.kraken.com/0/public/AssetPairs"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            print(f"❌ API Error: {data['error']}")
            return

        result = data.get("result", {})
        
        # Kraken pairs are keys in the 'result' object. 
        # We check the key itself, the 'altname', and the 'wsname'.
        found = False
        for key, info in result.items():
            altname = info.get("altname", "")
            wsname = info.get("wsname", "")
            
            if pair_name.upper() in [key.upper(), altname.upper(), wsname.upper()]:
                print(f"✅ Found Match!")
                print(f"   - Internal ID: {key}")
                print(f"   - Alt Name:    {altname}")
                print(f"   - WS Name:     {wsname}")
                print(f"   - Status:      {info.get('status')}")
                found = True
                break
        
        if not found:
            print(f"❓ Pair '{pair_name}' not found on Kraken.")

    except Exception as e:
        print(f"⚠️ Request failed: {e}")

# Test your problematic pairs
check_kraken_pair("XAUTUSD")
check_kraken_pair("XETHZUSD")
def btc_lotsize(balance_usd, risk_percent, entry_price, stop_price):
    """
    balance_usd: account balance in USD
    risk_percent: e.g. 0.2 means 0.2%
    entry_price: BTC entry price
    stop_price: BTC stop loss price
    """

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        raise ValueError("Entry and stop must be different")

    risk_usd = balance_usd * (risk_percent / 100.0)
    btc_size = risk_usd / stop_distance
    notional_usd = btc_size * entry_price

    return {
        "btc_size": btc_size,
        "risk_usd": risk_usd,
        "stop_distance": stop_distance,
        "notional_usd": notional_usd,
    }

result = btc_lotsize(
    balance_usd=5000,
    risk_percent=0.2,
    entry_price=71415,
    stop_price=71500
)

for k, v in result.items():
    print(k, round(v, 6))
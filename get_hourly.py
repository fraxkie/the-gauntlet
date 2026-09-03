"""
get_hourly.py — fetch HOURLY crypto bars from OKX (US-OK, free) for intraday stop testing
═══════════════════════════════════════════════════════════════════════════════
Pulls hourly OHLC for BTC/ETH/SOL so we can test intraday stop-losses on the weekend
sleeve at the granularity stops actually operate at (24 checks/day vs 1 with daily data).

USAGE: python get_hourly.py
Outputs: hourly_btc.csv, hourly_eth.csv, hourly_sol.csv  (upload these 3)
"""
import requests, time
import pandas as pd

COINS = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "SOL": "SOL-USDT"}
H = {"User-Agent": "Mozilla/5.0"}

def _get(url, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=20, headers=H)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.6)
    return None

def fetch_hourly(inst):
    """Hourly candles, paginated backwards. OKX caps ~100/call, page via 'after'."""
    out = []
    after = ""
    for _ in range(400):  # up to ~40,000 hours = ~4.5 years
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=1H&limit=100"
        if after:
            url += f"&after={after}"
        j = _get(url)
        if not j or j.get("code") != "0" or not j.get("data"):
            break
        data = j["data"]
        out += data
        after = data[-1][0]  # oldest timestamp -> page further back
        time.sleep(0.12)
        if len(data) < 100:
            break
    if not out:
        return None
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df[0].astype("int64"), unit="ms")
    for col, name in [(1, "open"), (2, "high"), (3, "low"), (4, "close")]:
        df[name] = df[col].astype(float)
    return df[["time", "open", "high", "low", "close"]].set_index("time").sort_index()

if __name__ == "__main__":
    for coin, inst in COINS.items():
        print(f"{coin}: fetching hourly...", end=" ", flush=True)
        df = fetch_hourly(inst)
        if df is not None:
            fname = f"hourly_{coin.lower()}.csv"
            df.to_csv(fname)
            print(f"{len(df)} bars  ({df.index.min()} to {df.index.max()}) -> {fname}")
        else:
            print("FAILED")
    print("\nDone. Upload hourly_btc.csv, hourly_eth.csv, hourly_sol.csv")

"""
get_funding.py — crypto funding rates + prices via OKX (US-OK), FULL history
═══════════════════════════════════════════════════════════════════════════════
Fix: OKX funding-rate-history pages BACKWARDS using `after`=oldest timestamp seen.
Loop until exhausted to get the full 2-3yr history (not just 100 points).
"""
import requests, time
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "XRP", "LTC", "ADA", "DOGE"]
H = {"User-Agent": "Mozilla/5.0"}

def _get(url, tries=3):
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=20, headers=H)
            if r.status_code == 200: return r.json()
        except Exception: pass
        time.sleep(0.6)
    return None

def okx_funding(coin):
    inst=f"{coin}-USDT-SWAP"; out=[]; after=""
    for _ in range(200):  # up to ~20000 pts
        url=f"https://www.okx.com/api/v5/public/funding-rate-history?instId={inst}&limit=100"
        if after: url+=f"&after={after}"
        j=_get(url)
        if not j or j.get("code")!="0" or not j.get("data"): break
        data=j["data"]; out+=data
        after=data[-1]["fundingTime"]   # oldest in this page -> page further back
        time.sleep(0.15)
        if len(data)<100: break
    if not out: return None
    df=pd.DataFrame(out)
    df["time"]=pd.to_datetime(df["fundingTime"].astype("int64"),unit="ms")
    df["rate"]=df["fundingRate"].astype(float)
    return df.set_index("time")["rate"].sort_index().groupby(level=0).last()

def okx_klines(coin, kind):
    inst=f"{coin}-USDT-SWAP" if kind=="perp" else f"{coin}-USDT"
    out=[]; after=""
    for _ in range(60):
        url=f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=1D&limit=100"
        if after: url+=f"&after={after}"
        j=_get(url)
        if not j or j.get("code")!="0" or not j.get("data"): break
        data=j["data"]; out+=data
        after=data[-1][0]
        time.sleep(0.15)
        if len(data)<100: break
    if not out: return None
    df=pd.DataFrame(out)
    df["time"]=pd.to_datetime(df[0].astype("int64"),unit="ms")
    df["close"]=df[4].astype(float)
    return df.set_index("time")["close"].sort_index().groupby(level=0).last()

if __name__=="__main__":
    funding,perp,spot={},{},{}
    for c in COINS:
        print(f"{c}: funding...",end=" ",flush=True)
        f=okx_funding(c); funding[c]=f
        print(f"{0 if f is None else len(f)} | perp...",end=" ",flush=True)
        p=okx_klines(c,"perp"); perp[c]=p
        print(f"{0 if p is None else len(p)} | spot...",end=" ",flush=True)
        s=okx_klines(c,"spot"); spot[c]=s
        print(f"{0 if s is None else len(s)}")
    def save(d,n):
        d={k:v for k,v in d.items() if v is not None}
        if d: pd.DataFrame(d).to_csv(n); print(f"  saved {n} ({len(d)} coins)")
    print(); save(funding,"funding_rates.csv"); save(perp,"perp_prices.csv"); save(spot,"spot_prices.csv")
    fr={k:v for k,v in funding.items() if v is not None}
    if fr:
        fr=pd.DataFrame(fr)
        print(f"\nFunding history: {fr.index.min()} to {fr.index.max()}  ({len(fr)} settlements)")
        print("Mean 8h funding by coin (annualized APR), full history:")
        for c in fr.columns:
            m=fr[c].mean(); print(f"  {c}: {m*100:+.4f}%/8h = {m*3*365*100:+.0f}% APR")

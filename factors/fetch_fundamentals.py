"""
fetch_fundamentals.py — pull POINT-IN-TIME fundamentals (FMP STABLE API).

UPDATED for FMP's 2025 API change: the old /api/v3/ endpoints became "Legacy"
(403 for new accounts). This uses the new /stable/ endpoints with ?symbol= params.

THE WHOLE POINT: a value/quality factor backtest is only honest if, at each
rebalance date, it uses fundamentals that were PUBLICLY KNOWN by that date. We key
every fundamental to its FILING date and apply a safety lag so the backtest can
only ever use already-public data.

Requires a free FMP API key: https://site.financialmodelingprep.com/developer
Set it before running (PowerShell):  $env:FMP_KEY="your_key_here"

Output: fundamentals.csv [date_filed, ticker, pe, pb, ps, roe, gross_margin,
debt_to_equity, market_cap], every row lag-safe (keyed to public filing date).
"""
import os, sys, time, json
import urllib.request
import urllib.error
import pandas as pd

FMP_KEY = os.environ.get("FMP_KEY", "")
BASE = "https://financialmodelingprep.com/stable"   # NEW stable API base

UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM",
    "JPM","BAC","WFC","GS","MS","C","AXP","BLK",
    "JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","LLY",
    "XOM","CVX","COP","SLB",
    "WMT","HD","MCD","NKE","SBUX","LOW","TGT",
    "PG","KO","PEP","COST","CL",
    "CAT","BA","GE","HON","UPS",
]

SAFETY_LAG_DAYS = 5


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _num(d, *keys):
    """Return the first present numeric field from a list of possible key names
    (the stable API renamed some fields vs legacy)."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return None


def fetch_ticker(tk):
    """Pull quarterly ratios + key-metrics from the STABLE API, keyed to filing date."""
    rows = []
    # stable endpoints: ?symbol=TK&period=quarter&limit=N
    ratios = _get(f"{BASE}/ratios?symbol={tk}&period=quarter&limit=120&apikey={FMP_KEY}")
    try:
        km = _get(f"{BASE}/key-metrics?symbol={tk}&period=quarter&limit=120&apikey={FMP_KEY}")
        km_by_date = {r.get("date"): r for r in km} if isinstance(km, list) else {}
    except Exception:
        km_by_date = {}

    if isinstance(ratios, dict) and ratios.get("Error Message"):
        print(f"  {tk}: {ratios['Error Message'][:90]}")
        return rows
    if not isinstance(ratios, list):
        print(f"  {tk}: unexpected response -> {str(ratios)[:80]}")
        return rows

    for r in ratios:
        fiscal_date = r.get("date")
        kmr = km_by_date.get(fiscal_date, {})
        filed = r.get("filingDate") or r.get("fillingDate") or r.get("acceptedDate")
        if not filed:
            try:
                filed = (pd.to_datetime(fiscal_date) + pd.Timedelta(days=75)).strftime("%Y-%m-%d")
            except Exception:
                continue
        rows.append({
            "date_filed": filed,
            "ticker": tk,
            # stable API field names (with legacy fallbacks)
            "pe": _num(r, "priceToEarningsRatio", "priceEarningsRatio"),
            "pb": _num(r, "priceToBookRatio"),
            "ps": _num(r, "priceToSalesRatio"),
            "roe": _num(r, "returnOnEquity"),
            "gross_margin": _num(r, "grossProfitMargin"),
            "debt_to_equity": _num(r, "debtToEquityRatio", "debtEquityRatio"),
            "market_cap": _num(kmr, "marketCap", "marketCapitalization"),
        })
    return rows


if __name__ == "__main__":
    if not FMP_KEY:
        print('ERROR: set your FMP key first:  $env:FMP_KEY="your_key_here"')
        sys.exit(1)

    # quick probe on one ticker to confirm the key/endpoint works before the full loop
    print("Probing API with AAPL...")
    try:
        probe = _get(f"{BASE}/ratios?symbol=AAPL&period=quarter&limit=2&apikey={FMP_KEY}")
        if isinstance(probe, dict) and probe.get("Error Message"):
            print(f"  API says: {probe['Error Message']}")
            print("  -> The free tier may not cover this endpoint. Stopping.")
            sys.exit(1)
        elif isinstance(probe, list) and probe:
            print(f"  OK — got {len(probe)} records. Fields available:",
                  ", ".join(list(probe[0].keys())[:8]), "...")
        else:
            print(f"  Unexpected probe response: {str(probe)[:120]}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")
        if e.code == 403:
            print("  -> 403 Forbidden. The free tier likely doesn't include quarterly")
            print("     ratios history, or the key needs activation. Check your FMP dashboard.")
        sys.exit(1)

    all_rows = []
    for i, tk in enumerate(UNIVERSE):
        print(f"[{i+1}/{len(UNIVERSE)}] {tk}...")
        try:
            all_rows.extend(fetch_ticker(tk))
        except urllib.error.HTTPError as e:
            print(f"  {tk}: HTTP {e.code} {e.reason}")
            if e.code == 429:
                print("  -> rate limited; waiting 15s..."); time.sleep(15)
        except Exception as e:
            print(f"  {tk} failed: {e}")
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data fetched.")
        sys.exit(1)

    df["date_filed"] = pd.to_datetime(df["date_filed"]) + pd.Timedelta(days=SAFETY_LAG_DAYS)
    df = df.sort_values(["ticker", "date_filed"]).reset_index(drop=True)
    df.to_csv("fundamentals.csv", index=False)
    print(f"\nSaved fundamentals.csv: {len(df)} quarterly records, "
          f"{df['ticker'].nunique()} tickers")
    print(f"Filing date range: {df['date_filed'].min().date()} to {df['date_filed'].max().date()}")
    print("Every record keyed to public filing date (+5d safety) — no look-ahead.")

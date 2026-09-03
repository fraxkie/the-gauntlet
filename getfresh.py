"""
getfresh.py — fetch fresh asset classes: crypto + sector ETFs.

Crypto: 24/7 markets with violent forced-liquidation cascades — the best possible
test of whether our panic-reversion edge generalizes (the mechanism is forced-
selling overshoot, and crypto has the most extreme version of that).

Sectors: the 11 SPDR sector ETFs — for rotation, dispersion, cross-sectional tests.
"""
import yfinance as yf

CRYPTO = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","DOGE-USD","ADA-USD","LTC-USD"]
SECTORS = ["XLK","XLF","XLE","XLV","XLI","XLP","XLY","XLU","XLB","XLRE","XLC"]

print("Downloading crypto (this is where panic-reversion should be strongest)...")
crypto = yf.download(CRYPTO, start="2017-01-01", group_by="ticker", auto_adjust=True)
crypto.to_csv("crypto_data.csv")
print(f"Saved crypto_data.csv: {len(crypto)} rows, {crypto.index.min().date()} to {crypto.index.max().date()}")

print("\nDownloading sector ETFs...")
sectors = yf.download(SECTORS, start="2005-01-01", group_by="ticker", auto_adjust=True)
sectors.to_csv("sector_data.csv")
print(f"Saved sector_data.csv: {len(sectors)} rows")
print("\nNote: crypto history is shorter (most coins post-2017) and SOL/some are even")
print("newer. The loader handles partial history. Crypto vol is HUGE — expect the")
print("regime detector to find dramatic crisis regimes.")

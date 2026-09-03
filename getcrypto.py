"""
getcrypto.py — fetch crypto data to test if panic-reversion generalizes to the most
violent forced-liquidation market that exists.

Mechanism prediction: crypto has the most extreme forced-liquidation cascades
(leverage liquidations, exchange margin calls) of any market. If panic-overshoot
mean-reversion is real, crypto crashes should show it STRONGLY. This is a clean
out-of-sample test of the mechanism on a totally different asset class.

Uses yfinance crypto tickers (free). Saves crypto_data.csv in the same multi-ticker
format as etf_data.csv so run_batch.py picks it up automatically.
"""
import yfinance as yf

# liquid crypto with multi-year history (yfinance uses -USD suffix)
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
           "SOL-USD", "DOGE-USD", "LTC-USD"]
print(f"Downloading {len(tickers)} crypto assets...")
data = yf.download(tickers, start="2017-01-01", group_by="ticker", auto_adjust=True)
data.to_csv("crypto_data.csv")
print("Saved crypto_data.csv")
print("Rows:", len(data))
print("Date range:", data.index.min(), "to", data.index.max())
print("Note: crypto trades 24/7, so 'daily' bars differ from equities, but the")
print("panic-overshoot mechanism (if real) should still appear in crash periods.")

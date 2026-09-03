"""
getstocks.py — pull a universe of individual stocks for cross-sectional testing.

Cross-sectional long-short is only market-neutral when the assets are COMPARABLE
(many stocks within one market), so the long and short legs cancel market exposure.
13 mixed-asset ETFs can't do that. This pulls ~45 large-cap US stocks so the
cross-sectional family can be tested fairly.
"""
import yfinance as yf

# ~45 liquid large-cap names across sectors (comparable universe)
tickers = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM",  # tech
    "JPM","BAC","WFC","GS","MS","C","AXP","BLK",                            # financials
    "JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","LLY",                       # healthcare
    "XOM","CVX","COP","SLB",                                                # energy
    "WMT","HD","MCD","NKE","SBUX","LOW","TGT",                              # consumer
    "PG","KO","PEP","COST","CL",                                           # staples
    "CAT","BA","GE","HON","UPS",                                          # industrials
]
print(f"Downloading {len(tickers)} stocks...")
data = yf.download(tickers, start="2005-01-01", group_by="ticker", auto_adjust=True)
data.to_csv("stock_data.csv")
print("Saved stock_data.csv")
print("Rows:", len(data))
print("Date range:", data.index.min(), "to", data.index.max())
print("Note: some names IPO'd after 2005 (META 2012, TSLA 2010, AVGO 2009) —")
print("the loader drops rows with missing data, so the usable window starts when")
print("all names have data. For a longer history, drop the newer tickers.")

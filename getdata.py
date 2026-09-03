import yfinance as yf

tickers = ["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","TLT","IEF","GLD","EFA","EEM"]
print("Downloading", len(tickers), "tickers...")

data = yf.download(tickers, start="2005-01-01", group_by="ticker", auto_adjust=True)
data.to_csv("etf_data.csv")

print("Saved etf_data.csv")
print("Rows:", len(data))
print("Date range:", data.index.min(), "to", data.index.max())

"""
data.py — load the yfinance multi-ticker CSV into clean per-ticker OHLC frames.

yfinance multi-ticker CSV layout:
  row 0: Ticker, QQQ, QQQ, QQQ, QQQ, QQQ, GLD, ...   (ticker repeated x5)
  row 1: Price, Open, High, Low, Close, Volume, ...  (the OHLCV field)
  row 2: Date,,,,,,...                               (blank)
  row 3+: 2005-01-03, <values...>

We parse with a 2-level column header and reshape into {ticker: DataFrame} where
each DataFrame has columns open/high/low/close indexed by date.
"""
import pandas as pd


def load_etf_data(path: str) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path, header=[0, 1], index_col=0, skiprows=[2])
    raw.index = pd.to_datetime(raw.index)

    out: dict[str, pd.DataFrame] = {}
    tickers = raw.columns.get_level_values(0).unique()
    for tk in tickers:
        sub = raw[tk].copy()
        sub.columns = [c.lower() for c in sub.columns]
        keep = sub[["open", "high", "low", "close"]].dropna()
        keep = keep.astype(float)
        out[tk] = keep
    return out


if __name__ == "__main__":
    d = load_etf_data("/mnt/user-data/uploads/etf_data.csv")
    print(f"Loaded {len(d)} tickers: {sorted(d.keys())}")
    spy = d["SPY"]
    print(f"\nSPY: {len(spy)} bars, {spy.index.min().date()} to {spy.index.max().date()}")
    print(spy.head(3))
    print(spy.tail(3))
    # sanity: SPY should roughly 4x over 2005->2026
    print(f"\nSPY close 2005: {spy['close'].iloc[0]:.2f}  ->  2026: {spy['close'].iloc[-1]:.2f}")

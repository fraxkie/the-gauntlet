"""
get_intraday_equity.py — pull intraday SPY/QQQ/XLK bars for real intraday strategy testing
═══════════════════════════════════════════════════════════════════════════════
Daily candles can't test VWAP/ORB/intraday-RSI — those need intraday bars + volume.
This pulls what yfinance offers free:
  - HOURLY bars, ~2 years  (for full-session intraday tests, walk-forward depth)
  - 15-MINUTE bars, ~60 days (for first-hour ORB + opening-range tests)
Both include VOLUME so VWAP is computable.

USAGE:  python get_intraday_equity.py
Outputs (upload all 6):
  spy_1h.csv  qqq_1h.csv  xlk_1h.csv      (hourly, ~2yr)
  spy_15m.csv qqq_15m.csv xlk_15m.csv     (15-min, ~60d)
"""
import sys
import pandas as pd

TICKERS = ["SPY", "QQQ", "XLK"]


def pull(tk, interval, period):
    import yfinance as yf
    df = yf.download(tk, interval=interval, period=period,
                     auto_adjust=False, prepost=False, progress=False)
    if df is None or len(df) == 0:
        return None
    # flatten possible multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    keep = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    keep.columns = ["open", "high", "low", "close", "volume"]
    keep.index.name = "datetime"
    return keep


if __name__ == "__main__":
    print("Pulling intraday equity bars (hourly ~2yr, 15-min ~60d) for SPY/QQQ/XLK...\n")
    specs = [("1h", "730d", "1h"), ("15m", "60d", "15m")]
    for tk in TICKERS:
        for suffix, period, interval in specs:
            try:
                df = pull(tk, interval, period)
                if df is not None and len(df) > 0:
                    fname = f"{tk.lower()}_{suffix}.csv"
                    df.to_csv(fname)
                    print(f"  {tk} {interval}: {len(df)} bars "
                          f"({df.index.min()} -> {df.index.max()}) -> {fname}")
                else:
                    print(f"  {tk} {interval}: NO DATA")
            except Exception as e:
                print(f"  {tk} {interval}: ERROR {e}")
    print("\nDone. Upload all 6 CSVs (spy/qqq/xlk _1h and _15m).")

# Fundamental Factor Testing — point-in-time, no look-ahead

This is Path B: testing value/quality factors, which (unlike everything tested so
far) have strong academic evidence for REAL, cost-survivable edges — because they
rebalance quarterly (low turnover) and the clean-data barrier keeps them less
arbitraged.

## Why this needs a (free) data source
Free price data (yfinance) only gives CURRENT valuations. A factor backtest needs
to know what a stock's P/B was AT THE TIME in 2015 — using today's value would be
look-ahead bias (the exact thing the gauntlet's audit catches). So we pull
point-in-time fundamentals keyed to FILING dates from Financial Modeling Prep.

## Setup (one time)
1. Get a free FMP API key: https://site.financialmodelingprep.com/developer
   (free tier = limited calls/day, enough for ~47 large-caps)
2. In PowerShell, set the key for your session:
   $env:FMP_KEY="your_key_here"

## Run order
```
# 1. prices for the same universe (if not already present)
python getstocks.py

# 2. point-in-time fundamentals (uses your FMP key)
python factors\fetch_fundamentals.py

# 3. run value/quality factors through the gauntlet
python factors\run_factors.py
```

## What's enforced
- Every fundamental is keyed to its public FILING date + a 5-day safety lag.
- At each monthly rebalance, the engine uses merge-asof backward: only fundamentals
  already filed as of that date. Verified leak-free with a known-answer test.
- Long top-quintile / short bottom-quintile, dollar-neutral, quarterly-ish rebalance.
- Full gauntlet: permutation p-value, walk-forward consistency, SPY-neutrality.

## If FMP's free tier is too limited
Alternatives with free/cheap tiers and point-in-time data: Tiingo, Alpha Vantage
(fundamentals endpoint), or Sharadar (via Nasdaq Data Link, paid but gold-standard).
The run_factors.py engine only needs fundamentals.csv in the right format
(date_filed, ticker, pe, pb, ps, roe, gross_margin, debt_to_equity, market_cap).

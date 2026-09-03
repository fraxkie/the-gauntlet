# Probability Modeling — quant-style

The shift from "strategies" to "probabilities": instead of buy/sell rules, forecast
the PROBABILITY DISTRIBUTION of future returns, then size bets by expected value.
This is how systematic funds actually think.

## The pipeline
```
features.py     -> normalized predictive features from OHLC (relative, no look-ahead)
model.py        -> logistic regression, WALK-FORWARD out-of-sample P(up)
calibration.py  -> the lie-detector: Brier score + reliability diagram + Murphy
                   decomposition (reliability / resolution / uncertainty)
run_probability.py -> end-to-end on real data
```

## The key idea: calibration is the validation
A probability is only worth money if it's CALIBRATED (60%-predicted happens 60% of
the time, out of sample) AND SHARP (resolution > 0 — it discriminates, not just
predicting the base rate). The Brier decomposition separates these:
  Brier = Reliability(calibration, low=good) - Resolution(sharpness, high=good) + Uncertainty
A model can be perfectly calibrated yet useless (always predict base rate = zero
resolution). The harness catches both failure modes. Verified with known-answer tests.

## First result (next-day direction, SPY/QQQ/EEM)
Calibrated but ZERO resolution, negative Brier skill — next-day direction is a coin
flip from price features alone. Honest result; the harness correctly refused to let
an impressive-looking "55% accuracy" (which is just the base rate) masquerade as edge.

## Where edges might actually be (next experiments)
- VOLATILITY prediction instead of direction — research says vol is far more
  predictable than direction (GARCH works). Same pipeline, different target.
- Longer horizons (5d, 20d) — may have more structure than 1d.
- Fundamental features (once fundamentals.csv exists).
- Stronger models (gradient boosting, LSTM w/ skewed-t distribution) — but only
  after the simple pipeline shows signal, so a positive result stays trustworthy.

## Run
```
python run_probability.py        # needs etf_data.csv (run getdata.py first)
python probability\test_calibration.py   # proves the harness works
```

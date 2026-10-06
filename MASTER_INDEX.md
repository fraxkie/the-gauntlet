# THE GAUNTLET — Master Index

> **ENGINE STATUS: v1.0 FINISHED & VERIFIED.** All 30 known-answer checks pass across
> 8 test files. The engine (validation gauntlet + probability pipeline) is frozen and
> trustworthy. One open lead — crisis mean-reversion — is documented in
> CRISIS_MEANREVERSION_LEAD.md, parked for rigorous testing ON this finished engine.


A quant research engine that finds what's real and kills what isn't. Built from
scratch in Python. Every component is proven on known-answer synthetic data before
being trusted on real markets.

═══════════════════════════════════════════════════════════════════════════════
## QUICK START
```
python getdata.py                  # fetch 21y of ETF data (needs internet)
python -m tests.test_core          # prove the foundations (run all 5 tests)
python run_macd.py                 # see a strategy get killed as beta
python probability/test_volatility.py   # prove volatility IS predictable
python run_voltarget.py            # volatility targeting (real risk benefit)
```

═══════════════════════════════════════════════════════════════════════════════
## THE TWO HALVES

### HALF 1 — The Strategy Gauntlet (kills false edges)
Five firewalls, each independently proven:
| File | Firewall | Question it answers |
|---|---|---|
| gauntlet/mcpt.py | Monte Carlo Permutation | Is the edge more than luck? |
| gauntlet/robustness.py | Walk-forward + regime | Does it hold over time? Is it just beta? |
| gauntlet/lookahead.py | Look-ahead audit | Does it peek at the future? |
| gauntlet/ledger.py | Multiple-comparison | Did we p-hack across many tries? |
| gauntlet/core.py | Cost model + bounded math | (foundation for all) |

Strategy runners (all families tested → all killed):
- run_macd.py (momentum), run_pairs.py + scan_pairs.py (stat-arb),
  run_seasonality.py, run_xsection.py (cross-sectional), run_timeseries.py
  (abs-momentum/breakout/allocation), run_overnight.py (overnight effect)

### HALF 2 — The Probability Pipeline (finds real signal)
| File | Component | Status |
|---|---|---|
| probability/volatility.py | Volatility forecasting (EWMA/GARCH) | ✅ REAL signal: 0.65 corr OOS, +27% vs naive |
| probability/regime.py | HMM regime detection (from scratch) | ✅ found every crisis 2008/2011/2020/2022 |
| probability/calibration.py | Brier/reliability lie-detector | ✅ proven |
| probability/features.py + model.py | Logistic direction model | direction = coin flip (honest) |
| run_voltarget.py | Volatility targeting | ✅ 5/5 steadier risk, drawdown −55%→−35% |
| run_regime_calibration.py | Regime-conditioned forecasts | ✅ +14% sharper vol forecasts |

### Factor scaffolding (needs your machine + free FMP key)
- factors/fetch_fundamentals.py + factors/run_factors.py — point-in-time value/
  quality factors, no-look-ahead verified, NOT YET RUN on real data

═══════════════════════════════════════════════════════════════════════════════
## THE STORY (what we found)
1. Eight simple strategy families on liquid ETFs → ALL killed, almost all as
   disguised market beta. Matches the literature: easy price-based alpha is gone.
2. The overnight effect: real (+432%) but uncapturable (−79% after costs).
3. Next-day DIRECTION: a coin flip (calibrated but zero resolution).
4. **Volatility: genuinely predictable** (0.65 corr OOS) — first real signal.
5. **Market regimes: detectable** — an HMM found every crisis from returns alone.
6. Volatility forecasting → steadier risk via vol-targeting; regime knowledge →
   sharper forecasts. The real edges are in RISK and STRUCTURE, not direction.

## METHODS
After Timothy Masters (permutation tests), López de Prado (multiple testing),
and standard volatility/regime econometrics (GARCH, Hidden Markov Models).
All core math — permutation, Baum-Welch/forward-backward/Viterbi, Brier
decomposition — implemented from scratch.

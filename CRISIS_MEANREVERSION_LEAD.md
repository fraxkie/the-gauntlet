# OPEN LEAD: Crisis Mean-Reversion (NOT yet confirmed — needs validation)

## Status: PROMISING LEAD, parked for rigorous testing on the finished engine.

## What we found
Within HMM-detected CRISIS regimes, next-day returns show significant negative
autocorrelation (mean-reversion) that does NOT appear in calm/normal regimes and
averages away in the full sample. The regime detector isolated it.

| Asset | Crisis autocorr | Permutation p | Strategy net (after basic costs) | Walk-forward |
|---|---|---|---|---|
| SPY | -0.204 | 0.0005 | +162% | 5/6 PASS |
| QQQ | -0.217 | 0.0005 | +73% | 3/6 PASS |
| EEM | -0.199 (crisis, p=0.066 NOT sig); -0.174 in NORMAL (p=0.0005) | mixed | +170% | 2/2 |

The strategy: in a crisis regime, fade yesterday's move (mean-revert). Flat otherwise.
Trades only ~5% of the time. Economically sensible (panic -> overreaction -> bounce).

## Why it is NOT confirmed — the open concerns
1. **COST MODEL UNDERSTATES CRISIS COSTS.** Used flat 3bps. Real bid-ask spreads
   blow out 5-10x during crises — exactly when this trades. The +162% could be much
   lower (or gone) with realistic crisis-period spreads. THIS IS THE #1 TEST.
2. **Inconsistent across assets** — EEM's effect is in NORMAL not CRISIS. It may be
   a "high-volatility-period" effect, not cleanly "crisis", and asset-dependent.
3. **Rare** — accrues entirely in ~5% of days; long waits between active periods.
4. **Possible single-event dependence** — is it real, or is it just 2008 (and 2020)
   wearing a trenchcoat? Must test excluding the financial crisis.
5. **Multiple-comparisons** — we ran many tests across the project; must apply the
   ledger's corrected bar (p=0.0005 likely survives, but verify).
6. **Execution realism** — backtest assumes calm fills during crashes; reality has
   gaps, halts, slippage, and the psychology of buying a crash.

## The validation plan (run on the FINISHED, frozen engine)
1. **Cost stress test** — re-run at 3x, 5x, 10x crisis spreads. At what cost does it
   die? Survives 5x = robust. Dies at 2x = mirage. (HIGHEST PRIORITY)
2. **Ledger + Bonferroni** — log every test; apply the multiple-comparison correction.
3. **Leave-one-crisis-out** — exclude 2008, then 2020, separately. Does it persist?
4. **Realistic crisis costs** — model spread as a function of the VIX/realized vol,
   not a flat number.
5. Only if it survives ALL of the above: consider it a real (if rare) edge.

## How to test it
`python run_regime_conditional.py` (the current exploratory version). The frozen
engine's gauntlet (mcpt, walk-forward, ledger, cost model) is the validator.

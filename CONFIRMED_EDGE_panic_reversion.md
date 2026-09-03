# CONFIRMED EDGE #1: Panic-Crash Mean-Reversion

## Status: CONFIRMED (survived all honesty gates). First validated edge found by the engine.
Real, rigorous, but SITUATIONAL and RARE — not a money-printer. ~5% of trading days.
A small genuine edge found by an engine built to kill fakes = proof the engine works.

## The finding
During FAST, panic-driven crash periods, next-day returns mean-revert: a down day
tends to be followed by an up day (and vice versa). Fade yesterday's move. This is
forced-liquidation OVERSHOOT correcting itself.

## Evidence (all gates passed)
| Test | Result | Verdict |
|---|---|---|
| Significance (permutation, SPY) | autocorr -0.20, p=0.0002 | REAL |
| Bonferroni (30 comparisons, bar=0.00167) | 0.0002 < 0.00167 | SURVIVES |
| Cross-asset replication | SPY p=0.0005, QQQ p=0.0005 | REPLICATES |
| Cross-episode replication | 2008 (-0.13), 2011 Euro (-0.29), 2020 COVID (-0.42) | 3 INDEPENDENT EVENTS |
| Cost stress test | SPY survives to 10x normal crisis spreads | SURVIVES REALISTIC COSTS |
| Mechanism (falsifiable) | FAST crash -0.227 vs SLOW grind -0.117 | MECHANISM CONFIRMED |

## THE MECHANISM (why it works)
Panic crashes (2008, 2020) = forced selling (margin calls, redemptions, stop-outs)
that OVERSHOOTS fair value, then snaps back. The snap-back is the mean-reversion.
Confirmed by a falsifiable prediction: the FASTER the crash, the STRONGER the
reversion (-0.227 for fast vs -0.117 for slow). Speed of panic predicts the edge.

## WHY IT DOESN'T (the boundary — equally important)
- SLOW grinding bears (2022: +0.05, no reversion) = rational repricing, not panic.
  No overshoot, no snap-back. The edge needs PANIC, not just a falling market.
- Our HMM regime detector keys on VOLATILITY, so it lumps panic crashes and slow
  grinds into one "crisis" bucket — too crude. The edge lives in the panic subset.
- RARE: ~5% of days. Long waits between opportunities.
- The headline +162% is mostly cost-cushion; at realistic 5-10x crisis spreads the
  real return is modest-positive, not huge.

## REMAINING REALISM CAVEATS (before risking real money)
- Execution during crashes: halts, gaps, slippage far worse than modeled.
- Psychological: requires buying into a 2008/2020-style crash in real time.
- Capital: fading a crash needs dry powder exactly when everything's down.
- The causal panic-detector (distinguishing fast crash from slow grind in REAL time,
  not hindsight) is built but could be sharpened — that's the path to making it
  more tradeable.

## How to reproduce
`python run_regime_conditional.py` + the gates in this session's cost/episode/
mechanism tests. Engine: gauntlet v1.0 (frozen, verified).

## What this PROVES
The engine can find a real edge AND rigorously characterize why it works and where
it fails. That's the whole point: not a money machine, but a truth machine that
happened to find a small real thing. The methodology is the deliverable.

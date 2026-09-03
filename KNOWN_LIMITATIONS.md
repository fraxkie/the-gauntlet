# KNOWN LIMITATIONS / OPEN ITEMS

## 1. Synthetic positive-control for MCPT detection (DEFERRED — diagnosed)

**Status:** Destroy-side proven and measured. Detection-side validated behaviorally
on real data but NOT via a clean synthetic positive control. Five construction
attempts pinned down exactly why.

**The precise diagnosis (why a synthetic positive control is hard):**
MCPT's permutation destroys *temporal ordering* (autocorrelation) while preserving
the data's *marginal statistics* (drift, volatility, the distribution of bar
shapes — all measured-preserved in test_permute.py). For MCPT to detect an edge,
the strategy's profitability must depend on the temporal ordering that permutation
destroys. Across five attempts, every synthetic edge I constructed leaked into the
*marginal* statistics instead — so the strategy stayed profitable on permuted data
(noise-mean tracked the real score), and MCPT correctly reported "no edge beyond
what survives shuffling." Changing the objective (profit factor -> mean return)
did not help, confirming it's a data-construction issue, not a scoring one.

In short: hand-building fake data whose edge lives PURELY in time-ordering (and not
at all in the preserved marginals) is genuinely difficult. This is a statement
about the difficulty of the synthetic fixture, not a defect in the engine.

**Why deferring is the right call (not a cop-out):**
1. Destroy-side is measured-correct: permutation kills autocorrelation
   (-0.35 -> -0.001); MCPT kills noise, no-mechanism signals, and pure beta.
   That's the half that prevents losing money, and it's proven.
2. On REAL data, MCPT discriminates by signal strength: across 11 ETFs it ranked
   XLF at p=0.07 vs junk at p=0.7-0.9. It is NOT blindly rejecting everything —
   it responds to real structure. That's behavioral evidence the detection side
   works.
3. A sixth attempt risks a MISLEADING pass — synthetic data where the edge leaks
   in a way that happens to get detected — giving false confidence. A clean pass
   I don't fully understand would be worse than an honest deferral.

**The real safeguard against false-negatives is the STACK, not MCPT alone.**
Four independent firewalls (MCPT, walk-forward consistency gate, regime check,
look-ahead audit) give a real edge multiple paths to prove itself. A strategy that
MCPT unfairly flags (e.g. one exploiting volatility clustering, which permutation
destroys — a known boundary noted by Masters/doc-40) can still pass the others. We
never let one p-value auto-reject; a kill that disagrees with the other firewalls
is a conversation-starter, not a verdict.

**Detection will be proven on real strategies:** when a real strategy passes MCPT
on real data with low p AND passes the other firewalls, that is the detection
proof — more convincing than any synthetic fixture because the edge is real, not
constructed.


"""
test_regime.py — proof the HMM recovers KNOWN regimes (the validation that matters).

If we plant a known two-regime structure (calm vs stormy, with known vols and known
switch points) and the HMM recovers it — the right vols, and the right labeling of
which periods were calm vs stormy — then we can trust what it infers in real markets.

Cases:
  1. RECOVERY: two regimes with distinct vol; model must recover both vols and label
     >85% of time steps to the correct regime.
  2. PERSISTENCE: a sticky regime process; model must learn high self-transition.
  3. CAUSAL FILTER: filter_proba must not use future data (shifting future values
     must not change the filtered belief at time t).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from probability.regime import GaussianHMM, detect_regimes


def make_two_regime(n=4000, p_stay=0.99, seed=0):
    """Generate returns from 2 known regimes:
       regime 0 = calm  (mean +0.05%/day, vol 0.7%/day)
       regime 1 = stormy(mean -0.10%/day, vol 2.5%/day)
    with sticky switching. Returns (returns, true_states)."""
    rng = np.random.default_rng(seed)
    means = [0.0005, -0.0010]; vols = [0.007, 0.025]
    states = np.zeros(n, int)
    s = 0
    for i in range(n):
        if rng.uniform() > p_stay:
            s = 1 - s
        states[i] = s
    r = np.array([rng.normal(means[s], vols[s]) for s in states])
    return r, states


def test_recovers_known_regimes():
    r, true_states = make_two_regime(seed=1)
    model, desc = detect_regimes(r, n_states=2, seed=0)
    # regimes are sorted by vol: regime 0 = calm, 1 = stormy
    calm_vol = desc[0]["ann_vol"]; storm_vol = desc[1]["ann_vol"]
    true_calm = 0.007 * np.sqrt(252) * 100
    true_storm = 0.025 * np.sqrt(252) * 100
    print(f"  recovered vols: calm={calm_vol:.1f}% (true {true_calm:.1f}%)  "
          f"stormy={storm_vol:.1f}% (true {true_storm:.1f}%)")
    assert abs(calm_vol - true_calm) / true_calm < 0.25, "calm vol should be recovered"
    assert abs(storm_vol - true_storm) / true_storm < 0.25, "stormy vol should be recovered"

    # labeling accuracy via Viterbi path
    path = model.viterbi(r)
    # path states already ordered by vol (0=calm,1=stormy) matches true_states coding
    acc = max((path == true_states).mean(), (path == (1 - true_states)).mean())
    print(f"  regime labeling accuracy: {acc*100:.1f}%")
    assert acc > 0.85, f"should label >85% of periods correctly, got {acc}"
    print("      [PASS] HMM recovers known regime vols AND labels periods correctly")


def test_learns_persistence():
    r, _ = make_two_regime(p_stay=0.995, seed=2)
    model, desc = detect_regimes(r, n_states=2, seed=0)
    avg_persist = np.mean([d["persistence"] for d in desc])
    print(f"  learned persistence: {avg_persist:.3f} (true 0.995)")
    assert avg_persist > 0.95, "should learn regimes are sticky"
    print("      [PASS] HMM learns high regime persistence")


def test_filter_is_causal():
    """filter_proba at time t must not depend on data after t."""
    r, _ = make_two_regime(seed=3)
    model, _ = detect_regimes(r, n_states=2, seed=0)
    t = 2000
    filt_full = model.filter_proba(r)
    # change the FUTURE (after t) drastically; filtered belief at t must be unchanged
    r2 = r.copy(); r2[t+1:] += 0.05
    filt_mod = model.filter_proba(r2)
    diff = np.abs(filt_full[t] - filt_mod[t]).max()
    print(f"  causal check: belief change at t when future altered = {diff:.2e}")
    assert diff < 1e-9, "filtered belief at t must not depend on future"
    print("      [PASS] filter_proba is causal (no look-ahead)")


if __name__ == "__main__":
    print("STEP 3 PROOF — Hidden Markov Model regime detector")
    test_recovers_known_regimes()
    test_learns_persistence()
    test_filter_is_causal()
    print("Regime detector verified: recovers known regimes, causal filter is leak-free.\n")

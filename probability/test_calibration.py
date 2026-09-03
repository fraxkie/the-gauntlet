"""
test_calibration.py — proof the calibration harness is a working lie-detector.

Three known-answer cases:
  1. PERFECT forecaster: probabilities exactly match outcome frequencies ->
     near-zero reliability, positive resolution, good Brier skill.
  2. OVERCONFIDENT forecaster (states 0.9, wins 0.6): high ECE, flagged.
  3. USELESS forecaster (always predicts base rate): calibrated but ZERO resolution
     -> must be flagged as having no discrimination.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from probability.calibration import (brier_score, murphy_decomposition,
                                     expected_calibration_error, calibration_report)


def test_perfect_forecaster():
    """Generate data where stated probability IS the true probability."""
    rng = np.random.default_rng(0)
    n = 20000
    # stated probs spread across [0,1]; outcome drawn with exactly that prob
    p = rng.uniform(0.05, 0.95, n)
    o = (rng.uniform(0, 1, n) < p).astype(float)
    rep = calibration_report(p, o, label="perfect")
    print(f"  perfect    : ECE={rep['ece']:.4f}  resolution={rep['resolution']:.4f}  "
          f"skill={rep['brier_skill_score']:+.4f}")
    assert rep["ece"] < 0.02, f"perfect forecaster should have tiny ECE, got {rep['ece']}"
    assert rep["resolution"] > 0.05, f"should have real resolution, got {rep['resolution']}"
    assert rep["brier_skill_score"] > 0.1, "should clearly beat base rate"
    print("      [PASS] perfect forecaster: calibrated + sharp + beats baseline")


def test_overconfident_forecaster():
    """States high confidence but is wrong more often than claimed."""
    rng = np.random.default_rng(1)
    n = 20000
    p = rng.uniform(0.05, 0.95, n)
    # outcome true prob is COMPRESSED toward 0.5 (overconfident: extreme statements
    # don't pan out). true_p = 0.5 + 0.4*(p-0.5)  -> states .9 but真 ~.66
    true_p = 0.5 + 0.4 * (p - 0.5)
    o = (rng.uniform(0, 1, n) < true_p).astype(float)
    rep = calibration_report(p, o, label="overconfident")
    print(f"  overconfid : ECE={rep['ece']:.4f}  reliability={rep['reliability']:.4f}")
    assert rep["ece"] > 0.05, f"overconfidence should show high ECE, got {rep['ece']}"
    # check a high-confidence bin specifically is miscalibrated
    high_bins = [b for b in rep["diagram"] if b["count"] > 0 and b["stated"] > 0.85]
    if high_bins:
        b = high_bins[0]
        assert b["observed"] < b["stated"] - 0.1, "high-conf bin should underperform"
    print("      [PASS] overconfident forecaster correctly flagged (high ECE)")


def test_useless_forecaster():
    """Always predicts the base rate. Perfectly calibrated, ZERO resolution."""
    rng = np.random.default_rng(2)
    n = 20000
    base = 0.54
    o = (rng.uniform(0, 1, n) < base).astype(float)
    p = np.full(n, o.mean())   # always predict the realized base rate
    rep = calibration_report(p, o, label="useless")
    print(f"  useless    : ECE={rep['ece']:.4f}  resolution={rep['resolution']:.6f}  "
          f"skill={rep['brier_skill_score']:+.4f}")
    assert rep["resolution"] < 1e-3, f"base-rate forecaster must have ~0 resolution, got {rep['resolution']}"
    # it IS calibrated (low ECE) but useless (no resolution) — the key distinction
    assert rep["ece"] < 0.02, "base-rate forecaster is trivially calibrated"
    assert abs(rep["brier_skill_score"]) < 0.02, "no skill over base rate"
    print("      [PASS] useless forecaster: calibrated but ZERO resolution (correctly distinguished)")


def test_brier_proper():
    """Sanity: Brier is minimized by stating the TRUE probability. Deviating worse."""
    rng = np.random.default_rng(3)
    n = 50000
    true_p = 0.7
    o = (rng.uniform(0, 1, n) < true_p).astype(float)
    honest = brier_score(np.full(n, 0.7), o)
    lie_high = brier_score(np.full(n, 0.9), o)
    lie_low = brier_score(np.full(n, 0.5), o)
    print(f"  brier proper: honest(.7)={honest:.4f}  lie(.9)={lie_high:.4f}  lie(.5)={lie_low:.4f}")
    assert honest < lie_high and honest < lie_low, "honest forecast must score best"
    print("      [PASS] Brier is strictly proper (honesty scores best)")


def test_decomposition_recombines():
    """Murphy decomposition must recombine to the actual Brier score."""
    rng = np.random.default_rng(4)
    n = 10000
    p = rng.uniform(0.05, 0.95, n)
    o = (rng.uniform(0, 1, n) < p).astype(float)
    bs = brier_score(p, o)
    dec = murphy_decomposition(p, o)
    print(f"  decomp     : brier={bs:.4f}  recombined={dec['brier_recombined']:.4f}")
    assert abs(bs - dec["brier_recombined"]) < 0.01, "decomposition must recombine to Brier"
    print("      [PASS] Reliability - Resolution + Uncertainty = Brier")


if __name__ == "__main__":
    print("CALIBRATION HARNESS PROOF — the probability lie-detector")
    test_brier_proper()
    test_decomposition_recombines()
    test_perfect_forecaster()
    test_overconfident_forecaster()
    test_useless_forecaster()
    print("\nCalibration harness verified: detects miscalibration AND uselessness.\n")

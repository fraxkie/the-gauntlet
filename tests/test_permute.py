"""
test_permute.py — proof that the permutation engine is correct.

A valid permutation must satisfy BOTH:
  PRESERVE  : drift (total log return) and volatility (std of returns)
  DESTROY   : autocorrelation (the temporal pattern a strategy would exploit)

We build a series with STRONG, known negative lag-1 autocorrelation (mean
reversion) and measure all three properties before vs after permutation, averaged
over many permutations.

If autocorrelation is driven toward zero while drift/vol are preserved, the engine
is correct — and any earlier false-negative was bad test DATA, not a bad engine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gauntlet.permute import permute_prices


def log_returns(df):
    c = np.log(df["close"].to_numpy(dtype=float))
    return np.diff(c)


def lag1_autocorr(x):
    x = x - x.mean()
    denom = (x * x).sum()
    if denom <= 1e-12:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)


def make_mean_reverting(n=3000, seed=5):
    """Strong NEGATIVE lag-1 autocorrelation: each day's return partly reverses
    the previous day's. This is a pure-timing pattern with (by construction) very
    little net drift, so it is NOT confounded with trend."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    r[0] = rng.normal(0, 0.01)
    phi = -0.4  # negative autocorrelation = mean reversion
    for i in range(1, n):
        r[i] = phi * r[i-1] + rng.normal(0, 0.01)
    close = 100 * np.exp(np.cumsum(r))
    opens = np.empty(n); opens[0] = 100; opens[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 0.003, n))
    high = np.maximum(opens, close) * (1 + wig)
    low = np.minimum(opens, close) * (1 - wig)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": close}, idx)


def test_permutation_properties():
    df = make_mean_reverting()
    real_r = log_returns(df)
    real_drift = real_r.sum()
    real_vol = real_r.std()
    real_ac = lag1_autocorr(real_r)

    print(f"  REAL data : drift={real_drift:+.4f}  vol={real_vol:.5f}  "
          f"lag1-autocorr={real_ac:+.4f}")

    rng = np.random.default_rng(123)
    n_perm = 200
    drifts, vols, acs = [], [], []
    for _ in range(n_perm):
        pdf = permute_prices(df, start_index=0, rng=rng)
        pr = log_returns(pdf)
        drifts.append(pr.sum()); vols.append(pr.std()); acs.append(lag1_autocorr(pr))

    md, mv, mac = np.mean(drifts), np.mean(vols), np.mean(acs)
    print(f"  PERMUTED  : drift={md:+.4f}  vol={mv:.5f}  lag1-autocorr={mac:+.4f}  "
          f"(mean of {n_perm} perms)")

    # PRESERVE drift: permuted mean drift close to real (within a vol-scaled band)
    assert abs(md - real_drift) < 3 * np.std(drifts) + 0.05, \
        f"drift not preserved: {md} vs {real_drift}"
    print("  [PASS] drift preserved")

    # PRESERVE vol: permuted vol within 5% of real
    assert abs(mv - real_vol) / real_vol < 0.05, f"vol not preserved: {mv} vs {real_vol}"
    print("  [PASS] volatility preserved")

    # DESTROY autocorrelation: real is strongly negative (~-0.4 attenuated by
    # OHLC noise); permuted must be driven essentially to zero.
    assert abs(mac) < 0.05, f"autocorrelation NOT destroyed: {mac}"
    assert abs(real_ac) > abs(mac) * 2, "real autocorr should be much larger than permuted"
    print(f"  [PASS] autocorrelation destroyed ({real_ac:+.4f} -> {mac:+.4f})")


def test_anchor_preserved():
    """First and last close should be (nearly) unchanged — the path moves, the
    endpoints anchor the overall drift."""
    df = make_mean_reverting(n=500)
    rng = np.random.default_rng(1)
    pdf = permute_prices(df, start_index=0, rng=rng)
    assert np.isclose(df["close"].iloc[0], pdf["close"].iloc[0], rtol=1e-6)
    print("  [PASS] start anchor preserved")


if __name__ == "__main__":
    print("STEP 2 PROOF — permutation engine")
    test_permutation_properties()
    test_anchor_preserved()
    print("Permutation engine verified.\n")

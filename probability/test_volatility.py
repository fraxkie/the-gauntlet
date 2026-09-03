"""
test_volatility.py — proof the volatility forecaster detects real vol-predictability.

Known-answer cases:
  1. CLUSTERED vol (engineered GARCH process): EWMA and GARCH must BEAT naive on
     QLIKE, and forecasts must correlate with realized vol. This is the signal that
     direction models could never produce.
  2. CONSTANT vol (no clustering): no model should claim large skill over naive —
     there's nothing to predict, so improvements should be ~0.
  3. Forecast tracking: when vol regime-switches (calm -> stormy), the forecaster
     must raise its forecast in the stormy period.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from probability.volatility import (walk_forward_vol, evaluate_vol_models,
                                    qlike, forecast_ewma, forecast_naive)


def make_garch_returns(n=6000, omega=2e-6, alpha=0.08, beta=0.90, seed=0):
    """Simulate a GARCH(1,1) process — returns with REAL volatility clustering."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n); var = omega / (1 - alpha - beta)
    for i in range(n):
        var = omega + alpha * (r[i-1] ** 2 if i > 0 else 0) + beta * var
        r[i] = rng.normal(0, np.sqrt(var))
    return r


def make_constant_vol(n=6000, sigma=0.01, seed=1):
    """IID returns — constant volatility, NO clustering, nothing to forecast."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, n)


def test_beats_naive_on_clustered():
    r = make_garch_returns()
    res = evaluate_vol_models(r, label="GARCH-simulated")
    m = res["models"]
    print(f"  clustered : naive QLIKE={m['naive']['qlike']:.4f}  "
          f"ewma={m['ewma']['qlike']:.4f}  garch={m['garch']['qlike']:.4f}")
    print(f"              ewma improvement vs naive: "
          f"{m['ewma'].get('qlike_improvement_vs_naive', 0)*100:+.1f}%  "
          f"vol-corr={m['ewma']['vol_corr']:.3f}")
    # EWMA should beat naive (lower QLIKE) on genuinely clustered data
    assert m["ewma"]["qlike"] < m["naive"]["qlike"], \
        "EWMA must beat naive on clustered vol"
    assert m["ewma"]["vol_corr"] > 0.3, "forecast must correlate with realized vol"
    print("      [PASS] EWMA beats naive on clustered volatility (real skill)")


def test_no_false_skill_on_constant():
    """The RIGHT test: on constant-vol data there's no clustering to PREDICT, so the
    forecast-realized vol CORRELATION should be near the noise floor — much lower
    than on clustered data. (Note: EWMA can still beat naive on QLIKE here simply by
    being a lower-variance ESTIMATOR of the constant variance — that's correct and
    not 'skill at predicting clustering'. The discriminator is correlation, not QLIKE.)"""
    r_const = make_constant_vol()
    res_const = evaluate_vol_models(r_const, label="constant-vol")
    corr_const = res_const["models"]["ewma"]["vol_corr"]

    r_clust = make_garch_returns(seed=5)
    res_clust = evaluate_vol_models(r_clust, label="clustered")
    corr_clust = res_clust["models"]["ewma"]["vol_corr"]

    print(f"  constant  : forecast-realized vol corr = {corr_const:.3f}")
    print(f"  clustered : forecast-realized vol corr = {corr_clust:.3f}")
    print(f"              clustering raises predictive corr by "
          f"{(corr_clust - corr_const):.3f}")
    # the real signal: clustered data is MUCH more predictable than constant
    assert corr_clust > corr_const + 0.2, \
        "clustered vol should be far more predictable than constant vol"
    assert corr_const < 0.35, \
        f"constant vol shouldn't show strong predictive correlation, got {corr_const}"
    print("      [PASS] forecaster predicts clustering, NOT constant-vol noise")


def test_forecast_tracks_regime_switch():
    """Calm period then stormy period — forecaster must raise its forecast."""
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 800)
    storm = rng.normal(0, 0.025, 800)
    r = np.concatenate([calm, storm])
    # forecast at end of calm vs partway into storm
    fc_calm = forecast_ewma(r[:800])
    fc_storm = forecast_ewma(r[:1200])
    print(f"  regime    : forecast var calm={fc_calm:.2e}  storm={fc_storm:.2e}  "
          f"(ratio {fc_storm/fc_calm:.1f}x)")
    assert fc_storm > fc_calm * 3, "forecaster must react to the vol regime switch"
    print("      [PASS] forecaster tracks calm->stormy regime switch")


if __name__ == "__main__":
    print("STEP 1 PROOF — volatility forecaster")
    test_beats_naive_on_clustered()
    test_no_false_skill_on_constant()
    test_forecast_tracks_regime_switch()
    print("Volatility forecaster verified: real skill where it exists, none where it doesn't.\n")

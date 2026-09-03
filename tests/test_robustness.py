"""
test_robustness.py — proof for walk-forward consistency gate and regime check.

Walk-forward: a strategy profitable in a MAJORITY of folds passes; one whose
profit comes from a single freak fold FAILS.

Regime: an always-long strategy on a rising market must trip the beta warning
(makes money in UP, not in DOWN).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gauntlet.core import Costs
from gauntlet.robustness import walk_forward, regime_check


def _trend_df(n=2400, seed=2):
    """Steady uptrend across all folds — a consistent long strategy wins everywhere."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0007, 0.009, n)
    close = 100 * np.exp(np.cumsum(r))
    opens = np.empty(n); opens[0] = 100; opens[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 0.002, n))
    high = np.maximum(opens, close) * (1 + wig)
    low = np.minimum(opens, close) * (1 - wig)
    idx = pd.date_range("2012-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": close}, idx)


def _one_lucky_fold_df(n=2400, seed=4):
    """Flat/choppy everywhere EXCEPT one fold with a big rally. A long strategy is
    net-positive only because of that one fold — the gate should FAIL it."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.008, n)
    # inject a strong rally in fold 3 of 6 (bars 1200..1600)
    r[1200:1600] += 0.004
    close = 100 * np.exp(np.cumsum(r))
    opens = np.empty(n); opens[0] = 100; opens[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 0.002, n))
    high = np.maximum(opens, close) * (1 + wig)
    low = np.minimum(opens, close) * (1 - wig)
    idx = pd.date_range("2012-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": close}, idx)


def strat_long(df):
    return pd.Series(np.ones(len(df)), index=df.index)


def test_consistency_gate_passes_consistent():
    df = _trend_df()
    res = walk_forward(df, strat_long, Costs())
    print(f"  consistent uptrend : {res['n_profitable']}/{res['n_folds']} folds "
          f"profitable -> gate {res['consistency_gate']}")
    assert res["consistency_gate"] == "PASS"
    print("      [PASS] consistent strategy passes the gate")


def test_consistency_gate_fails_one_lucky_fold():
    df = _one_lucky_fold_df()
    res = walk_forward(df, strat_long, Costs())
    prof = [round(f["total_return_pct"], 1) for f in res["folds"]]
    print(f"  one-lucky-fold     : per-fold returns {prof}")
    print(f"                       {res['n_profitable']}/{res['n_folds']} "
          f"profitable -> gate {res['consistency_gate']}")
    assert res["consistency_gate"] == "FAIL", \
        "gate should FAIL a strategy carried by one freak fold"
    print("      [PASS] one-lucky-fold strategy correctly FAILS the gate")


def test_regime_flags_beta():
    df = _trend_df()
    rg = regime_check(df, strat_long, Costs())
    print(f"  regime (always-long on uptrend): UP {rg['UP']['mean_bps']:+.1f}bps  "
          f"DOWN {rg['DOWN']['mean_bps']:+.1f}bps  "
          f"beta_warning={rg['_beta_warning']}")
    assert rg["_beta_warning"], "always-long on uptrend should trip the beta warning"
    print("      [PASS] beta-in-a-costume correctly flagged")


if __name__ == "__main__":
    print("STEP 5 PROOF — walk-forward consistency gate + regime check")
    test_consistency_gate_passes_consistent()
    test_consistency_gate_fails_one_lucky_fold()
    test_regime_flags_beta()
    print("Robustness firewalls verified.\n")

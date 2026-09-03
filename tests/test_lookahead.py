"""
test_lookahead.py — proof the audit catches future-leaks.

The decisive test: a blatant cheater that uses TOMORROW's close to decide today
must be CAUGHT (the old audit falsely passed it). A legitimate strategy that uses
only past data must PASS.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gauntlet.lookahead import lookahead_audit


def _df(n=500, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.011, n)
    close = 100 * np.exp(np.cumsum(r))
    opens = np.empty(n); opens[0] = 100; opens[1:] = close[:-1]
    wig = np.abs(rng.normal(0, 0.003, n))
    high = np.maximum(opens, close) * (1 + wig)
    low = np.minimum(opens, close) * (1 - wig)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": close}, idx)


def strat_legit(df):
    """Legitimate: long if close > 20-bar moving average. Uses only past/current."""
    close = df["close"]
    ma = close.rolling(20).mean()
    pos = (close > ma).astype(float)
    pos[ma.isna()] = 0.0
    return pd.Series(pos.to_numpy(), index=df.index)


def strat_cheater(df):
    """CHEATER: long if TOMORROW's close is higher. Pure look-ahead leak. The
    position at bar i depends on bar i+1, so hiding the future MUST change it."""
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    pos = np.zeros(n)
    # bar i looks at i+1; the last bar has no tomorrow, so it stays 0
    pos[:-1] = (close[1:] > close[:-1]).astype(float)
    return pd.Series(pos, index=df.index)


def strat_subtle_leak(df):
    """A subtler, realistic leak: uses a CENTERED moving average (includes future
    bars in the window). The kind of bug that happens by accident with the wrong
    pandas call."""
    close = df["close"]
    # center=True makes the window straddle the current bar -> uses future
    ma = close.rolling(20, center=True).mean()
    pos = (close > ma).astype(float)
    pos[ma.isna()] = 0.0
    return pd.Series(pos.to_numpy(), index=df.index)


def test_legit_passes():
    df = _df()
    res = lookahead_audit(df, strat_legit)
    print(f"  legit strategy   : {res['note']}")
    assert res["look_ahead_clean"], f"legit strategy wrongly flagged: {res['mismatches']}"
    print("      [PASS] legitimate strategy passes audit")


def test_cheater_caught():
    df = _df()
    res = lookahead_audit(df, strat_cheater)
    print(f"  blatant cheater  : {res['note']}")
    assert not res["look_ahead_clean"], "CHEATER PASSED — audit still broken!"
    assert res["n_mismatches"] > 0
    print(f"      [PASS] peek-at-tomorrow cheater CAUGHT "
          f"({res['n_mismatches']} mismatches)")


def test_subtle_leak_caught():
    df = _df()
    res = lookahead_audit(df, strat_subtle_leak)
    print(f"  subtle (centered): {res['note']}")
    assert not res["look_ahead_clean"], "subtle centered-MA leak slipped through!"
    print(f"      [PASS] subtle centered-window leak CAUGHT "
          f"({res['n_mismatches']} mismatches)")


if __name__ == "__main__":
    print("STEP 4 PROOF — look-ahead audit (rebuilt)")
    test_legit_passes()
    test_cheater_caught()
    test_subtle_leak_caught()
    print("Look-ahead audit verified: catches blatant AND subtle future-leaks.\n")

"""
test_core.py — proof that the foundation is correct.

Three claims to verify:
  1. Costs are charged correctly on each turn.
  2. A position earns the NEXT bar's return, not the current one (no look-ahead).
  3. profit_factor / total_return behave sanely on hand-checkable inputs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gauntlet.core import (Costs, strategy_returns, bar_returns,
                           profit_factor, total_return_pct)


def _toy_df(closes):
    closes = np.array(closes, dtype=float)
    n = len(closes)
    opens = np.empty(n); opens[0] = closes[0]; opens[1:] = closes[:-1]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": opens, "high": closes, "low": closes,
                         "close": closes}, index=idx)


def test_costs_arithmetic():
    # flat prices so returns are zero; only costs should show up
    df = _toy_df([100, 100, 100, 100])
    # go long at bar 1 (flat->long = one turn), stay long, exit at bar 3
    pos = pd.Series([0, 1, 1, 0], index=df.index)
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)  # 3 bps per side
    net = strategy_returns(df, pos, costs).to_numpy()
    # turns: bar1 flat->long (|1-0|=1), bar3 long->flat (|0-1|=1). Each costs 3bps.
    # prices flat so gross=0; net should be -3bps on bar1 and -3bps on bar3.
    assert np.isclose(net[1], -0.0003), net[1]
    assert np.isclose(net[3], -0.0003), net[3]
    assert np.isclose(net[0], 0.0) and np.isclose(net[2], 0.0)
    print("  [PASS] costs charged correctly on each turn")


def test_no_look_ahead():
    """The decisive test. A strategy that goes long on bar i must earn bar i+1's
    return, NOT bar i's. We build a price series with one big up-move and confirm
    the position must be set the bar BEFORE to capture it."""
    # prices: flat, then +10% on bar 2, then flat
    df = _toy_df([100, 100, 110, 110, 110])
    costs = Costs(0, 0)  # no costs, isolate the timing

    # Strategy A: long ON bar 2 (the up bar itself). With no look-ahead, the
    # position is held into bar 3, so it earns bar 3's return (0%), NOT the +10%.
    posA = pd.Series([0, 0, 1, 0, 0], index=df.index)
    netA = strategy_returns(df, posA, costs).to_numpy()
    assert np.isclose(netA.sum(), 0.0), f"A captured {netA.sum()} — LOOK-AHEAD LEAK"

    # Strategy B: long on bar 1 (the bar BEFORE the move). Position held into
    # bar 2, earns bar 2's +10%. This is the legitimate way to capture it.
    posB = pd.Series([0, 1, 0, 0, 0], index=df.index)
    netB = strategy_returns(df, posB, costs).to_numpy()
    assert netB.sum() > 0.09, f"B should capture ~+10%, got {netB.sum()}"
    print("  [PASS] no-look-ahead: a position earns the NEXT bar's return only")


def test_profit_factor():
    # three wins of +1, two losses of -1  -> PF = 3/2 = 1.5
    r = np.array([0.0, 1, -1, 1, -1, 1])
    assert np.isclose(profit_factor(r), 1.5), profit_factor(r)
    # all positive -> inf
    assert profit_factor(np.array([0.1, 0.2])) == float("inf")
    print("  [PASS] profit_factor arithmetic correct")


def test_total_return():
    # +10% then -10% compounded = 0.99 -> -1.0%
    r = np.array([0.10, -0.10])
    assert np.isclose(total_return_pct(r), -1.0), total_return_pct(r)
    print("  [PASS] total_return compounding correct")


if __name__ == "__main__":
    print("STEP 1 PROOF — foundation")
    test_costs_arithmetic()
    test_no_look_ahead()
    test_profit_factor()
    test_total_return()
    print("All foundation checks passed.\n")

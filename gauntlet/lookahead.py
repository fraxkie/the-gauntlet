"""
lookahead.py  —  STEP 4: the look-ahead audit (rebuilt)
═══════════════════════════════════════════════════════════════════════════════
A future-leak makes every p-value a lie: a strategy that peeks at tomorrow looks
incredible in backtest and dies live. The first version of this audit FALSELY
PASSED a blatant cheater, so it's rebuilt here with a stronger, harder-to-fool
test.

The principle: a legitimate strategy's decision on bar i depends ONLY on data
through bar i. So if we hide everything after bar i, the decision for bar i must
not change. We test this by truncation:

  For several cut points k:
    • run the strategy on the FULL data, record the position at bar k-1
    • run the strategy on data TRUNCATED at bar k (bars 0..k-1 only)
    • the position at bar k-1 must be IDENTICAL

A strategy that peeks ahead will decide bar k-1 differently when bars >= k are
present vs absent — and this catches it. We test at many cut points because a
leak might only manifest at certain bars.

Why the old version failed: it compared the full vs truncated positions over the
whole overlap but the cheater I wrote recomputed its leak relative to each array's
OWN length, so a uniform shift hid the mismatch. The fix: compare the position at
the LAST bar of each truncation (bar k-1) specifically — the bar that, in the
truncated run, has NO future after it. A leaker is forced to differ there.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from gauntlet.core import Strategy


def lookahead_audit(df: pd.DataFrame, strat: Strategy,
                    n_cuts: int = 12, warmup: int = 50) -> dict:
    """
    Returns look_ahead_clean=True iff the strategy's decision for the final bar of
    each truncation matches its decision for that same bar with full data present.

    warmup : skip the first `warmup` bars so indicators that need history aren't
             flagged for the legitimate reason of not having warmed up yet.
    """
    full_pos = strat(df).to_numpy(dtype=float)
    n = len(df)

    cut_points = np.linspace(warmup + 5, n - 1, n_cuts, dtype=int)
    cut_points = np.unique(cut_points)

    mismatches = []
    for k in cut_points:
        # truncated run sees bars 0..k-1; its LAST decided bar is k-1, which has
        # no future after it in this run. A clean strategy decides k-1 the same
        # way whether or not bars >= k exist.
        trunc_pos = strat(df.iloc[:k]).to_numpy(dtype=float)
        if len(trunc_pos) < k:
            continue
        full_val = full_pos[k - 1]
        trunc_val = trunc_pos[k - 1]
        if not np.isclose(full_val, trunc_val, equal_nan=True):
            mismatches.append({
                "bar": int(k - 1),
                "full_decision": float(full_val),
                "decision_with_future_hidden": float(trunc_val),
            })

    clean = len(mismatches) == 0
    return {
        "look_ahead_clean": clean,
        "n_cuts_tested": len(cut_points),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:5],  # first few for the report
        "note": ("PASS — decisions identical when future is hidden (no leak)"
                 if clean else
                 f"FAIL — {len(mismatches)} decision(s) changed when future was "
                 f"hidden (LOOK-AHEAD LEAK)"),
    }

"""
robustness.py  —  STEP 5: walk-forward + consistency gate, and regime check
═══════════════════════════════════════════════════════════════════════════════
Two more independent firewalls, each catching a different lie:

WALK-FORWARD + CONSISTENCY GATE (doc-41's lesson):
  Split the timeline into folds; measure each fold's out-of-sample result
  INDEPENDENTLY. A real edge wins a MAJORITY of folds. A strategy that's net-
  positive only because one freak fold carried it FAILS the gate. "Did enough
  folds make money independently?" — one or two could be luck; a majority is a
  fingerprint.

REGIME SEGMENTATION (the council's death):
  Classify each bar as UP / DOWN / FLAT market regime and report the strategy's
  return within each. A strategy that ONLY makes money in UP regimes is beta in a
  costume — it's long the market, not exploiting an edge. The tell is whether it
  makes money when the market ISN'T going up.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from gauntlet.core import Costs, Strategy, strategy_returns, profit_factor, total_return_pct


def walk_forward(df: pd.DataFrame, strat: Strategy, costs: Costs,
                 n_folds: int = 6, gate_threshold: float = 0.6) -> dict:
    """Independent out-of-sample folds + consistency gate."""
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        lo = f * fold_size
        hi = n if f == n_folds - 1 else (f + 1) * fold_size
        sub = df.iloc[lo:hi]
        if len(sub) < 10:
            continue
        net = strategy_returns(sub, strat(sub), costs).to_numpy()
        folds.append({
            "fold": f,
            "total_return_pct": total_return_pct(net),
            "profit_factor": profit_factor(net),
            "n_bars": len(sub),
        })

    n_prof = sum(1 for r in folds if r["total_return_pct"] > 0)
    total = len(folds)
    frac = n_prof / total if total else 0.0
    return {
        "folds": folds,
        "n_profitable": n_prof,
        "n_folds": total,
        "pct_profitable": round(100 * frac, 1),
        "consistency_gate": "PASS" if frac >= gate_threshold else "FAIL",
        "gate_threshold_pct": int(gate_threshold * 100),
    }


def regime_check(df: pd.DataFrame, strat: Strategy, costs: Costs,
                 ma_window: int = 50) -> dict:
    """Performance segmented by market regime. Flags beta-in-a-costume.

    IMPORTANT: we report MEAN return per bar (in bps), NOT compounded total return.
    Compounding a non-contiguous SUBSET of days (e.g. all UP-regime days scattered
    across 20 years) is not a real equity curve — a handful of large-magnitude bars
    multiply into astronomical, meaningless numbers. Mean-per-bar is bounded, honest,
    and directly answers 'does the strategy make money in this regime?'. Profit
    factor (already bounded) is kept as a secondary read."""
    close = df["close"]
    ma = close.rolling(ma_window).mean()
    slope = ma.diff()

    regime = pd.Series("FLAT", index=df.index)
    regime[(close > ma) & (slope > 0)] = "UP"
    regime[(close < ma) & (slope < 0)] = "DOWN"

    net = strategy_returns(df, strat(df), costs)
    out = {}
    for r in ("UP", "DOWN", "FLAT"):
        mask = (regime == r).to_numpy()
        rr = net.to_numpy()[mask]
        rr = rr[np.isfinite(rr)]
        out[r] = ({"n_bars": 0, "mean_bps": 0.0, "profit_factor": 1.0}
                  if len(rr) == 0 else
                  {"n_bars": int(len(rr)),
                   "mean_bps": float(rr.mean() * 1e4),   # mean return per bar, bps
                   "profit_factor": profit_factor(rr)})
    # beta warning: positive mean in UP, non-positive in DOWN
    out["_beta_warning"] = (out["UP"]["mean_bps"] > 0
                            and out["DOWN"]["mean_bps"] <= 0)
    return out

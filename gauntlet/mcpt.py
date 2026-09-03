"""
mcpt.py  —  STEP 3: the Monte Carlo Permutation Test (the heart firewall)
═══════════════════════════════════════════════════════════════════════════════
Built on the VERIFIED permutation engine (step 2). One question:

   "Could a worthless strategy have scored as well as ours, just by luck?"

Method:
  1. Score the strategy on the REAL data (profit factor or sharpe, after costs).
  2. Generate N permutations (drift/vol preserved, pattern destroyed).
  3. Re-run the SAME strategy on each permutation and score it.
  4. p-value = (# permutations scoring >= real + 1) / (N + 1).

Interpretation:
  • low p  (< 0.01): the real score is hard to achieve on patternless data, so
                     the edge is unlikely to be luck. KEEP investigating.
  • high p (>= 0.05): the strategy scores just as well on noise — the "edge" is
                     indistinguishable from chance. DISCARD, however pretty.

This is the ONLY firewall that re-runs the strategy logic itself on synthetic
data, and the only one that catches the council's death (a long-biased strategy
riding drift fails here, because permutation preserves the drift but the strategy
has no real pattern to exploit once the path is shuffled).
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from gauntlet.core import Costs, Strategy, strategy_returns, profit_factor, sharpe
from gauntlet.permute import permute_prices


def mcpt(df: pd.DataFrame, strat: Strategy, costs: Costs,
         n_perm: int = 1000, objective: str = "pf",
         start_index: int = 0, seed: int = 42) -> dict:
    """
    In-sample (start_index=0) or walk-forward (start_index>0) permutation test.

    objective : "pf" (profit factor) or "sharpe".
    start_index : if >0, only permute bars from here on; the strategy is still
                  scored on the full series. Used for the walk-forward MCPT.
    """
    obj = profit_factor if objective == "pf" else sharpe
    rng = np.random.default_rng(seed)

    real_pos = strat(df)
    real_score = obj(strategy_returns(df, real_pos, costs).to_numpy())

    perm_scores = np.empty(n_perm)
    count_ge = 1  # include the real result (standard; prevents p=0)
    for j in range(n_perm):
        pdf = permute_prices(df, start_index=start_index, rng=rng)
        ppos = strat(pdf)
        pscore = obj(strategy_returns(pdf, ppos, costs).to_numpy())
        perm_scores[j] = pscore
        if pscore >= real_score:
            count_ge += 1

    p_value = count_ge / (n_perm + 1)
    return {
        "real_score": float(real_score),
        "p_value": float(p_value),
        "perm_mean": float(np.nanmean(perm_scores)),
        "perm_95th": float(np.nanpercentile(perm_scores, 95)),
        "n_perm": n_perm,
        "objective": objective,
        "verdict": ("REAL (unlikely luck)" if p_value < 0.01
                    else "SUSPECT" if p_value < 0.05
                    else "NOISE (discard)"),
    }

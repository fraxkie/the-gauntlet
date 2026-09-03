"""
scan_pairs.py — systematic search for a LIVE market-neutral pairs edge.

Your grinding idea, done the disciplined way:
  • Test every pair across the 13-ETF universe (78 pairs).
  • Judge each on RECENT consistency (last 3 folds ≈ 2015-2026), NOT the full
    21-year history — because EFA/EEM taught us an edge can be real-but-decayed,
    and we want one that's alive NOW.
  • Log EVERY test to the persistent trial ledger.
  • Apply the multiple-comparisons correction: with ~78 trials, the p-value bar to
    claim a real find becomes far harder than the naive 0.01. This is what stops
    the scan from manufacturing fool's gold.

A pair is a CANDIDATE only if it clears the corrected bar AND is profitable in the
recent folds AND is genuinely market-neutral. We expect very few — possibly none.
That's the honest outcome, not a disappointment.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, total_return_pct, profit_factor
from gauntlet.data import load_etf_data
from gauntlet.mcpt import mcpt
from gauntlet.lookahead import lookahead_audit
from gauntlet.ledger import log_trial, assess, LEDGER_PATH
from run_pairs import make_spread_df, strat_spread_reversion


def pairs_pnl(sdf, costs):
    pos = strat_spread_reversion(sdf).to_numpy(float)
    spread_ret = np.nan_to_num(sdf.attrs["ret_a"] - sdf.attrs["ret_b"])
    pos_held = np.zeros_like(pos); pos_held[1:] = pos[:-1]
    gross = pos_held * spread_ret
    turn = np.zeros_like(pos); turn[1:] = np.abs(pos[1:] - pos[:-1])
    net = gross - turn * costs.per_turn * 2.0
    return net, pos


def recent_consistency(net, n_folds=6, recent=3):
    """Profitable-fold count over the LAST `recent` folds only."""
    n = len(net); fold = n // n_folds
    flags = []
    for f in range(n_folds - recent, n_folds):
        lo = f * fold; hi = n if f == n_folds - 1 else (f + 1) * fold
        flags.append(total_return_pct(net[lo:hi]) > 0)
    return sum(flags), recent


if __name__ == "__main__":
    # fresh ledger for this scan (so the count reflects THIS search honestly)
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)

    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)
    tickers = sorted(data.keys())
    all_pairs = list(itertools.combinations(tickers, 2))

    print("#" * 72)
    print(f"#  PAIRS SCAN — {len(all_pairs)} pairs, judged on RECENT consistency")
    print(f"#  Every test logged. Multiple-comparisons correction applied at the end.")
    print("#" * 72)

    results = []
    for a, b in all_pairs:
        sdf = make_spread_df(data[a], data[b])
        net, pos = pairs_pnl(sdf, costs)
        if (pos != 0).sum() < 50:        # too few trades to judge
            continue
        full_ret = total_return_pct(net)
        rec_prof, rec_total = recent_consistency(net)
        # recent-only return (last 3 folds)
        n = len(net); fold = n // 6
        recent_net = net[(6 - 3) * fold:]
        rec_ret = total_return_pct(recent_net)

        mc = mcpt(sdf, strat_spread_reversion, costs, n_perm=300, objective="pf")
        log_trial(f"PAIRS {a}/{b}", mc["p_value"], full_ret,
                  extra={"recent_return": rec_ret,
                         "recent_folds_profitable": f"{rec_prof}/{rec_total}"})
        results.append({
            "pair": f"{a}/{b}", "p": mc["p_value"], "full_ret": full_ret,
            "recent_ret": rec_ret, "rec_prof": rec_prof,
        })

    # apply the correction
    a_assess = assess()
    corrected_bar = a_assess["corrected_bar"]

    print(f"\n  Tested {a_assess['n_trials']} pairs (with enough trades).")
    print(f"  Naive bar p<0.01 would let ~{a_assess['expected_false_positives_at_naive']} "
          f"pairs through BY LUCK.")
    print(f"  Corrected bar (Bonferroni): p < {corrected_bar:.2e}\n")

    # rank by recent return among those that are recently-consistent (3/3 folds)
    live = [r for r in results if r["rec_prof"] == 3 and r["recent_ret"] > 0]
    live.sort(key=lambda r: r["p"])

    print("  " + "═" * 68)
    print("  PAIRS THAT ARE PROFITABLE IN ALL 3 RECENT FOLDS (alive now):")
    print("  " + "═" * 68)
    if not live:
        print("  None. No pair is consistently profitable in recent folds.")
    else:
        print(f"  {'pair':10} {'recent ret':>11} {'full ret':>10} {'MCPT p':>9} {'verdict':>20}")
        for r in live:
            passes = "CANDIDATE" if r["p"] < corrected_bar else "fails corrected bar"
            print(f"  {r['pair']:10} {r['recent_ret']:>+10.1f}% {r['full_ret']:>+9.1f}% "
                  f"{r['p']:>9.4f} {passes:>20}")

    # also show the top by MCPT p regardless, for context
    print("\n  " + "─" * 68)
    print("  Lowest MCPT p-values overall (context — most may be decayed):")
    for r in sorted(results, key=lambda x: x["p"])[:8]:
        print(f"    {r['pair']:10} p={r['p']:.4f}  full={r['full_ret']:+.1f}%  "
              f"recent={r['recent_ret']:+.1f}%  recent-folds={r['rec_prof']}/3")

    n_candidates = sum(1 for r in live if r["p"] < corrected_bar)
    print("\n  " + "═" * 68)
    print(f"  TRUE CANDIDATES (recent-consistent AND clear corrected bar): {n_candidates}")
    print("  " + "═" * 68)

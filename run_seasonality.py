"""
run_seasonality.py — calendar/seasonality strategies through the gauntlet.

A DIFFERENT family than momentum (beta) or pairs (decayed). Seasonality bets on
recurring CALENDAR patterns in flows — not on trend, not on reversion. Documented
effects we test:

  • TURN-OF-MONTH: equities tend to drift up around the last/first few trading days
    of a month (pension/401k inflows, window dressing). Long the last N and first M
    trading days; flat otherwise.
  • DAY-OF-WEEK: historical Monday-weakness / midweek-strength patterns.
  • SELL-IN-MAY: long Nov-Apr, flat/short May-Oct (the "Halloween" effect).

IMPORTANT CAVEAT we keep front of mind: a long-only seasonal strategy on an asset
that rose over 21 years will show profit partly from being long during an uptrend —
so the REGIME/beta check and MCPT matter as much here as anywhere. MCPT is well-
suited to seasonality: permutation destroys the calendar alignment (the pattern
lives in WHICH days, which shuffling scrambles), so a real seasonal edge should
show a low p-value while preserved drift means pure-beta seasonals get rejected.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, strategy_returns, profit_factor, total_return_pct
from gauntlet.data import load_etf_data
from gauntlet.mcpt import mcpt
from gauntlet.lookahead import lookahead_audit
from gauntlet.robustness import walk_forward, regime_check
from gauntlet.ledger import log_trial


def strat_turn_of_month(df, days_before=3, days_after=2):
    """Long around the turn of month (last `days_before` + first `days_after`
    trading days), flat otherwise. Uses only the calendar — no future data."""
    idx = df.index
    month = idx.to_period("M")
    pos = np.zeros(len(idx))
    # for each month, mark the last days_before and first days_after trading days
    df_tmp = pd.DataFrame({"m": month}, index=idx)
    for _, grp in df_tmp.groupby("m"):
        positions_in_month = [idx.get_loc(t) for t in grp.index]
        for p in positions_in_month[:days_after]:        # first days of month
            pos[p] = 1.0
        for p in positions_in_month[-days_before:]:       # last days of month
            pos[p] = 1.0
    return pd.Series(pos, index=idx)


def strat_sell_in_may(df):
    """Long November through April, flat May through October."""
    idx = df.index
    pos = np.where(idx.month.isin([11, 12, 1, 2, 3, 4]), 1.0, 0.0)
    return pd.Series(pos, index=idx)


def strat_day_of_week(df, long_days=(1, 2, 3)):
    """Long on chosen weekdays (default Tue/Wed/Thu = 1,2,3), flat otherwise."""
    idx = df.index
    pos = np.where(np.isin(idx.dayofweek, long_days), 1.0, 0.0)
    return pd.Series(pos, index=idx)


def run(name, df, strat, market_df, costs, n_perm=500, log=True):
    audit = lookahead_audit(df, strat)
    mc = mcpt(df, strat, costs, n_perm=n_perm, objective="pf")
    wf = walk_forward(df, strat, costs)
    rg = regime_check(df, strat, costs)
    net = strategy_returns(df, strat(df), costs).to_numpy()

    survives = (audit["look_ahead_clean"] and mc["p_value"] < 0.01
                and wf["consistency_gate"] == "PASS")
    if log:
        log_trial(name, mc["p_value"], total_return_pct(net),
                  extra={"family": "seasonality"})

    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(net):+.1f}%  "
          f"PF {profit_factor(net):.3f}  ({(strat(df).to_numpy()!=0).sum()} bars long)")
    print(f"  LOOK-AHEAD AUDIT : {audit['note']}")
    print(f"  MCPT ({mc['n_perm']} perms): real PF={mc['real_score']:.3f}  "
          f"noise-mean={mc['perm_mean']:.3f}  p-value={mc['p_value']:.4f}  -> {mc['verdict']}")
    print(f"  WALK-FORWARD     : {wf['n_profitable']}/{wf['n_folds']} folds "
          f"({wf['pct_profitable']}%)  -> gate {wf['consistency_gate']}")
    print(f"  REGIME           : UP {rg['UP']['mean_bps']:+.1f}bps  | "
          f"DOWN {rg['DOWN']['mean_bps']:+.1f}bps  | FLAT {rg['FLAT']['mean_bps']:+.1f}bps"
          + ("   [BETA WARNING]" if rg["_beta_warning"] else ""))
    print("  " + "-" * 68)
    print(f"  VERDICT          : {'*** SURVIVES ***' if survives else 'KILLED'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "p": mc["p_value"],
            "ret": total_return_pct(net), "beta": rg["_beta_warning"]}


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)

    print("#" * 72)
    print("#  SEASONALITY — calendar strategies through the gauntlet")
    print("#  A different family: betting on calendar patterns, not trend/reversion.")
    print("#" * 72)

    results = []
    # test each seasonal effect on the broad equity ETFs where flows concentrate
    for tk in ["SPY", "QQQ", "IWM", "DIA"]:
        results.append(run(f"TURN-OF-MONTH {tk}", data[tk], strat_turn_of_month,
                           data["SPY"], costs, n_perm=500))
    for tk in ["SPY", "QQQ"]:
        results.append(run(f"SELL-IN-MAY {tk}", data[tk], strat_sell_in_may,
                           data["SPY"], costs, n_perm=500))
    results.append(run("DAY-OF-WEEK SPY", data["SPY"], strat_day_of_week,
                       data["SPY"], costs, n_perm=500))

    print("\n\n" + "═" * 72)
    print("  SEASONALITY SUMMARY")
    print("═" * 72)
    ns = sum(1 for r in results if r["survives"])
    print(f"  Survivors: {ns}/{len(results)}")
    for r in sorted(results, key=lambda x: x["p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        beta = " [BETA]" if r["beta"] else ""
        print(f"    {r['name']:22s} return {r['ret']:+7.1f}%  MCPT p={r['p']:.4f}  "
              f"[{flag}]{beta}")
    print("═" * 72)

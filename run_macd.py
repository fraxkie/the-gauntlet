"""
run_macd.py — FIRST REAL TEST: the MACD/200MA strategy through the full gauntlet.

Strategy (the cleanly-codeable Tier-1 candidate from the YouTube batch, doc 33):
  • MACD line (12,26 EMA diff) crosses ABOVE its 9-EMA signal line
  • AND price is above the 200-day moving average (trend filter)
  • -> go long; flat otherwise
  This is a trend-following momentum entry. We run it on SPY first, then the whole
  universe, through all five firewalls.

Expectation, stated up front: most published strategies FAIL the gauntlet. Watching
the engine kill a real-but-mediocre strategy is itself the result — it proves the
firewalls bite on real data, not just synthetic tests.
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


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def strat_macd_200ma(df: pd.DataFrame) -> pd.Series:
    """Long when MACD crosses above signal AND close > 200-day MA. Flat otherwise.
    Position persists while MACD stays above signal and price stays above the MA
    (a held trend position), exits when either condition breaks."""
    close = df["close"]
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    ma200 = close.rolling(200).mean()

    long_ok = (macd_line > signal_line) & (close > ma200)
    pos = long_ok.astype(float)
    # warmup: no position until the 200-MA exists
    pos[ma200.isna()] = 0.0
    return pd.Series(pos.to_numpy(), index=df.index)


def run_full_gauntlet(name, df, strat, costs, n_perm=1000):
    audit = lookahead_audit(df, strat)
    mc = mcpt(df, strat, costs, n_perm=n_perm, objective="pf")
    wf = walk_forward(df, strat, costs)
    rg = regime_check(df, strat, costs)
    net = strategy_returns(df, strat(df), costs).to_numpy()

    survives = (audit["look_ahead_clean"]
                and mc["p_value"] < 0.01
                and wf["consistency_gate"] == "PASS")

    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(net):+.1f}%  "
          f"profit factor {profit_factor(net):.3f}  over {len(df)} bars")
    print(f"  LOOK-AHEAD AUDIT : {audit['note']}")
    print(f"  MCPT ({mc['n_perm']} perms): real PF={mc['real_score']:.3f}  "
          f"noise-mean={mc['perm_mean']:.3f}  p-value={mc['p_value']:.4f}  "
          f"-> {mc['verdict']}")
    print(f"  WALK-FORWARD     : {wf['n_profitable']}/{wf['n_folds']} folds "
          f"profitable ({wf['pct_profitable']}%)  -> gate {wf['consistency_gate']}")
    print(f"  REGIME           : UP {rg['UP']['mean_bps']:+.1f}bps  | "
          f"DOWN {rg['DOWN']['mean_bps']:+.1f}bps  | "
          f"FLAT {rg['FLAT']['mean_bps']:+.1f}bps"
          + ("   [BETA WARNING]" if rg["_beta_warning"] else ""))
    print("  " + "-" * 68)
    print(f"  VERDICT          : "
          f"{'*** SURVIVES ***' if survives else 'KILLED — does not deserve real money'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "mcpt_p": mc["p_value"],
            "total_return": total_return_pct(net), "pf": profit_factor(net)}


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)

    print("#" * 72)
    print("#  FIRST REAL TEST — MACD/200MA strategy on real ETF data (2005-2026)")
    print("#  Expectation: most published strategies fail. Let's see.")
    print("#" * 72)

    # SPY first — the headline test
    run_full_gauntlet("MACD/200MA on SPY", data["SPY"], strat_macd_200ma, costs, n_perm=1000)

    # then the rest of the liquid majors to see if it's a fluke on one ticker
    print("\n\n" + "#" * 72)
    print("#  SAME STRATEGY across the universe (does it work broadly or just SPY?)")
    print("#" * 72)
    summary = []
    for tk in ["QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "TLT", "GLD", "EFA", "EEM"]:
        r = run_full_gauntlet(f"MACD/200MA on {tk}", data[tk], strat_macd_200ma,
                              costs, n_perm=500)
        summary.append(r)

    print("\n\n" + "═" * 72)
    print("  UNIVERSE SUMMARY")
    print("═" * 72)
    n_survive = sum(1 for r in summary if r["survives"])
    print(f"  Survivors: {n_survive}/{len(summary)} tickers")
    for r in sorted(summary, key=lambda x: x["mcpt_p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        print(f"    {r['name'][-3:]}: return {r['total_return']:+7.1f}%  "
              f"PF {r['pf']:.3f}  MCPT p={r['mcpt_p']:.3f}  [{flag}]")
    print("═" * 72)

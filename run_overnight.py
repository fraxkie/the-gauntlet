"""
run_overnight.py — the OVERNIGHT EFFECT through the gauntlet.

A genuinely different angle: every prior strategy used only CLOSE prices. This one
uses the gap structure in the full OHLC — information we'd been throwing away.

THE PHENOMENON (real, documented, visible in our data): the large majority of many
equity ETFs' long-run return accrues OVERNIGHT (prior close -> next open) rather
than INTRADAY (open -> close). SPY: +432% overnight vs +70% intraday over 21y.
QQQ: +940% vs +103%. Candidate explanations: overnight concentration of news/
earnings, futures-driven gaps, liquidity-premium for holding overnight risk.

STRATEGIES:
  • OVERNIGHT-HOLD: hold from each close to the next open, flat during the day.
    Captures the overnight return. The question is whether it survives COSTS (a
    round-trip every single day) and whether it's more than beta.
  • INTRADAY-SHORT- a control: hold open->close. If overnight is where the return
    is, intraday should be weak/negative — a useful sanity check, not a strategy.
  • OVERNIGHT-MINUS-INTRADAY: long overnight, short intraday (same instrument).
    This is the interesting one — it's structurally MARKET-NEUTRAL-ish because
    it's long and short the SAME asset on different clocks, isolating the
    overnight premium from the asset's overall drift.

IMPORTANT: the existing gauntlet (mcpt/walk-forward) operates on a single 'close'
series. Overnight/intraday returns aren't close-to-close, so we compute the net
return streams directly and run permutation + walk-forward + regime on those
streams via small adapters here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, profit_factor, total_return_pct, mean_bps
from gauntlet.data import load_etf_data
from gauntlet.ledger import log_trial, assess


def overnight_returns(df):
    """prior close -> today open."""
    o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
    r = np.zeros(len(c)); r[1:] = o[1:] / c[:-1] - 1.0
    return r


def intraday_returns(df):
    """today open -> today close."""
    o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
    return c / o - 1.0


def apply_costs_daily(gross, costs, turns_per_bar=1.0):
    """Overnight-hold trades once per day (enter at close, exit at open). Charge a
    round-trip cost each active bar."""
    cost = costs.per_turn * 2.0 * turns_per_bar   # round trip
    return gross - cost


def permutation_pvalue(returns_stream, n_perm=1000, seed=42):
    """Permutation test for a raw return STREAM (not OHLC): shuffle the per-bar
    returns to destroy any time structure, compare profit factor. For a pure
    'always hold' overnight stream this mainly tests whether the SIGN/consistency
    is more than chance given the same return distribution."""
    rng = np.random.default_rng(seed)
    r = returns_stream[np.isfinite(returns_stream)]
    real_pf = profit_factor(r)
    count_ge = 1
    for _ in range(n_perm):
        perm = rng.permutation(r)
        # a permuted always-on stream has identical PF (sum-invariant) — so for the
        # always-hold case we instead test against a SIGN-RANDOMIZED null: randomly
        # flip the sign of each day's return (is being LONG better than random
        # direction?). This is the meaningful null for a directional hold.
        signs = rng.choice([-1.0, 1.0], size=len(r))
        ppf = profit_factor(r * signs)
        if ppf >= real_pf:
            count_ge += 1
    return real_pf, count_ge / (n_perm + 1)


def walk_forward_stream(net, n_folds=6, thr=0.6):
    n = len(net); fold = n // n_folds
    prof = []
    for f in range(n_folds):
        lo = f*fold; hi = n if f==n_folds-1 else (f+1)*fold
        seg = net[lo:hi]; seg = seg[np.isfinite(seg)]
        prof.append(total_return_pct(seg) > 0)
    return sum(prof), n_folds, ("PASS" if sum(prof)/n_folds >= thr else "FAIL")


def regime_stream(net, spy_close, index):
    spy_ret = spy_close.reindex(index).pct_change().to_numpy()
    up = spy_ret > 0; dn = spy_ret < 0
    return mean_bps(net[up & np.isfinite(net)]), mean_bps(net[dn & np.isfinite(net)])


def run_overnight(name, df, kind, spy, costs, n_perm=1000, log=True):
    if kind == "overnight":
        gross = overnight_returns(df)
        net = apply_costs_daily(gross, costs)        # daily round-trip cost
    elif kind == "intraday":
        gross = intraday_returns(df)
        net = apply_costs_daily(gross, costs)
    elif kind == "on_minus_id":
        gross = overnight_returns(df) - intraday_returns(df)
        net = apply_costs_daily(gross, costs, turns_per_bar=2.0)  # both legs
    else:
        raise ValueError(kind)

    netf = net[np.isfinite(net)]
    real_pf, p = permutation_pvalue(net, n_perm=n_perm)
    nprof, nf, gate = walk_forward_stream(net)
    up_bps, dn_bps = regime_stream(net, spy["close"], df.index)
    beta = (up_bps > 0 and dn_bps <= 0)

    survives = (p < 0.01 and gate == "PASS")
    if log:
        log_trial(name, p, total_return_pct(netf),
                  extra={"family": "overnight", "kind": kind})

    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(netf):+.1f}%  "
          f"PF {profit_factor(netf):.3f}  (after daily round-trip costs)")
    print(f"  PERMUTATION      : real PF={real_pf:.3f}  p={p:.4f}  "
          f"-> {'REAL' if p<0.01 else 'SUSPECT' if p<0.05 else 'NOISE'}")
    print(f"  WALK-FORWARD     : {nprof}/{nf} folds  -> gate {gate}")
    print(f"  REGIME (vs SPY)  : UP-days {up_bps:+.1f}bps | DOWN-days {dn_bps:+.1f}bps"
          + ("  [BETA WARNING]" if beta else "  [not pure beta]"))
    print("  " + "-" * 68)
    print(f"  VERDICT          : {'*** SURVIVES ***' if survives else 'KILLED'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "p": p,
            "ret": total_return_pct(netf), "pf": profit_factor(netf),
            "up": up_bps, "dn": dn_bps, "beta": beta}


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)
    spy = data["SPY"]

    print("#" * 72)
    print("#  THE OVERNIGHT EFFECT — using the OHLC gap structure prior strategies")
    print("#  ignored. Most ETF return accrues overnight; is it tradeable after costs?")
    print("#" * 72)

    results = []
    # overnight-hold on the major equity ETFs
    for tk in ["SPY", "QQQ", "XLK", "IWM", "EEM", "EFA"]:
        results.append(run_overnight(f"OVERNIGHT-HOLD {tk}", data[tk], "overnight",
                                     spy, costs, n_perm=1000))
    # the market-neutral-ish version: overnight minus intraday
    for tk in ["SPY", "QQQ", "XLK"]:
        results.append(run_overnight(f"ON-MINUS-INTRADAY {tk}", data[tk], "on_minus_id",
                                     spy, costs, n_perm=1000))

    print("\n\n" + "═" * 72)
    print("  OVERNIGHT SUMMARY")
    print("═" * 72)
    ns = sum(1 for r in results if r["survives"])
    print(f"  Survivors: {ns}/{len(results)}")
    for r in sorted(results, key=lambda x: x["p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        beta = " [BETA]" if r["beta"] else " [neutral-ish]"
        print(f"    {r['name']:24s} ret {r['ret']:+8.1f}%  PF {r['pf']:.3f}  "
              f"p={r['p']:.4f}  [{flag}]{beta}")
    print("═" * 72)
    a = assess()
    print(f"\n  Trial ledger: {a['n_trials']} cumulative tests, "
          f"corrected bar p<{a['corrected_bar']:.2e}")
    print("  NOTE: overnight-hold has real-world frictions beyond modeled costs —")
    print("  overnight gap risk, and the open price may not be achievable at scale.")

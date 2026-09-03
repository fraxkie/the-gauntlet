"""
run_xsection.py — CROSS-SECTIONAL strategies through the gauntlet.

THE STRUCTURAL DIFFERENCE: momentum/seasonality/pairs all got killed largely as
beta. Cross-sectional strategies are LONG-SHORT by construction — each period you
rank all assets and go long the top, short the bottom, in equal dollar amounts. If
the whole market moves, the longs and shorts cancel. That's the structural escape
from the beta trap, and it's what real equity quants run.

Families implemented here:
  • XS-MOMENTUM   : long top-K by trailing return, short bottom-K (relative strength)
  • XS-REVERSAL   : long bottom-K, short top-K (short-term mean reversion)
  • XS-LOWVOL     : long lowest-vol K, short highest-vol K (low-vol anomaly)

MCPT FOR CROSS-SECTIONAL: the signal lives in the RELATIVE RANKING across assets.
We permute each asset's return series independently — this destroys the
cross-sectional structure (which asset is strong WHEN) while preserving each
asset's own drift/vol. A real cross-sectional edge should beat that null.

P&L is dollar-neutral: +1 weight on longs, -1 on shorts, normalized so gross
exposure = 1 each side. Regime check vs SPY confirms market-neutrality.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, total_return_pct, profit_factor, mean_bps
from gauntlet.data import load_etf_data
from gauntlet.ledger import log_trial, assess, LEDGER_PATH


def build_panel(data: dict, min_assets: int = 10) -> pd.DataFrame:
    """Align tickers into a close-price panel (dates x tickers).

    For a mixed-history universe (e.g. stocks that IPO'd at different times),
    requiring ALL to be present from day one throws away decades. Instead we keep
    the window from when at least `min_assets` have data, and within that window
    keep only assets with full coverage — maximizing usable cross-sectional breadth.
    """
    closes = {tk: df["close"] for tk, df in data.items()}
    panel = pd.DataFrame(closes)
    # find the earliest date where >= min_assets have data
    n_avail = panel.notna().sum(axis=1)
    start = n_avail[n_avail >= min_assets].index.min()
    if pd.isna(start):
        return panel.dropna()
    panel = panel.loc[start:]
    # keep only columns fully populated over this window (comparable universe)
    panel = panel.loc[:, panel.notna().all()]
    return panel.dropna()


def xs_signal(panel: pd.DataFrame, kind: str, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Return a DataFrame of per-asset SCORES used to rank cross-sectionally.
    lookback : window for the signal. skip : skip most-recent bars (momentum often
    skips the last month to avoid short-term reversal contamination)."""
    if kind == "momentum":
        # trailing return over [t-lookback-skip, t-skip]
        past = panel.shift(skip)
        score = past / past.shift(lookback) - 1.0
    elif kind == "reversal":
        # NEGATIVE of recent short return -> long recent losers
        score = -(panel / panel.shift(lookback) - 1.0)
    elif kind == "lowvol":
        # NEGATIVE of trailing volatility -> long low-vol
        rets = panel.pct_change()
        score = -rets.rolling(lookback).std()
    else:
        raise ValueError(kind)
    return score


def xs_portfolio_returns(panel: pd.DataFrame, score: pd.DataFrame,
                         k: int, costs: Costs, rebal: int = 21):
    """Long top-k, short bottom-k by score, rebalanced every `rebal` bars.
    Returns the dollar-neutral net return stream (after costs) as a numpy array,
    indexed like panel. No look-ahead: positions set from score at rebal date are
    held forward."""
    dates = panel.index
    n = len(dates)
    asset_rets = panel.pct_change().to_numpy()       # [n, m]
    score_np = score.to_numpy()
    m = panel.shape[1]

    weights = np.zeros((n, m))
    current_w = np.zeros(m)
    for i in range(n):
        if i % rebal == 0 and i > 0:
            s = score_np[i]
            if np.isfinite(s).sum() >= 2 * k:
                order = np.argsort(np.where(np.isfinite(s), s, -np.inf))
                longs = order[-k:]
                shorts = order[:k]
                w = np.zeros(m)
                w[longs] = 1.0 / k       # equal-weight long leg, gross = 1
                w[shorts] = -1.0 / k     # equal-weight short leg, gross = 1
                current_w = w
        weights[i] = current_w

    # P&L: weight decided at bar i earns bar i+1's return (no look-ahead)
    w_held = np.zeros_like(weights)
    w_held[1:] = weights[:-1]
    gross = np.nansum(w_held * asset_rets, axis=1)

    # costs on turnover (sum of |weight changes| across assets each bar)
    turn = np.zeros(n)
    turn[1:] = np.nansum(np.abs(weights[1:] - weights[:-1]), axis=1)
    cost = turn * costs.per_turn
    net = gross - cost
    return net, weights


def xs_mcpt(panel, kind, lookback, skip, k, costs, rebal, n_perm=300, seed=42):
    """Permute each asset's returns independently to destroy cross-sectional
    structure, rebuild a synthetic price panel, recompute signal+P&L, compare."""
    rng = np.random.default_rng(seed)
    score = xs_signal(panel, kind, lookback, skip)
    real_net, _ = xs_portfolio_returns(panel, score, k, costs, rebal)
    real_pf = profit_factor(real_net[np.isfinite(real_net)])

    rets = panel.pct_change()
    count_ge = 1
    for _ in range(n_perm):
        # independently permute each column's returns
        permuted = rets.apply(lambda col: col.sample(frac=1, random_state=rng.integers(1e9)).values
                              if col.notna().sum() else col)
        # rebuild prices from permuted returns
        pprice = (1 + permuted.fillna(0)).cumprod() * 100
        pscore = xs_signal(pprice, kind, lookback, skip)
        pnet, _ = xs_portfolio_returns(pprice, pscore, k, costs, rebal)
        ppf = profit_factor(pnet[np.isfinite(pnet)])
        if ppf >= real_pf:
            count_ge += 1
    return real_pf, count_ge / (n_perm + 1)


def run_xs(name, panel, spy, kind, lookback, k, costs, skip=0, rebal=21,
           n_perm=300, log=True):
    score = xs_signal(panel, kind, lookback, skip)
    net, weights = xs_portfolio_returns(panel, score, k, costs, rebal)
    net_f = net[np.isfinite(net)]

    # walk-forward consistency on the net stream
    nlen = len(net); fold = nlen // 6
    fprof = []
    for f in range(6):
        lo = f * fold; hi = nlen if f == 5 else (f + 1) * fold
        seg = net[lo:hi]; seg = seg[np.isfinite(seg)]
        fprof.append(total_return_pct(seg) > 0)
    gate = "PASS" if sum(fprof) / 6 >= 0.6 else "FAIL"

    # regime / market-neutrality vs SPY
    spy_ret = spy["close"].reindex(panel.index).pct_change().to_numpy()
    up = spy_ret > 0; dn = spy_ret < 0
    up_r = mean_bps(net[up & np.isfinite(net)])
    dn_r = mean_bps(net[dn & np.isfinite(net)])

    real_pf, p = xs_mcpt(panel, kind, lookback, skip, k, costs, rebal, n_perm=n_perm)

    survives = (p < 0.01 and gate == "PASS")
    if log:
        log_trial(name, p, total_return_pct(net_f),
                  extra={"family": f"xsection-{kind}",
                         "neutral": bool(up_r > 0 and dn_r > 0)})

    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(net_f):+.1f}%  "
          f"PF {profit_factor(net_f):.3f}  (long-short, rebal {rebal}d)")
    print(f"  MCPT ({n_perm} perms): real PF={real_pf:.3f}  p-value={p:.4f}  "
          f"-> {'REAL' if p<0.01 else 'SUSPECT' if p<0.05 else 'NOISE'}")
    print(f"  WALK-FORWARD     : {sum(fprof)}/6 folds  -> gate {gate}")
    print(f"  MARKET-NEUTRAL?  : SPY-up days {up_r:+.1f}bps/bar  | SPY-down days {dn_r:+.1f}bps/bar"
          + ("   [truly neutral!]" if (up_r > 0 and dn_r > 0)
             else "   [directional]" if (up_r > 0) != (dn_r > 0) else ""))
    print("  " + "-" * 68)
    print(f"  VERDICT          : {'*** SURVIVES ***' if survives else 'KILLED'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "p": p,
            "ret": total_return_pct(net_f), "up": up_r, "dn": dn_r}


if __name__ == "__main__":
    # accept a data file argument: `python run_xsection.py stock_data.csv`
    # defaults to stock_data.csv if it exists (proper cross-sectional universe),
    # else falls back to etf_data.csv
    data_file = sys.argv[1] if len(sys.argv) > 1 else (
        "stock_data.csv" if os.path.exists("stock_data.csv") else "etf_data.csv")
    data = load_etf_data(data_file)
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)
    panel = build_panel(data)

    # market proxy for the regime/neutrality check: SPY if present, else an
    # equal-weight basket of the panel (the universe's own "market")
    if "SPY" in data:
        spy = data["SPY"]
    else:
        ew = panel.mean(axis=1)
        spy = pd.DataFrame({"close": ew})

    print("#" * 72)
    print(f"#  CROSS-SECTIONAL strategies — {panel.shape[1]} assets, long-short")
    print(f"#  Data: {data_file}  |  window: {panel.index.min().date()} to {panel.index.max().date()}")
    print("#  The structural escape from beta: long winners, short losers.")
    print("#" * 72)

    results = []
    # scale K to universe size: ~quintile each side
    k = max(3, panel.shape[1] // 5)
    results.append(run_xs(f"XS-MOMENTUM 12m (top/bot {k})", panel, spy, "momentum",
                          lookback=252, k=k, costs=costs, skip=21))
    results.append(run_xs(f"XS-MOMENTUM 6m (top/bot {k})", panel, spy, "momentum",
                          lookback=126, k=k, costs=costs, skip=21))
    results.append(run_xs(f"XS-REVERSAL 1m (top/bot {k})", panel, spy, "reversal",
                          lookback=21, k=k, costs=costs))
    results.append(run_xs(f"XS-LOWVOL 3m (top/bot {k})", panel, spy, "lowvol",
                          lookback=63, k=k, costs=costs))

    print("\n\n" + "═" * 72)
    print("  CROSS-SECTIONAL SUMMARY")
    print("═" * 72)
    ns = sum(1 for r in results if r["survives"])
    print(f"  Survivors: {ns}/{len(results)}")
    for r in sorted(results, key=lambda x: x["p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        neutral = "neutral" if (r["up"] > 0 and r["dn"] > 0) else "directional"
        print(f"    {r['name']:30s} ret {r['ret']:+7.1f}%  p={r['p']:.4f}  "
              f"[{neutral}]  [{flag}]")
    print("═" * 72)

    a = assess()
    print(f"\n  Trial ledger now holds {a['n_trials']} cumulative tests "
          f"(corrected bar: p<{a['corrected_bar']:.2e})")

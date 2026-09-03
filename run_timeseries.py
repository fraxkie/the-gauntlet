"""
run_timeseries.py — TIME-SERIES family strategies through the gauntlet.

These judge each asset against ITS OWN history (not against other assets). They run
fine on our ETF data. Families:

  • ABS-MOMENTUM (time-series momentum): long an asset if its own trailing return is
    positive, else flat. The Moskowitz/Ooi/Pedersen "time series momentum" effect.
  • DONCHIAN BREAKOUT: long when price makes a new N-day high (the doc-40 strategy).
  • DUAL MOMENTUM (Antonacci): combine absolute + relative — only hold the asset if
    it beats both T-bills (absolute) AND its alternative (relative). Tested as
    SPY-vs-bonds rotation.
  • INVERSE-VOL (risk parity lite): weight assets by inverse volatility. This is an
    ALLOCATION scheme, not an edge — included for completeness; expect it to look
    like a smoother version of long-the-market.

Each is run on its own asset(s) through the single-asset gauntlet where applicable,
or as a portfolio stream (dual momentum, inverse-vol).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, strategy_returns, profit_factor, total_return_pct, mean_bps
from gauntlet.data import load_etf_data
from gauntlet.mcpt import mcpt
from gauntlet.lookahead import lookahead_audit
from gauntlet.robustness import walk_forward, regime_check
from gauntlet.ledger import log_trial, assess
from run_xsection import build_panel


# ── single-asset strategies (use existing gauntlet) ──
def strat_abs_momentum(df, lookback=252):
    """Long if trailing `lookback`-day return > 0, else flat. Time-series momentum."""
    close = df["close"]
    trailing = close / close.shift(lookback) - 1.0
    pos = (trailing > 0).astype(float)
    pos[trailing.isna()] = 0.0
    return pd.Series(pos.to_numpy(), index=df.index)


def strat_donchian(df, lookback=50):
    """Long when close == highest close over trailing `lookback` (new high breakout).
    Stays long until a new `lookback`-day low. Classic trend breakout."""
    close = df["close"]
    hi = close.rolling(lookback).max()
    lo = close.rolling(lookback).min()
    pos = pd.Series(0.0, index=df.index)
    state = 0.0
    c = close.to_numpy(); h = hi.to_numpy(); l = lo.to_numpy()
    out = np.zeros(len(c))
    for i in range(len(c)):
        if np.isnan(h[i]):
            out[i] = 0.0; continue
        if c[i] >= h[i]:
            state = 1.0
        elif c[i] <= l[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=df.index)


def run_single(name, df, strat, spy, costs, n_perm=500, log=True):
    audit = lookahead_audit(df, strat)
    mc = mcpt(df, strat, costs, n_perm=n_perm, objective="pf")
    wf = walk_forward(df, strat, costs)
    rg = regime_check(df, strat, costs)
    net = strategy_returns(df, strat(df), costs).to_numpy()
    survives = (audit["look_ahead_clean"] and mc["p_value"] < 0.01
                and wf["consistency_gate"] == "PASS")
    if log:
        log_trial(name, mc["p_value"], total_return_pct(net), extra={"family": "timeseries"})
    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(net):+.1f}%  "
          f"PF {profit_factor(net):.3f}")
    print(f"  LOOK-AHEAD AUDIT : {audit['note']}")
    print(f"  MCPT ({mc['n_perm']}): real PF={mc['real_score']:.3f}  p={mc['p_value']:.4f}  "
          f"-> {mc['verdict']}")
    print(f"  WALK-FORWARD     : {wf['n_profitable']}/{wf['n_folds']} -> {wf['consistency_gate']}")
    print(f"  REGIME           : UP {rg['UP']['mean_bps']:+.0f}bps | "
          f"DOWN {rg['DOWN']['mean_bps']:+.0f}bps | FLAT {rg['FLAT']['mean_bps']:+.0f}bps"
          + ("  [BETA WARNING]" if rg["_beta_warning"] else ""))
    print("  " + "-" * 68)
    print(f"  VERDICT          : {'*** SURVIVES ***' if survives else 'KILLED'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "p": mc["p_value"],
            "ret": total_return_pct(net), "beta": rg["_beta_warning"]}


# ── portfolio strategies ──
def dual_momentum(panel, spy, costs, lookback=252, rebal=21):
    """Antonacci dual momentum, simplified: each rebal, between SPY and TLT (bonds),
    hold whichever has higher trailing return — but only if that return beats cash
    (absolute filter); else hold bonds. Returns net stream."""
    if "SPY" not in panel or "TLT" not in panel:
        return None
    spy_p = panel["SPY"]; tlt_p = panel["TLT"]
    spy_r = spy_p / spy_p.shift(lookback) - 1.0
    tlt_r = tlt_p / tlt_p.shift(lookback) - 1.0
    asset_rets = panel.pct_change()
    n = len(panel)
    held = np.zeros(n)  # 0 = cash/none, 1 = SPY, 2 = TLT
    state = 0
    spy_rn = spy_r.to_numpy(); tlt_rn = tlt_r.to_numpy()
    for i in range(n):
        if i % rebal == 0 and i > 0 and not np.isnan(spy_rn[i]):
            if spy_rn[i] > 0 and spy_rn[i] >= tlt_rn[i]:
                state = 1
            else:
                state = 2  # defensive: bonds
        held[i] = state
    spy_ret = asset_rets["SPY"].to_numpy()
    tlt_ret = asset_rets["TLT"].to_numpy()
    pos_ret = np.where(held == 1, spy_ret, np.where(held == 2, tlt_ret, 0.0))
    held_shift = np.zeros(n); held_shift[1:] = held[:-1]
    gross = np.where(held_shift == 1, spy_ret, np.where(held_shift == 2, tlt_ret, 0.0))
    turn = np.zeros(n); turn[1:] = (held[1:] != held[:-1]).astype(float)
    net = gross - turn * costs.per_turn
    return net


def inverse_vol(panel, costs, lookback=63, rebal=21):
    """Risk-parity-lite: weight each asset by inverse trailing vol, long-only,
    rebalanced. Allocation scheme, not an edge."""
    rets = panel.pct_change()
    vol = rets.rolling(lookback).std()
    inv = 1.0 / vol
    weights = inv.div(inv.sum(axis=1), axis=0)  # normalize to sum 1
    w = weights.to_numpy()
    # hold constant between rebalances
    n, m = w.shape
    held = np.zeros((n, m)); cur = np.zeros(m)
    for i in range(n):
        if i % rebal == 0 and i > 0 and np.isfinite(w[i]).all():
            cur = w[i]
        held[i] = cur
    ar = rets.to_numpy()
    hs = np.zeros((n, m)); hs[1:] = held[:-1]
    gross = np.nansum(hs * ar, axis=1)
    turn = np.zeros(n); turn[1:] = np.nansum(np.abs(held[1:] - held[:-1]), axis=1)
    return gross - turn * costs.per_turn


def run_portfolio(name, net, spy, panel, family, log=True):
    netf = net[np.isfinite(net)]
    nlen = len(net); fold = nlen // 6
    fprof = [total_return_pct(net[f*fold:(nlen if f==5 else (f+1)*fold)]) > 0 for f in range(6)]
    gate = "PASS" if sum(fprof)/6 >= 0.6 else "FAIL"
    spy_ret = spy["close"].reindex(panel.index).pct_change().to_numpy()
    up = spy_ret > 0; dn = spy_ret < 0
    up_r = mean_bps(net[up & np.isfinite(net)])
    dn_r = mean_bps(net[dn & np.isfinite(net)])
    beta = (up_r > 0 and dn_r <= 0)
    if log:
        log_trial(name, 1.0, total_return_pct(netf), extra={"family": family, "note": "no-MCPT allocation"})
    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(netf):+.1f}%  PF {profit_factor(netf):.3f}")
    print(f"  WALK-FORWARD     : {sum(fprof)}/6 -> {gate}")
    print(f"  REGIME (vs SPY)  : UP-days {up_r:+.1f}bps/bar | DOWN-days {dn_r:+.1f}bps/bar"
          + ("  [BETA WARNING]" if beta else ""))
    print("  " + "-" * 68)
    note = "allocation scheme — judged on consistency/neutrality, not MCPT"
    print(f"  NOTE             : {note}")
    print("═" * 72)
    return {"name": name, "ret": total_return_pct(netf), "up": up_r, "dn": dn_r, "beta": beta}


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)
    panel = build_panel(data)
    spy = data["SPY"]

    print("#" * 72)
    print("#  TIME-SERIES & PORTFOLIO families through the gauntlet")
    print("#" * 72)

    results = []
    # absolute momentum on the broad equity indices + gold + bonds
    for tk in ["SPY", "QQQ", "GLD", "TLT", "EEM"]:
        results.append(run_single(f"ABS-MOMENTUM 12m {tk}", data[tk], strat_abs_momentum,
                                  spy, costs, n_perm=500))
    # donchian breakout
    for tk in ["SPY", "QQQ", "GLD"]:
        results.append(run_single(f"DONCHIAN-50 {tk}", data[tk], strat_donchian,
                                  spy, costs, n_perm=500))

    # portfolio strategies
    dm = dual_momentum(panel, spy, costs)
    if dm is not None:
        run_portfolio("DUAL-MOMENTUM SPY/TLT", dm, spy, panel, "dual-momentum")
    iv = inverse_vol(panel, costs)
    run_portfolio("INVERSE-VOL (risk parity lite)", iv, spy, panel, "risk-parity")

    print("\n\n" + "═" * 72)
    print("  TIME-SERIES SUMMARY (MCPT-tested strategies)")
    print("═" * 72)
    ns = sum(1 for r in results if r["survives"])
    print(f"  Survivors: {ns}/{len(results)}")
    for r in sorted(results, key=lambda x: x["p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        beta = " [BETA]" if r["beta"] else ""
        print(f"    {r['name']:24s} ret {r['ret']:+7.1f}%  p={r['p']:.4f}  [{flag}]{beta}")
    print("═" * 72)
    a = assess()
    print(f"\n  Trial ledger: {a['n_trials']} cumulative tests, "
          f"corrected bar p<{a['corrected_bar']:.2e}")

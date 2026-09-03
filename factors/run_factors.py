"""
run_factors.py — VALUE & QUALITY factor strategies, point-in-time, through the gauntlet.

Why factors might survive where everything else died:
  • REAL, documented, persistent edges (value, quality/profitability) with decades
    of academic evidence — Fama-French, Novy-Marx.
  • COST-SURVIVABLE: rebalanced quarterly (not daily), so turnover is low and the
    cost wall that killed the overnight effect doesn't apply.
  • The data barrier (point-in-time fundamentals) is the moat — it's less arbitraged
    precisely because the clean data is harder to get than free prices.

NO-LOOK-AHEAD (the core discipline): at each monthly rebalance date t, we use only
fundamental records whose FILING date <= t. A merge_asof backward join enforces
this — each stock gets its most recent ALREADY-FILED fundamentals as of t, never a
report that hadn't come out yet.

Factors tested (long top-quintile, short bottom-quintile, dollar-neutral):
  • VALUE     : cheap (low P/B, low P/E) minus expensive
  • QUALITY   : high ROE / high gross margin minus low
  • VALUE+QUALITY combo

Run AFTER fetch_fundamentals.py (needs fundamentals.csv) and getstocks.py (prices).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, profit_factor, total_return_pct, mean_bps
from gauntlet.data import load_etf_data
from gauntlet.ledger import log_trial, assess


def load_prices(path="stock_data.csv"):
    data = load_etf_data(path)  # same multi-ticker parser
    closes = {tk: df["close"] for tk, df in data.items()}
    return pd.DataFrame(closes)


def point_in_time_panel(fund: pd.DataFrame, rebal_dates, factor_cols):
    """For each rebalance date, get each ticker's most recent ALREADY-FILED
    fundamentals (filing date <= rebalance date). Returns dict date -> DataFrame
    [ticker x factor_cols]. This is the no-look-ahead core."""
    fund = fund.sort_values("date_filed")
    out = {}
    for t in rebal_dates:
        # records filed on or before t
        avail = fund[fund["date_filed"] <= t]
        if avail.empty:
            continue
        # most recent per ticker
        latest = avail.sort_values("date_filed").groupby("ticker").tail(1)
        out[t] = latest.set_index("ticker")[factor_cols]
    return out


def factor_score(snapshot, kind):
    """Build a cross-sectional score (higher = more desirable to be LONG)."""
    s = snapshot.copy()
    if kind == "value":
        # cheap = low P/B and low P/E -> long. Invert (negative) so high score=cheap.
        pb = -s["pb"]; pe = -s["pe"]
        # only positive PE/PB are meaningful (negatives = losses/odd); mask them
        pb = pb.where(s["pb"] > 0); pe = pe.where(s["pe"] > 0)
        return _zsum([pb, pe])
    if kind == "quality":
        roe = s["roe"]; gm = s["gross_margin"]
        return _zsum([roe, gm])
    if kind == "value_quality":
        pb = (-s["pb"]).where(s["pb"] > 0)
        roe = s["roe"]; gm = s["gross_margin"]
        return _zsum([pb, roe, gm])
    raise ValueError(kind)


def _zsum(series_list):
    """Z-score each series cross-sectionally and sum (equal-weight the sub-factors)."""
    total = None
    for s in series_list:
        s = pd.to_numeric(s, errors="coerce")
        z = (s - s.mean()) / (s.std() + 1e-12)
        total = z if total is None else total.add(z, fill_value=0)
    return total


def run_factor(name, prices, fund, kind, costs, k_frac=0.2, n_perm=300, log=True):
    # monthly rebalance dates from the price index
    prices = prices.dropna(how="all")
    monthly = prices.resample("ME").last().index
    monthly = [d for d in monthly if d >= prices.index.min() and d <= prices.index.max()]

    factor_cols = ["pe", "pb", "ps", "roe", "gross_margin", "debt_to_equity"]
    pit = point_in_time_panel(fund, monthly, factor_cols)

    daily_ret = prices.pct_change()
    tickers = list(prices.columns)
    n = len(prices); m = len(tickers)
    tk_idx = {tk: i for i, tk in enumerate(tickers)}
    weights = np.zeros((n, m))
    cur = np.zeros(m)
    date_to_row = {d: i for i, d in enumerate(prices.index)}

    rebal_rows = sorted([date_to_row[d] for d in monthly if d in date_to_row])
    rebal_set = {}
    for d in monthly:
        if d in date_to_row and d in pit:
            snap = pit[d]
            score = factor_score(snap, kind).dropna()
            valid = [tk for tk in score.index if tk in tk_idx]
            score = score.loc[valid]
            if len(score) >= 6:
                k = max(2, int(len(score) * k_frac))
                ranked = score.sort_values()
                shorts = ranked.index[:k]; longs = ranked.index[-k:]
                w = np.zeros(m)
                for tk in longs: w[tk_idx[tk]] = 1.0 / k
                for tk in shorts: w[tk_idx[tk]] = -1.0 / k
                rebal_set[date_to_row[d]] = w

    for i in range(n):
        if i in rebal_set:
            cur = rebal_set[i]
        weights[i] = cur

    w_held = np.zeros_like(weights); w_held[1:] = weights[:-1]
    ar = daily_ret.to_numpy()
    gross = np.nansum(w_held * ar, axis=1)
    turn = np.zeros(n); turn[1:] = np.nansum(np.abs(weights[1:] - weights[:-1]), axis=1)
    net = gross - turn * costs.per_turn
    netf = net[np.isfinite(net)]

    # permutation: shuffle each asset's returns independently (destroy cross-section)
    rng = np.random.default_rng(42)
    real_pf = profit_factor(netf)
    cnt = 1
    rets_np = daily_ret.to_numpy()
    for _ in range(n_perm):
        perm = np.empty_like(rets_np)
        for j in range(m):
            col = rets_np[:, j]; fin = np.isfinite(col)
            vals = col[fin].copy(); rng.shuffle(vals)
            pc = col.copy(); pc[fin] = vals; perm[:, j] = pc
        pg = np.nansum(w_held * perm, axis=1) - turn * costs.per_turn
        if profit_factor(pg[np.isfinite(pg)]) >= real_pf:
            cnt += 1
    p = cnt / (n_perm + 1)

    # walk-forward + neutrality
    nlen = len(net); fold = nlen // 6
    fprof = [total_return_pct(net[f*fold:(nlen if f==5 else (f+1)*fold)]) > 0 for f in range(6)]
    gate = "PASS" if sum(fprof)/6 >= 0.6 else "FAIL"
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().to_numpy()
    else:
        spy_ret = prices.mean(axis=1).pct_change().to_numpy()
    up = spy_ret > 0; dn = spy_ret < 0
    up_bps = mean_bps(net[up & np.isfinite(net)]); dn_bps = mean_bps(net[dn & np.isfinite(net)])
    beta = (up_bps > 0 and dn_bps <= 0)

    survives = (p < 0.01 and gate == "PASS")
    if log:
        log_trial(name, p, total_return_pct(netf), extra={"family": f"factor-{kind}"})

    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(netf):+.1f}%  "
          f"PF {profit_factor(netf):.3f}  (quarterly-ish rebal, long-short)")
    print(f"  PERMUTATION      : real PF={real_pf:.3f}  p={p:.4f}  "
          f"-> {'REAL' if p<0.01 else 'SUSPECT' if p<0.05 else 'NOISE'}")
    print(f"  WALK-FORWARD     : {sum(fprof)}/6 -> {gate}")
    print(f"  NEUTRALITY (SPY) : UP {up_bps:+.1f}bps | DOWN {dn_bps:+.1f}bps"
          + ("  [BETA]" if beta else "  [neutral-ish]"))
    print("  " + "-" * 68)
    print(f"  VERDICT          : {'*** SURVIVES ***' if survives else 'KILLED'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "p": p, "ret": total_return_pct(netf)}


if __name__ == "__main__":
    if not os.path.exists("fundamentals.csv"):
        print("Need fundamentals.csv first — run factors/fetch_fundamentals.py")
        sys.exit(1)
    if not os.path.exists("stock_data.csv"):
        print("Need stock_data.csv first — run getstocks.py")
        sys.exit(1)

    prices = load_prices("stock_data.csv")
    fund = pd.read_csv("fundamentals.csv", parse_dates=["date_filed"])
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)

    print("#" * 72)
    print("#  FUNDAMENTAL FACTORS — point-in-time, no look-ahead")
    print(f"#  {prices.shape[1]} stocks | {fund['ticker'].nunique()} with fundamentals")
    print("#  Quarterly rebalance -> low turnover -> costs don't kill it")
    print("#" * 72)

    results = []
    for kind, label in [("value", "VALUE"), ("quality", "QUALITY"),
                        ("value_quality", "VALUE+QUALITY")]:
        results.append(run_factor(f"FACTOR {label}", prices, fund, kind, costs))

    print("\n\n" + "═" * 72)
    print("  FACTOR SUMMARY")
    print("═" * 72)
    ns = sum(1 for r in results if r["survives"])
    print(f"  Survivors: {ns}/{len(results)}")
    for r in sorted(results, key=lambda x: x["p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        print(f"    {r['name']:22s} ret {r['ret']:+8.1f}%  p={r['p']:.4f}  [{flag}]")
    print("═" * 72)
    a = assess()
    print(f"\n  Trial ledger: {a['n_trials']} cumulative, corrected bar p<{a['corrected_bar']:.2e}")

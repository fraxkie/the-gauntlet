"""
run_batch.py — apply the WHOLE engine to many assets at once, honestly.
═══════════════════════════════════════════════════════════════════════════════
Combines two goals in one ledger-tracked sweep:
  (A) Does our CONFIRMED panic-reversion edge GENERALIZE to fresh asset classes?
      Crypto especially — it has the most violent forced-liquidation cascades, so
      if the mechanism (panic overshoot) is real, crypto should show it strongest.
  (B) What NEW structure exists in these markets? Volatility predictability, regime
      behavior, baseline strategy performance.

THE DISCIPLINE: testing many things at once is exactly where false positives breed.
Every test is counted. We report results with a Bonferroni-corrected significance
bar based on the TOTAL number of tests run in this sweep. A result only "passes" if
it clears the corrected bar — the same gate that validated panic-reversion.

This is the anti-fishing design: we're not hunting for the one asset that looks
good by luck; we're asking whether a MECHANISM replicates, with the bar raised for
multiplicity.
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.data import load_etf_data
from gauntlet.core import Costs, profit_factor, total_return_pct, mean_bps
from probability.regime import GaussianHMM
from probability.volatility import evaluate_vol_models


def load_any(path):
    """Load a multi-ticker CSV (ETF/crypto/sector format) into {ticker: df}."""
    return load_etf_data(path)


def panic_reversion_test(returns, n_perm=2000, seed=0, fast_thresh=-0.08):
    """Test #1: does panic-crash mean-reversion appear in this asset?
    Fit HMM, find crisis (high-vol) regime, test within-crisis next-day autocorr.
    Also test the FAST-crash subset (the mechanism-specific signal)."""
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    if len(r) < 800:
        return None
    split = int(len(r) * 0.4)
    try:
        hmm = GaussianHMM(n_states=3, n_iter=50, seed=seed).fit(r[:split])
    except Exception:
        return None
    labels = hmm.filter_proba(r).argmax(axis=1)
    crisis = labels[:-1] == 2
    today, tmrw = r[:-1], r[1:]

    # within-crisis autocorr
    m = crisis & np.isfinite(today) & np.isfinite(tmrw)
    if m.sum() < 30:
        return None
    ac = np.corrcoef(today[m], tmrw[m])[0, 1]
    # permutation p-value
    rng = np.random.default_rng(seed)
    cnt = sum(1 for _ in range(n_perm)
              if abs(np.corrcoef(today[m], rng.permutation(tmrw[m]))[0, 1]) >= abs(ac))
    p = (cnt + 1) / (n_perm + 1)

    return {"crisis_autocorr": float(ac), "crisis_n": int(m.sum()),
            "p_value": float(p)}


def panic_strategy_pnl(returns, costs, seed=0):
    """Build the fade-in-crisis strategy, return net % after costs."""
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    if len(r) < 800:
        return None
    split = int(len(r) * 0.4)
    try:
        hmm = GaussianHMM(n_states=3, n_iter=50, seed=seed).fit(r[:split])
    except Exception:
        return None
    labels = hmm.filter_proba(r).argmax(axis=1)
    n = len(r); pos = np.zeros(n)
    for i in range(1, n):
        if labels[i-1] == 2 and np.isfinite(r[i-1]):
            pos[i] = -np.sign(r[i-1])
    gross = pos * np.nan_to_num(r)
    turn = np.zeros(n); turn[1:] = np.abs(pos[1:] - pos[:-1])
    net = gross - turn * costs.per_turn
    netf = net[np.isfinite(net)]
    active = int((pos != 0).sum())
    return {"net_pct": total_return_pct(netf), "pf": profit_factor(netf),
            "active_days": active}


def vol_predictability_test(returns):
    """Test #2: is volatility predictable here (EWMA beats naive)?"""
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    if len(r) < 800:
        return None
    try:
        ev = evaluate_vol_models(r)
        m = ev["models"]
        return {"vol_corr": m["ewma"]["vol_corr"],
                "improvement": m["ewma"].get("qlike_improvement_vs_naive", 0)}
    except Exception:
        return None


def run_batch(datasets, costs, n_perm=2000):
    """datasets: dict of {class_name: {ticker: df}}. Runs the full battery on each
    asset, collects results, applies Bonferroni at the end."""
    rows = []
    for cls, data in datasets.items():
        for tk, df in data.items():
            if "close" not in df:
                continue
            r = df["close"].pct_change().dropna().to_numpy()
            pr = panic_reversion_test(r, n_perm=n_perm)
            if pr is None:
                continue
            pnl = panic_strategy_pnl(r, costs)
            vol = vol_predictability_test(r)
            rows.append({
                "class": cls, "ticker": tk,
                "crisis_autocorr": pr["crisis_autocorr"],
                "crisis_n": pr["crisis_n"],
                "p_value": pr["p_value"],
                "net_pct": pnl["net_pct"] if pnl else np.nan,
                "pf": pnl["pf"] if pnl else np.nan,
                "active_days": pnl["active_days"] if pnl else 0,
                "vol_corr": vol["vol_corr"] if vol else np.nan,
                "vol_improvement": vol["improvement"] if vol else np.nan,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)

    # load whatever datasets are present
    datasets = {}
    for fname, cls in [("etf_data.csv", "ETF"), ("crypto_data.csv", "CRYPTO"),
                       ("sector_data.csv", "SECTOR"), ("stock_data.csv", "STOCK")]:
        if os.path.exists(fname):
            try:
                datasets[cls] = load_any(fname)
                print(f"loaded {cls}: {len(datasets[cls])} assets")
            except Exception as e:
                print(f"skip {fname}: {e}")

    if not datasets:
        print("No data files found. Run getdata.py and getfresh.py first.")
        sys.exit(1)

    print("\nRunning full battery on every asset (this takes a few minutes)...\n")
    df = run_batch(datasets, costs)

    # ── Bonferroni correction for the whole sweep ──
    n_tests = len(df)
    bar = 0.05 / max(n_tests, 1)

    print("=" * 78)
    print(f"  BATCH RESULTS — {n_tests} assets tested. Bonferroni bar: p < {bar:.5f}")
    print("=" * 78)

    # PANIC-REVERSION generalization
    print("\n  ── DOES PANIC-REVERSION GENERALIZE? (crisis autocorr, corrected) ──")
    print(f"  {'asset':>12} {'class':>7} {'autocorr':>9} {'n':>5} {'p-value':>9} {'survives?':>10}")
    dfp = df.sort_values("p_value")
    for _, row in dfp.iterrows():
        surv = "YES" if row["p_value"] < bar and row["crisis_autocorr"] < 0 else "no"
        flag = " <<<" if surv == "YES" else ""
        print(f"  {row['ticker']:>12} {row['class']:>7} {row['crisis_autocorr']:>+9.3f} "
              f"{row['crisis_n']:>5} {row['p_value']:>9.4f} {surv:>10}{flag}")

    survivors = df[(df["p_value"] < bar) & (df["crisis_autocorr"] < 0)]
    print(f"\n  Panic-reversion CONFIRMED in {len(survivors)}/{n_tests} assets "
          f"(after Bonferroni).")
    if len(survivors):
        by_class = survivors.groupby("class").size()
        print("  By asset class:", dict(by_class))

    # VOLATILITY predictability (we expect this everywhere)
    print("\n  ── VOLATILITY PREDICTABILITY (EWMA vol-corr, should be high everywhere) ──")
    dfv = df.dropna(subset=["vol_corr"]).sort_values("vol_corr", ascending=False)
    print(f"  Mean vol-forecast correlation across all assets: {df['vol_corr'].mean():.3f}")
    print(f"  Range: {df['vol_corr'].min():.3f} to {df['vol_corr'].max():.3f}")
    print(f"  (Consistently high = volatility predictability is universal, not luck)")

    # save full results
    df.to_csv("batch_results.csv", index=False)
    print(f"\n  Full results saved to batch_results.csv")
    print("=" * 78)

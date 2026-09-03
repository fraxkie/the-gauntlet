"""
run_regime_conditional.py — does the market behave EXPLOITABLY differently per regime?
═══════════════════════════════════════════════════════════════════════════════
The fusion of everything we built. We proved direction is a coin flip OVERALL — but
that mixed all 21 years together. The regime detector lets us finally ask: is
direction a coin flip WITHIN each regime, or does mixing regimes hide real structure?

Exploratory finding that motivated this: SPY next-day autocorrelation is ~0 in calm
and normal regimes, but ~-0.20 in CRISIS regimes — a mean-reversion signal (violent
bounces) that only appears inside crises and averages away in the full sample.

This engine tests that rigorously:
  • SIGNIFICANCE: is the within-regime autocorrelation real, or small-sample noise?
    (permutation test: shuffle returns within the regime, see if |autocorr| as large
    arises by chance)
  • TRADEABILITY: build the implied strategy (in crisis, fade yesterday's move) and
    run it through the gauntlet — walk-forward AND realistic costs. Crisis mean-
    reversion is notoriously cost-sensitive (spreads blow out exactly then), so this
    is the real test.

NO-LOOK-AHEAD: regime labels come from the causal filter (past data only). The HMM
is fit on a training prefix; regimes for the test period are filtered forward.
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gauntlet.data import load_etf_data
from gauntlet.core import Costs, profit_factor, total_return_pct, mean_bps
from probability.regime import GaussianHMM


def regime_labels_causal(returns, n_states=3, train_frac=0.4, seed=0):
    """Fit HMM on a training prefix, then filter forward to label every bar using
    only past data. Returns (labels, hmm)."""
    r = np.asarray(returns, float)
    split = int(len(r) * train_frac)
    hmm = GaussianHMM(n_states=n_states, n_iter=60, seed=seed).fit(r[:split])
    filt = hmm.filter_proba(r)
    return filt.argmax(axis=1), hmm


def within_regime_autocorr_test(returns, labels, regime, n_perm=2000, seed=0):
    """Is the next-day autocorrelation within `regime` statistically real?
    Permutation: shuffle the within-regime returns and recompute autocorr; p-value
    is the fraction of shuffles with |autocorr| >= the real one."""
    r = np.asarray(returns, float)
    mask = labels[:-1] == regime
    today = r[:-1][mask]; tmrw = r[1:][mask]
    valid = np.isfinite(today) & np.isfinite(tmrw)
    today, tmrw = today[valid], tmrw[valid]
    if len(today) < 30:
        return None
    real_ac = np.corrcoef(today, tmrw)[0, 1]

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shuf = rng.permutation(tmrw)
        ac = np.corrcoef(today, shuf)[0, 1]
        if abs(ac) >= abs(real_ac):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"regime": regime, "n": len(today), "autocorr": float(real_ac),
            "p_value": float(p)}


def crisis_meanreversion_strategy(returns, labels, crisis_regime, costs,
                                  scale=1.0):
    """The implied strategy: when in the crisis regime, take a position OPPOSITE to
    yesterday's return (fade the move), sized by `scale`. Flat otherwise. Then apply
    realistic costs — the real test, since crisis trading is expensive.
    Returns the net return stream."""
    r = np.asarray(returns, float)
    n = len(r)
    pos = np.zeros(n)
    for i in range(1, n):
        if labels[i-1] == crisis_regime and np.isfinite(r[i-1]):
            # fade yesterday: short if yesterday up, long if yesterday down
            pos[i] = -np.sign(r[i-1]) * scale
    # position decided using info through i-1, earns return at i (no look-ahead)
    gross = pos * np.nan_to_num(r)
    turn = np.zeros(n); turn[1:] = np.abs(pos[1:] - pos[:-1])
    net = gross - turn * costs.per_turn
    return net, pos


def walk_forward_stream(net, n_folds=6, thr=0.5):
    n = len(net); fold = n // n_folds
    prof = []
    for f in range(n_folds):
        lo = f*fold; hi = n if f == n_folds-1 else (f+1)*fold
        seg = net[lo:hi]; seg = seg[np.isfinite(seg)]
        # only count folds that actually had crisis activity
        if np.any(seg != 0):
            prof.append(total_return_pct(seg) > 0)
    if not prof:
        return 0, 0, "N/A"
    return sum(prof), len(prof), ("PASS" if sum(prof)/len(prof) >= thr else "FAIL")


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)

    print("#" * 72)
    print("#  REGIME-CONDITIONAL ANALYSIS — does structure hide inside regimes?")
    print("#  The fusion: HMM regimes + return structure + the gauntlet.")
    print("#" * 72)

    for tk in ["SPY", "QQQ", "EEM"]:
        r = data[tk]["close"].pct_change().dropna().to_numpy()
        labels, hmm = regime_labels_causal(r, n_states=3)
        # crisis = highest-vol regime = index 2 (HMM sorts by vol)
        crisis = 2

        print("\n" + "═" * 72)
        print(f"  {tk}")
        print("═" * 72)
        # 1. significance of within-regime autocorr
        print("  Within-regime next-day autocorrelation (permutation-tested):")
        for k, nm in [(0, "CALM"), (1, "NORMAL"), (2, "CRISIS")]:
            res = within_regime_autocorr_test(r, labels, k, n_perm=2000)
            if res:
                sig = ("REAL" if res["p_value"] < 0.01 else
                       "marginal" if res["p_value"] < 0.05 else "noise")
                print(f"    {nm:>7}: autocorr={res['autocorr']:>+.3f}  "
                      f"n={res['n']:>5}  p={res['p_value']:.4f}  -> {sig}")

        # 2. is the crisis mean-reversion TRADEABLE after costs?
        net, pos = crisis_meanreversion_strategy(r, labels, crisis, costs)
        active = (pos != 0).sum()
        netf = net[np.isfinite(net)]
        nprof, nf, gate = walk_forward_stream(net)
        # gross (no cost) for comparison — shows how much cost eats
        gross_net, _ = crisis_meanreversion_strategy(r, labels, crisis, Costs(0, 0))
        print(f"\n  Crisis mean-reversion strategy (fade yesterday in crisis regime):")
        print(f"    Active days     : {active} ({active/len(r)*100:.1f}% of time)")
        print(f"    Gross return    : {total_return_pct(gross_net[np.isfinite(gross_net)]):>+8.1f}%  (no costs)")
        print(f"    Net return      : {total_return_pct(netf):>+8.1f}%  (after costs)")
        print(f"    Profit factor   : {profit_factor(netf):.3f}")
        print(f"    Walk-forward    : {nprof}/{nf} crisis-active folds -> {gate}")
        print(f"    Cost impact     : {total_return_pct(gross_net[np.isfinite(gross_net)]) - total_return_pct(netf):>+8.1f}% eaten by costs")

    print("\n" + "═" * 72)
    print("  The honest test: does within-crisis mean-reversion (a) test as")
    print("  statistically real, and (b) survive the brutal costs of trading during")
    print("  crises? Gross-vs-net shows whether the signal clears the cost wall.")
    print("═" * 72)

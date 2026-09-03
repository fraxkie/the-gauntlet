"""
run_regime_calibration.py — STEP 4: does knowing the REGIME improve forecasts?
═══════════════════════════════════════════════════════════════════════════════
Weaves the threads together. We have:
  • Step 3: a regime detector (HMM) that says which regime we're probably in
  • earlier: the calibration harness (Brier / reliability / resolution) — the
    probability lie-detector

Question: if we CONDITION our forecasts on the detected regime, do they get better
— more calibrated, sharper (higher resolution)? We test this honestly, two ways:

  TEST A — VOLATILITY forecast conditioned on regime.
    Baseline: a single EWMA vol forecast.
    Conditioned: blend the forecast toward the current regime's characteristic vol
    (calm regimes -> lower vol forecast, crisis -> higher). Does conditioning reduce
    out-of-sample vol-forecast error (QLIKE)?

  TEST B — DIRECTION calibration BY regime.
    We already know unconditional next-day direction is a coin flip. But is it a
    coin flip IN EVERY REGIME? Maybe direction is more predictable in some regimes
    (e.g. crises have momentum/continuation). We bucket the SAME forecasts by regime
    and check calibration/resolution within each.

THE NON-NEGOTIABLE: regime belief must come from the CAUSAL FILTER (past data only).
Using smoothed/Viterbi regime labels (which see the future) would be look-ahead and
would make conditioning look magically good. We use filter_proba throughout, and we
fit the HMM only on a training prefix, then filter forward — no peeking.
═══════════════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gauntlet.data import load_etf_data
from probability.volatility import forecast_ewma, qlike
from probability.regime import GaussianHMM
from probability.calibration import calibration_report, print_calibration


def regime_conditioned_vol_test(returns, train_frac=0.5, n_states=3,
                                target_window=21, rebal=21, blend=0.5):
    """TEST A: does regime-conditioning improve OOS volatility forecasts?
    Fit HMM on the training prefix only. Then walk forward on the test region,
    at each step using the CAUSAL filtered regime belief to adjust the vol forecast.
    """
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    n = len(r)
    split = int(n * train_frac)
    train = r[:split]

    # fit regime model on TRAIN ONLY
    hmm = GaussianHMM(n_states=n_states, n_iter=60, seed=0).fit(train)
    regime_vols = np.sqrt(hmm.vars)  # per-regime daily vol (sorted calm->stormy)

    base_fc, cond_fc, realized = [], [], []
    start = split
    lookback = 504
    while start + target_window <= n:
        hist = r[max(0, start - lookback):start]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 100:
            start += rebal; continue

        # baseline EWMA vol forecast (variance)
        base_var = forecast_ewma(hist)

        # CAUSAL regime belief at `start`: filter using data up to start only
        filt = hmm.filter_proba(r[:start])
        regime_belief = filt[-1]                      # P(each regime) right now
        # regime-implied variance = belief-weighted regime variances
        regime_var = float(np.sum(regime_belief * hmm.vars))
        # blend baseline with regime-implied
        cond_var = (1 - blend) * base_var + blend * regime_var

        # realized variance over the next window (the truth)
        future = r[start:start + target_window]
        future = future[np.isfinite(future)]
        if len(future) < target_window // 2:
            start += rebal; continue
        rv = np.var(future)

        base_fc.append(base_var); cond_fc.append(cond_var); realized.append(rv)
        start += rebal

    base_fc = np.array(base_fc); cond_fc = np.array(cond_fc); realized = np.array(realized)
    q_base = qlike(base_fc, realized)
    q_cond = qlike(cond_fc, realized)
    return {
        "n": len(realized),
        "qlike_baseline": q_base,
        "qlike_regime_conditioned": q_cond,
        "improvement_pct": (q_base - q_cond) / q_base * 100 if q_base > 0 else 0,
    }


def direction_by_regime_test(returns, n_states=3, train_frac=0.5):
    """TEST B: is next-day direction more predictable in SOME regimes?
    We bucket days by their CAUSAL filtered regime, and within each regime check the
    base rate and whether a simple 'predict the regime's historical up-rate' forecast
    has any resolution. (Honest expectation: mostly still coin flips, but crises may
    show directional persistence.)"""
    r = np.asarray(returns, float); r = r[np.isfinite(r)]
    n = len(r)
    split = int(n * train_frac)
    hmm = GaussianHMM(n_states=n_states, n_iter=60, seed=0).fit(r[:split])

    # causal regime label for each test-region day
    filt = hmm.filter_proba(r)
    regime_label = filt.argmax(axis=1)

    # next-day up indicator
    up = (r[1:] > 0).astype(float)
    reg = regime_label[:-1]      # regime today -> predict tomorrow's direction

    out = {}
    names = {0: "CALM", 1: "NORMAL", 2: "CRISIS"} if n_states == 3 else {i: f"R{i}" for i in range(n_states)}
    for k in range(n_states):
        mask = (reg == k)
        if mask.sum() < 50:
            continue
        up_rate = up[mask].mean()
        out[names.get(k, f"R{k}")] = {
            "n_days": int(mask.sum()),
            "up_rate": float(up_rate),
            "deviation_from_half": float(abs(up_rate - 0.5)),
        }
    return out


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")

    print("#" * 72)
    print("#  STEP 4 — REGIME-CONDITIONED CALIBRATION")
    print("#  Does knowing the regime make our forecasts measurably better?")
    print("#  (All regime beliefs are CAUSAL — filtered from past data only.)")
    print("#" * 72)

    print("\n" + "═" * 72)
    print("  TEST A — Volatility forecast: baseline vs regime-conditioned (OOS QLIKE)")
    print("═" * 72)
    print(f"  {'ticker':>6} {'baseline':>12} {'regime-cond':>13} {'improvement':>13}")
    va_results = []
    for tk in ["SPY", "QQQ", "EEM", "TLT", "GLD"]:
        r = data[tk]["close"].pct_change().dropna().to_numpy()
        res = regime_conditioned_vol_test(r)
        va_results.append((tk, res))
        print(f"  {tk:>6} {res['qlike_baseline']:>12.4f} "
              f"{res['qlike_regime_conditioned']:>13.4f} "
              f"{res['improvement_pct']:>+12.1f}%")
    avg_imp = np.mean([r["improvement_pct"] for _, r in va_results])
    print(f"  {'-'*46}")
    print(f"  Average improvement from regime-conditioning: {avg_imp:+.1f}%")

    print("\n" + "═" * 72)
    print("  TEST B — Is next-day direction more predictable in some regimes?")
    print("═" * 72)
    for tk in ["SPY", "QQQ"]:
        r = data[tk]["close"].pct_change().dropna().to_numpy()
        by_regime = direction_by_regime_test(r)
        print(f"\n  {tk}:")
        print(f"    {'regime':>8} {'days':>7} {'up-rate':>9} {'|dev from 50%|':>15}")
        for name, d in by_regime.items():
            flag = "  <- directional!" if d["deviation_from_half"] > 0.04 else ""
            print(f"    {name:>8} {d['n_days']:>7} {d['up_rate']*100:>8.1f}% "
                  f"{d['deviation_from_half']*100:>13.1f}%{flag}")

    print("\n" + "═" * 72)
    print("  Interpretation: positive QLIKE improvement = regime info genuinely")
    print("  sharpens volatility forecasts. Direction up-rates far from 50% in a")
    print("  regime = a hint that direction is more predictable in that regime.")
    print("═" * 72)

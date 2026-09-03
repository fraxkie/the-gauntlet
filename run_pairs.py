"""
run_pairs.py — pairs trading / statistical arbitrage through the gauntlet.

THE BET (structurally opposite to MACD's fatal flaw):
  Two historically-linked ETFs (e.g. SPY/DIA, both large-cap US equity) occasionally
  drift apart. When the spread stretches unusually wide (z-score beyond a threshold),
  bet it reverts: short the rich leg, long the cheap leg. Profit when they reconverge.

WHY IT MIGHT BEAT THE BETA WARNING:
  Long one leg + short the other = market-neutral. If the whole market moves, both
  legs move together and cancel. The edge (if real) comes from the SPREAD reverting,
  not from the market going up. That's the thing MACD catastrophically lacked.

MECHANISM: cointegration — economically-linked assets can diverge short-term but
are tethered long-term. Real, named, and not just "it backtested well."

NOTE ON THE GAUNTLET FOR PAIRS:
  The spread is synthetic (leg A minus hedge*leg B), so we run the firewalls on the
  spread's strategy returns directly. MCPT permutes the SPREAD series. Regime is
  measured against SPY (the market) to check the market-neutral claim: a true pairs
  edge should make money in DOWN markets too, since it's hedged.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from gauntlet.core import Costs, profit_factor, total_return_pct, mean_bps
from gauntlet.data import load_etf_data
from gauntlet.permute import permute_prices
from gauntlet.mcpt import mcpt
from gauntlet.robustness import walk_forward


def make_spread_df(a: pd.DataFrame, b: pd.DataFrame, hedge_lookback: int = 60) -> pd.DataFrame:
    """Build the spread as the LOG-PRICE RATIO between A and B — a stable,
    bounded, mean-reverting series (not a cumulative price that can explode).

    ratio_t = log(A_t) - log(B_t).  For cointegrated assets this oscillates around
    a slowly-moving mean. We hand the engine an OHLC frame whose 'close' is this
    ratio shifted into positive territory, so the existing z-score machinery works.
    The strategy trades the ratio's mean reversion; we reconstruct true dollar-
    neutral P&L separately in the runner.
    """
    idx = a.index.intersection(b.index)
    la = np.log(a.loc[idx, "close"].to_numpy(float))
    lb = np.log(b.loc[idx, "close"].to_numpy(float))
    ratio = la - lb                      # log price ratio = the spread
    # shift to a positive "price" centered ~100 for the OHLC engine (monotonic
    # transform, preserves z-scores and reversion structure)
    spread_price = 100.0 * np.exp(ratio - np.mean(ratio))
    df = pd.DataFrame({
        "open": spread_price, "high": spread_price,
        "low": spread_price, "close": spread_price,
    }, index=idx)
    # keep the raw leg returns for true P&L reconstruction
    df.attrs["ret_a"] = a.loc[idx, "close"].pct_change().to_numpy()
    df.attrs["ret_b"] = b.loc[idx, "close"].pct_change().to_numpy()
    return df


def strat_spread_reversion(df: pd.DataFrame, z_lookback: int = 30,
                           z_entry: float = 2.0) -> pd.Series:
    """Mean-reversion on the spread's z-score. When the spread is z_entry std ABOVE
    its mean -> short it (expect down). z_entry std BELOW -> long it. Flat in
    between. Uses only past data for the rolling mean/std (no look-ahead)."""
    close = np.log(df["close"])
    mean = close.rolling(z_lookback).mean()
    std = close.rolling(z_lookback).std()
    z = (close - mean) / std

    pos = pd.Series(0.0, index=df.index)
    pos[z > z_entry] = -1.0      # spread rich -> short
    pos[z < -z_entry] = 1.0      # spread cheap -> long
    pos[z.isna()] = 0.0
    return pd.Series(pos.to_numpy(), index=df.index)


def run_pairs_gauntlet(name, spread_df, market_df, costs, n_perm=500):
    from gauntlet.core import strategy_returns, profit_factor as _pf
    from gauntlet.lookahead import lookahead_audit

    strat = strat_spread_reversion

    # TRUE dollar-neutral P&L: position +1 on spread = long A / short B, earning
    # (ret_a - ret_b) next bar; -1 = the reverse. Costs charged on position turns.
    pos = strat(spread_df).to_numpy(float)
    ret_a = spread_df.attrs["ret_a"]
    ret_b = spread_df.attrs["ret_b"]
    spread_ret = np.nan_to_num(ret_a - ret_b)   # long-A/short-B return per bar
    pos_held = np.zeros_like(pos); pos_held[1:] = pos[:-1]   # no look-ahead
    gross = pos_held * spread_ret
    turn = np.zeros_like(pos); turn[1:] = np.abs(pos[1:] - pos[:-1])
    # a pairs turn trades BOTH legs, so cost is 2x a single-leg turn
    cost = turn * costs.per_turn * 2.0
    net = gross - cost

    audit = lookahead_audit(spread_df, strat)
    # MCPT: permute the spread series, re-run; score on the TRUE pairs P&L proxy.
    # We pass a wrapper objective by permuting the ratio and recomputing positions,
    # but P&L needs leg returns — so for MCPT we use the spread-price returns as a
    # faithful proxy (permutation destroys the ratio's reversion structure).
    mc = mcpt(spread_df, strat, costs, n_perm=n_perm, objective="pf")
    # walk-forward on the TRUE net P&L, fold by fold
    n = len(net); fold = n // 6
    fold_prof = []
    for f in range(6):
        lo = f * fold; hi = n if f == 5 else (f + 1) * fold
        seg = net[lo:hi]
        fold_prof.append(total_return_pct(seg) > 0)
    n_prof = sum(fold_prof)
    gate = "PASS" if n_prof / 6 >= 0.6 else "FAIL"

    # market-neutrality vs SPY
    mkt = market_df["close"].reindex(spread_df.index).pct_change().to_numpy()
    up_mask = mkt > 0; dn_mask = mkt < 0
    up_ret = mean_bps(net[up_mask]) if up_mask.any() else 0.0
    dn_ret = mean_bps(net[dn_mask]) if dn_mask.any() else 0.0

    survives = (audit["look_ahead_clean"] and mc["p_value"] < 0.01 and gate == "PASS")

    print("\n" + "═" * 72)
    print(f"  {name}")
    print("═" * 72)
    print(f"  Raw performance  : total return {total_return_pct(net):+.1f}%  "
          f"profit factor {_pf(net):.3f}  over {len(spread_df)} bars  "
          f"({(pos!=0).sum()} bars in trade)")
    print(f"  LOOK-AHEAD AUDIT : {audit['note']}")
    print(f"  MCPT ({mc['n_perm']} perms): real PF={mc['real_score']:.3f}  "
          f"noise-mean={mc['perm_mean']:.3f}  p-value={mc['p_value']:.4f}  "
          f"-> {mc['verdict']}")
    print(f"  WALK-FORWARD     : {n_prof}/6 folds profitable "
          f"({round(100*n_prof/6,1)}%)  -> gate {gate}")
    print(f"  MARKET-NEUTRAL?  : on SPY-up days {up_ret:+.1f}bps/bar  | "
          f"on SPY-down days {dn_ret:+.1f}bps/bar"
          + ("   [truly neutral — makes money both ways]"
             if (up_ret > 0 and dn_ret > 0) else
             "   [directional]" if (up_ret > 0) != (dn_ret > 0) else ""))
    print("  " + "-" * 68)
    print(f"  VERDICT          : {'*** SURVIVES ***' if survives else 'KILLED'}")
    print("═" * 72)
    return {"name": name, "survives": survives, "p": mc["p_value"],
            "ret": total_return_pct(net), "up": up_ret, "dn": dn_ret}


if __name__ == "__main__":
    data = load_etf_data("etf_data.csv")
    costs = Costs(commission_bps=1.0, slippage_bps=2.0)

    print("#" * 72)
    print("#  PAIRS TRADING — market-neutral spread reversion through the gauntlet")
    print("#  The bet: beat the beta warning that killed MACD, via hedged legs.")
    print("#" * 72)

    # candidate cointegrated pairs (economically linked)
    pairs = [
        ("SPY", "DIA"),   # large-cap US equity — nearly the same thing
        ("SPY", "IWM"),   # large vs small cap US
        ("QQQ", "XLK"),   # nasdaq vs tech sector — heavy overlap
        ("XLF", "XLE"),   # financials vs energy (weaker link — control)
        ("EFA", "EEM"),   # developed vs emerging intl
        ("IEF", "TLT"),   # 7-10yr vs 20yr treasuries — strong link
    ]
    summary = []
    for a, b in pairs:
        sdf = make_spread_df(data[a], data[b])
        r = run_pairs_gauntlet(f"PAIRS {a}/{b}", sdf, data["SPY"], costs, n_perm=500)
        summary.append(r)

    print("\n\n" + "═" * 72)
    print("  PAIRS SUMMARY")
    print("═" * 72)
    ns = sum(1 for r in summary if r["survives"])
    print(f"  Survivors: {ns}/{len(summary)} pairs")
    for r in sorted(summary, key=lambda x: x["p"]):
        flag = "SURVIVES" if r["survives"] else "killed"
        neutral = "neutral" if (r["up"] > 0 and r["dn"] > 0) else "directional"
        print(f"    {r['name'][6:]:9s}: return {r['ret']:+7.1f}%  MCPT p={r['p']:.3f}  "
              f"[{neutral}]  [{flag}]")
    print("═" * 72)

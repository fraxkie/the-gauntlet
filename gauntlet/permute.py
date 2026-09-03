"""
permute.py  —  STEP 2: the permutation engine (Masters / doc-40)
═══════════════════════════════════════════════════════════════════════════════
ONE job: take a real price series and produce a synthetic one that has the SAME
statistical fingerprint (drift, volatility, bar-shape distribution) but with the
PATH shuffled so any real temporal pattern — the thing a strategy claims to
exploit — is destroyed.

Why log space + relative decomposition (Masters' method):
  • Work in log prices so returns are additive and the reassembly is exact.
  • Split each bar into:
        gap_i      = open_i  - close_{i-1}   (overnight move)
        rel_high_i = high_i  - open_i        (intrabar shape) }
        rel_low_i  = low_i   - open_i                          } shuffled together
        rel_close_i= close_i - open_i                          }
  • Shuffle the gaps independently from the intrabar shapes, then string the bars
    back together. The first bar is an anchor and is left in place.

Result: the multiset of daily moves is identical (so drift & vol are preserved),
but their ORDER is randomized (so autocorrelation / patterns are destroyed).

This file does NOT know what a strategy is. It only makes noise that looks real.
Proven in tests/test_permute.py by measuring drift, vol, and autocorrelation.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd


def permute_prices(df: pd.DataFrame, start_index: int = 0,
                   rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Return a permuted copy of an OHLC dataframe.

    start_index : bars strictly before this index are left UNCHANGED. Used by the
                  walk-forward MCPT so only the out-of-sample tail is permuted
                  while the in-sample head (the anchor history) stays put.
    rng         : pass a seeded Generator for reproducibility; default fresh one.
    """
    if rng is None:
        rng = np.random.default_rng()

    o = np.log(df["open"].to_numpy(dtype=float))
    h = np.log(df["high"].to_numpy(dtype=float))
    l = np.log(df["low"].to_numpy(dtype=float))
    c = np.log(df["close"].to_numpy(dtype=float))
    n = len(c)

    if n - start_index < 3:
        return df.copy()

    # decompose
    rel_h = h - o
    rel_l = l - o
    rel_c = c - o
    gap = np.zeros(n)
    gap[1:] = o[1:] - c[:-1]

    # indices eligible to shuffle: from start_index+1 to end (start bar = anchor)
    idx = np.arange(start_index + 1, n)
    perm_shape = rng.permutation(idx)   # shuffles intrabar (h,l,c relative to o)
    perm_gap = rng.permutation(idx)     # shuffles overnight gaps independently

    new_o = o.copy(); new_h = h.copy(); new_l = l.copy(); new_c = c.copy()

    prev_close = c[start_index]
    for k in range(len(idx)):
        i = start_index + 1 + k
        new_o[i] = prev_close + gap[perm_gap[k]]
        new_h[i] = new_o[i] + rel_h[perm_shape[k]]
        new_l[i] = new_o[i] + rel_l[perm_shape[k]]
        new_c[i] = new_o[i] + rel_c[perm_shape[k]]
        prev_close = new_c[i]

    return pd.DataFrame({
        "open":  np.exp(new_o),
        "high":  np.exp(new_h),
        "low":   np.exp(new_l),
        "close": np.exp(new_c),
    }, index=df.index)

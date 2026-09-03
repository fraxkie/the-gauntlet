"""
core.py  —  STEP 1: the foundation
═══════════════════════════════════════════════════════════════════════════════
The pieces every firewall stands on:
  • Costs        — no free fills; commission + slippage charged on every turn
  • Strategy     — a function (OHLC df) -> position series in {-1, 0, +1}
  • strategy_returns — converts positions to per-bar NET returns, with the
                       no-look-ahead rule enforced in ONE place so no strategy
                       can accidentally cheat through it
  • profit_factor / sharpe — the objective functions

Keep this file small and boring. If it's wrong, everything downstream is wrong,
so it has a dedicated proof in tests/test_core.py.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Costs:
    """Transaction costs in basis points, charged per side."""
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def per_turn(self) -> float:
        """Fraction charged on the notional that changes hands when position moves."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0


# A strategy maps an OHLC dataframe to a position series in {-1, 0, +1}.
# The position at bar i is the position HELD GOING INTO the next bar.
Strategy = Callable[[pd.DataFrame], pd.Series]


def bar_returns(df: pd.DataFrame) -> np.ndarray:
    """Close-to-close simple returns. ret[0] = 0; ret[i] = close[i]/close[i-1] - 1."""
    close = df["close"].to_numpy(dtype=float)
    rets = np.zeros_like(close)
    rets[1:] = close[1:] / close[:-1] - 1.0
    return rets


def strategy_returns(df: pd.DataFrame, positions: pd.Series, costs: Costs) -> pd.Series:
    """
    Positions -> per-bar NET returns, after costs.

    NO-LOOK-AHEAD RULE (enforced here, once): a position decided using data
    through bar i earns the return from bar i -> i+1. We shift the position
    forward by one bar before multiplying by that bar's return. This makes it
    structurally impossible for a strategy to earn a bar's return using that
    same bar's (future) close.

    COSTS: charged whenever the held position changes, proportional to the size
    of the change (so flat->long, long->short etc. all cost correctly).
    """
    rets = bar_returns(df)
    pos = positions.to_numpy(dtype=float)

    pos_held = np.zeros_like(pos)
    pos_held[1:] = pos[:-1]            # <-- the shift that prevents look-ahead
    gross = pos_held * rets

    turn = np.zeros_like(pos)
    turn[1:] = np.abs(pos[1:] - pos[:-1])
    cost = turn * costs.per_turn

    return pd.Series(gross - cost, index=df.index)


def total_return_pct(net_rets: np.ndarray) -> float:
    """Compounded total return of a net-return series, in percent.

    Guards against degenerate inputs: a single bar return <= -100% would make
    log1p undefined/-inf and blow up the product. We floor per-bar returns just
    above -100% (a -100% bar already means total wipeout) so the compounding stays
    finite and the number is interpretable rather than astronomical."""
    r = np.asarray(net_rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    r = np.clip(r, -0.999999, None)        # a bar can't lose more than ~100%
    return float((np.exp(np.log1p(r).sum()) - 1.0) * 100.0)


def mean_bps(net_rets: np.ndarray) -> float:
    """Mean return per bar in basis points. BOUNDED — the correct way to measure a
    non-contiguous subset of days (a market regime), unlike compounding a subset
    into a fake equity curve which explodes. Answers 'avg per-bar P&L in this set'."""
    r = np.asarray(net_rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return float(r.mean() * 1e4)


def profit_factor(net_rets: np.ndarray) -> float:
    """Gross wins / gross losses. >1 means profitable. The doc-40 objective."""
    wins = net_rets[net_rets > 0].sum()
    losses = -net_rets[net_rets < 0].sum()
    if losses <= 1e-12:
        return float("inf") if wins > 0 else 1.0
    return float(wins / losses)


def sharpe(net_rets: np.ndarray, periods_per_year: int = 252) -> float:
    sd = net_rets.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float((net_rets.mean() / sd) * np.sqrt(periods_per_year))

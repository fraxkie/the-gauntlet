"""
corr_engine.py — a cross-sectional correlation regime engine
═══════════════════════════════════════════════════════════════════════════════
Reads market regime from how TOGETHER stocks move (dispersion), independent of the
HMM which reads it from volatility. Low avg correlation = "market of stocks" (calm,
idiosyncratic); high correlation = "stock market" (crisis, everything moves as one).
"""
import numpy as np, pandas as pd

class CorrelationEngine:
    def __init__(self, window=21):
        self.window = window

    def realized_correlation(self, stock_returns):
        """Avg pairwise correlation over a rolling window, via the index-variance method:
        sigma_idx^2 = (1/N)avg_var + (1-1/N)*rho*avg_var  ->  solve for rho."""
        N = stock_returns.shape[1]
        idx_var = stock_returns.mean(axis=1).rolling(self.window).var()
        avg_var = stock_returns.rolling(self.window).var().mean(axis=1)
        rho = ((idx_var/avg_var) - 1/N) / (1 - 1/N)
        return rho.clip(0, 1)

    def regime(self, stock_returns):
        """Label each day: 'dispersed' (calm), 'normal', or 'correlated' (crisis)."""
        rho = self.realized_correlation(stock_returns).dropna()
        lo, hi = rho.quantile(0.33), rho.quantile(0.67)
        lab = pd.Series("normal", index=rho.index)
        lab[rho <= lo] = "dispersed"
        lab[rho >= hi] = "correlated"
        return lab, rho

if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from probability.regime import GaussianHMM
    from gauntlet.data import load_etf_data
    stk = pd.read_csv("stock_close.csv", index_col=0, parse_dates=True)
    rets = stk.pct_change()
    d = load_etf_data("etf_data.csv")
    vix = pd.read_csv("vix_clean.csv", index_col=0, parse_dates=True)["VIX"]

    eng = CorrelationEngine(window=21)
    lab, rho = eng.regime(rets)
    print("="*76)
    print("CORRELATION ENGINE — validation")
    print("="*76)
    print(f"  Avg realized correlation: {rho.mean():.2f}")
    print(f"  Calm (VIX<18): corr={rho[vix.reindex(rho.index)<18].mean():.2f}  |  "
          f"Crisis (VIX>30): corr={rho[vix.reindex(rho.index)>30].mean():.2f}")
    print()

    # does the correlation regime LEAD the HMM crisis (like GEX did)?
    spy = d["SPY"]["close"]; r = spy.pct_change()
    rv = r.dropna().values
    hl = GaussianHMM(n_states=3, n_iter=60, seed=0).fit(rv[:int(len(rv)*.4)]).filter_proba(rv).argmax(axis=1)
    hmm = pd.Series(hl, index=r.dropna().index)
    cs = np.argmax([rv[hl==k].std() for k in range(3)])
    hmm_crisis = (hmm == cs).astype(int).reindex(lab.index).fillna(0)
    corr_high = (lab == "correlated").astype(int)

    onsets = hmm_crisis.diff() == 1
    leads = []
    for od in hmm_crisis.index[onsets.values]:
        w = corr_high.loc[:od].tail(10)
        if w.sum() > 0:
            leads.append((od - w[w==1].index[-1]).days)
    leads = np.array(leads)
    print(f"  HMM crisis onsets: {onsets.sum()}")
    print(f"  Correlation-engine 'correlated' fired in prior 10d: {len(leads)} ({len(leads)/max(onsets.sum(),1)*100:.0f}%)")
    if len(leads): print(f"  Avg lead time before HMM: {leads.mean():.1f} days")
    print()

    # do the two regimes agree, and does correlation add NEW info?
    agree = ((corr_high==1)&(hmm_crisis==1)).sum() + ((corr_high==0)&(hmm_crisis==0)).sum()
    print(f"  Agreement with HMM: {agree/len(lab)*100:.0f}%")
    print(f"  (moderate = they see related-but-different things = correlation adds info)")

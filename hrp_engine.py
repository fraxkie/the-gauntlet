"""
hrp_engine.py — Hierarchical Risk Parity (Lopez de Prado 2016) from scratch
═══════════════════════════════════════════════════════════════════════════════
Allocates capital across assets by: (1) hierarchical clustering on the correlation
distance matrix, (2) quasi-diagonalization (reorder by cluster), (3) recursive
bisection allocating inverse-variance risk down the tree. Avoids matrix inversion
(unlike mean-variance), giving more stable, less concentrated out-of-sample weights.
"""
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def correl_dist(corr):
    """Distance matrix from correlation: d = sqrt((1-corr)/2)."""
    d = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    np.fill_diagonal(d, 0.0)
    return d

def quasi_diag(link, n):
    """Return the leaf order from the linkage (groups correlated assets together)."""
    return list(leaves_list(link))

def hrp_weights(returns):
    """Compute HRP weights for a returns DataFrame (assets in columns)."""
    cov = returns.cov()
    corr = returns.corr().values
    n = cov.shape[0]
    if n == 1:
        return pd.Series([1.0], index=returns.columns)
    d = correl_dist(corr)
    link = linkage(squareform(d, checks=False), method="single")
    order = quasi_diag(link, n)
    sort_ix = returns.columns[order].tolist()

    # recursive bisection
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    cov_df = cov.loc[sort_ix, sort_ix]
    while clusters:
        new = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            def cluster_var(items):
                c = cov_df.loc[items, items].values
                iv = 1.0 / np.diag(c)
                iv /= iv.sum()
                return float(iv @ c @ iv)
            vL, vR = cluster_var(left), cluster_var(right)
            alpha = 1 - vL / (vL + vR)  # inverse-variance: less risky side gets more
            w[left] *= alpha
            w[right] *= (1 - alpha)
            new += [left, right]
        clusters = new
    return w.reindex(returns.columns)

if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from gauntlet.data import load_etf_data
    d = load_etf_data("etf_data.csv")

    # the overnight basket names + the panic sleeve as "assets" to weight
    # build their daily net-return streams (simplified) and HRP-weight them
    def metrics(s):
        s = np.asarray(s); s = s[np.isfinite(s)]
        cg = (np.prod(1+np.clip(s,-.5,.5))**(252/len(s))-1)*100
        sh = s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0
        eq=np.cumprod(1+np.clip(s,-.5,.5)); dd=((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)).min()*100
        return cg, sh, dd

    from probability.regime import GaussianHMM
    def overnight_net(tk):
        df=d[tk]; o=df['open'].values; c=df['close'].values; ov=o[1:]/c[:-1]-1
        rv=df['close'].pct_change().dropna().values
        lab=GaussianHMM(n_states=3,n_iter=50,seed=0).fit(rv[:int(len(rv)*.4)]).filter_proba(rv).argmax(axis=1)[:len(ov)]
        vol=pd.Series(rv).rolling(20).std().values[:len(ov)]; vm=np.nanmedian(vol)
        cs=np.argmax([rv[GaussianHMM(n_states=3,n_iter=50,seed=0).fit(rv[:int(len(rv)*.4)]).filter_proba(rv).argmax(axis=1)==k].std() for k in range(3)])
        m=(lab!=2)&(vol<=vm)
        return pd.Series(np.where(m,ov-1e-4,0.0),index=df.index[1:len(ov)+1])

    streams=pd.DataFrame({tk:overnight_net(tk) for tk in ['SPY','QQQ','XLK']}).fillna(0)
    # only use active days for the covariance (the days the edge is on)
    active=streams[(streams!=0).any(axis=1)]

    print("="*72)
    print("HRP vs EQUAL-WEIGHT — weighting the overnight basket (SPY/QQQ/XLK)")
    print("="*72)
    w_hrp=hrp_weights(active)
    print(f"  HRP weights:        {dict(w_hrp.round(3))}")
    print(f"  Equal weights:      {{'SPY': 0.333, 'QQQ': 0.333, 'XLK': 0.333}}")
    print()
    eq_port=streams.mean(axis=1)
    hrp_port=(streams*w_hrp).sum(axis=1)
    for name,p in [("Equal-weight basket",eq_port),("HRP-weighted basket",hrp_port)]:
        for lo,lbl in [(2015,'2015-26'),(2022,'2022-26')]:
            seg=p[p.index.year>=lo]
            cg,sh,dd=metrics(seg.values)
            print(f"  {name:>22} {lbl}: CAGR={cg:+.1f}% Sharpe={sh:.2f} maxDD={dd:.1f}%")
        print()

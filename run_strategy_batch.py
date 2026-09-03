"""
run_strategy_batch.py — many STRATEGY IDEAS x many assets, through the gauntlet.

The companion to run_batch.py (which tests panic-reversion generalization). This
tests a REGISTRY of strategy ideas — each with a required mechanism — across the
whole asset universe, with honest Bonferroni correction. The anti-fishing design:
no strategy enters without an economic mechanism, every test is counted, and a
"winner" must clear the CORRECTED bar (not raw p<0.05).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from gauntlet.core import Costs, strategy_returns, profit_factor, total_return_pct
from gauntlet.data import load_etf_data
from gauntlet.mcpt import mcpt
from gauntlet.robustness import walk_forward, regime_check
from gauntlet.lookahead import lookahead_audit

def s_macd(df):
    c=df["close"];e12=c.ewm(span=12).mean();e26=c.ewm(span=26).mean()
    macd=e12-e26;sig=macd.ewm(span=9).mean();ma200=c.rolling(200).mean()
    return pd.Series(((macd>sig)&(c>ma200)).astype(float).to_numpy(),index=df.index)
def s_rsi_rev(df):
    c=df["close"];d=c.diff();up=d.clip(lower=0).rolling(14).mean();dn=(-d.clip(upper=0)).rolling(14).mean()
    rsi=(100-100/(1+up/(dn+1e-12))).to_numpy();out=np.zeros(len(rsi));st=0.0
    for i in range(len(rsi)):
        if not np.isnan(rsi[i]):
            if rsi[i]<30:st=1.0
            elif rsi[i]>50:st=0.0
        out[i]=st
    return pd.Series(out,index=df.index)
def s_donchian(df):
    c=df["close"];hi=c.rolling(50).max();lo=c.rolling(50).min()
    cv=c.to_numpy();hv=hi.to_numpy();lv=lo.to_numpy();out=np.zeros(len(cv));st=0.0
    for i in range(len(cv)):
        if not np.isnan(hv[i]):
            if cv[i]>=hv[i]:st=1.0
            elif cv[i]<=lv[i]:st=0.0
        out[i]=st
    return pd.Series(out,index=df.index)
def s_5d_rev(df):
    r5=df["close"].pct_change(5).to_numpy();return pd.Series(np.nan_to_num(-np.sign(r5)),index=df.index)
def s_gap_fade(df):
    o=df["open"].to_numpy();c=df["close"].to_numpy();gap=np.zeros(len(c));gap[1:]=o[1:]/c[:-1]-1
    return pd.Series(np.nan_to_num(-np.sign(gap)),index=df.index)
def s_trend100(df):
    c=df["close"];ma=c.rolling(100).mean();return pd.Series((c>ma).astype(float).to_numpy(),index=df.index)

REGISTRY=[
    ("MACD+200MA","trend momentum",s_macd),
    ("RSI-reversion","oversold bounce",s_rsi_rev),
    ("Donchian-50","breakout continuation",s_donchian),
    ("5d-reversal","short-term overreaction",s_5d_rev),
    ("Gap-fade","gap overshoot fills",s_gap_fade),
    ("Trend-100MA","time-series momentum",s_trend100),
]

if __name__=="__main__":
    data=load_etf_data("etf_data.csv")
    costs=Costs(commission_bps=1.0,slippage_bps=2.0)
    universe=["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","TLT","IEF","GLD","EFA","EEM"]
    n_tests=len(REGISTRY)*len(universe);bar=0.05/n_tests
    print("#"*74)
    print(f"#  STRATEGY BATCH — {len(REGISTRY)} strategies x {len(universe)} assets = {n_tests} tests")
    print(f"#  Bonferroni-corrected bar: p < {bar:.5f}")
    print("#"*74)
    rows=[]
    for sname,mech,sfn in REGISTRY:
        for tk in universe:
            df=data[tk]
            try:
                audit=lookahead_audit(df,sfn);mc=mcpt(df,sfn,costs,n_perm=200,objective="pf")
                wf=walk_forward(df,sfn,costs);rg=regime_check(df,sfn,costs)
                net=strategy_returns(df,sfn(df),costs).to_numpy();p=mc["p_value"]
                raw=(p<0.05 and wf["consistency_gate"]=="PASS" and audit["look_ahead_clean"])
                corr=(p<bar and wf["consistency_gate"]=="PASS" and audit["look_ahead_clean"] and not rg["_beta_warning"])
                rows.append({"strategy":sname,"mechanism":mech,"asset":tk,
                    "return_pct":round(total_return_pct(net),1),"pf":round(profit_factor(net),3),
                    "mcpt_p":round(p,4),"wf":wf["consistency_gate"],"beta":rg["_beta_warning"],
                    "raw_sig":raw,"corrected_sig":corr})
            except Exception as e:
                rows.append({"strategy":sname,"asset":tk,"error":str(e)[:40]})
    df_res=pd.DataFrame(rows)
    valid=df_res[df_res["mcpt_p"].notna()]
    print("\n"+"═"*74)
    print("  FULL GRID — winners AND losers (no cherry-picking)")
    print("═"*74)
    nraw=int(valid["raw_sig"].sum());ncorr=int(valid["corrected_sig"].sum())
    print(f"  Tests run: {len(valid)}   Pass RAW p<0.05: {nraw} (expect ~{len(valid)*0.05:.0f} by chance)   Pass CORRECTED: {ncorr}")
    print()
    print("  Per-strategy: assets where it passed RAW (replication signal):")
    for sname,mech,_ in REGISTRY:
        sub=valid[valid["strategy"]==sname];hits=int(sub["raw_sig"].sum())
        flag="  <- replicates!" if hits>=4 else ""
        print(f"    {sname:>14} ({mech:<24}): {hits}/{len(sub)}{flag}")
    if ncorr>0:
        print("\n  *** SURVIVORS (cleared corrected bar) ***")
        for _,r in valid[valid["corrected_sig"]].iterrows():
            print(f"    {r['strategy']} on {r['asset']}: {r['return_pct']:+.0f}%  p={r['mcpt_p']:.4f}")
    else:
        print("\n  No strategy cleared the corrected bar. Honest result: these")
        print("  unconditional strategies are beta-or-noise (as we found before).")
    df_res.to_csv("strategy_batch_results.csv",index=False)
    print(f"\n  Saved strategy_batch_results.csv")

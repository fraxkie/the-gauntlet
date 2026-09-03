"""
volatility.py — STEP 1: volatility PREDICTION (where real signal actually lives)
═══════════════════════════════════════════════════════════════════════════════
Three independent sources (academic research, hedge-fund-models list, the de-hyped
"quantum" post) all converge here: predicting DIRECTION is a coin flip, but
predicting VOLATILITY is genuinely tractable — because volatility CLUSTERS. Big
moves follow big moves; calm follows calm. That clustering is real, persistent
autocorrelation, and it's what makes volatility forecastable where direction isn't.

Models (simplest -> most standard):
  • NAIVE      : tomorrow's vol = today's realized vol (the baseline to beat)
  • EWMA       : exponentially-weighted moving average of squared returns
                 (RiskMetrics' approach; decay lambda ~0.94). Cheap, strong.
  • GARCH(1,1) : the standard econometric volatility model. Variance tomorrow =
                 omega + alpha*(today's shock^2) + beta*(today's variance). Captures
                 clustering AND mean-reversion of volatility. If `arch` is
                 installed we use it; otherwise a from-scratch MLE-lite fallback.

Evaluation is WALK-FORWARD and OUT-OF-SAMPLE: fit on past, predict next chunk, roll.
A real vol model must beat NAIVE on out-of-sample forecast error (we use QLIKE and
RMSE on the variance, the standard vol-forecast loss functions). Beating naive is a
low bar that direction models can't clear — vol models can, and that's the point.
═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd

try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False


def realized_vol(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Trailing realized volatility (std of returns) over `window`. This is what we
    try to predict: next period's realized vol."""
    s = pd.Series(returns)
    return s.rolling(window).std().to_numpy()


# ── forecasters: each maps a history of returns -> next-step variance forecast ──
def forecast_naive(returns: np.ndarray, window: int = 21) -> float:
    """Tomorrow's variance = the most recent realized variance."""
    r = returns[-window:]
    return float(np.nanvar(r))


def forecast_ewma(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA: var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2, iterated."""
    r = returns[np.isfinite(returns)]
    if len(r) < 5:
        return float(np.nanvar(r)) if len(r) else np.nan
    var = np.nanvar(r[: min(20, len(r))])
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(var)


def forecast_garch(returns: np.ndarray) -> float:
    """GARCH(1,1) one-step-ahead variance forecast. Uses `arch` if available, else
    a lightweight grid-fit fallback so the module runs anywhere."""
    r = returns[np.isfinite(returns)]
    if len(r) < 100:
        return forecast_ewma(r)
    if HAVE_ARCH:
        # arch wants percent returns for numerical stability
        am = arch_model(r * 100, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=1, reissue=False)
        var_pct = fc.variance.values[-1, 0]
        return float(var_pct / (100 ** 2))
    # fallback: crude GARCH(1,1) via a small parameter grid maximizing pseudo-LL
    return _garch_fallback(r)


def _garch_fallback(r: np.ndarray) -> float:
    uncond = np.var(r)
    best_ll, best_var = -np.inf, uncond
    # small grid over (alpha, beta); omega pinned to match unconditional variance
    for alpha in (0.03, 0.05, 0.08, 0.1, 0.15):
        for beta in (0.8, 0.85, 0.9, 0.92):
            if alpha + beta >= 0.999:
                continue
            omega = uncond * (1 - alpha - beta)
            var = uncond
            ll = 0.0
            for x in r:
                var = omega + alpha * x * x + beta * var
                if var <= 1e-12:
                    ll = -np.inf; break
                ll += -0.5 * (np.log(var) + x * x / var)
            if ll > best_ll:
                best_ll = ll
                # one more step for the forecast
                best_var = omega + alpha * r[-1] ** 2 + beta * var
    return float(best_var)


# ── walk-forward out-of-sample evaluation ──
def walk_forward_vol(returns: np.ndarray, method: str = "ewma",
                     train_window: int = 504, step: int = 21,
                     target_window: int = 21):
    """Produce OOS next-period-variance forecasts and the realized target.

    At each step we forecast the variance of the NEXT `target_window` bars using
    only data up to now, then compare to what actually happened. Returns
    (forecast_var, realized_var, index_positions) for the OOS region.
    """
    r = np.asarray(returns, float)
    n = len(r)
    fc_fn = {"naive": forecast_naive, "ewma": forecast_ewma, "garch": forecast_garch}[method]

    fc_list, rv_list, pos_list = [], [], []
    start = train_window
    while start + target_window <= n:
        hist = r[max(0, start - train_window):start]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 50:
            start += step; continue
        f = fc_fn(hist)
        # realized variance over the NEXT target_window bars (the truth)
        future = r[start:start + target_window]
        future = future[np.isfinite(future)]
        if len(future) < target_window // 2:
            start += step; continue
        rv = np.var(future)
        fc_list.append(f); rv_list.append(rv); pos_list.append(start)
        start += step

    return np.array(fc_list), np.array(rv_list), np.array(pos_list)


# ── forecast loss functions (standard in the vol literature) ──
def qlike(forecast_var, realized_var):
    """QLIKE loss: realized/forecast - log(realized/forecast) - 1. Robust, the
    preferred loss for variance forecasts. Lower = better."""
    f = np.asarray(forecast_var, float); rv = np.asarray(realized_var, float)
    mask = (f > 1e-12) & (rv > 1e-12) & np.isfinite(f) & np.isfinite(rv)
    f, rv = f[mask], rv[mask]
    ratio = rv / f
    return float(np.mean(ratio - np.log(ratio) - 1))


def rmse_var(forecast_var, realized_var):
    f = np.asarray(forecast_var, float); rv = np.asarray(realized_var, float)
    mask = np.isfinite(f) & np.isfinite(rv)
    return float(np.sqrt(np.mean((f[mask] - rv[mask]) ** 2)))


def evaluate_vol_models(returns: np.ndarray, label: str = "") -> dict:
    """Compare naive / ewma / garch out-of-sample. The test: do EWMA and GARCH
    beat NAIVE on QLIKE? Volatility's predictability means they should — the thing
    direction models could never do."""
    out = {"label": label, "models": {}}
    naive_q = None
    for m in ["naive", "ewma", "garch"]:
        fc, rv, _ = walk_forward_vol(returns, method=m)
        if len(fc) == 0:
            continue
        q = qlike(fc, rv); rm = rmse_var(fc, rv)
        # correlation between forecast and realized vol (forecast skill)
        corr = float(np.corrcoef(np.sqrt(np.abs(fc)), np.sqrt(np.abs(rv)))[0, 1])
        out["models"][m] = {"qlike": q, "rmse_var": rm, "vol_corr": corr, "n": len(fc)}
        if m == "naive":
            naive_q = q
    # skill vs naive
    for m, d in out["models"].items():
        if naive_q and naive_q > 0:
            d["qlike_improvement_vs_naive"] = (naive_q - d["qlike"]) / naive_q
    return out

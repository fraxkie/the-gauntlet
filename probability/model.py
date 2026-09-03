"""
model.py — the probability model with WALK-FORWARD out-of-sample prediction.

Quant-grade discipline: a probability model is only honest if its forecasts are
OUT-OF-SAMPLE. In-sample probabilities always look calibrated (the model fit them),
so they're worthless for validation. We use walk-forward: train on a window of
past data, predict the next (unseen) chunk, then roll forward — exactly how the
model would have to operate live.

Baseline model: LOGISTIC REGRESSION. Chosen deliberately as the honest starting
point — it's linear, interpretable, hard to overfit, and outputs a genuine
probability. The research uses everything up to LSTMs, but starting simple means a
positive result is trustworthy rather than a black-box fluke. Upgradeable later to
gradient boosting / neural nets once the pipeline is proven.

Output: an out-of-sample probability for (almost) every bar, ready for the
calibration harness and Kelly sizing.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from probability.features import feature_target_matrix


def walk_forward_proba(df: pd.DataFrame, horizon: int = 1,
                       train_window: int = 756, step: int = 63,
                       min_train: int = 504, model_fn=None):
    """Produce out-of-sample P(up) for each bar via expanding/rolling walk-forward.

    train_window : bars of history to train on (756 ≈ 3 years).
    step         : how many bars to predict before retraining (63 ≈ 1 quarter).
    min_train    : minimum history before the first prediction.
    model_fn     : callable returning a fresh sklearn-style classifier with
                   predict_proba; defaults to standardized logistic regression.

    Returns (probs, outcomes, index) aligned — only OOS predictions are included.
    """
    if model_fn is None:
        model_fn = lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0))

    X, y, idx = feature_target_matrix(df, horizon)
    Xv = X.to_numpy(); yv = y.to_numpy()
    n = len(Xv)

    probs = np.full(n, np.nan)
    start = min_train
    while start < n:
        end = min(start + step, n)
        # rolling window: train on the most recent train_window bars before `start`
        lo = max(0, start - train_window)
        Xtr, ytr = Xv[lo:start], yv[lo:start]
        Xte = Xv[start:end]
        # need both classes present to fit a classifier
        if len(np.unique(ytr)) < 2 or len(Xtr) < min_train:
            start = end
            continue
        model = model_fn()
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]   # P(class 1 = up)
        probs[start:end] = p
        start = end

    mask = np.isfinite(probs)
    return probs[mask], yv[mask], idx[mask]


def probs_to_dataframe(df, horizon=1, **kw):
    """Convenience: return a DataFrame with the OOS probability and realized outcome
    indexed by date, for inspection / feeding downstream modules."""
    p, o, idx = walk_forward_proba(df, horizon=horizon, **kw)
    return pd.DataFrame({"prob_up": p, "outcome": o}, index=idx)

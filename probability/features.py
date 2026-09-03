"""
features.py — feature engineering for the probability model.

The gold-bot video and the asset-pricing literature both insist: features must be
RELATIVE / NORMALIZED, never absolute. "Gold is $4100" means nothing; "today's
range is 1.3x its 20-day average" means something. Absolute price levels are
meaningless to a model and induce spurious nonstationarity.

Every feature here uses ONLY past/current data (computed from values available at
the bar), so a feature at bar t never peeks at t+1. The target (what we predict) is
the NEXT bar's direction, shifted so training never sees the answer.

Features (all normalized):
  • ret_1, ret_5, ret_20  : trailing returns over windows
  • vol_20                : trailing realized volatility (normalized by its own mean)
  • mom_norm              : 50-day momentum normalized by volatility (a z-score)
  • rsi_14                : RSI (already bounded 0-100, rescaled to 0-1)
  • range_norm            : today's high-low range / its 20-day average
  • dist_ma               : (close - 50MA) / close, how far above/below trend
  • gap                   : overnight gap (open vs prior close), normalized by vol
"""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of normalized features indexed like df. NaNs in the warmup
    region are expected (callers drop them)."""
    close = df["close"]; high = df["high"]; low = df["low"]; open_ = df["open"]

    feat = pd.DataFrame(index=df.index)
    # trailing returns
    feat["ret_1"] = close.pct_change(1)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_20"] = close.pct_change(20)

    # volatility, normalized by its own long-run level (so it's regime-relative)
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20).std()
    feat["vol_norm"] = vol20 / vol20.rolling(252).mean()

    # momentum normalized by volatility = a z-score of the 50d move
    mom50 = close / close.shift(50) - 1.0
    feat["mom_norm"] = mom50 / (vol20 * np.sqrt(50) + 1e-12)

    # RSI rescaled to 0-1
    feat["rsi"] = rsi(close, 14) / 100.0

    # today's range relative to its 20-day average
    rng = (high - low) / close
    feat["range_norm"] = rng / (rng.rolling(20).mean() + 1e-12)

    # distance from 50-day trend
    ma50 = close.rolling(50).mean()
    feat["dist_ma"] = (close - ma50) / close

    # overnight gap normalized by volatility
    gap = open_ / close.shift(1) - 1.0
    feat["gap_norm"] = gap / (vol20 + 1e-12)

    return feat


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Binary target: does the close `horizon` bars ahead exceed today's close?
    SHIFTED so that the target at row t refers to the FUTURE — training pairs
    feature[t] with target[t] = (close[t+horizon] > close[t]). The model learns
    to predict the future from the present; no look-ahead because at prediction
    time we only USE features[t] and never the target."""
    close = df["close"]
    future = close.shift(-horizon)
    return (future > close).astype(float)


def feature_target_matrix(df: pd.DataFrame, horizon: int = 1):
    """Aligned (X, y, index) with warmup and trailing NaNs dropped."""
    X = build_features(df)
    y = make_target(df, horizon)
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid], df.index[valid]

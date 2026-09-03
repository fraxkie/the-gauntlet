"""
regime.py — STEP 3: probabilistic REGIME DETECTION (Hidden Markov Model)
═══════════════════════════════════════════════════════════════════════════════
This is the legitimate core of the "quantum probability surface" post, stripped of
the marketing: a Hidden Markov Model. No quantum anything — just the standard,
powerful technique for inferring hidden states a system switches between.

The idea: the market is in one of K hidden REGIMES at each time (e.g. calm-bull,
calm-bear, high-vol-crisis). We never observe the regime directly — we only see
returns. The HMM infers, from the sequence of returns, the PROBABILITY of being in
each regime at each time, plus how regimes transition. Unlike our old crude
up/down/flat MA-slope check, this is:
  • PROBABILISTIC: "73% likely in the high-vol regime" not a hard label
  • LEARNED: the regimes' characteristics (mean, vol) and transition odds are fit
    from data, not hand-specified
  • CONTINUOUS in belief: a smooth probability surface over regimes through time
    (exactly what that post's 3D surface was gesturing at)

Built from scratch (no hmmlearn): Gaussian emissions, with
  • forward_backward : filtered/smoothed regime probabilities (the E-step)
  • baum_welch       : EM training of transition matrix + emission params
  • viterbi          : single most-likely regime PATH

NO-LOOK-AHEAD for live use: `filter_proba` gives the regime belief using only data
up to time t (causal). Smoothed/Viterbi use the whole series and are for ANALYSIS
of history, not for trading signals — we keep that distinction explicit.
═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np


def _gaussian_logpdf(x, mean, var):
    var = max(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


class GaussianHMM:
    """Hidden Markov Model with univariate Gaussian emissions per state."""

    def __init__(self, n_states=2, n_iter=50, tol=1e-4, seed=0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        # parameters (initialized in fit)
        self.pi = None          # initial state distribution [K]
        self.A = None           # transition matrix [K,K]
        self.means = None       # emission means [K]
        self.vars = None        # emission variances [K]

    # ── initialization ──
    def _init_params(self, x):
        K = self.K
        # initialize means by quantiles of the data, vars by global var, uniform-ish A
        qs = np.quantile(x, np.linspace(0.1, 0.9, K))
        self.means = qs.copy()
        self.vars = np.full(K, np.var(x) + 1e-8)
        self.pi = np.full(K, 1.0 / K)
        A = np.full((K, K), 0.1 / (K - 1)) if K > 1 else np.array([[1.0]])
        np.fill_diagonal(A, 0.9)        # regimes are persistent (sticky)
        self.A = A / A.sum(axis=1, keepdims=True)

    # ── emission log-likelihoods for all states at all times ──
    def _log_emissions(self, x):
        T = len(x); K = self.K
        logB = np.empty((T, K))
        for k in range(K):
            logB[:, k] = _gaussian_logpdf(x, self.means[k], self.vars[k])
        return logB

    # ── forward-backward in log space (numerically stable) ──
    def forward_backward(self, x):
        T = len(x); K = self.K
        logB = self._log_emissions(x)
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)

        # forward (alpha)
        log_alpha = np.empty((T, K))
        log_alpha[0] = logpi + logB[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logB[t, k] + _logsumexp(log_alpha[t-1] + logA[:, k])
        # backward (beta)
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = _logsumexp(logA[k, :] + logB[t+1] + log_beta[t+1])

        log_likelihood = _logsumexp(log_alpha[-1])
        # smoothed posteriors gamma[t,k] = P(state k at t | all data)
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma, log_alpha, log_beta, logB, log_likelihood

    # ── Baum-Welch EM training ──
    def fit(self, x):
        x = np.asarray(x, float)
        self._init_params(x)
        prev_ll = -np.inf
        for it in range(self.n_iter):
            gamma, log_alpha, log_beta, logB, ll = self.forward_backward(x)
            logA = np.log(self.A + 1e-300)
            T, K = len(x), self.K

            # xi summed over t: expected transitions
            log_xi_acc = np.full((K, K), -np.inf)
            for t in range(T - 1):
                denom = _logsumexp(
                    (log_alpha[t][:, None] + logA + logB[t+1][None, :] + log_beta[t+1][None, :]).ravel()
                )
                for i in range(K):
                    vals = log_alpha[t, i] + logA[i, :] + logB[t+1] + log_beta[t+1] - denom
                    log_xi_acc[i] = np.logaddexp(log_xi_acc[i], vals)

            # updates
            self.pi = gamma[0] / gamma[0].sum()
            xi = np.exp(log_xi_acc)
            self.A = xi / (xi.sum(axis=1, keepdims=True) + 1e-300)
            for k in range(K):
                w = gamma[:, k]
                wsum = w.sum() + 1e-300
                self.means[k] = (w * x).sum() / wsum
                self.vars[k] = (w * (x - self.means[k]) ** 2).sum() / wsum + 1e-10

            if ll - prev_ll < self.tol and it > 5:
                break
            prev_ll = ll
        # order states by variance so "regime 0 = calm, regime K-1 = stormy"
        order = np.argsort(self.vars)
        self.means = self.means[order]; self.vars = self.vars[order]
        self.A = self.A[np.ix_(order, order)]
        self.pi = self.pi[order]
        return self

    def smoothed_proba(self, x):
        """P(regime | ALL data) — for analyzing history (uses future; not causal)."""
        gamma, *_ = self.forward_backward(np.asarray(x, float))
        return gamma

    def filter_proba(self, x):
        """P(regime_t | data up to t) — CAUSAL, no look-ahead, safe for live signals.
        Runs only the forward pass, normalized per time step."""
        x = np.asarray(x, float)
        T, K = len(x), self.K
        logB = self._log_emissions(x)
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        log_alpha = np.empty((T, K))
        log_alpha[0] = logpi + logB[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logB[t, k] + _logsumexp(log_alpha[t-1] + logA[:, k])
        # normalize each row to get filtered posterior
        filt = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        filt /= filt.sum(axis=1, keepdims=True)
        return filt

    def viterbi(self, x):
        """Most-likely single regime PATH (hard labels) given all data."""
        x = np.asarray(x, float)
        T, K = len(x), self.K
        logB = self._log_emissions(x)
        logA = np.log(self.A + 1e-300)
        logpi = np.log(self.pi + 1e-300)
        delta = np.empty((T, K)); psi = np.zeros((T, K), int)
        delta[0] = logpi + logB[0]
        for t in range(1, T):
            for k in range(K):
                seq = delta[t-1] + logA[:, k]
                psi[t, k] = np.argmax(seq)
                delta[t, k] = logB[t, k] + seq.max()
        path = np.zeros(T, int); path[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path


def _logsumexp(a):
    a = np.asarray(a, float)
    m = a.max()
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(a - m)))


# ── convenience: fit a regime model to returns and describe the regimes ──
def detect_regimes(returns, n_states=2, n_iter=60, seed=0):
    """Fit a Gaussian HMM to a return series. Returns the model plus a description
    of each regime (annualized mean/vol, persistence)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    model = GaussianHMM(n_states=n_states, n_iter=n_iter, seed=seed).fit(r)
    desc = []
    for k in range(n_states):
        persistence = model.A[k, k]
        desc.append({
            "regime": k,
            "ann_return": float(model.means[k] * 252 * 100),
            "ann_vol": float(np.sqrt(model.vars[k] * 252) * 100),
            "persistence": float(persistence),
            "avg_duration_days": float(1 / (1 - persistence + 1e-9)),
        })
    return model, desc

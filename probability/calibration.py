"""
calibration.py — the CALIBRATION HARNESS (heart of the probability gauntlet)
═══════════════════════════════════════════════════════════════════════════════
Calibration is the lie-detector for probability forecasts. A model that says "60%
up" is only worth money if, out of sample, 60%-predicted events happen ~60% of the
time. This is the probability analog of MCPT: it's the firewall that separates a
real probabilistic edge from an overconfident fantasy.

The core tool is the BRIER SCORE — strictly proper, meaning a forecaster cannot
improve their expected score by stating anything other than their true beliefs.
Murphy's decomposition splits it into three meaningful pieces:

    Brier = Reliability - Resolution + Uncertainty

  • RELIABILITY (lower = better): calibration. Mean squared gap between stated
    probability and actual frequency within each bin. Zero = perfectly honest.
  • RESOLUTION (higher = better): discrimination/sharpness. How much the binned
    frequencies deviate from the base rate. A model that always predicts the base
    rate is perfectly calibrated but USELESS — zero resolution, no edge.
  • UNCERTAINTY: the irreducible variance of the outcome itself (data-dependent,
    not a property of the model).

A useful model needs BOTH: calibrated (low reliability term) AND sharp (high
resolution). The reliability DIAGRAM (observed freq vs stated prob, per bin) is the
money visual — points on the 45-degree line = perfectly calibrated; below = over-
confident (the classic retail failure: state 85%, win 70%).

Everything here must be computed OUT-OF-SAMPLE (walk-forward). In-sample calibration
is meaningless — any model looks calibrated on data it was fit to.
═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between forecast probabilities and binary outcomes.
    Lower is better; 0 is perfect. Strictly proper scoring rule."""
    p = np.asarray(probs, float)
    o = np.asarray(outcomes, float)
    mask = np.isfinite(p) & np.isfinite(o)
    p, o = p[mask], o[mask]
    if p.size == 0:
        return np.nan
    return float(np.mean((p - o) ** 2))


def reliability_diagram(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Bin forecasts by stated probability; return per-bin (stated, observed, count).
    Perfect calibration => observed == stated (points on the 45-degree line)."""
    p = np.asarray(probs, float)
    o = np.asarray(outcomes, float)
    mask = np.isfinite(p) & np.isfinite(o)
    p, o = p[mask], o[mask]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin closed on the right
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(sel.sum())
        if n == 0:
            bins.append({"bin_mid": (lo + hi) / 2, "stated": np.nan,
                         "observed": np.nan, "count": 0})
        else:
            bins.append({"bin_mid": (lo + hi) / 2,
                         "stated": float(p[sel].mean()),
                         "observed": float(o[sel].mean()),
                         "count": n})
    return bins


def murphy_decomposition(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Decompose Brier = Reliability - Resolution + Uncertainty.
    Returns dict with all three plus the recombined Brier (should match brier_score).
    """
    p = np.asarray(probs, float)
    o = np.asarray(outcomes, float)
    mask = np.isfinite(p) & np.isfinite(o)
    p, o = p[mask], o[mask]
    N = p.size
    if N == 0:
        return {"reliability": np.nan, "resolution": np.nan,
                "uncertainty": np.nan, "brier_recombined": np.nan}

    base_rate = o.mean()                      # overall outcome frequency
    uncertainty = base_rate * (1 - base_rate)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        nk = int(sel.sum())
        if nk == 0:
            continue
        pk = p[sel].mean()         # mean stated prob in bin
        ok = o[sel].mean()         # observed freq in bin
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base_rate) ** 2
    reliability /= N
    resolution /= N

    return {
        "reliability": float(reliability),    # lower better (calibration error)
        "resolution": float(resolution),      # higher better (discrimination)
        "uncertainty": float(uncertainty),    # data-fixed
        "base_rate": float(base_rate),
        "brier_recombined": float(reliability - resolution + uncertainty),
    }


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE: count-weighted average |stated - observed| across bins. A single-number
    summary of miscalibration (0 = perfect). Complements the reliability term
    (which is squared); ECE is in probability units, more interpretable."""
    bins = reliability_diagram(probs, outcomes, n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return np.nan
    ece = sum(b["count"] * abs(b["stated"] - b["observed"])
              for b in bins if b["count"] > 0)
    return float(ece / total)


def calibration_report(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10,
                       label: str = "") -> dict:
    """Full calibration assessment. The verdict a probability model must pass:
    well-calibrated (low ECE/reliability) AND sharp (resolution clearly > 0)."""
    bs = brier_score(probs, outcomes)
    dec = murphy_decomposition(probs, outcomes, n_bins)
    ece = expected_calibration_error(probs, outcomes, n_bins)
    diag = reliability_diagram(probs, outcomes, n_bins)

    # baseline Brier: always predict the base rate (the "no-skill" forecast).
    # A real model must beat this — beating it means positive resolution.
    base = dec["base_rate"]
    o = np.asarray(outcomes, float); o = o[np.isfinite(o)]
    brier_baseline = float(np.mean((base - o) ** 2)) if o.size else np.nan
    skill = (1 - bs / brier_baseline) if brier_baseline and brier_baseline > 0 else np.nan

    return {
        "label": label,
        "n": int(np.isfinite(probs).sum()),
        "brier": bs,
        "brier_baseline": brier_baseline,    # always-base-rate forecast
        "brier_skill_score": skill,          # >0 means better than no-skill
        "ece": ece,
        "reliability": dec["reliability"],
        "resolution": dec["resolution"],
        "uncertainty": dec["uncertainty"],
        "base_rate": dec["base_rate"],
        "diagram": diag,
    }


def print_calibration(rep: dict):
    print(f"\n  CALIBRATION REPORT{(' — ' + rep['label']) if rep['label'] else ''}")
    print(f"  {'-'*60}")
    print(f"  Brier score        : {rep['brier']:.4f}  "
          f"(baseline {rep['brier_baseline']:.4f})")
    ss = rep['brier_skill_score']
    print(f"  Brier skill score  : {ss:+.4f}  "
          f"({'BEATS no-skill' if (ss is not None and ss>0) else 'no better than base rate'})")
    print(f"  ECE (miscalib)     : {rep['ece']:.4f}  (0 = perfect calibration)")
    print(f"  Reliability  (low) : {rep['reliability']:.5f}")
    print(f"  Resolution   (high): {rep['resolution']:.5f}  "
          f"({'has discrimination' if rep['resolution']>1e-4 else 'NO discrimination — useless'})")
    print(f"  Base rate          : {rep['base_rate']:.3f}")
    print(f"  Reliability diagram (stated -> observed):")
    for b in rep["diagram"]:
        if b["count"] > 0:
            bar = "█" * max(1, int(b["count"] / max(1, rep["n"]) * 40))
            flag = ""
            if abs(b["stated"] - b["observed"]) > 0.1:
                flag = "  <- miscalibrated"
            print(f"    {b['stated']:.2f} -> {b['observed']:.2f}  "
                  f"(n={b['count']:>4}) {bar}{flag}")

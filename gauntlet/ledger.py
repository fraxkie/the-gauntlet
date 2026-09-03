"""
ledger.py — the persistent trial ledger (multiple-comparisons honesty layer)
═══════════════════════════════════════════════════════════════════════════════
The danger your own instinct surfaced: grind enough strategy variations and some
will pass MCPT at p<0.01 BY PURE CHANCE (that's literally what p<0.01 means — a
1-in-100 false-positive rate). Show only the survivors and you've manufactured
fool's gold with a certificate of authenticity.

This ledger is the antidote. It does two things:

1. PERSISTS every test ever run, across all sessions, to disk. The trial count is
   cumulative — testing 20 pairs tonight and 20 more next week counts as 40, not
   two separate sets of 20. The slow hand-search can't escape the math by being
   spread over time.

2. Applies a multiple-comparisons correction. If you test N strategies, the p-value
   bar to claim a real find gets HARDER. We use the Šidák / Bonferroni-style
   correction: to claim family-wise significance at 0.05 across N tests, an
   individual test must clear roughly 0.05/N (Bonferroni) — and we also report the
   expected number of false positives (N * alpha) so you can see how many "winners"
   pure luck would produce at the naive threshold.

A survivor that clears the CORRECTED bar is meaningfully more likely to be real.
One that only clears the naive 0.01 after 500 trials is probably the luckiest noise.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
from datetime import datetime, timezone

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "trial_ledger.jsonl")


def log_trial(name: str, p_value: float, total_return: float,
              extra: dict | None = None, ledger_path: str = LEDGER_PATH):
    """Append one trial to the persistent ledger."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "p_value": float(p_value),
        "total_return": float(total_return),
    }
    if extra:
        rec.update(extra)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_trials(ledger_path: str = LEDGER_PATH) -> list[dict]:
    if not os.path.exists(ledger_path):
        return []
    out = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def bonferroni_threshold(n_trials: int, family_alpha: float = 0.05) -> float:
    """The per-test p-value bar so the FAMILY-WIDE false-positive rate stays at
    family_alpha across n_trials. Bonferroni: alpha / n."""
    return family_alpha / max(n_trials, 1)


def assess(ledger_path: str = LEDGER_PATH, family_alpha: float = 0.05) -> dict:
    """Summarize the ledger and apply the multiple-comparisons correction to the
    cumulative trial count."""
    trials = load_trials(ledger_path)
    n = len(trials)
    if n == 0:
        return {"n_trials": 0, "note": "ledger empty"}

    naive_bar = 0.01
    corrected_bar = bonferroni_threshold(n, family_alpha)
    expected_false_positives = n * naive_bar  # how many pass naive 0.01 by luck

    passes_naive = [t for t in trials if t["p_value"] < naive_bar]
    passes_corrected = [t for t in trials if t["p_value"] < corrected_bar]

    return {
        "n_trials": n,
        "naive_bar": naive_bar,
        "corrected_bar": corrected_bar,
        "expected_false_positives_at_naive": round(expected_false_positives, 2),
        "n_pass_naive": len(passes_naive),
        "n_pass_corrected": len(passes_corrected),
        "survivors_corrected": sorted(passes_corrected, key=lambda t: t["p_value"]),
        "note": (f"After {n} cumulative trials, a real find must clear "
                 f"p<{corrected_bar:.2e} (not the naive 0.01). Pure luck would "
                 f"produce ~{expected_false_positives:.1f} 'winners' at the naive bar."),
    }


if __name__ == "__main__":
    a = assess()
    print(json.dumps(a, indent=2))

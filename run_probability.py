"""
run_probability.py — the full probability pipeline, end to end.

  features -> walk-forward logistic regression -> OOS P(up) -> calibration gauntlet

This answers the real quant question: can we produce CALIBRATED and SHARP
probability forecasts of next-day direction, out of sample? The calibration harness
is the judge. Expectation, stated honestly up front: next-day direction is close to
a coin flip (base rate ~53% up for equities), so we expect calibration to be OK but
RESOLUTION to be very low — i.e. honest probabilities with little edge. That's the
likely truth, and the harness will show it cleanly rather than letting us fool
ourselves with an impressive-looking accuracy number.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gauntlet.data import load_etf_data
from probability.model import walk_forward_proba
from probability.calibration import calibration_report, print_calibration


if __name__ == "__main__":
    data_file = "etf_data.csv"
    if not os.path.exists(data_file):
        # fall back to the uploads copy if running in a fresh env
        alt = "/mnt/user-data/uploads/etf_data.csv"
        data_file = alt if os.path.exists(alt) else data_file
    data = load_etf_data(data_file)

    print("#" * 72)
    print("#  PROBABILITY PIPELINE — walk-forward logistic regression -> calibration")
    print("#  Predicting next-day direction. Judge: the calibration harness.")
    print("#" * 72)

    for tk in ["SPY", "QQQ", "EEM"]:
        df = data[tk]
        probs, outcomes, idx = walk_forward_proba(df, horizon=1)
        rep = calibration_report(probs, outcomes, label=f"{tk} next-day up")
        print_calibration(rep)
        # the honest bottom line
        ss = rep["brier_skill_score"]; res = rep["resolution"]
        verdict = ("HAS A REAL EDGE (sharp + calibrated + beats baseline)"
                   if (ss is not None and ss > 0.01 and res > 1e-3)
                   else "calibrated but no exploitable edge (resolution ~0)")
        print(f"  >>> {tk}: {verdict}")

    print("\n" + "═" * 72)
    print("  Note: a positive Brier skill score with real resolution would be the")
    print("  first genuine predictive edge found. Low resolution = honest coin-flip.")
    print("═" * 72)

"""Diagnose DML undercoverage from an existing raw_replicates.csv.

Reads the per-rep output of run_simulation.py (no re-run needed) and, per
(estimator, N, dgp) cell, reconciles the three quantities that should agree
under a normal CI:

    empirical SD of theta_hat   vs   mean reported SE   vs   mean CI width

and recomputes coverage two ways (from the stored ci_low/ci_high, and from
theta +/- 1.96*se) so we can tell WHICH piece is wrong.

Outcomes and what they imply for the fix:
  (A) mean_se ~= emp_sd, but coverage still < 0.95  -> truth miscalibration
      (the single n_pop draw used to define `truth` is itself noisy); fix by
      recalibrating truth with more draws / larger n_pop.
  (B) mean_se << emp_sd (ratio well below 1)        -> SE genuinely too small.
      The IF variance underestimates real sampling variability -- the
      generated-regressor (PCA-estimated treatment) problem. Fix: cross-fit the
      PCA or report bootstrap CIs.
  (C) coverage_from_ci != coverage_from_theta_se, or mean_ci_width != 3.92*mean_se
      -> the CI/coverage CODE is inconsistent (a bug), not the estimator. Fix
      the formula; DML may be fine once corrected.

Usage:
    python3 data/scripts/simulation/diagnose_coverage.py \
        [--raw results/simulation/raw_replicates.csv] \
        [--truths results/simulation/truths.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "results" / "simulation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DEFAULT_DIR / "raw_replicates.csv"))
    ap.add_argument("--truths", default=str(DEFAULT_DIR / "truths.json"))
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    truths = {}
    tp = Path(args.truths)
    if tp.exists():
        truths = json.loads(tp.read_text())

    raw = raw.dropna(subset=["theta", "se", "ci_low", "ci_high"])
    z = 1.959963985

    rows = []
    for (est, N, dgp), g in raw.groupby(["estimator", "N", "dgp"]):
        theta = g["theta"].to_numpy(float)
        se = g["se"].to_numpy(float)
        lo = g["ci_low"].to_numpy(float)
        hi = g["ci_high"].to_numpy(float)
        truth = float(truths.get(f"{est}|{dgp}", 0.0))

        emp_sd = float(theta.std(ddof=1))
        mean_se = float(np.nanmean(se))
        mean_width = float(np.nanmean(hi - lo))
        implied_se_from_width = mean_width / (2 * z)

        cov_stored = float(np.mean((lo <= truth) & (truth <= hi)))
        cov_from_se = float(np.mean(np.abs(theta - truth) <= z * se))
        # coverage you'd get with a CORRECTLY-sized SE = the empirical SD:
        cov_if_se_were_empsd = float(np.mean(np.abs(theta - truth) <= z * emp_sd))

        rows.append({
            "cell": f"{est}|N={N}|{dgp}",
            "emp_sd": round(emp_sd, 5),
            "mean_se": round(mean_se, 5),
            "se/sd": round(mean_se / emp_sd, 3) if emp_sd else float("nan"),
            "mean_ci_w": round(mean_width, 5),
            "se_from_w": round(implied_se_from_width, 5),
            "w==3.92se?": "yes" if abs(implied_se_from_width - mean_se) < 1e-3 else "NO",
            "cov_stored": round(cov_stored, 3),
            "cov_theta±1.96se": round(cov_from_se, 3),
            "cov_if_se=sd": round(cov_if_se_were_empsd, 3),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(out.to_string(index=False))

    print("\nReading guide:")
    print("  se/sd << 1            -> SE underestimates true spread (outcome B: generated-regressor).")
    print("  w==3.92se? == NO      -> CI width inconsistent with reported SE (outcome C: CI/code bug).")
    print("  cov_stored != cov_theta±1.96se -> coverage code uses a different interval than theta±1.96se (outcome C).")
    print("  cov_if_se=sd ~ 0.95   -> the ESTIMATOR is fine; only the SE is mis-sized. Fixing the SE fixes coverage.")
    print("  cov_if_se=sd still <0.95 -> residual bias vs truth (outcome A: recalibrate truth, or real bias).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

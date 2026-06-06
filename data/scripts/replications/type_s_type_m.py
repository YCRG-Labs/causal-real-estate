"""Gelman-Carlin (2014) Type-S and Type-M analysis for per-market Shen-Ross.

Computes:
  - Power: probability of obtaining a statistically significant result
    at α=0.05 given the true effect size θ_true and standard error s.
  - Type-S error rate: P(sign(θ̂) ≠ sign(θ_true) | |θ̂| > critical, θ_true).
  - Type-M exaggeration ratio: E[|θ̂| | |θ̂| > critical, θ_true] / |θ_true|.

The Gelman-Carlin framework is the canonical defense for reporting low-
powered estimates: at low power the expected outcomes are exactly what
the observed pattern shows (a small number of false positives with
exaggerated magnitudes and a non-trivial sign-error rate).  Reporting
these statistics inoculates against the "you have nothing" critique by
demonstrating that the analysis acknowledges what the per-market power
actually buys.

Reference: Gelman, A. & Carlin, J. (2014) "Beyond Power Calculations:
Assessing Type S (Sign) and Type M (Magnitude) Errors", Perspectives on
Psychological Science 9(6):641-651.

Input: results/replications/shen_12city_table.csv
Output: results/replications/type_s_type_m_shen.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]

# Shen & Ross (2021), "Information Value of Property Description: A Machine
# Learning Approach," J. Urban Econ. 121:103299.  Verified against the
# published abstract: a one-standard-deviation increase in listing uniqueness
# raises sale price by 15% in the hedonic price model and by 10% in the
# repeat-sales model.  This replication is a cross-sectional hedonic DML, so
# the anchor is the hedonic figure, converted to log points via ln(1.15) =
# 0.1398.  (The conservative repeat-sales figure maps to ln(1.10) = 0.0953 and
# can be supplied through --theta_true.)
SHEN_PUBLISHED_THETA = 0.140


def retrodesign(theta_true: float, s: float, alpha: float = 0.05):
    """Gelman & Carlin 2014 retrodesign.  At true effect θ_true and
    estimator standard error s, computes the power, Type-S error rate,
    and Type-M exaggeration ratio for a two-sided test at level α.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    abs_theta = abs(theta_true)

    # Power: P(|θ̂| > z*s | θ_true) under N(θ_true, s²).
    upper = 1 - stats.norm.cdf(z_crit - abs_theta / s)
    lower = stats.norm.cdf(-z_crit - abs_theta / s)
    power = upper + lower

    # Type-S: probability the significant estimate has the wrong sign.
    if power <= 0:
        return dict(power=0.0, type_s=float("nan"),
                    type_m=float("nan"), z_crit=z_crit)
    type_s = lower / power if abs_theta > 0 else 0.5

    # Type-M: E[|θ̂| | |θ̂| > z*s, θ_true] / |θ_true|.
    # Compute via direct numerical integration over the truncated
    # normal tails.
    grid = np.linspace(-20, 20, 20001) * s + theta_true
    if grid.size == 0 or abs_theta == 0:
        type_m = float("nan")
    else:
        pdf = stats.norm.pdf(grid, loc=theta_true, scale=s)
        mask = np.abs(grid) > z_crit * s
        if mask.sum() == 0 or pdf[mask].sum() == 0:
            type_m = float("nan")
        else:
            type_m = (np.sum(np.abs(grid[mask]) * pdf[mask])
                       / np.sum(pdf[mask])) / abs_theta

    return dict(power=power, type_s=type_s, type_m=type_m, z_crit=z_crit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shen",
                    default=str(REPO / "results" / "replications"
                                  / "shen_12city_table.csv"))
    ap.add_argument("--theta_true", type=float,
                    default=SHEN_PUBLISHED_THETA,
                    help="assumed true per-σ effect; default is the Shen "
                         "& Ross 2021 hedonic published effect (15%, "
                         "0.140 log points)")
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications"
                                  / "type_s_type_m_shen.csv"))
    args = ap.parse_args()

    shen = pd.read_csv(args.shen)[["city", "n", "dml_theta", "dml_se"]]

    print(f"\n=== Gelman-Carlin Type-S/Type-M analysis ===")
    print(f"  assumed θ_true = {args.theta_true:+.4f} (per σ, "
          "Shen & Ross 2021 hedonic published: 15% ≈ 0.140 log pts)")
    print(f"  per-market estimators are Shen-Ross uniqueness DML "
          "with bootstrap SE\n")

    rows = []
    for _, r in shen.iterrows():
        rd = retrodesign(args.theta_true, r["dml_se"])
        rows.append({
            "city": r["city"], "n": r["n"],
            "dml_theta": r["dml_theta"], "dml_se": r["dml_se"],
            "power": rd["power"], "type_s": rd["type_s"],
            "type_m": rd["type_m"],
        })
    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  median power across 12 markets: {out['power'].median():.3f}")
    print(f"  median Type-S error rate: {out['type_s'].median():.3f}")
    print(f"  median Type-M magnification: {out['type_m'].median():.2f}x")
    n_min, n_max = int(out["n"].min()), int(out["n"].max())
    print(f"\n  Interpretation: at the per-market sample sizes in this "
          f"panel (n = {n_min:,}-{n_max:,}),")
    print(f"  the analysis has ~{out['power'].median()*100:.0f}% power to "
          f"detect Shen's published effect; significant findings would "
          f"exaggerate")
    print(f"  the true magnitude by ~{out['type_m'].median():.1f}x and "
          f"would have the wrong sign in ~{out['type_s'].median()*100:.0f}% "
          f"of cases.")
    print(f"\nCSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Threshold sensitivity for the doubly-robust ATE.

The paper's Appendix Table 13 currently reports ATE = -0.018 across three
price-outlier thresholds with N ~ 60K. Those numbers do not come from any
script in this repository, and they conflict with the main-text DR estimates
(BOS -0.001, NYC -0.138, SF -0.105, all on the ~1000-row description sample).

This script regenerates the threshold sensitivity table from the actual DR
estimator on the description sample, so the appendix matches the main text.
It runs the DR estimator at each (low, high) price threshold and reports the
ATE, IF-based 95% CI, MDE, and N retained, for every requested city.

Usage:
    python threshold_sensitivity.py            # all cities
    python threshold_sensitivity.py nyc        # one city
    python threshold_sensitivity.py nyc sf     # multiple
"""
import sys
import numpy as np
from config import CITIES
from causal_inference import (
    load_analysis_data,
    get_features_and_target,
    doubly_robust_estimation,
)

THRESHOLDS = [
    (25_000, 25_000_000, "$25K – $25M"),
    (50_000, 20_000_000, "$50K – $20M"),
    (75_000, 15_000_000, "$75K – $15M"),
]


def run_city(city):
    print(f"\n{'#'*72}")
    print(f"THRESHOLD SENSITIVITY: {city.upper()}")
    print(f"{'#'*72}")

    data = load_analysis_data(city)
    if data is None:
        print(f"  No data for {city}, skipping")
        return None

    emb_df, parcels = data
    feat = get_features_and_target(emb_df, parcels)
    if feat is None:
        print(f"  Could not build features for {city}, skipping")
        return None

    T_full, conf_full, Y_full, meta_full = feat
    prices = np.exp(Y_full)

    rows = []
    for lo, hi, label in THRESHOLDS:
        mask = (prices >= lo) & (prices <= hi)
        n_kept = int(mask.sum())
        n_dropped = int((~mask).sum())
        if n_kept < 50:
            print(f"\n  {label}: only {n_kept} retained, skipping")
            rows.append({
                "label": label, "n": n_kept, "ate": None,
                "ci_low": None, "ci_high": None, "mde": None,
            })
            continue

        T = T_full[mask]
        conf = conf_full[mask]
        Y = Y_full[mask]

        print(f"\n  {label}  (N retained={n_kept}, dropped={n_dropped})")
        ate, boot_ci, extras = doubly_robust_estimation(T, conf, Y)
        rows.append({
            "label": label,
            "n": n_kept,
            "ate": float(ate),
            "if_ci_low": float(extras["if_ci"][0]),
            "if_ci_high": float(extras["if_ci"][1]),
            "boot_ci_low": float(boot_ci[0]),
            "boot_ci_high": float(boot_ci[1]),
            "if_se": float(extras["if_se"]),
            "mde": float(extras["mde"]),
        })

    print(f"\n  {'─'*72}")
    print(f"  Sensitivity summary for {city}")
    print(f"  {'─'*72}")
    header = f"  {'Thresholds':<14} {'N':>7} {'ATE':>10} {'IF 95% CI':>22} {'MDE':>10}"
    print(header)
    print(f"  {'─'*72}")
    for r in rows:
        if r["ate"] is None:
            print(f"  {r['label']:<14} {r['n']:>7d}  {'(skipped)':>10}")
            continue
        ci = f"[{r['if_ci_low']:+.4f},{r['if_ci_high']:+.4f}]"
        print(f"  {r['label']:<14} {r['n']:>7d} {r['ate']:>+10.4f} {ci:>22} {r['mde']:>+10.4f}")

    if all(r["ate"] is not None for r in rows):
        ate_range = max(r["ate"] for r in rows) - min(r["ate"] for r in rows)
        print(f"\n  Max−min ATE across thresholds: {ate_range:.4f}")
        if ate_range < 0.05:
            print("  → Estimate stable across thresholds (robust to outlier filtering)")
        else:
            print("  → Estimate sensitive to thresholds; report range, not point estimate")

    return rows


def emit_latex_table(all_results):
    """Prints a LaTeX-formatted table that drops directly into the appendix."""
    print(f"\n\n{'='*72}")
    print("LATEX REPLACEMENT FOR APPENDIX TABLE 13 (sensitivity_results)")
    print(f"{'='*72}")
    print(r"\begin{table}[!htbp]")
    print(r"\centering")
    print(r"\caption{Sensitivity of doubly-robust ATE estimates to price outlier "
          r"thresholds, regenerated from the same DR pipeline as the main text. "
          r"All confidence intervals are influence-function based.}")
    print(r"\label{tab:sensitivity_results}")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"\textbf{City} & \textbf{Thresholds} & \textbf{N} & "
          r"$\hat{\tau}_{\text{causal}}$ & \textbf{95\% IF CI} \\")
    print(r"\midrule")
    for city, rows in all_results.items():
        if rows is None:
            continue
        for i, r in enumerate(rows):
            if r["ate"] is None:
                continue
            city_cell = city.upper() if i == 0 else ""
            ci_cell = f"[{r['if_ci_low']:+.3f}, {r['if_ci_high']:+.3f}]"
            label = r["label"].replace("$", r"\$")
            print(f"  {city_cell} & {label} & {r['n']:,} & "
                  f"${r['ate']:+.4f}$ & {ci_cell} \\\\")
        print(r"  \midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    all_results = {}
    for city in cities:
        all_results[city] = run_city(city)
    emit_latex_table(all_results)


if __name__ == "__main__":
    main()

"""Render the candidate-method coverage table and appendix figure from the
output of dk_inference_lab.py.

Usage:
  python3 dk_lab_report.py --raw results/simulation/davis_kahan_diag/dk_lab_raw.csv
  python3 dk_lab_report.py --raw results/simulation/davis_kahan_diag/dk_lab_raw.csv \
                           --lam_boot_raw /tmp/dk_lam_boot/davis_kahan_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


TAU_TRUE = 0.10


def _coverage_columns(df: pd.DataFrame) -> dict:
    """For each candidate method, build a coverage-by-spike series."""
    methods = {}
    for nm in ["classical", "hc1", "swm_v_off_only", "swm_fix", "swm_sweep",
               "conley_lehner", "fixed_b"]:
        col = f"se_{nm}"
        if col not in df.columns:
            continue
        se = df[col]
        lo = df["theta_rs"] - 1.96 * se
        hi = df["theta_rs"] + 1.96 * se
        methods[nm] = (lo, hi)
    t4 = float(stats.t.ppf(0.975, df=4))
    methods["bch_cluster"] = (df["theta_rs"] - t4 * df["se_bch"],
                              df["theta_rs"] + t4 * df["se_bch"])
    if "ci_sn_lo" in df.columns:
        methods["self_normalized"] = (df["ci_sn_lo"], df["ci_sn_hi"])
    if "ci_wcb_lo" in df.columns:
        methods["wcb"] = (df["ci_wcb_lo"], df["ci_wcb_hi"])
    if "ci_bb_lo" in df.columns:
        methods["block_boot"] = (df["ci_bb_lo"], df["ci_bb_hi"])
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("results/simulation/davis_kahan_diag/dk_lab_raw.csv"))
    ap.add_argument("--lam_boot_raw", type=Path, default=None,
                    help="Optional: davis_kahan_raw.csv from a run with --n_boot>0 "
                         "to splice in the Lam-2022 cheap-bootstrap coverage column.")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/simulation/davis_kahan_diag"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw)
    methods = _coverage_columns(df)

    lam_cov: dict[float, float] | None = None
    if args.lam_boot_raw is not None and args.lam_boot_raw.exists():
        lam = pd.read_csv(args.lam_boot_raw)
        if "ci_rs_boot_lo" in lam.columns:
            lam_cov = {}
            for lamval, sub in lam.groupby("spike_strength"):
                lo = sub["ci_rs_boot_lo"]; hi = sub["ci_rs_boot_hi"]
                lam_cov[float(lamval)] = float(
                    ((lo <= TAU_TRUE) & (TAU_TRUE <= hi)).mean()
                )

    rows = []
    for lamval, sub in df.groupby("spike_strength"):
        row = {"lambda": float(lamval), "n_reps": len(sub)}
        for nm, (lo_arr, hi_arr) in methods.items():
            mask = (df["spike_strength"] == lamval)
            lo_g = lo_arr[mask].to_numpy()
            hi_g = hi_arr[mask].to_numpy()
            row[f"cov_{nm}"] = float(((lo_g <= TAU_TRUE) & (TAU_TRUE <= hi_g)).mean())
        if lam_cov is not None and float(lamval) in lam_cov:
            row["cov_lam_iid_boot"] = lam_cov[float(lamval)]
        for nm, (lo_arr, hi_arr) in methods.items():
            mask = (df["spike_strength"] == lamval)
            lo_g = lo_arr[mask].to_numpy()
            hi_g = hi_arr[mask].to_numpy()
            row[f"cov0_{nm}"] = float(((lo_g <= 0.0) & (0.0 <= hi_g)).mean())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    table.to_csv(args.out_dir / "candidate_coverage_table.csv", index=False)

    keep_cov = [c for c in table.columns if c.startswith("cov_") and not c.startswith("cov0_")]
    summary = (
        table[keep_cov].mean().rename("mean_coverage_across_spikes").to_frame()
    )
    summary["mean_coverage_at_truth_0"] = (
        table[[c.replace("cov_", "cov0_") for c in keep_cov]].mean().to_numpy()
    )
    summary = summary.sort_values("mean_coverage_across_spikes", ascending=False)
    summary.to_csv(args.out_dir / "candidate_summary.csv")
    print("=== Coverage summary (averaged across 15 spikes) ===")
    print(summary.to_string())

    headline = "swm_v_off_only"
    rest = [m for m in summary.index.tolist() if m != f"cov_{headline}"]
    top = rest[:3]
    methods_to_plot = [f"cov_{headline}"] + top
    fig, ax = plt.subplots(figsize=(8, 4.8))
    color_cycle = plt.cm.tab10.colors
    for i, m in enumerate(methods_to_plot):
        nm = m.replace("cov_", "")
        ax.plot(table["lambda"], table[m], "-o",
                label=nm, color=color_cycle[i])
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="Nominal 0.95")
    ax.axhspan(0.93, 0.97, color="lightgreen", alpha=0.20)
    ax.set_xlabel(r"spike strength $\lambda$")
    ax.set_ylabel("empirical 95% CI coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("Candidate inference methods for residual-subspace DML\n"
                 f"(n={int(348)}, d={int(768)}, 500 reps/spike, Matern-2.5 GP rho=0.15)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_appendix_candidate_coverage.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'fig_appendix_candidate_coverage.png'}")
    print(f"Wrote {args.out_dir / 'candidate_coverage_table.csv'}")


if __name__ == "__main__":
    main()

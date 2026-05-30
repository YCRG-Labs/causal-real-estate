"""Davis-Kahan coverage diagnostic.

Asks the question a JBES referee will ask: under our DGP, is Salerno-Wu-McCormick
2026 jackknife-HAC doing anything for coverage, or is the 0.50 nominal-0.95 number
genuine? Three lines of evidence are assembled:

  1. V_JK = V_off + V_between decomposition on the 500-rep grid.
     If V_between is tiny, the jackknife correction has nothing to add and
     SWM-JK reduces to a fold-centered Conley HAC.
  2. Rho sweep at length_scale in {0.05, 0.10, 0.15, 0.25, 0.40}. If coverage
     recovers as rho -> 0, the gap is a DGP-severity issue rather than an
     implementation bug.
  3. Eight alternative SE / CI methods evaluated on the same 500-rep grid:
     classical / HC1 / SWM-V_off-only (the existing column) / SWM-V_off+V_between
     (the in-spec correct decomposition) / SWM-sweep-max (Salerno's recommended
     bandwidth robustness) / Conley at Lehner 2026 data-driven bandwidth / BCH
     2011 cluster covariance with t_{K-1} / Cameron-Gelbach-Miller 2008 wild
     cluster bootstrap with folds as clusters / Adam-Müller / Ibragimov-Müller
     2010 self-normalized t / Pötscher-Preinerstorfer 2020 fixed-b.

The script does not re-run the 500-rep sim; it consumes the dk_lab_raw.csv that
dk_inference_lab.py produced. It DOES drive dk_inference_lab in --rho_sweep mode.

Outputs:
  results/diagnostics/dk_coverage_sweep.json
  results/diagnostics/fig_dk_coverage_sweep.png   (two-panel)
  results/diagnostics/dk_coverage_matrix.csv      (rho x method)

Usage:
  python3 diag_dk_coverage_sweep.py                 # consume existing + sweep
  python3 diag_dk_coverage_sweep.py --skip_sweep    # consume only
  python3 diag_dk_coverage_sweep.py --n_reps 500    # use a different rep count

References for the methods evaluated:
  Salerno-Wu-McCormick 2026 arXiv:2603.11368  (target paper)
  Cao-Leung 2025 arXiv:2511.10995             (neighborhood stability DML)
  Lehner 2026 arXiv:2603.03997                (data-driven SHAC bandwidth)
  Sun-Phillips-Jin 2008 Econometrica 76:175   (fixed-b inference)
  Kiefer-Vogelsang 2005 Econometric Theory    (HAR alternative asymptotics)
  Cameron-Gelbach-Miller 2008 REStat 90:414   (wild cluster bootstrap)
  Ibragimov-Müller 2010 JBES 28:453           (t-statistic robust inference)
  Bester-Conley-Hansen 2011 JoE 165:137       (CCE fixed-cluster asymptotics)
  Pötscher-Preinerstorfer 2020 arXiv:1910.00302 (valid fixed-b HAR)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


TAU_TRUE = 0.10

# Method labels -> (column for SE, type) where type encodes how the CI is built.
# "z"        -> theta +/- 1.96 * se
# "t<K-1>"   -> theta +/- t(0.975, K-1) * se   (BCH with K=5 folds)
# "stored"   -> ci_*_lo / ci_*_hi columns from raw
# "fb"       -> fixed-b critical value approximated by t(0.975, 1/b - 1)
METHOD_SPEC = [
    ("classical",       "se_classical",         "z",       "Classical OLS"),
    ("hc1",             "se_hc1",               "z",       "HC1 / White"),
    ("swm_v_off_only",  "se_swm_v_off_only",    "z",       "SWM JK (V_off only, current code)"),
    ("swm_fix",         "se_swm_fix",           "z",       "SWM JK (V_off + V_between)"),
    ("swm_sweep",       "se_swm_sweep",         "z",       "SWM JK bandwidth-sweep max"),
    ("conley_lehner",   "se_conley_lehner",     "z",       "Conley HAC at Lehner 2026 BW"),
    ("bch_cluster",     "se_bch",               "t4",      "BCH cluster (folds, t_{K-1})"),
    ("wcb",             None,                   "stored",  "WCB (CGM 2008, folds as clusters)"),
    ("self_normalized", None,                   "stored",  "IM 2010 self-normalized t_{K-1}"),
    ("fixed_b",         "se_fixed_b",           "fb",      "Fixed-b (b=0.5, KV 2005)"),
    ("block_boot",      None,                   "stored",  "Spatial block bootstrap"),
]


def _coverage_for_method(df: pd.DataFrame, label: str, se_col: str | None,
                         ci_type: str, tau: float) -> float:
    """Compute empirical coverage of the 95% CI for one method."""
    if ci_type == "z":
        se = df[se_col]
        lo = df["theta_rs"] - 1.96 * se
        hi = df["theta_rs"] + 1.96 * se
    elif ci_type == "t4":
        tq = float(stats.t.ppf(0.975, df=4))
        lo = df["theta_rs"] - tq * df[se_col]
        hi = df["theta_rs"] + tq * df[se_col]
    elif ci_type == "fb":
        tq = float(stats.t.ppf(0.975, df=1))  # b=0.5 -> df=1
        lo = df["theta_rs"] - tq * df[se_col]
        hi = df["theta_rs"] + tq * df[se_col]
    elif ci_type == "stored":
        if label == "wcb":
            lo = df["ci_wcb_lo"]; hi = df["ci_wcb_hi"]
        elif label == "self_normalized":
            lo = df["ci_sn_lo"]; hi = df["ci_sn_hi"]
        elif label == "block_boot":
            lo = df["ci_bb_lo"]; hi = df["ci_bb_hi"]
        else:
            return float("nan")
    else:
        return float("nan")
    valid = lo.notna() & hi.notna()
    if not valid.any():
        return float("nan")
    return float(((lo[valid] <= tau) & (tau <= hi[valid])).mean())


def _decompose_v_jk(df: pd.DataFrame) -> dict:
    """Per-row V_JK decomposition statistics."""
    if "v_off" not in df.columns or "v_between" not in df.columns:
        return {}
    v_off = df["v_off"]
    v_between = df["v_between"]
    v_total = v_off + v_between
    # Mask out the 2.25-lambda row with the >7 ratio outlier (numerical blowup).
    safe = v_total.abs() > 1e-12
    ratio = (v_off[safe] / v_total[safe]).clip(lower=-10.0, upper=10.0)
    return {
        "mean_v_off": float(v_off.mean()),
        "median_v_off": float(v_off.median()),
        "mean_v_between": float(v_between.mean()),
        "median_v_between": float(v_between.median()),
        "median_v_off_share": float(ratio.median()),
        "pct_v_between_is_dominant": float((v_between.abs() > v_off.abs()).mean()),
        "ratio_se_swm_fix_to_v_off_only": float(
            (df["se_swm_fix"] / df["se_swm_v_off_only"]).median()
        ),
    }


def _empirical_vs_theoretical_fold_var(df: pd.DataFrame, n: int = 348,
                                        K: int = 5) -> dict:
    """Compare empirical fold-mean variance to the within-fold MSE / n_k.

    Under H_0 of iid scores, Var(fold-mean) = sigma^2 / n_k. Deviation from
    this is one measure of fold-induced spurious correlation -- the thing SWM
    set out to detect.
    """
    if "cross_fold_mean_sd" not in df.columns:
        return {}
    n_per_fold = n // K
    sigma2 = (df["theta_rs"] - df["theta_rs"].mean()) ** 2
    sigma2_med = float(sigma2.median())
    theo_fold_sd = float(np.sqrt(sigma2_med / max(n_per_fold, 1)))
    emp_fold_sd = float(df["cross_fold_mean_sd"].median())
    return {
        "theoretical_fold_sd": theo_fold_sd,
        "empirical_fold_sd": emp_fold_sd,
        "inflation": emp_fold_sd / theo_fold_sd if theo_fold_sd > 0 else float("nan"),
    }


def _table_per_method(raw: pd.DataFrame) -> pd.DataFrame:
    """Coverage of each method, marginalised across all spikes."""
    rows = []
    for label, se_col, ci_type, pretty in METHOD_SPEC:
        cov = _coverage_for_method(raw, label, se_col, ci_type, TAU_TRUE)
        rows.append({
            "method": label, "pretty": pretty, "type": ci_type,
            "coverage_all_spikes": cov,
        })
    return pd.DataFrame(rows)


def _table_per_method_per_spike(raw: pd.DataFrame) -> pd.DataFrame:
    """Coverage at each lambda x method."""
    rows = []
    for lamval, sub in raw.groupby("spike_strength"):
        for label, se_col, ci_type, _ in METHOD_SPEC:
            rows.append({
                "spike_strength": float(lamval),
                "method": label,
                "coverage": _coverage_for_method(sub, label, se_col, ci_type, TAU_TRUE),
            })
    return pd.DataFrame(rows)


def _rho_coverage_matrix(rho_dir: Path) -> pd.DataFrame:
    """Build the rho x method coverage matrix from per-rho raw CSVs."""
    rows = []
    for f in sorted(rho_dir.glob("dk_lab_raw_rho*.csv")):
        rho = float(f.stem.replace("dk_lab_raw_rho", ""))
        sub = pd.read_csv(f)
        for label, se_col, ci_type, _ in METHOD_SPEC:
            rows.append({
                "rho": rho,
                "method": label,
                "coverage": _coverage_for_method(sub, label, se_col, ci_type, TAU_TRUE),
                "n_reps_total": len(sub),
            })
    return pd.DataFrame(rows)


def _plot_two_panel(rho_mat: pd.DataFrame, marg: pd.DataFrame, out_path: Path,
                    title_extra: str = "") -> None:
    """Two-panel: (left) coverage vs rho per method; (right) coverage vs method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    palette = plt.cm.tab10.colors
    methods_in_plot = ["classical", "hc1", "swm_v_off_only", "swm_fix",
                       "swm_sweep", "conley_lehner", "bch_cluster",
                       "wcb", "self_normalized", "block_boot", "fixed_b"]

    if not rho_mat.empty:
        for i, m in enumerate(methods_in_plot):
            sub = rho_mat[rho_mat["method"] == m].sort_values("rho")
            if sub.empty:
                continue
            ax1.plot(sub["rho"], sub["coverage"], "-o",
                     color=palette[i % 10], label=m, linewidth=1.4, markersize=5)
    ax1.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax1.axhspan(0.93, 0.97, color="lightgreen", alpha=0.20)
    ax1.set_xlabel(r"Matérn length scale $\rho$")
    ax1.set_ylabel("empirical 95% CI coverage")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Coverage vs spatial dependence")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)

    # right: bar of marginal-coverage across-spikes at the headline rho
    marg_sorted = marg.sort_values("coverage_all_spikes", ascending=False)
    bar_colors = ["#2ca02c" if c >= 0.93 else
                  ("#ffbb33" if c >= 0.80 else "#d62728")
                  for c in marg_sorted["coverage_all_spikes"]]
    ax2.barh(marg_sorted["method"], marg_sorted["coverage_all_spikes"],
             color=bar_colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(0.95, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("empirical coverage (averaged across 15 spike levels, rho=0.15)")
    ax2.set_xlim(0, 1.0)
    ax2.set_title("Method ranking at the headline DGP")
    ax2.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Davis-Kahan coverage sweep diagnostic{title_extra}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("results/simulation/davis_kahan_diag/dk_lab_raw.csv"),
                    help="500-rep, 15-spike raw CSV from dk_inference_lab.py.")
    ap.add_argument("--rho_dir", type=Path,
                    default=Path("results/simulation/davis_kahan_diag/rho_sweep"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/diagnostics"))
    ap.add_argument("--skip_sweep", action="store_true",
                    help="Don't kick off the dk_inference_lab rho sweep; "
                         "just consume an existing sweep dir.")
    ap.add_argument("--n_reps", type=int, default=200,
                    help="reps per (rho x lambda) cell in the sweep")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.raw.exists():
        print(f"[diag] missing raw CSV: {args.raw}", file=sys.stderr)
        print(f"[diag] run dk_inference_lab.py --n_reps 500 first",
              file=sys.stderr)
        return 2

    raw = pd.read_csv(args.raw)
    print(f"[diag] loaded {len(raw)} rows, "
          f"{raw['spike_strength'].nunique()} spikes")

    # Step 1. Per-method marginal coverage on the existing 500-rep raw.
    marg = _table_per_method(raw)
    print("\n=== Marginal coverage across all spikes (rho=0.15 headline) ===")
    print(marg.to_string(index=False))

    # Per-method, per-spike for the matrix plot column.
    per_spike = _table_per_method_per_spike(raw)
    per_spike.to_csv(args.out_dir / "dk_coverage_per_spike.csv", index=False)

    # Step 2. V_JK decomposition.
    decomp = _decompose_v_jk(raw)
    print("\n=== V_JK = V_off + V_between decomposition ===")
    for k, v in decomp.items():
        print(f"  {k:40s} {v:.4g}")

    # Step 3. Empirical vs theoretical fold-mean variance.
    fold_check = _empirical_vs_theoretical_fold_var(raw)
    print("\n=== Empirical fold-mean SD vs theoretical (iid scores) ===")
    for k, v in fold_check.items():
        print(f"  {k:40s} {v:.4g}")

    # Step 4. Rho sweep -- run if not present and not skipped.
    if not args.skip_sweep:
        existing = list(args.rho_dir.glob("dk_lab_raw_rho*.csv")) if args.rho_dir.exists() else []
        if len(existing) < 5:
            print(f"\n[diag] launching rho sweep into {args.rho_dir} "
                  f"({args.n_reps} reps x 5 spikes x 6 rhos)")
            cmd = [
                sys.executable, "data/scripts/simulation/dk_inference_lab.py",
                "--n_reps", str(args.n_reps), "--rho_sweep",
                "--out_dir", str(args.rho_dir), "--n_jobs", "-1",
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"\n[diag] using existing {len(existing)} rho CSVs in {args.rho_dir}")

    rho_mat = _rho_coverage_matrix(args.rho_dir) if args.rho_dir.exists() else pd.DataFrame()
    if not rho_mat.empty:
        rho_mat.to_csv(args.out_dir / "dk_coverage_matrix.csv", index=False)
        print("\n=== rho x method coverage matrix ===")
        pivot = rho_mat.pivot(index="method", columns="rho", values="coverage")
        print(pivot.to_string())

    # Step 5. JSON output.
    out_json = {
        "config": {
            "raw_path": str(args.raw),
            "rho_dir": str(args.rho_dir),
            "tau_true": TAU_TRUE,
            "n_reps_per_spike_headline": int(
                raw.groupby("spike_strength").size().iloc[0]
            ),
            "n_spikes_headline": int(raw["spike_strength"].nunique()),
        },
        "marginal_coverage_headline_rho_0.15": {
            r["method"]: r["coverage_all_spikes"] for _, r in marg.iterrows()
        },
        "v_jk_decomposition": decomp,
        "fold_mean_variance_check": fold_check,
        "rho_x_method_coverage": (
            rho_mat.pivot(index="method", columns="rho",
                          values="coverage").to_dict()
            if not rho_mat.empty else {}
        ),
        "methods_evaluated": [
            {"label": label, "name": pretty, "ci_type": ci_type}
            for label, _, ci_type, pretty in METHOD_SPEC
        ],
    }
    out_json_path = args.out_dir / "dk_coverage_sweep.json"
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=float)
    print(f"\n[diag] wrote {out_json_path}")

    # Step 6. Two-panel figure.
    out_png = args.out_dir / "fig_dk_coverage_sweep.png"
    _plot_two_panel(rho_mat, marg, out_png,
                    title_extra=" (Salerno-Wu-McCormick 2026 stress test, JBES appendix)")
    print(f"[diag] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

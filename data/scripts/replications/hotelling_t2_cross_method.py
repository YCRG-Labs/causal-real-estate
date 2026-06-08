"""Hotelling T² joint null test per market across Shen, Baur, CF.

For each metropolitan market m we have a 3-vector of per-σ DML
estimates from three independent identification strategies (Shen-Ross
uniqueness, pooled-PCA Baur sentence-BERT axis, counterfactual rewriting
total effect), each with its own bootstrap standard error.  We test
the joint null  H_0: (θ_Shen_m, θ_Baur_m, θ_CF_m) = (0, 0, 0)
against the two-sided alternative.  The three estimates are NOT
independent: they share the same listings, outcome, and confounders, and
three correlated text representations.  Hence the naive statistic
T²_indep = Σ_j (θ̂_j / SE_j)² referenced to χ²(3) is anti-conservative
(it over-rejects); its true null law is Σ_j λ_j χ²(1) with λ_j the
eigenvalues of the 3×3 cross-method correlation Σ, and χ²(3) only obtains
when Σ = I.

FIX: we estimate the 3×3 cross-method correlation R of the per-method
z-statistics — the across-city sample correlation of the z's, used as a
proxy for the within-market sampling correlation because no per-city
joint bootstrap draws are available here — and report the Mahalanobis
quadratic form T²_corr = z' R⁻¹ z, which is χ²(3) under H₀ when
z ~ N(0, R).  The naive χ²(3) column is retained for comparison but
explicitly flagged as anti-conservative, and we report the effective
number of independent method dimensions d_eff = tr(R)² / tr(R²).

Output: results/replications/hotelling_t2_cross_method.csv with one row
per market (naive and correlation-corrected T²/p plus BH-FDR q-values
across the 12 markets), and a companion *_meta.json holding R, its
eigenvalues, d_eff, and the cross-method dependence caveat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shen",
                    default=str(REPO / "results" / "replications"
                                  / "shen_12city_table.csv"))
    ap.add_argument("--baur_pooled",
                    default=str(REPO / "results" / "replications"
                                  / "baur_pooled_pca"
                                  / "baur_pooled_pca_table.csv"))
    ap.add_argument("--cf",
                    default=str(REPO / "results" / "counterfactual"
                                  / "counterfactual_12city_table.csv"))
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications"
                                  / "hotelling_t2_cross_method.csv"))
    args = ap.parse_args()

    shen = pd.read_csv(args.shen)[["city", "dml_theta", "dml_se"]].rename(
        columns={"dml_theta": "shen_theta", "dml_se": "shen_se"})
    baur = pd.read_csv(args.baur_pooled)[["city", "dml_theta", "dml_se"]].rename(
        columns={"dml_theta": "baur_theta", "dml_se": "baur_se"})
    cf = pd.read_csv(args.cf)[["city", "te_mean", "te_ci_low", "te_ci_high"]].rename(
        columns={"te_mean": "cf_theta"})
    cf["cf_se"] = (cf["te_ci_high"] - cf["te_ci_low"]) / (2 * 1.96)

    df = shen.merge(baur, on="city").merge(
        cf[["city", "cf_theta", "cf_se"]], on="city")

    df["shen_z"] = np.where(df["shen_se"] > 0,
                            df["shen_theta"] / df["shen_se"], 0.0)
    df["baur_z"] = np.where(df["baur_se"] > 0,
                            df["baur_theta"] / df["baur_se"], 0.0)
    df["cf_z"] = np.where(df["cf_se"] > 0,
                          df["cf_theta"] / df["cf_se"], 0.0)

    Zmat = df[["shen_z", "baur_z", "cf_z"]].to_numpy()
    R = np.corrcoef(Zmat, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    Rinv = np.linalg.pinv(R)
    evals = np.clip(np.linalg.eigvalsh(R), 0.0, None)
    d_eff = float(evals.sum() ** 2 / max(float(np.sum(evals ** 2)), 1e-12))

    rows = []
    for _, r in df.iterrows():
        zv = np.array([r["shen_z"], r["baur_z"], r["cf_z"]], dtype=float)
        T2_indep = float(zv @ zv)
        p_indep = float(1 - stats.chi2.cdf(T2_indep, df=3))
        T2_corr = float(zv @ Rinv @ zv)
        p_corr = float(1 - stats.chi2.cdf(T2_corr, df=3))
        rows.append({
            "city": r["city"],
            "shen_theta": r["shen_theta"], "shen_z": r["shen_z"],
            "baur_theta": r["baur_theta"], "baur_z": r["baur_z"],
            "cf_theta": r["cf_theta"], "cf_z": r["cf_z"],
            "hotelling_T2": T2_indep,
            "p_chi2_3df_indep": p_indep,
            "T2_mahalanobis": T2_corr,
            "p_chi2_3df_corr": p_corr,
        })

    out = pd.DataFrame(rows)
    out["bh_q_corr"] = bh_qvalues(out["p_chi2_3df_corr"].to_numpy())
    out["bh_q_indep"] = bh_qvalues(out["p_chi2_3df_indep"].to_numpy())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    caveat = (
        "The naive p_chi2_3df_indep (= Sum_j z_j^2 ~ chi^2_3) assumes the "
        "three methods' z's are independent. They are not (shared listings, "
        "outcome, confounders, and correlated text representations), so the "
        "naive reference is anti-conservative and OVER-rejects. Primary "
        "inference should use p_chi2_3df_corr (Mahalanobis z' R^-1 z ~ "
        "chi^2_3). Caveat: R is estimated from only k cities, and the "
        "across-city z-correlation is a proxy for the within-market sampling "
        "correlation; co-movement of the true effects across markets can bias "
        "it. If per-city joint bootstrap draws become available, estimate R "
        "from them instead."
    )
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta = {
        "k": int(len(out)),
        "R_methods_order": ["shen", "baur", "cf"],
        "cross_method_correlation_R": R.tolist(),
        "R_eigenvalues": evals.tolist(),
        "effective_independent_dimensions": d_eff,
        "reference_distribution_naive": "chi2_3 (anti-conservative)",
        "reference_distribution_corrected": "chi2_3 via Mahalanobis z' R^-1 z",
        "caveat": caveat,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    print(f"\n=== Hotelling T² joint-null cross-method test (k={len(out)}) ===")
    print(out.sort_values("p_chi2_3df_corr").to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n  cross-method z-correlation R (shen, baur, cf):")
    for nm, row in zip(["shen", "baur", "cf"], R):
        print(f"    {nm:>5}: " + " ".join(f"{c:+.3f}" for c in row))
    print(f"  R eigenvalues: " + ", ".join(f"{e:.3f}" for e in evals))
    print(f"  effective independent method dimensions "
          f"d_eff = tr(R)^2/tr(R^2) = {d_eff:.3f} of 3")
    print(f"\n  markets rejecting joint null at corrected p<0.05: "
          f"{(out['p_chi2_3df_corr'] < 0.05).sum()} / {len(out)} "
          f"(naive p<0.05: {(out['p_chi2_3df_indep'] < 0.05).sum()})")
    print(f"  markets rejecting joint null at corrected BH q<0.05: "
          f"{(out['bh_q_corr'] < 0.05).sum()} / {len(out)}")
    print(f"  markets rejecting joint null at corrected BH q<0.10: "
          f"{(out['bh_q_corr'] < 0.10).sum()} / {len(out)}")
    print(f"\n  CAVEAT: {caveat}")
    print(f"\nCSV  -> {out_path}")
    print(f"META -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

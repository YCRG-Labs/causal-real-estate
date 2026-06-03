"""Hotelling T² joint null test per market across Shen, Baur, CF.

For each metropolitan market m we have a 3-vector of per-σ DML
estimates from three independent identification strategies (Shen-Ross
uniqueness, pooled-PCA Baur sentence-BERT axis, counterfactual rewriting
total effect), each with its own bootstrap standard error.  We test
the joint null  H_0: (θ_Shen_m, θ_Baur_m, θ_CF_m) = (0, 0, 0)
against the two-sided alternative, treating the three estimates as
asymptotically independent (off-diagonal sampling covariances are
zero because the three methods use disjoint sources of variation:
Shen uses Doc2Vec local uniqueness, Baur uses the sentence-BERT pooled
PC1 score, CF uses LLM-rewriting differences).

Under the joint null and the independence assumption the Hotelling-
style statistic T² = Σ_j (θ̂_j / SE_j)² is distributed as χ²(3) and
the per-market p-value is 1 - F_{χ²(3)}(T²).  We also report the
3-vector and per-method z-statistics for diagnostic.

Output: results/replications/hotelling_t2_cross_method.csv with one row
per market, the T² statistic, the asymptotic p-value, and a BH-FDR
q-value across the 12 markets.
"""
from __future__ import annotations

import argparse
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

    rows = []
    for _, r in df.iterrows():
        z_shen = r["shen_theta"] / r["shen_se"] if r["shen_se"] > 0 else 0
        z_baur = r["baur_theta"] / r["baur_se"] if r["baur_se"] > 0 else 0
        z_cf = r["cf_theta"] / r["cf_se"] if r["cf_se"] > 0 else 0
        T2 = z_shen ** 2 + z_baur ** 2 + z_cf ** 2
        p = float(1 - stats.chi2.cdf(T2, df=3))
        rows.append({
            "city": r["city"],
            "shen_theta": r["shen_theta"], "shen_z": z_shen,
            "baur_theta": r["baur_theta"], "baur_z": z_baur,
            "cf_theta": r["cf_theta"], "cf_z": z_cf,
            "hotelling_T2": T2,
            "p_chi2_3df": p,
        })

    out = pd.DataFrame(rows)
    out["bh_q"] = bh_qvalues(out["p_chi2_3df"].to_numpy())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n=== Hotelling T² joint-null cross-method test (k={len(out)}) ===")
    print(out.sort_values("p_chi2_3df").to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n  markets rejecting joint null at raw p<0.05: "
          f"{(out['p_chi2_3df'] < 0.05).sum()} / {len(out)}")
    print(f"  markets rejecting joint null at BH q<0.05: "
          f"{(out['bh_q'] < 0.05).sum()} / {len(out)}")
    print(f"  markets rejecting joint null at BH q<0.10: "
          f"{(out['bh_q'] < 0.10).sum()} / {len(out)}")
    print(f"\nCSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

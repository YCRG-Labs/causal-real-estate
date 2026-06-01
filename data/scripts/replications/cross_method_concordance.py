"""Cross-method per-city concordance: Shen θ, Baur θ, counterfactual TE.

For each of the 12 metros, join the three estimators that target the
text-as-spatial-proxy mechanism on different representations:
  - Shen 2021 Doc2Vec uniqueness DML (scalar treatment)
  - Baur 2023 BERT-PC1 DML (768-dim → PC1)
  - Counterfactual rewrite TE on the BERT-PC1 DML predictor

Report Spearman + Kendall rank correlations across the 12 cities. Report
sign-concordance (in how many cities do two methods agree on θ > 0?). For
the bootstrap-normalized CIs, also test whether the methods agree on
which cities exclude zero.

The interpretation is descriptive: with k=12 cities the rank statistic
has standard error ~1/sqrt(12) ≈ 0.29, so we cannot formally reject the
null of independence at conventional α even with perfect agreement
(r=1 → z=2.04). The point is to surface the heterogeneity pattern, not
to test it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "replications"
CF_RESULTS = REPO / "results" / "counterfactual"

CITY_ORDER = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
              "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
DISPLAY = {"boston": "Boston", "nyc": "New York", "sf": "San Francisco",
           "dc": "Washington DC", "philadelphia": "Philadelphia",
           "chicago": "Chicago", "seattle": "Seattle", "denver": "Denver",
           "atlanta": "Atlanta", "portland": "Portland", "phoenix": "Phoenix",
           "dallas": "Dallas"}


def _load_shen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index("city")[["dml_theta", "dml_se", "dml_ci_low",
                                 "dml_ci_high", "dml_excludes_zero"]]


def _load_baur(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index("city")[["dml_theta", "dml_se", "dml_ci_low",
                                 "dml_ci_high", "dml_excludes_zero"]]


def _load_cf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index("city")
    df["te_excludes_zero"] = (df["te_ci_low"] > 0) | (df["te_ci_high"] < 0)
    return df[["te_mean", "te_ci_low", "te_ci_high", "te_excludes_zero",
               "dml_theta", "dml_se"]]


def concordance_correlation(x, y):
    """Lin's concordance correlation coefficient.

    Combines Pearson correlation with bias and scale adjustments. Penalizes
    departures from the y=x identity line. Standard in agreement-statistics
    literature; preferred over plain Pearson when units are commensurate.
    """
    x = np.asarray(x); y = np.asarray(y)
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=0), y.var(ddof=0)
    cov = np.mean((x - mx) * (y - my))
    return float(2 * cov / (sx2 + sy2 + (mx - my) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shen", default=str(RESULTS / "shen_12city_table.csv"))
    ap.add_argument("--baur", default=str(RESULTS / "baur_12city_table.csv"))
    ap.add_argument("--cf",   default=str(CF_RESULTS / "counterfactual_12city_table.csv"))
    args = ap.parse_args()

    shen = _load_shen(Path(args.shen)).rename(columns=lambda c: f"shen_{c}")
    baur = _load_baur(Path(args.baur)).rename(columns=lambda c: f"baur_{c}")
    cf = _load_cf(Path(args.cf)).rename(columns={
        "te_mean": "cf_te", "te_ci_low": "cf_te_lo",
        "te_ci_high": "cf_te_hi", "te_excludes_zero": "cf_te_excludes_zero",
        "dml_theta": "cf_dml_theta", "dml_se": "cf_dml_se",
    })

    df = shen.join(baur, how="inner").join(cf, how="inner")
    df = df.reindex([c for c in CITY_ORDER if c in df.index])
    df["city_display"] = [DISPLAY[c] for c in df.index]

    methods = ["shen_dml_theta", "baur_dml_theta", "cf_te"]
    method_labels = ["Shen DML θ", "Baur DML θ", "CF TE"]

    print(f"\n{'city':<14}  {'Shen θ':>10}{'(se)':>10}{'≠0':>4}  "
          f"{'Baur θ':>10}{'(se)':>10}{'≠0':>4}  "
          f"{'CF TE':>10}{'(95% CI half)':>18}{'≠0':>4}")
    print("-" * 110)
    for c, r in df.iterrows():
        sh_ex = "Y" if r["shen_dml_excludes_zero"] else "n"
        ba_ex = "Y" if r["baur_dml_excludes_zero"] else "n"
        cf_ex = "Y" if r["cf_te_excludes_zero"] else "n"
        cf_half = (r["cf_te_hi"] - r["cf_te_lo"]) / 2
        print(f"{r['city_display']:<14}  "
              f"{r['shen_dml_theta']:>+10.4f}{r['shen_dml_se']:>10.4f}{sh_ex:>4}  "
              f"{r['baur_dml_theta']:>+10.4f}{r['baur_dml_se']:>10.4f}{ba_ex:>4}  "
              f"{r['cf_te']:>+10.4f}{cf_half:>18.4f}{cf_ex:>4}")

    print("\n" + "=" * 110)
    print("PAIRWISE CONCORDANCE STATISTICS (k=12 cities)")
    print("=" * 110)
    print(f"{'pair':<28}  {'Spearman ρ':>12}{'p':>8}  "
          f"{'Kendall τ':>12}{'p':>8}  {'Pearson r':>12}{'p':>8}  {'Lin CCC':>10}")
    print("-" * 110)
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if j <= i:
                continue
            xi, xj = df[mi].values, df[mj].values
            sp = stats.spearmanr(xi, xj)
            kt = stats.kendalltau(xi, xj)
            pe = stats.pearsonr(xi, xj)
            ccc = concordance_correlation(xi, xj)
            label = f"{method_labels[i]} vs {method_labels[j]}"
            print(f"{label:<28}  "
                  f"{sp.statistic:>+12.3f}{sp.pvalue:>8.3f}  "
                  f"{kt.statistic:>+12.3f}{kt.pvalue:>8.3f}  "
                  f"{pe.statistic:>+12.3f}{pe.pvalue:>8.3f}  "
                  f"{ccc:>+10.3f}")

    print("\n" + "=" * 110)
    print("SIGN-AGREEMENT MATRIX (counts of cities where two methods agree on sign of θ)")
    print("=" * 110)
    print(f"{'pair':<28}  {'agree':>8}  {'agree>0':>10}  {'agree<0':>10}")
    print("-" * 110)
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if j <= i:
                continue
            xi, xj = df[mi].values, df[mj].values
            both_pos = int(((xi > 0) & (xj > 0)).sum())
            both_neg = int(((xi < 0) & (xj < 0)).sum())
            agree = both_pos + both_neg
            label = f"{method_labels[i]} vs {method_labels[j]}"
            print(f"{label:<28}  {agree:>5}/12  {both_pos:>5}/12{' ':>4}  {both_neg:>5}/12")

    print("\n" + "=" * 110)
    print("EXCLUDES-ZERO AGREEMENT (cities where each method's CI excludes 0)")
    print("=" * 110)
    shen_ex = df["shen_dml_excludes_zero"].astype(bool)
    baur_ex = df["baur_dml_excludes_zero"].astype(bool)
    cf_ex = df["cf_te_excludes_zero"].astype(bool)
    print(f"  Shen DML excludes 0: {int(shen_ex.sum())}/12  "
          f"cities: {df.index[shen_ex].tolist()}")
    print(f"  Baur DML excludes 0: {int(baur_ex.sum())}/12  "
          f"cities: {df.index[baur_ex].tolist()}")
    print(f"  CF TE   excludes 0: {int(cf_ex.sum())}/12  "
          f"cities: {df.index[cf_ex].tolist()}")
    all_three = shen_ex & baur_ex & cf_ex
    any_two = ((shen_ex.astype(int) + baur_ex.astype(int) + cf_ex.astype(int)) >= 2)
    none_three = ~(shen_ex | baur_ex | cf_ex)
    print(f"  Identified by all three: {int(all_three.sum())}/12  "
          f"({df.index[all_three].tolist()})")
    print(f"  Identified by ≥ two:     {int(any_two.sum())}/12  "
          f"({df.index[any_two].tolist()})")
    print(f"  Identified by none:      {int(none_three.sum())}/12  "
          f"({df.index[none_three].tolist()})")

    out_csv = RESULTS / "cross_method_concordance.csv"
    df.to_csv(out_csv)
    print(f"\nCSV -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

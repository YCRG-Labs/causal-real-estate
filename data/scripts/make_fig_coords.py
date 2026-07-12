"""Emit pgfplots coordinate blocks for the v3 figures, straight from results/.

Run: python3 data/scripts/make_fig_coords.py
Prints, per figure, the exact `(x,y)` / `(x,y) +- (e,0)` lines to paste into the
causalre TikZ blocks, so the figures carry the canonical numbers with no manual
transcription.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results"
z = 1.959963985

CITY = {"boston":"Boston","nyc":"New York","sf":"San Francisco","dc":"Washington",
        "philadelphia":"Philadelphia","chicago":"Chicago","seattle":"Seattle",
        "denver":"Denver","atlanta":"Atlanta","portland":"Portland",
        "phoenix":"Phoenix","dallas":"Dallas"}


def coverage():
    print("\n===== FIG 1: COVERAGE (eta, coverage) per estimator x N =====")
    df = pd.read_csv(f"{R}/simulation_baseline/coverage_table.csv")
    eta = {"scm0":0.0, "scm1_0.01":0.01, "scm1_0.05":0.05, "scm1_0.10":0.10}
    df["eta"] = df["dgp"].map(eta)
    for N in (500, 2000):
        for est in ("DML", "DML_cfpca", "DR"):
            sub = df[(df.N==N)&(df.estimator==est)].sort_values("eta")
            pts = " ".join(f"({r.eta:.2f},{r.coverage:.3f})" for r in sub.itertuples())
            print(f"  N={N:<4} {est:<10} {pts}")


def forest():
    print("\n===== FIG 2: FOREST (theta, idx) +- (halfwidth,0), idx 12=top =====")
    shen = pd.read_csv(f"{R}/replications/shen_12city_table.csv").set_index("city")
    order = shen["dml_theta"].sort_values(ascending=True).index.tolist()
    print("  CITY ORDER (idx 1..12, bottom->top):", [CITY[c] for c in order])
    print("  -- Shen --")
    for i,c in enumerate(order,1):
        r=shen.loc[c]; hw=(r.dml_ci_high-r.dml_ci_low)/2
        print(f"    ({r.dml_theta:+.3f},{i}) +- ({hw:.3f},0)  % {CITY[c]}")
    baur = pd.read_csv(f"{R}/replications/baur_pooled_pca/baur_pooled_pca_table.csv").set_index("city")
    print("  -- Baur (oriented) --")
    for i,c in enumerate(order,1):
        r=baur.loc[c]; hw=(r.dml_ci_high-r.dml_ci_low)/2
        print(f"    ({r.dml_theta:+.3f},{i}) +- ({hw:.3f},0)  % {CITY[c]}")
    cf = pd.read_csv(f"{R}/counterfactual/counterfactual_12city_table.csv").set_index("city")
    print("  -- CF style-stripped (theta-SE folded) --")
    for i,c in enumerate(order,1):
        r=cf.loc[c]
        boot_se=(r.nde_ci_high-r.nde_ci_low)/(2*z)
        cse=np.sqrt(boot_se**2 + (r.nde_mean**2)*(r.dml_se/r.dml_theta)**2)
        print(f"    ({r.nde_mean:+.3f},{i}) +- ({z*cse:.3f},0)  % {CITY[c]}")


def erasure():
    print("\n===== FIG 5: ERASURE (city, R2) raw vs erased, ybar =====")
    df = pd.read_csv(f"{R}/leace12_igbp/leace_12city_table.csv").set_index("city")
    order = df["ridge_r2_raw"].sort_values(ascending=False).index.tolist()
    raw = " ".join(f"({CITY[c]},{df.loc[c].ridge_r2_raw:.3f})" for c in order)
    era = " ".join(f"({CITY[c]},{max(df.loc[c].ridge_r2_erased,0):.3f})" for c in order)
    print("  symbolic x order:", [CITY[c] for c in order])
    print("  raw   :", raw)
    print("  erased:", era)


def sensitivity():
    print("\n===== FIG 4: SENSITIVITY robustness value per city (rv, idx) =====")
    df = pd.read_csv(f"{R}/replications/confounder_sensitivity_12.csv")
    none = df[df.block_dropped=="none"].drop_duplicates("city").set_index("city")
    order = none["rv"].sort_values(ascending=True).index.tolist()
    print("  CITY ORDER (idx 1..12):", [CITY[c] for c in order])
    for i,c in enumerate(order,1):
        print(f"    ({none.loc[c].rv:.3f},{i})  % {CITY[c]}  (theta={none.loc[c].theta:+.3f})")


def concordance():
    print("\n===== SUPP: CONCORDANCE (shen_theta, baur_theta) =====")
    df = pd.read_csv(f"{R}/replications/cross_method_concordance.csv").set_index("city")
    pts = " ".join(f"({df.loc[c].shen_dml_theta:+.3f},{df.loc[c].baur_dml_theta:+.3f})"
                   for c in df.index)
    print("  pts:", pts)
    print("  pearson r =", round(np.corrcoef(df.shen_dml_theta, df.baur_dml_theta)[0,1],3))


if __name__ == "__main__":
    coverage(); forest(); erasure(); sensitivity(); concordance()

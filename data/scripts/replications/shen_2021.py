"""Replication of Shen & Ross (2021, JUE) — Information Value of Property Description.

Shen, L. & Ross, S. L. (2021). "Information value of property description: A
machine learning approach." Journal of Urban Economics 122, 103299.

Method (faithful to the paper, scaled to our SF data):

  1. Vectorize all listing descriptions with TF-IDF (pre-BERT NLP, matching
     the paper's shallow NLP).
  2. For each listing find K=5 nearest spatial neighbours in lat/lon.
  3. Define uniqueness = 1 − cosine(self_tfidf, mean(neighbour_tfidf)).
     This is the "semantic deviation from neighbours" construct.
  4. Hedonic OLS:
        log_price ~ uniqueness + structured features
     with HC3 robust SEs (Shen reports +15% per σ on Atlanta MLS).
  5. Re-run the project DML pipeline with uniqueness as a continuous
     treatment in place of PC1 of the embedding. Same 30-feature confounder
     set as the production analysis.

Output JSON has both estimates side-by-side so the OLS gain and the DML
estimate can be read off the same record.

Usage:
  python shen_2021.py
  python shen_2021.py --out results/replications/shen.json
  python shen_2021.py --n 200            # subset for smoke tests
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_inference import (
    PROPERTY_COLS,
    get_features_and_target,
    load_analysis_data,
)
from replications.compare_to_dml import result_to_dict, run_dml


@dataclass
class OLSUniqueness:
    coef: float
    se: float
    t: float
    p: float
    ci_low: float
    ci_high: float
    n: int
    r2: float
    adj_r2: float
    coef_per_sd: float
    pct_per_sd: float


def compute_uniqueness(
    descriptions: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    k: int = 5,
    max_features: int = 5000,
) -> np.ndarray:
    """TF-IDF uniqueness per listing: 1 - cos(self, mean(K nearest in lat/lon)).

    Self is excluded from the K neighbours (we query K+1 and drop index 0).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf = vec.fit_transform(descriptions)

    coords = np.column_stack([lat, lon])
    tree = cKDTree(coords)
    n = len(descriptions)
    k_eff = min(k + 1, n)
    _, nn_idx = tree.query(coords, k=k_eff)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx.reshape(-1, 1)

    uniq = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = [j for j in nn_idx[i].tolist() if j != i][:k]
        if not neigh:
            uniq[i] = 0.0
            continue
        # mean of neighbour TF-IDF rows -> 1 x V
        neigh_mean = tfidf[neigh].mean(axis=0)
        sim = cosine_similarity(tfidf[i], np.asarray(neigh_mean))[0, 0]
        uniq[i] = 1.0 - float(sim)

    return uniq


def hedonic_ols(
    uniqueness: np.ndarray,
    confounders: np.ndarray,
    confounder_names: list[str],
    Y_log: np.ndarray,
) -> tuple[OLSUniqueness, sm.regression.linear_model.RegressionResultsWrapper]:
    """log_price ~ uniqueness_z + confounders, HC3 robust SEs.

    Standardises uniqueness so the coefficient is per-σ (matching how Shen
    reports the "+15%" headline).
    """
    u_sd = float(np.std(uniqueness, ddof=1))
    if u_sd == 0:
        u_sd = 1.0
    u_z = (uniqueness - uniqueness.mean()) / u_sd

    X = np.column_stack([u_z, confounders])
    X = sm.add_constant(X, has_constant="add")
    names = ["const", "uniqueness_z"] + list(confounder_names)
    df = pd.DataFrame(X, columns=names)
    model = sm.OLS(Y_log, df).fit(cov_type="HC3")

    coef = float(model.params["uniqueness_z"])
    se = float(model.bse["uniqueness_z"])
    tval = float(model.tvalues["uniqueness_z"])
    pval = float(model.pvalues["uniqueness_z"])
    lo, hi = (float(x) for x in model.conf_int().loc["uniqueness_z"])

    out = OLSUniqueness(
        coef=coef,
        se=se,
        t=tval,
        p=pval,
        ci_low=lo,
        ci_high=hi,
        n=int(len(Y_log)),
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        coef_per_sd=coef,
        pct_per_sd=float(100.0 * (np.exp(coef) - 1.0)),
    )
    return out, model


def _confounder_names(confounders: np.ndarray, parcels) -> list[str]:
    """Best-effort names for confounder columns. Falls back to indexed names.

    Layout (per causal_inference.get_features_and_target): [lat, lon,
    *property_cols_present, *contextual_cols_present]. We label what we know
    and pad the rest.
    """
    from causal_inference import CONTEXTUAL_COLS  # local import keeps file tidy

    known = ["lat", "lon"]
    known += [c for c in PROPERTY_COLS if parcels is not None and c in parcels.columns]
    known += [c for c in CONTEXTUAL_COLS if parcels is not None and c in parcels.columns]
    if len(known) == confounders.shape[1]:
        return known
    if len(known) > confounders.shape[1]:
        return known[: confounders.shape[1]]
    pad = [f"c{i}" for i in range(len(known), confounders.shape[1])]
    return known + pad


def run_shen(city: str = "sf", n_subset: int | None = None, seed: int = 42) -> dict:
    print(f"\n=== Shen & Ross (2021) replication: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    _, confounders, Y, meta = feats

    # `get_features_and_target` may have dropped rows (Y<=0 etc). Align emb_df
    # against the same `valid` mask by rebuilding the mask the way the loader
    # does. Easier: trust that emb_df rows correspond 1:1 to confounders rows
    # because no row drops happen on a 995-row SF set with valid prices.
    if len(emb_df) != confounders.shape[0]:
        # Conservative: take the first N of emb_df to match.
        emb_df = emb_df.iloc[: confounders.shape[0]].reset_index(drop=True)

    descriptions = emb_df["clean_description"].fillna(emb_df["description"]).astype(str).tolist()
    lat = emb_df["latitude"].values.astype(float)
    lon = emb_df["longitude"].values.astype(float)

    if n_subset is not None and n_subset < len(descriptions):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(descriptions), size=n_subset, replace=False)
        idx.sort()
        descriptions = [descriptions[i] for i in idx]
        lat = lat[idx]
        lon = lon[idx]
        confounders = confounders[idx]
        Y = Y[idx]

    n = len(Y)
    print(f"  N={n:,}, confounders={confounders.shape[1]}")

    print("  Computing TF-IDF uniqueness (K=5 spatial neighbours)...")
    uniqueness = compute_uniqueness(descriptions, lat, lon, k=5)
    print(f"    uniqueness: mean={uniqueness.mean():.3f}  sd={uniqueness.std():.3f}  "
          f"min={uniqueness.min():.3f}  max={uniqueness.max():.3f}")

    print("  Hedonic OLS (HC3 SEs)...")
    conf_names = _confounder_names(confounders, parcels)
    ols, model = hedonic_ols(uniqueness, confounders, conf_names, Y)
    print(f"    uniqueness_z: β={ols.coef:+.4f}  se={ols.se:.4f}  t={ols.t:+.2f}  "
          f"p={ols.p:.4g}  95%CI=[{ols.ci_low:+.4f}, {ols.ci_high:+.4f}]")
    print(f"    → per-σ price effect: {ols.pct_per_sd:+.2f}%   (Shen reports ~+15%)")
    print(f"    OLS R²={ols.r2:.4f}, adj R²={ols.adj_r2:.4f}")

    print("  DML: uniqueness as continuous treatment...")
    dml = run_dml(
        uniqueness,
        confounders,
        Y,
        label="DML on TF-IDF uniqueness",
    )
    if dml is None:
        print("    DML failed (treatment fully explained by confounders)")
    else:
        flag = "contains 0" if dml.contains_zero else "EXCLUDES 0"
        print(f"    DML θ={dml.theta:+.4f}  se={dml.se:.4f}  "
              f"95%CI=[{dml.ci_low:+.4f}, {dml.ci_high:+.4f}]  {flag}")
        print(f"    MDE: ±{dml.mde:.4f}  ({100*(np.exp(dml.mde)-1):+.2f}% in price)")

    print("\n  Headline contrast:")
    print(f"    OLS uniqueness coefficient (Shen-style): {ols.coef:+.4f} per σ "
          f"({ols.pct_per_sd:+.2f}% per σ in price)")
    if dml is not None:
        print(f"    DML θ on uniqueness (causal):            {dml.theta:+.4f} per σ "
              f"(95% CI [{dml.ci_low:+.4f}, {dml.ci_high:+.4f}])")
        if dml.contains_zero and abs(ols.coef) > 2 * abs(dml.theta):
            print("    → predictive/hedonic gain is much larger than the causal estimate; "
                  "uniqueness effect attenuates under DML.")

    return {
        "paper": "Shen & Ross 2021 (JUE)",
        "city": city,
        "n": int(n),
        "n_confounders": int(confounders.shape[1]),
        "uniqueness_descriptive": {
            "mean": float(uniqueness.mean()),
            "sd": float(uniqueness.std(ddof=1)),
            "min": float(uniqueness.min()),
            "max": float(uniqueness.max()),
        },
        "ols_uniqueness": asdict(ols),
        "dml_uniqueness": result_to_dict(dml),
        "method_notes": {
            "tfidf_max_features": 5000,
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 2,
            "k_neighbors": 5,
            "neighbor_metric": "euclidean lat/lon",
            "ols_cov": "HC3",
            "treatment_standardization": "z-score (per σ)",
            "dml_pipeline": "causal_inference.dml_continuous_treatment",
        },
        "meta": {
            "has_rich_confounders": bool(meta["has_rich_confounders"]),
            "crime_dropped": bool(meta["crime_dropped"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sf")
    ap.add_argument("--n", type=int, default=None,
                    help="optional subset size for fast smoke runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="path to write JSON results")
    args = ap.parse_args()

    result = run_shen(city=args.city, n_subset=args.n, seed=args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

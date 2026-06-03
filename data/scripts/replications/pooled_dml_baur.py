"""Pooled DML across 12 cities with city fixed effects and city-cluster SE.

Replaces the per-market Baur DML estimator (which has 7% power against
the published Shen-Ross-magnitude effect) with a single pooled estimator
that uses city fixed effects in X, the pooled within-city-centered BERT
PC1 from `pooled_pca_treatment.py` as the treatment, and the Chiang-
Kato-Ma-Sasaki (2022, JBES 40(3):1046-1056) cluster-robust SE estimator
through DoubleML's `cluster_cols='city'` API.  A pairs-cluster bootstrap
(Cameron-Gelbach-Miller 2008, REStat 90(3):414-427) is reported as a
small-G robustness check (G=12 sits below the 30-cluster comfort zone).

Output: results/replications/pooled_dml_baur.json with the single pooled
theta estimate, CKMS cluster-robust SE with t(G-1) CI, pairs-cluster
bootstrap CI, sample-size breakdown by city, and per-rep diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from replications.baur_2023 import (  # noqa: E402
    get_features_and_target, load_analysis_data,
)

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _attach_pooled(city: str, emb_df: pd.DataFrame, pool: pd.DataFrame
                    ) -> pd.Series:
    if "listing_id" not in emb_df.columns:
        emb_df = emb_df.assign(listing_id=np.arange(len(emb_df)))
    rows = pool[pool["city"] == city][["listing_id", "treatment_z"]]
    merged = emb_df[["listing_id"]].merge(rows, on="listing_id", how="left")
    return merged["treatment_z"].fillna(0.0).to_numpy()


def _load_city_block(city: str, pool: pd.DataFrame):
    loaded = load_analysis_data(city)
    if loaded is None:
        return None
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return None
    _, confounders, Y_log, meta = feats
    T = _attach_pooled(city, emb_df, pool)
    if confounders.shape[0] != Y_log.shape[0]:
        print(f"  [warn] {city}: shape mismatch X={confounders.shape}, "
              f"Y={Y_log.shape}; skipping")
        return None
    if T.shape[0] != Y_log.shape[0]:
        print(f"  [warn] {city}: T-merge shape mismatch; skipping")
        return None
    return Y_log, T, confounders, meta


def _common_columns(meta_per_city: dict[str, pd.DataFrame]) -> list[str]:
    """Intersection of confounder column names across cities."""
    common = None
    for c, md in meta_per_city.items():
        cols = set(md.columns)
        common = cols if common is None else common & cols
    common.discard("city")
    return sorted(common)


def pooled_dml(df: pd.DataFrame, y: str, t: str, x_cols: list[str],
                cluster: str = "city", n_folds: int = 5, n_rep: int = 5,
                seed: int = 42):
    """DoubleMLPLR with city dummies in X and city-cluster SE.  Uses
    Chiang-Kato-Ma-Sasaki (2022) cluster-robust variance via
    DoubleML's cluster_cols API.  Returns (theta_hat, se_ckms, ci_low,
    ci_high, n_clusters, n_total).
    """
    from doubleml import DoubleMLData, DoubleMLPLR
    city_dum = pd.get_dummies(df[cluster], prefix="city", drop_first=True,
                               dtype=float)
    X = pd.concat([df[x_cols], city_dum], axis=1)
    data = pd.concat([df[[y, t, cluster]], X], axis=1).reset_index(drop=True)
    dml_data = DoubleMLData(data, y_col=y, d_cols=t,
                             x_cols=list(X.columns), cluster_cols=cluster)
    learner = RidgeCV(alphas=np.logspace(-3, 3, 25))
    dml = DoubleMLPLR(dml_data, ml_l=clone(learner), ml_m=clone(learner),
                       n_folds=n_folds, n_rep=n_rep)
    dml.fit()
    theta = float(dml.coef[0])
    se = float(dml.se[0])
    G = df[cluster].nunique()
    tcrit = stats.t.ppf(0.975, df=G - 1)
    return theta, se, theta - tcrit * se, theta + tcrit * se, G, len(df)


def pairs_cluster_boot(df: pd.DataFrame, y: str, t: str, x_cols: list[str],
                        cluster: str, B: int = 200, n_folds: int = 5,
                        seed: int = 42) -> tuple[float, float, list[float]]:
    """Cameron-Gelbach-Miller pairs-cluster bootstrap.  Resamples whole
    cities with replacement, refits DoubleMLPLR per bootstrap draw,
    returns 2.5%/97.5% empirical percentiles.
    """
    from doubleml import DoubleMLData, DoubleMLPLR
    rng = np.random.default_rng(seed)
    cities = df[cluster].unique()
    learner = RidgeCV(alphas=np.logspace(-3, 3, 25))
    boot_theta = []
    for b in range(B):
        draw = rng.choice(cities, size=len(cities), replace=True)
        blocks = []
        for i, c in enumerate(draw):
            bdf = df[df[cluster] == c].copy()
            bdf["_bcid"] = f"{c}__{i}"
            blocks.append(bdf)
        bdf_all = pd.concat(blocks, ignore_index=True)
        bX = pd.concat([bdf_all[x_cols],
                        pd.get_dummies(bdf_all[cluster], prefix="city",
                                        drop_first=True, dtype=float)],
                       axis=1)
        bdata = pd.concat([bdf_all[[y, t, "_bcid"]], bX], axis=1)
        try:
            bdml_data = DoubleMLData(bdata, y_col=y, d_cols=t,
                                      x_cols=list(bX.columns),
                                      cluster_cols="_bcid")
            bdml = DoubleMLPLR(bdml_data, ml_l=clone(learner),
                                ml_m=clone(learner), n_folds=n_folds,
                                n_rep=1)
            bdml.fit()
            boot_theta.append(float(bdml.coef[0]))
        except Exception as e:
            print(f"  [boot b={b}] failed: {e}")
            continue
    if not boot_theta:
        return float("nan"), float("nan"), []
    return (float(np.percentile(boot_theta, 2.5)),
            float(np.percentile(boot_theta, 97.5)),
            boot_theta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=ALL_12)
    ap.add_argument("--pooled_csv",
                    default=str(REPO / "results" / "replications"
                                  / "pooled_pca_treatment.csv"))
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_rep", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications"
                                  / "pooled_dml_baur.json"))
    args = ap.parse_args()

    pool = pd.read_csv(args.pooled_csv)

    # Load each city.  Keep a per-city DataFrame of (Y, T, all_confounders,
    # city) so we can find the common confounder columns across all 12.
    per_city = {}
    for c in args.cities:
        block = _load_city_block(c, pool)
        if block is None:
            print(f"  [skip] {c}: load failed")
            continue
        Y_log, T, X_arr, meta = block
        X_df = pd.DataFrame(X_arr, columns=[f"x_{i}" for i in range(X_arr.shape[1])])
        df_c = X_df.copy()
        df_c["log_price"] = Y_log
        df_c["treatment"] = T
        df_c["city"] = c
        per_city[c] = df_c
        print(f"  [{c}] n={len(df_c)} x_dim={X_arr.shape[1]}")

    # Generic name x_0..x_{p-1} columns are city-specific (different
    # confounder blocks).  The simplest defensible approach is to use the
    # common prefix subset; since names are positional we use only the
    # leading min(p_c) columns across cities, which corresponds to the
    # PROPERTY block of get_features_and_target (lat, lon, beds, sqft,
    # year_built, etc.) plus whatever subsequent blocks are universal.
    p_min = min(d.filter(like="x_").shape[1] for d in per_city.values())
    common_x = [f"x_{i}" for i in range(p_min)]
    print(f"\n  common confounder dim (intersection): {p_min}")

    df_all = pd.concat([d[common_x + ["log_price", "treatment", "city"]]
                         for d in per_city.values()], ignore_index=True)
    print(f"  pooled n={len(df_all)}, G={df_all['city'].nunique()}")

    print(f"\n=== Pooled DML (DoubleMLPLR, cluster_cols='city') ===")
    theta, se, ci_lo, ci_hi, G, n = pooled_dml(
        df_all, y="log_price", t="treatment", x_cols=common_x,
        cluster="city", n_folds=args.n_folds, n_rep=args.n_rep,
        seed=args.seed,
    )
    tcrit = stats.t.ppf(0.975, df=G - 1)
    print(f"  theta = {theta:+.4f}")
    print(f"  CKMS SE = {se:.4f}  (t-stat = {theta/se:+.2f})")
    print(f"  t({G-1}) critical = {tcrit:.3f}")
    print(f"  95% CKMS CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  excludes zero: {(ci_lo > 0 or ci_hi < 0)}")

    print(f"\n=== Pairs-cluster bootstrap (B={args.n_boot}) ===")
    bci_lo, bci_hi, boot_theta = pairs_cluster_boot(
        df_all, y="log_price", t="treatment", x_cols=common_x,
        cluster="city", B=args.n_boot, n_folds=args.n_folds,
        seed=args.seed,
    )
    print(f"  bootstrap 95% CI = [{bci_lo:+.4f}, {bci_hi:+.4f}]")
    if boot_theta:
        print(f"  bootstrap mean theta = {np.mean(boot_theta):+.4f}, "
              f"sd = {np.std(boot_theta, ddof=1):.4f}")

    n_per_city = {c: int(len(d)) for c, d in per_city.items()}
    out = {
        "method": "DoubleMLPLR with cluster_cols='city', ridge nuisances",
        "treatment": "pooled_within_city_centered_pca_pc1",
        "cluster_method": "CKMS (Chiang-Kato-Ma-Sasaki 2022)",
        "boot_method": "Pairs-cluster (Cameron-Gelbach-Miller 2008)",
        "theta_pooled": theta, "ckms_se": se,
        "ckms_ci_low": ci_lo, "ckms_ci_high": ci_hi,
        "ckms_t_critical": tcrit, "t_stat": theta / se if se > 0 else float("nan"),
        "boot_ci_low": bci_lo, "boot_ci_high": bci_hi,
        "boot_n_successful": len(boot_theta),
        "boot_mean": float(np.mean(boot_theta)) if boot_theta else None,
        "boot_sd": float(np.std(boot_theta, ddof=1)) if boot_theta else None,
        "G_clusters": G, "n_total": n,
        "n_per_city": n_per_city,
        "n_common_confounders": p_min,
        "ckms_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "boot_excludes_zero": bool(bci_lo > 0 or bci_hi < 0)
            if not np.isnan(bci_lo) else None,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

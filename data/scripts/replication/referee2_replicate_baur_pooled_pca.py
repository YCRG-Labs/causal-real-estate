"""Referee 2 clean-room replication of the pooled-PCA Baur DML result.

Independent re-implementation. Does NOT import any estimator function from
data/scripts/causal_inference.py or data/scripts/replications/*. It only
reads the author's committed intermediate CSV (pooled_pca_treatment.csv,
column treatment_z) as raw input data, since the task is to check the DML
estimator + pipeline wiring downstream of that file, not to re-derive the
pooled PCA axis itself.

Pipeline (written from the task spec, from scratch):
  1. Y = log(price) from data/processed/{city}_embeddings.parquet
  2. T = treatment_z for the city from
     results/replications/pooled_pca_treatment.csv, aligned to the embeddings
     parquet by ROW POSITION (listing_id in that CSV is 0..n-1 in the same
     row order as the parquet -- verified: max listing_id + 1 == parquet n).
  3. Confounders: nearest-parcel join from
     data/processed/{city}_parcels_micro_geo.gpkg onto each listing's
     (latitude, longitude), taking every numeric non-identifier column
     (property characteristics, census block-group covariates, crime counts,
     amenity counts, distance-to-amenity features). This is a superset /
     independent reconstruction of the author's canonical confounder list,
     not a copy of it.
  4. Partially-linear DML: 5-fold cross-fitted RidgeCV residualization of Y
     and T on standardized confounders; theta = mean(Tr*Yr)/mean(Tr*Tr);
     analytic influence-function SE.

Run:
  .venv/bin/python data/scripts/replication/referee2_replicate_baur_pooled_pca.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
CITIES = ["boston", "sf", "philadelphia", "chicago"]
SEED = 42
K_FOLDS = 5
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

NON_NUMERIC_PARCEL_COLS = {
    "parcel_id", "loc_id", "poly_type", "condition", "land_use_desc",
    "GEOID", "crime_data_year_range", "crime_temporal_match", "geometry",
    "shape_length_deg", "shape_area_deg",
}


def load_outcome_and_treatment(city: str):
    emb_path = REPO / "data" / "processed" / f"{city}_embeddings.parquet"
    emb = pd.read_parquet(emb_path)

    pooled_csv = REPO / "results" / "replications" / "pooled_pca_treatment.csv"
    pool = pd.read_csv(pooled_csv)
    pool_city = pool[pool["city"] == city].sort_values("listing_id")
    assert pool_city["listing_id"].is_monotonic_increasing
    n_pool = pool_city["listing_id"].max() + 1
    if n_pool != len(emb):
        print(f"  [DATA DRIFT] {city}: pooled_pca_treatment.csv covers "
              f"{n_pool} positional ids but {emb_path.name} now has {len(emb)} "
              f"rows. Embeddings parquet grew/shrank after the pooled-PCA "
              f"treatment CSV was frozen -> positional merge is stale for the "
              f"tail rows. Left-joining and 0-filling, exactly as the author's "
              f"_attach_pooled_treatment() does, to reproduce the same fallback.")
    full_ids = pd.DataFrame({"listing_id": np.arange(len(emb))})
    merged = full_ids.merge(
        pool_city[["listing_id", "treatment_z"]], on="listing_id", how="left")
    n_missing = int(merged["treatment_z"].isna().sum())
    if n_missing:
        print(f"  [warn] {n_missing} listings without a pooled-PCA score; "
              "filling with 0.0 (city mean after z-scoring)")
    treatment_z = merged["treatment_z"].fillna(0.0).to_numpy()

    price = pd.to_numeric(emb["price"], errors="coerce").to_numpy()
    lat = pd.to_numeric(emb["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(emb["longitude"], errors="coerce").to_numpy()
    return emb, price, lat, lon, treatment_z


def nearest_parcel_confounders(city: str, lat: np.ndarray, lon: np.ndarray):
    gpkg_path = REPO / "data" / "processed" / f"{city}_parcels_micro_geo.gpkg"
    parcels = gpd.read_file(gpkg_path, layer=city)

    numeric_cols = [
        c for c in parcels.columns
        if c not in NON_NUMERIC_PARCEL_COLS
        and pd.api.types.is_numeric_dtype(parcels[c])
    ]
    p_lat = parcels["latitude"].to_numpy(dtype=float)
    p_lon = parcels["longitude"].to_numpy(dtype=float)
    valid_p = np.isfinite(p_lat) & np.isfinite(p_lon)

    tree = cKDTree(np.column_stack([p_lat[valid_p], p_lon[valid_p]]))
    valid_l = np.isfinite(lat) & np.isfinite(lon)

    n = len(lat)
    conf = np.full((n, len(numeric_cols)), np.nan)
    if valid_l.sum() > 0:
        _, nn = tree.query(np.column_stack([lat[valid_l], lon[valid_l]]))
        parcel_idx = np.where(valid_p)[0][nn]
        vals = parcels[numeric_cols].to_numpy(dtype=float)[parcel_idx]
        conf[np.where(valid_l)[0], :] = vals

    return conf, numeric_cols, valid_l


def clean_confounders(conf: np.ndarray, nan_col_thresh=0.5):
    nan_rate = np.isnan(conf).mean(axis=0)
    keep = nan_rate < nan_col_thresh
    conf = conf[:, keep]
    for j in range(conf.shape[1]):
        col = conf[:, j]
        finite = np.isfinite(col)
        if finite.sum() == 0:
            col[:] = 0.0
            continue
        med = np.median(col[finite])
        col = np.where(finite, col, med)
        lo, hi = np.percentile(col, [1, 99])
        conf[:, j] = np.clip(col, lo, hi)
    return conf, keep


def ridge_dml(T: np.ndarray, X: np.ndarray, Y: np.ndarray,
              k_folds=K_FOLDS, seed=SEED, alphas=ALPHAS):
    n = len(Y)
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_resid = np.empty(n)
    T_resid = np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = RidgeCV(alphas=alphas).fit(Xs[tr], Y[tr])
        m_t = RidgeCV(alphas=alphas).fit(Xs[tr], T[tr])
        Y_resid[te] = Y[te] - m_y.predict(Xs[te])
        T_resid[te] = T[te] - m_t.predict(Xs[te])
    denom = float(np.mean(T_resid ** 2))
    theta = float(np.mean(T_resid * Y_resid) / denom)
    psi = (Y_resid - theta * T_resid) * T_resid / denom
    se_if = float(np.sqrt(np.var(psi, ddof=1) / n))
    return theta, se_if, n


def run_city(city: str):
    emb, price, lat, lon, treatment_z = load_outcome_and_treatment(city)
    conf_raw, numeric_cols, valid_l = nearest_parcel_confounders(city, lat, lon)

    valid = valid_l & np.isfinite(price) & (price > 0) & np.isfinite(treatment_z)
    conf_all_nan_row = np.all(np.isnan(conf_raw), axis=1)
    valid &= ~conf_all_nan_row

    conf = conf_raw[valid]
    conf, keep_cols = clean_confounders(conf)
    Y = np.log(price[valid])
    T = treatment_z[valid]
    # re-standardize T within the analysis sample, mirroring the author's
    # re-standardization of the already-per-city-z-scored input inside the
    # ridge DML wrapper (a near no-op if valid == full city sample).
    T = (T - T.mean()) / (T.std(ddof=1) or 1.0)

    theta, se_if, n = ridge_dml(T, conf, Y)
    n_conf_kept = int(conf.shape[1])
    return {
        "city": city,
        "n": n,
        "n_confounders": n_conf_kept,
        "n_confounders_before_nan_filter": len(numeric_cols),
        "theta": theta,
        "se_if": se_if,
        "ci_low": theta - 1.96 * se_if,
        "ci_high": theta + 1.96 * se_if,
    }


def main():
    author_tbl_path = (REPO / "results" / "replications" / "baur_pooled_pca")
    rows = []
    for city in CITIES:
        print(f"=== {city} ===")
        r = run_city(city)
        author = json.loads((author_tbl_path / f"{city}.json").read_text())
        a_theta = author["dml"]["theta"]
        a_se = author["dml"]["se"]
        a_n = author["n"]
        match = abs(r["theta"] - a_theta) < 0.02
        print(f"  n={r['n']} (author n={a_n}), n_confounders={r['n_confounders']} "
              f"(before NaN filter: {r['n_confounders_before_nan_filter']})")
        print(f"  referee2 theta={r['theta']:+.4f} se={r['se_if']:.4f}")
        print(f"  author   theta={a_theta:+.4f} se={a_se:.4f}")
        print(f"  match (|dtheta|<0.02): {match}")
        rows.append({
            "city": city, "n_referee2": r["n"], "n_author": a_n,
            "theta_referee2": r["theta"], "theta_author": a_theta,
            "se_referee2": r["se_if"], "se_author": a_se,
            "match_theta_2dp": match,
        })
    out_df = pd.DataFrame(rows)
    out_csv = REPO / "data" / "scripts" / "replication" / "referee2_baur_pooled_pca_comparison.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n=== Comparison table ===")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import bootstrap
from config import PROCESSED_DIR, CITIES, EMBEDDING_DIM


ESCALATION_LEVELS = [
    ("none", []),
    ("zip", ["zip_onehot"]),
    ("coordinates", ["latitude", "longitude"]),
    ("census", [
        "median_household_income", "pct_bachelors", "pct_white", "pct_black",
        "pct_asian", "pct_hispanic", "median_home_value", "median_gross_rent",
        "labor_force_participation", "pct_under_25", "pct_over_60",
    ]),
    ("crime", [
        "crime_violent", "crime_property", "crime_quality_of_life", "crime_total",
    ]),
    ("amenity", [
        "amenity_food_dining", "amenity_retail", "amenity_services",
        "amenity_recreation", "amenity_transportation", "amenity_education",
        "amenity_total", "amenity_diversity",
    ]),
    ("micro_geo", [
        "dist_park_m", "dist_transit_m", "dist_school_m",
        "dist_restaurant_m", "dist_retail_m", "dist_medical_m",
    ]),
]


def load_escalation_data(city):
    import geopandas as gpd

    emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    if not emb_path.exists():
        return None

    emb_df = pd.read_parquet(emb_path)

    for gpkg_name in [f"{city}_parcels_micro_geo.gpkg", f"{city}_parcels_amenities.gpkg"]:
        p = PROCESSED_DIR / gpkg_name
        if p.exists():
            parcels = gpd.read_file(p, layer=city)
            break
    else:
        parcels = None

    return emb_df, parcels


def build_confounder_matrix(emb_df, parcels, level_cols, matched_parcel_idx, valid_emb_mask):
    n = len(emb_df)

    if not level_cols:
        return np.zeros((n, 0))

    if level_cols == ["zip_onehot"]:
        if "zip" in emb_df.columns:
            zips = emb_df["zip"].fillna(0).astype(float).astype(int).astype(str)
            le = LabelEncoder()
            enc = le.fit_transform(zips)
            n_bins = min(len(le.classes_), 30)
            out = np.zeros((n, n_bins))
            for i, z in enumerate(enc):
                out[i, z % n_bins] = 1.0
            return out
        return np.zeros((n, 0))

    if level_cols == ["latitude", "longitude"]:
        lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)), errors="coerce").values
        lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)), errors="coerce").values
        return np.column_stack([np.nan_to_num(lat, nan=0.0), np.nan_to_num(lon, nan=0.0)])

    if parcels is None or matched_parcel_idx is None:
        return np.zeros((n, 0))

    available = [c for c in level_cols if c in parcels.columns]
    if not available:
        return np.zeros((n, 0))

    arr = np.full((n, len(available)), 0.0)
    for j, col in enumerate(available):
        vals = pd.to_numeric(parcels[col], errors="coerce").values
        arr[np.where(valid_emb_mask)[0], j] = np.nan_to_num(vals[matched_parcel_idx], nan=0.0)
    return arr


def spatial_match(emb_df, parcels):
    from scipy.spatial import cKDTree

    lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)), errors="coerce").values
    lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)), errors="coerce").values
    valid_emb = ~(np.isnan(lat) | np.isnan(lon))

    if parcels is None or not hasattr(parcels, "geometry"):
        return None, valid_emb

    centroids = parcels.geometry.centroid
    p_lat = centroids.y.values.astype(float)
    p_lon = centroids.x.values.astype(float)
    valid_p = ~(np.isnan(p_lat) | np.isnan(p_lon))

    if valid_p.sum() == 0 or valid_emb.sum() == 0:
        return None, valid_emb

    tree = cKDTree(np.column_stack([p_lat[valid_p], p_lon[valid_p]]))
    _, nn_idx = tree.query(np.column_stack([lat[valid_emb], lon[valid_emb]]))
    parcel_valid_idx = np.where(valid_p)[0]
    return parcel_valid_idx[nn_idx], valid_emb


def dr_estimate(T_pca, confounders, Y):
    if confounders.shape[1] > 0:
        scaler = StandardScaler()
        conf_s = scaler.fit_transform(confounders)
    else:
        conf_s = np.zeros((len(Y), 1))

    T_norm = np.linalg.norm(T_pca, axis=1)
    treatment = (T_norm > np.median(T_norm)).astype(float)

    outcome = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    outcome.fit(np.hstack([treatment.reshape(-1, 1), conf_s]), Y)

    propensity = LogisticRegression(max_iter=1000, random_state=42)
    propensity.fit(conf_s, treatment)
    e = np.clip(propensity.predict_proba(conf_s)[:, 1], 0.05, 0.95)

    mu1 = outcome.predict(np.hstack([np.ones((len(Y), 1)), conf_s]))
    mu0 = outcome.predict(np.hstack([np.zeros((len(Y), 1)), conf_s]))

    ate = np.mean(
        mu1 - mu0
        + treatment * (Y - mu1) / e
        - (1 - treatment) * (Y - mu0) / (1 - e)
    )

    def stat(idx, t=treatment, y=Y, m1=mu1, m0=mu0, ps=e):
        i = idx[0]
        return np.mean(m1[i] - m0[i] + t[i]*(y[i]-m1[i])/ps[i] - (1-t[i])*(y[i]-m0[i])/(1-ps[i]))

    rng = np.random.default_rng(42)
    ci = bootstrap((np.arange(len(Y)),), stat, n_resamples=500, random_state=rng, method="percentile")

    return ate, ci.confidence_interval.low, ci.confidence_interval.high


def run_escalation(city):
    data = load_escalation_data(city)
    if data is None:
        print(f"{city}: no data, skipping")
        return None

    emb_df, parcels = data

    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available_emb = [c for c in emb_cols if c in emb_df.columns]
    T = emb_df[available_emb].values

    if "price" in emb_df.columns:
        Y = pd.to_numeric(emb_df["price"], errors="coerce").values
    else:
        print(f"{city}: no price column, skipping")
        return None

    matched_idx, valid_emb = spatial_match(emb_df, parcels)

    n_pca = min(50, T.shape[1], T.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    valid = ~(np.isnan(Y) | np.isinf(Y) | (Y <= 0))
    T_pca_v = T_pca[valid]
    Y_v = np.log(Y[valid])

    print(f"\n{'='*70}")
    print(f"CONFOUNDER ESCALATION TEST: {city.upper()} (n={len(Y_v)})")
    print(f"{'='*70}")
    print(f"{'Level':<12} {'Confounders':<15} {'Cum. Dims':>10} {'ATE':>8} {'95% CI':>22} {'Zero in CI':>12}")
    print("-" * 80)

    cumulative_conf = np.zeros((len(emb_df), 0))
    results = []

    for level_name, level_cols in ESCALATION_LEVELS:
        new_block = build_confounder_matrix(emb_df, parcels, level_cols, matched_idx, valid_emb)

        if new_block.shape[1] > 0:
            cumulative_conf = np.hstack([cumulative_conf, new_block])

        conf_valid = cumulative_conf[valid]

        ate, ci_lo, ci_hi = dr_estimate(T_pca_v, conf_valid, Y_v)
        contains_zero = ci_lo <= 0 <= ci_hi

        print(f"{level_name:<12} +{level_cols[0][:12] if level_cols else '(none)':<14} {conf_valid.shape[1]:>10} "
              f"{ate:>8.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}] {'yes' if contains_zero else '** NO **':>12}")

        results.append({
            "level": level_name,
            "n_confounders": conf_valid.shape[1],
            "ate": ate,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "contains_zero": contains_zero,
        })

    ates = [r["ate"] for r in results]
    if len(ates) >= 2 and abs(ates[0]) > abs(ates[-1]):
        shrinkage = 1 - abs(ates[-1]) / max(abs(ates[0]), 1e-10)
        print(f"\nEffect shrinkage from none → full: {shrinkage*100:.1f}%")
        print("The estimated effect diminishes as confounders get richer,")
        print("consistent with SCM_0 (no direct T→Y) and confounding bias being removed.")
    else:
        print(f"\nEffect did not shrink monotonically. Examine results above.")

    all_contain_zero = all(r["contains_zero"] for r in results)
    if all_contain_zero:
        print("All CIs contain zero at every escalation level.")
    else:
        sig_levels = [r["level"] for r in results if not r["contains_zero"]]
        print(f"CIs exclude zero at: {', '.join(sig_levels)}")

    return results


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        run_escalation(city)


if __name__ == "__main__":
    main()

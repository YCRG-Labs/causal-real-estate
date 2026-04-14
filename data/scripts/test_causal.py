import sys
import numpy as np
import pandas as pd

np.random.seed(42)

N = 200
N_ZIPS = 5
EMB_DIM = 50

print("=" * 60)
print("SYNTHETIC DATA TEST FOR CAUSAL INFERENCE PIPELINE")
print("=" * 60)

lat = 40.7 + np.random.randn(N) * 0.05
lon = -74.0 + np.random.randn(N) * 0.05
zips = np.random.choice([f"1000{i}" for i in range(N_ZIPS)], N)

income = 50000 + lat * 10000 + np.random.randn(N) * 5000
crime = np.maximum(0, 100 - lat * 50 + np.random.randn(N) * 20)
amenity = 20 + lon * 5 + np.random.randn(N) * 5

price = np.exp(
    12.0
    + 0.5 * (lat - 40.7)
    + 0.3 * (lon + 74.0)
    + 0.2 * (income - 50000) / 10000
    - 0.1 * crime / 50
    + np.random.randn(N) * 0.3
)

text_emb = np.random.randn(N, EMB_DIM)
for i in range(N):
    text_emb[i, :5] += lat[i] * 0.5
    text_emb[i, 5:10] += lon[i] * 0.3

emb_df = pd.DataFrame({
    "latitude": lat,
    "longitude": lon,
    "zip": zips,
    "price": price,
    "clean_description": [f"nice property in area {z}" for z in zips],
})
for j in range(EMB_DIM):
    emb_df[f"emb_{j}"] = text_emb[:, j]

try:
    import geopandas as gpd
    from shapely.geometry import Point
    parcels = gpd.GeoDataFrame(
        {
            "parcel_id": [f"P{i}" for i in range(N)],
            "bedrooms": np.random.choice([2, 3, 4, 5], N),
            "bldg_area_sqft": np.random.uniform(1000, 5000, N),
            "lot_area_sqft": np.random.uniform(2000, 10000, N),
            "year_built": np.random.randint(1920, 2020, N),
            "median_household_income": income,
            "pct_bachelors": np.random.uniform(0.1, 0.6, N),
            "pct_white": np.random.uniform(0.2, 0.8, N),
            "pct_black": np.random.uniform(0.05, 0.4, N),
            "pct_asian": np.random.uniform(0.02, 0.3, N),
            "pct_hispanic": np.random.uniform(0.05, 0.4, N),
            "median_home_value": price * np.random.uniform(0.8, 1.2, N),
            "median_gross_rent": np.random.uniform(1000, 3000, N),
            "labor_force_participation": np.random.uniform(0.5, 0.8, N),
            "pct_under_25": np.random.uniform(0.1, 0.3, N),
            "pct_over_60": np.random.uniform(0.1, 0.3, N),
            "crime_violent": np.maximum(0, crime * 0.3 + np.random.randn(N) * 5).astype(int),
            "crime_property": np.maximum(0, crime * 0.5 + np.random.randn(N) * 5).astype(int),
            "crime_quality_of_life": np.maximum(0, crime * 0.2 + np.random.randn(N) * 3).astype(int),
            "crime_total": np.maximum(0, crime + np.random.randn(N) * 10).astype(int),
            "amenity_food_dining": np.maximum(0, amenity * 0.4 + np.random.randn(N) * 2).astype(int),
            "amenity_retail": np.maximum(0, amenity * 0.2 + np.random.randn(N)).astype(int),
            "amenity_services": np.maximum(0, amenity * 0.15 + np.random.randn(N)).astype(int),
            "amenity_recreation": np.maximum(0, amenity * 0.1 + np.random.randn(N)).astype(int),
            "amenity_transportation": np.maximum(0, amenity * 0.1 + np.random.randn(N)).astype(int),
            "amenity_education": np.maximum(0, amenity * 0.05 + np.random.randn(N)).astype(int),
            "amenity_total": np.maximum(0, amenity + np.random.randn(N) * 3).astype(int),
            "amenity_diversity": np.random.uniform(0.5, 2.5, N),
            "dist_park_m": np.random.uniform(50, 2000, N),
            "dist_transit_m": np.random.uniform(100, 3000, N),
            "dist_school_m": np.random.uniform(200, 2500, N),
            "dist_restaurant_m": np.random.uniform(50, 1500, N),
            "dist_retail_m": np.random.uniform(100, 2000, N),
            "dist_medical_m": np.random.uniform(200, 5000, N),
        },
        geometry=[Point(lo, la) for la, lo in zip(lat, lon)],
        crs="EPSG:4326",
    )
    HAS_PARCELS = True
    print(f"Created synthetic parcels GeoDataFrame: {len(parcels)} rows, {len(parcels.columns)} cols")
except ImportError:
    parcels = None
    HAS_PARCELS = False
    print("geopandas not available, testing without parcels")


sys.path.insert(0, str(pd.io.common.Path(__file__).resolve().parent))

from causal_inference import (
    get_features_and_target,
    backdoor_adjustment,
    doubly_robust_estimation,
    adversarial_deconfounding,
    randomization_test,
    cate_by_price_quantile,
    EMBEDDING_DIM as CI_EMB_DIM,
)

old_emb_dim = CI_EMB_DIM

import causal_inference
causal_inference.EMBEDDING_DIM = EMB_DIM

print("\n--- Testing get_features_and_target ---")
data = get_features_and_target(emb_df, parcels)
assert data is not None, "get_features_and_target returned None"
T, confounders, Y, meta = data
print(f"  T shape: {T.shape}")
print(f"  Confounders shape: {confounders.shape}")
print(f"  Y shape: {Y.shape}")
print(f"  Rich confounders: {meta['has_rich_confounders']}")
assert T.shape[0] == confounders.shape[0] == Y.shape[0], "Shape mismatch"
assert T.shape[1] == EMB_DIM, f"Expected {EMB_DIM} embedding dims, got {T.shape[1]}"
if HAS_PARCELS:
    assert confounders.shape[1] > 5, f"Expected rich confounders, got {confounders.shape[1]} dims"
    assert meta["has_rich_confounders"], "Should have rich confounders with parcels"
print("  PASSED")

print("\n--- Testing backdoor_adjustment ---")
delta_r2 = backdoor_adjustment(T, confounders, Y, n_pca=10)
assert isinstance(delta_r2, float), "backdoor should return float"
assert not np.isnan(delta_r2), "backdoor returned NaN"
print("  PASSED")

print("\n--- Testing doubly_robust_estimation ---")
dr_effect, dr_ci, dr_extras = doubly_robust_estimation(T, confounders, Y, n_pca=10)
assert isinstance(dr_effect, float), "DR should return float"
assert len(dr_ci) == 2, "DR CI should be tuple of 2"
assert not np.isnan(dr_effect), "DR returned NaN"
assert "if_se" in dr_extras and "mde" in dr_extras, "DR extras missing IF SE / MDE"
assert dr_extras["mde"] > 0, "MDE should be positive"
print("  PASSED")

print("\n--- Testing dml_continuous_treatment ---")
from causal_inference import dml_continuous_treatment
dml = dml_continuous_treatment(T, confounders, Y, n_pca=10)
assert dml is None or "theta" in dml, "DML should return dict with theta"
if dml is not None:
    assert "se" in dml and "ci" in dml and "mde" in dml, "DML extras missing"
print("  PASSED")

print("\n--- Testing adversarial_deconfounding (multi-head) ---")
pred_r2, disc_metrics = adversarial_deconfounding(T, Y, meta, n_pca=10, epochs=20)
assert isinstance(pred_r2, float), "adversarial should return float R2"
assert "zip_acc" in disc_metrics, "Missing zip_acc"
assert "geo_r2" in disc_metrics, "Missing geo_r2"
assert "inc_acc" in disc_metrics, "Missing inc_acc"
print("  PASSED")

print("\n--- Testing randomization_test ---")
r2_orig, r2_perm, p_val = randomization_test(T, confounders, Y, n_permutations=5, n_pca=10)
assert isinstance(r2_orig, float), "randomization should return float"
assert 0 <= p_val <= 1, f"p-value out of range: {p_val}"
print("  PASSED")

print("\n--- Testing cate_by_price_quantile ---")
cate_results = cate_by_price_quantile(T, confounders, Y, n_quantiles=3, n_pca=10)
assert isinstance(cate_results, list), "CATE should return list"
assert len(cate_results) == 3, f"Expected 3 quantiles, got {len(cate_results)}"
for r in cate_results:
    assert "quantile" in r, "Missing quantile key"
    assert "ate" in r, "Missing ate key"
print("  PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

causal_inference.EMBEDDING_DIM = old_emb_dim

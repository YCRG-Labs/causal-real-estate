"""Referee pre-emption: does the pooled-PCA Baur text effect survive
conditioning on independent property condition/quality signals?

The strongest referee critique of the "semantic content" story is that the
surviving DML theta on the pooled BERT PC1 treatment is actually laundering
unobserved property CONDITION (renovation state, finish quality, deferred
maintenance) through the listing language: nicer listings both write
differently and are objectively higher-condition properties, and condition
is not in the canonical confounder set (canonical_confounders.py: spatial +
bedrooms/bldg_area/lot_area/year_built + census + crime + amenity + micro-geo
distances -- no assessor condition/grade/value field anywhere in that list).

This script inventories which of the 12 markets have an *assessor-authored*
condition or value signal in the parcels_micro_geo.gpkg files (i.e. a field
authored by the county/city assessor's office on its own administrative
schedule, not derived from the current listing or its price), builds the
best available proxy per market, and re-runs the baur_pooled_pca DML with
that proxy appended to the confounder set. Reports theta/SE/CI with vs.
without the proxy.

Markets with a proxy (inventory result):
  boston -> `condition` (assessor condition grade, ordinal string:
            Unsound/Very Poor/Poor/Fair/Average/Good/Very Good/Excellent)
  sf     -> `assessed_improvement` (Prop 13 assessed building value, distinct
            from `assessed_land`; fixed at last change-of-ownership/new
            construction and inflated <=2%/yr by CA law, so for the median
            long-held property it reflects a purchase price from years/
            decades before the current listing, not the current outcome)
  nyc    -> `assessed_total` - `assessed_land` (NYC DOF's own improvement
            assessment, published on the city's annual assessment roll
            independent of any particular listing)

Markets WITHOUT any such field in the current pipeline (checked all 12
`data/processed/{city}_parcels_micro_geo.gpkg` schemas and the
`results/assessor/{city}_assessor.parquet` structural extracts):
  philadelphia, chicago, seattle, denver, atlanta, portland, phoenix, dallas,
  dc -- these files carry only bedrooms/bldg_area_sqft/lot_area_sqft/
  year_built (all already in PROPERTY_COLS) plus census/crime/amenity/micro
  confounders. No condition, grade, or assessed-value column survived
  ingestion for these nine markets, so no independent quality proxy can be
  built for them without new data collection.

CRITICAL: none of these proxies are a function of the CURRENT sale/asking
price being modeled (Y_log in this script). They are pre-existing,
administratively-scheduled assessor records that predate the current
listing. This is not the disallowed "price-per-sqft residual of log price"
bad control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from causal_inference import get_features_and_target, load_analysis_data
from replications.baur_pooled_pca import _attach_pooled_treatment
from replications.compare_to_dml import result_to_dict, run_dml

NO_PROXY_CITIES = ["philadelphia", "chicago"]


def _partial_r2(target: np.ndarray, proxy: np.ndarray, X_conf: np.ndarray,
                 k_folds: int = 5, seed: int = 42) -> float:
    """Squared partial correlation of `target` and `proxy` net of X_conf,
    via k-fold cross-fitted RidgeCV residualization (same nuisance family
    as the ridge-DML backend used throughout this project). This is the
    Cinelli-Hazlett-style partial R^2 used to benchmark an omitted
    confounder's plausible strength against a DML robustness value."""
    n = len(target)
    conf_s = StandardScaler().fit_transform(np.asarray(X_conf, dtype=np.float64))
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    t_res = np.empty(n)
    p_res = np.empty(n)
    target = np.asarray(target, dtype=np.float64).ravel()
    proxy = np.asarray(proxy, dtype=np.float64).ravel()
    for tr, te in kf.split(np.arange(n)):
        m_t = RidgeCV(alphas=alphas).fit(conf_s[tr], target[tr])
        t_res[te] = target[te] - m_t.predict(conf_s[te])
        m_p = RidgeCV(alphas=alphas).fit(conf_s[tr], proxy[tr])
        p_res[te] = proxy[te] - m_p.predict(conf_s[te])
    if np.std(t_res) < 1e-12 or np.std(p_res) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(t_res, p_res)[0, 1])
    return float(corr ** 2)


def _rv(theta: float, se: float, n: int, k_x: int, q_alpha: float = 1.96):
    """Cinelli-Hazlett-style Robustness Value for a DML estimate (matches
    replications/confounder_sensitivity_12.py's `_rv`, duplicated here to
    keep this script a self-contained artifact). Returns (RV, RV_alpha)."""
    if se == 0 or math.isnan(se) or math.isnan(theta):
        return float("nan"), float("nan")
    t = abs(theta) / se
    df = max(n - k_x - 1, 1)
    f2 = (t ** 2) / df
    rv = 0.5 * (math.sqrt(f2 ** 2 + 4 * f2) - f2)
    f2_q = max((t ** 2 - q_alpha ** 2), 0) / df
    rv_alpha = 0.5 * (math.sqrt(f2_q ** 2 + 4 * f2_q) - f2_q) if f2_q > 0 else 0.0
    return float(rv), float(rv_alpha)


def run_city_no_proxy(city: str, pooled_csv: Path, seed: int = 42) -> dict:
    """For markets with no assessor quality signal (philadelphia, chicago):
    report the DML robustness value instead, so it can be benchmarked
    against the quality-proxy partial R^2 measured in the markets where a
    proxy *is* available."""
    print(f"\n=== RV benchmark (no quality proxy available): {city} ===")
    loaded = load_analysis_data(city)
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    _T_emb, confounders_base, Y_log, meta = feats
    T_pooled = _attach_pooled_treatment(city, emb_df, pooled_csv)
    valid_mask = np.asarray(meta["valid_mask"], dtype=bool)
    T_pooled = T_pooled[valid_mask]

    dml = run_dml(T_pooled, confounders_base, Y_log,
                   label="baseline", ci_method="if", use_ridge=True,
                   seed=seed, n_pca=1)
    n = len(Y_log)
    k_x = confounders_base.shape[1]
    rv, rv_alpha = _rv(dml.theta, dml.se, n, k_x)
    print(f"    theta={dml.theta:+.4f} se={dml.se:.4f} n={n} k_x={k_x}")
    print(f"    RV={rv:.4f}  RV_alpha(0.05)={rv_alpha:.4f}")
    return {
        "city": city, "n": int(n), "n_confounders": int(k_x),
        "theta": dml.theta, "se": dml.se,
        "ci_low": dml.ci_low, "ci_high": dml.ci_high,
        "rv": rv, "rv_alpha": rv_alpha,
    }

CONDITION_MAP = {
    "US - Unsound": 1.0,
    "VP - Very Poor": 2.0,
    "P - Poor": 3.0,
    "F - Fair": 4.0,
    "A - Average": 5.0,
    "G - Good": 6.0,
    "VG - Very Good": 7.0,
    "E - Excellent": 8.0,
    "EX - Excellent": 8.0,
}


def _quality_boston(parcels: pd.DataFrame) -> tuple[np.ndarray, str]:
    raw = parcels["condition"].map(CONDITION_MAP).to_numpy(dtype=float)
    return raw, "assessor_condition_grade_ordinal_1to8"


def _quality_sf(parcels: pd.DataFrame) -> tuple[np.ndarray, str]:
    val = pd.to_numeric(parcels["assessed_improvement"], errors="coerce").to_numpy(dtype=float)
    out = np.where(val > 0, np.log(val), np.nan)
    return out, "log_assessed_improvement_value_prop13"


def _quality_nyc(parcels: pd.DataFrame) -> tuple[np.ndarray, str]:
    total = pd.to_numeric(parcels["assessed_total"], errors="coerce").to_numpy(dtype=float)
    land = pd.to_numeric(parcels["assessed_land"], errors="coerce").to_numpy(dtype=float)
    imp = total - land
    out = np.where(imp > 0, np.log(imp), np.nan)
    return out, "log_assessed_improvement_value_dof"


QUALITY_BUILDERS = {"boston": _quality_boston, "sf": _quality_sf, "nyc": _quality_nyc}


def _nn_join_quality(emb_df: pd.DataFrame, parcels, quality_raw: np.ndarray) -> np.ndarray:
    """Nearest-centroid join of a parcel-level column onto emb_df's row
    order, mirroring causal_inference._spatial_join_parcels's own KDTree
    match so the quality proxy lines up with the confounders built from the
    same parcels file."""
    lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    valid_emb = ~(np.isnan(lat) | np.isnan(lon))

    centroids = parcels.geometry.centroid
    p_lat = centroids.y.to_numpy(float)
    p_lon = centroids.x.to_numpy(float)
    valid_parcels = ~(np.isnan(p_lat) | np.isnan(p_lon)) & ~np.isnan(quality_raw)

    tree = cKDTree(np.column_stack([p_lat[valid_parcels], p_lon[valid_parcels]]))
    _, nn_idx = tree.query(np.column_stack([lat[valid_emb], lon[valid_emb]]))
    parcel_valid_idx = np.where(valid_parcels)[0]
    matched_idx = parcel_valid_idx[nn_idx]

    out = np.full(len(emb_df), np.nan)
    out[np.where(valid_emb)[0]] = quality_raw[matched_idx]
    return out


def run_city(city: str, pooled_csv: Path, seed: int = 42) -> dict:
    print(f"\n=== quality-proxy robustness: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    _T_emb, confounders_base, Y_log, meta = feats

    T_pooled = _attach_pooled_treatment(city, emb_df, pooled_csv)
    valid_mask = np.asarray(meta["valid_mask"], dtype=bool)
    T_pooled = T_pooled[valid_mask]
    assert len(T_pooled) == len(confounders_base) == len(Y_log)

    builder = QUALITY_BUILDERS[city]
    quality_raw, proxy_name = builder(parcels)
    quality_all = _nn_join_quality(emb_df, parcels, quality_raw)
    quality_v = quality_all[valid_mask]

    missing = np.isnan(quality_v)
    n_missing = int(missing.sum())
    med = float(np.nanmedian(quality_v)) if (~missing).any() else 0.0
    quality_filled = np.where(missing, med, quality_v)
    quality_block = np.column_stack([quality_filled, missing.astype(float)])
    confounders_aug = np.hstack([confounders_base, quality_block])

    n = len(Y_log)
    print(f"  N={n:,}, base confounders={confounders_base.shape[1]}, "
          f"proxy={proxy_name}, missing={n_missing}/{n} ({100*n_missing/n:.1f}%)")

    common = dict(ci_method="if", use_ridge=True, seed=seed, n_pca=1)
    dml_base = run_dml(T_pooled, confounders_base, Y_log,
                        label="baseline (no quality proxy)", **common)
    dml_aug = run_dml(T_pooled, confounders_aug, Y_log,
                       label="with quality proxy", **common)

    if dml_base is None or dml_aug is None:
        return {"city": city, "error": "DML failed", "proxy": proxy_name}

    print(f"    baseline   theta={dml_base.theta:+.4f} se={dml_base.se:.4f} "
          f"CI=[{dml_base.ci_low:+.4f}, {dml_base.ci_high:+.4f}]")
    print(f"    +proxy     theta={dml_aug.theta:+.4f} se={dml_aug.se:.4f} "
          f"CI=[{dml_aug.ci_low:+.4f}, {dml_aug.ci_high:+.4f}]")
    pct_atten = 100.0 * (1.0 - dml_aug.theta / dml_base.theta) if dml_base.theta != 0 else float("nan")
    print(f"    attenuation vs baseline: {pct_atten:+.1f}%  "
          f"(baseline excludes 0: {not dml_base.contains_zero}, "
          f"+proxy excludes 0: {not dml_aug.contains_zero})")

    cf_d = _partial_r2(T_pooled, quality_filled, confounders_base, seed=seed)
    cf_y = _partial_r2(Y_log, quality_filled, confounders_base, seed=seed)
    print(f"    quality-proxy partial R^2: with T={cf_d:.4f}, with Y={cf_y:.4f} "
          f"(benchmark for the RV comparison in philadelphia/chicago, where "
          f"no independent proxy exists)")

    return {
        "city": city,
        "n": int(n),
        "proxy_name": proxy_name,
        "proxy_n_missing": n_missing,
        "proxy_pct_missing": 100.0 * n_missing / n,
        "n_confounders_base": int(confounders_base.shape[1]),
        "n_confounders_with_proxy": int(confounders_aug.shape[1]),
        "dml_baseline": result_to_dict(dml_base),
        "dml_with_proxy": result_to_dict(dml_aug),
        "pct_attenuation": pct_atten,
        "proxy_partial_r2_with_T": cf_d,
        "proxy_partial_r2_with_Y": cf_y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(QUALITY_BUILDERS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pooled_csv",
                    default=str(REPO / "results" / "replications"
                                  / "pooled_pca_treatment_keyed.csv"))
    ap.add_argument("--no_proxy_cities", nargs="+", default=list(NO_PROXY_CITIES))
    ap.add_argument("--out_dir", type=Path,
                    default=REPO / "results" / "replications" / "quality_proxy_robustness")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in args.cities:
        if c not in QUALITY_BUILDERS:
            print(f"skipping {c}: no assessor quality proxy available in this pipeline")
            continue
        r = run_city(c, Path(args.pooled_csv), seed=args.seed)
        (args.out_dir / f"{c}.json").write_text(json.dumps(r, indent=2, default=float))
        rows.append(r)

    tbl_rows = []
    for r in rows:
        if "error" in r:
            tbl_rows.append({"city": r["city"], "error": r["error"]})
            continue
        b, a = r["dml_baseline"], r["dml_with_proxy"]
        tbl_rows.append({
            "city": r["city"], "n": r["n"], "proxy": r["proxy_name"],
            "pct_missing": r["proxy_pct_missing"],
            "theta_base": b["theta"], "se_base": b["se"],
            "ci_low_base": b["ci_low"], "ci_high_base": b["ci_high"],
            "excludes_zero_base": not b["contains_zero"],
            "theta_proxy": a["theta"], "se_proxy": a["se"],
            "ci_low_proxy": a["ci_low"], "ci_high_proxy": a["ci_high"],
            "excludes_zero_proxy": not a["contains_zero"],
            "pct_attenuation": r["pct_attenuation"],
            "proxy_partial_r2_with_T": r["proxy_partial_r2_with_T"],
            "proxy_partial_r2_with_Y": r["proxy_partial_r2_with_Y"],
        })
    df = pd.DataFrame(tbl_rows)
    tbl_csv = args.out_dir / "quality_proxy_robustness_table.csv"
    df.to_csv(tbl_csv, index=False)
    print("\n=== Table (direct conditioning: boston/sf/nyc) ===")
    print(df.to_string(index=False))
    print(f"\nTable -> {tbl_csv}")

    rv_rows = []
    for c in args.no_proxy_cities:
        r = run_city_no_proxy(c, Path(args.pooled_csv), seed=args.seed)
        (args.out_dir / f"{c}_rv.json").write_text(json.dumps(r, indent=2, default=float))
        rv_rows.append(r)
    if rv_rows:
        rv_df = pd.DataFrame(rv_rows)
        rv_csv = args.out_dir / "quality_proxy_rv_benchmark_table.csv"
        rv_df.to_csv(rv_csv, index=False)
        print("\n=== Table (RV benchmark: philadelphia/chicago, no direct proxy) ===")
        print(rv_df.to_string(index=False))
        if not df.empty and "proxy_partial_r2_with_T" in df.columns:
            max_cf_d = df["proxy_partial_r2_with_T"].max()
            max_cf_y = df["proxy_partial_r2_with_Y"].max()
            print(f"\nBenchmark: observed quality-proxy partial R^2 across "
                  f"boston/sf/nyc tops out at {max_cf_d:.4f} (with T) / "
                  f"{max_cf_y:.4f} (with Y).")
            for _, rv_row in rv_df.iterrows():
                print(f"  {rv_row['city']}: RV={rv_row['rv']:.4f}, "
                      f"RV_alpha={rv_row['rv_alpha']:.4f} -> "
                      f"{'>> ' if rv_row['rv'] > max(max_cf_d, max_cf_y) else ''}"
                      f"observed proxy strength is "
                      f"{'well below' if rv_row['rv'] > 3*max(max_cf_d, max_cf_y) else 'comparable to or above'} "
                      f"the RV needed to overturn this city's estimate")
        print(f"RV table -> {rv_csv}")


if __name__ == "__main__":
    sys.exit(main())

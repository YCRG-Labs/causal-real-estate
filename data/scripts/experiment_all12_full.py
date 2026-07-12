"""Uniform clean decomposition across all twelve markets.

Confounders use the LISTING's own structured attributes (exact, not a
nearest-parcel guess) plus block-group demographics attached by a point-in-polygon
join on the listing coordinate:

    lat, lon,
    beds, baths, sqft, year_built     (median-imputed, with missingness indicators)
    + 11 ACS block-group demographics

Treatment is the HTML-fixed pooled PC1 (results/experiment_length_full). For each
market we peel the naive estimate apart:

    base                      full confounders
    + length                  add log_len + log_len^2
    drop non-residential      remove land/lot/vacant/parking listings
    drop non-res + length     both

    python data/scripts/experiment_all12_clean.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from replications.compare_to_dml import run_dml

PROC = REPO / "data" / "processed"
FULL = REPO / "results" / "experiment_length_full"
CENSUS = REPO / "results" / "census_bg"
OSM = REPO / "results" / "osm_context"
OUT = REPO / "results" / "experiment_length" / "all12_full.json"
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
EMB = [f"emb_{i}" for i in range(768)]
STRUCT = ["beds", "baths", "sqft", "year_built"]
DEMO = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60"]
AMENITY = ["amenity_food_dining", "amenity_retail", "amenity_services",
           "amenity_recreation", "amenity_transportation", "amenity_education",
           "amenity_total", "amenity_diversity"]
MICRO = ["dist_park_m", "dist_transit_m", "dist_school_m",
         "dist_restaurant_m", "dist_retail_m", "dist_medical_m"]
NON_RES = r"land|lot|vacant|parking"


def pooled_treatment():
    blocks, sizes = [], []
    for c in ALL_12:
        r = pd.read_parquet(FULL / f"{c}_reembed.parquet").sort_values("row").reset_index(drop=True)
        blocks.append(r[EMB].to_numpy(np.float32))
        sizes.append((c, len(r)))
    Xc = np.vstack([b - b.mean(0, keepdims=True) for b in blocks]).astype(np.float64)
    del blocks
    gc.collect()
    d = PCA(1, random_state=0).fit(Xc).components_[0]
    if d.sum() < 0:
        d = -d
    scores = Xc @ d
    del Xc
    gc.collect()
    out, i = {}, 0
    for c, n in sizes:
        s = scores[i:i + n]
        out[c] = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        i += n
    return out


def confounders(city):
    lst = pd.read_parquet(PROC / f"{city}_embeddings.parquet",
                          columns=["latitude", "longitude", "price", "property_type"] + STRUCT)
    lat = pd.to_numeric(lst.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(lst.longitude, errors="coerce").to_numpy(float)

    pts = gpd.GeoDataFrame(lst.assign(row=np.arange(len(lst))),
                           geometry=gpd.points_from_xy(lon, lat), crs=4326)
    bg = gpd.read_file(CENSUS / f"{city}_census_bg.gpkg", layer="bg")
    joined = gpd.sjoin(pts, bg[DEMO + ["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].sort_values("row")

    S = joined[STRUCT].apply(pd.to_numeric, errors="coerce")
    D = joined[DEMO].apply(pd.to_numeric, errors="coerce")
    cols = [lat, lon,
            S.fillna(S.median()).to_numpy(float), S.isna().to_numpy(float),
            D.fillna(D.median()).to_numpy(float)]
    osm_path = OSM / f"{city}_osm.parquet"
    if osm_path.exists():
        o = pd.read_parquet(osm_path).set_index("row")
        o = o.reindex(range(len(joined)))
        A = o[AMENITY + MICRO].apply(pd.to_numeric, errors="coerce")
        cols += [A.fillna(A.median()).to_numpy(float)]
    X = np.column_stack(cols)
    Y = np.log(pd.to_numeric(joined.price, errors="coerce").to_numpy(float))
    nonres = joined.property_type.astype(str).str.contains(NON_RES, case=False, na=False).to_numpy()
    ok = np.isfinite(Y) & np.isfinite(lat) & np.isfinite(lon)
    return X, Y, nonres, ok


def th(T, X, Y):
    r = run_dml(T.reshape(-1, 1), StandardScaler().fit_transform(X), Y, label="x",
                ci_method="if", n_boot=None, use_ridge=True, seed=42, n_pca=1)
    if r is None:
        return None
    return {"abs_theta": abs(float(r.theta)), "se": float(r.se),
            "covers_zero": bool(r.ci_low < 0 < r.ci_high)}


def main():
    T = pooled_treatment()
    logmap = {c: pd.read_parquet(FULL / f"{c}_reembed.parquet").sort_values("row")["log_len"].to_numpy()
              for c in ALL_12}
    rows = []
    for c in ALL_12:
        X, Y, nonres, ok = confounders(c)
        Tc = T[c]
        L = logmap[c]
        if not (len(Tc) == len(Y) == len(nonres) == len(L)):
            print(f"  {c}: length mismatch T{len(Tc)} Y{len(Y)} nr{len(nonres)} L{len(L)}")
            continue
        base = ok
        res = {"city": c, "n": int(base.sum()), "pct_nonres": float(100 * nonres[base].mean()),
               "n_confounders": X.shape[1]}
        specs = {
            "base": (base, False),
            "plus_length": (base, True),
            "drop_nonres": (base & ~nonres, False),
            "drop_nonres_plus_length": (base & ~nonres, True),
        }
        for name, (m, addlen) in specs.items():
            Xm = X[m]
            if addlen:
                Xm = np.column_stack([Xm, L[m], L[m] ** 2])
            res[name] = th(Tc[m], Xm, Y[m])
        rows.append(res)
        gc.collect()

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n{'city':13s}{'nconf':>6s}{'%nonres':>8s}{'base':>8s}{'+len':>8s}"
          f"{'-nonres':>9s}{'-nonres+len':>13s}")
    print("-" * 65)
    for r in rows:
        def f(k):
            x = r.get(k)
            return "  -  " if not x else f"{x['abs_theta']:.3f}" + ("*" if x['covers_zero'] else " ")
        print(f"{r['city']:13s}{r['n_confounders']:6d}{r['pct_nonres']:7.1f}%"
              f"{f('base'):>8s}{f('plus_length'):>8s}{f('drop_nonres'):>9s}{f('drop_nonres_plus_length'):>13s}")
    print("\n* = 95% CI covers zero.  Treatment = HTML-fixed pooled PC1.")
    print("Confounders = lat/lon + listing beds/baths/sqft/year (own, imputed+indicators) + 11 ACS demographics.")


if __name__ == "__main__":
    main()

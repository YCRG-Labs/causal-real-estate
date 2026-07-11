from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
PANEL = REPO / "results" / "soldprice"
OUT = REPO / "results" / "soldprice" / "confounders"
OUT.mkdir(parents=True, exist_ok=True)
CENSUS = REPO / "results" / "census_bg"
ASSR = REPO / "results" / "assessor"
PROC = REPO / "data" / "processed"

CITIES = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]
DEMO = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60"]
UTM = {"boston": 32619, "sf": 32610, "dc": 32618, "philadelphia": 32618,
       "chicago": 32616, "seattle": 32610, "denver": 32613, "atlanta": 32616,
       "portland": 32610, "phoenix": 32612}
PARCEL_COLS = {"boston": ("bedrooms", "living_area_sqft", "year_built"),
               "sf": ("bedrooms", "property_area_sqft", "year_built"),
               "philadelphia": ("bedrooms", "bldg_area_sqft", "year_built")}


def _project(lon, lat, epsg):
    g = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(epsg)
    return np.column_stack([g.x.values, g.y.values])


def census_join(city, lat, lon):
    bg = gpd.read_file(CENSUS / f"{city}_census_bg.gpkg", layer="bg")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    have = [c for c in DEMO if c in bg.columns]
    j = gpd.sjoin(pts, bg[have + ["GEOID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    miss = j["GEOID"].isna().to_numpy()
    tract = j["GEOID"].astype("string").str.slice(0, 11)
    if miss.any():
        epsg = UTM[city]
        cent = bg.to_crs(epsg).geometry.centroid
        tree = cKDTree(np.column_stack([cent.x.values, cent.y.values]))
        q = _project(lon, lat, epsg)
        _, idx = tree.query(q)
        fill_demo = bg[have].to_numpy(float)[idx]
        fill_tract = bg["GEOID"].astype("string").str.slice(0, 11).to_numpy()[idx]
        demo = j[have].to_numpy(float).copy()
        demo[miss] = fill_demo[miss]
        tr = tract.to_numpy().copy()
        tr[miss] = fill_tract[miss]
    else:
        demo = j[have].to_numpy(float)
        tr = tract.to_numpy()
    full = np.full((len(lat), len(DEMO)), np.nan)
    for k, c in enumerate(DEMO):
        if c in have:
            full[:, k] = demo[:, have.index(c)]
    return full, miss, tr.astype(str)


def assessor_join(city, lat, lon):
    epsg = UTM[city]
    q = _project(lon, lat, epsg)
    p = ASSR / f"{city}_assessor.parquet"
    if p.exists():
        a = pd.read_parquet(p)
        good = np.isfinite(a[["lat", "lon"]].to_numpy(float)).all(1)
        a = a[good].reset_index(drop=True)
        src = _project(a["lon"].to_numpy(), a["lat"].to_numpy(), epsg)
        _, idx = cKDTree(src).query(q)
        return np.column_stack([pd.to_numeric(a["bedrooms"], errors="coerce").to_numpy()[idx],
                                pd.to_numeric(a["bldg_area_sqft"], errors="coerce").to_numpy()[idx],
                                pd.to_numeric(a["year_built"], errors="coerce").to_numpy()[idx]])
    gpkg = PROC / f"{city}_parcels_micro_geo.gpkg"
    if gpkg.exists() and city in PARCEL_COLS:
        bcol, scol, ycol = PARCEL_COLS[city]
        g = gpd.read_file(gpkg, layer=city).to_crs(epsg)
        cc = g.geometry.centroid
        ok = np.isfinite(cc.x.values) & np.isfinite(cc.y.values)
        _, idx = cKDTree(np.column_stack([cc.x.values[ok], cc.y.values[ok]])).query(q)
        src = np.where(ok)[0][idx]

        def col(name):
            return (pd.to_numeric(g[name], errors="coerce").to_numpy()[src]
                    if name in g.columns else np.full(len(lat), np.nan))
        return np.column_stack([col(bcol), col(scol), col(ycol)])
    return None


def build(city):
    d = pd.read_parquet(PANEL / f"{city}_panel.parquet")
    lat = d["latitude"].to_numpy(float)
    lon = d["longitude"].to_numpy(float)
    n = len(d)
    out = pd.DataFrame({"row": np.arange(n), "lat": lat, "lon": lon,
                        "sold_quarter": d["sold_quarter"].to_numpy()})

    demo, demo_miss, tract = census_join(city, lat, lon)
    for k, c in enumerate(DEMO):
        out[f"demo_{c}"] = demo[:, k]
    out["demo_missing"] = demo_miss.astype(int)
    out["tract"] = tract

    lb = pd.to_numeric(d["beds"], errors="coerce").to_numpy()
    lbath = pd.to_numeric(d["baths"], errors="coerce").to_numpy()
    ls = pd.to_numeric(d["sqft"], errors="coerce").to_numpy()
    ly = pd.to_numeric(d["year_built"], errors="coerce").to_numpy()
    out["list_beds"], out["list_baths"] = lb, lbath
    out["list_sqft"], out["list_year"] = ls, ly
    out["list_beds_missing"] = np.isnan(lb).astype(int)
    out["list_sqft_missing"] = np.isnan(ls).astype(int)
    out["list_year_missing"] = np.isnan(ly).astype(int)

    ass = assessor_join(city, lat, lon)
    if ass is not None:
        out["ass_beds"], out["ass_sqft"], out["ass_year"] = ass[:, 0], ass[:, 1], ass[:, 2]
        out["ass_missing"] = np.isnan(ass).any(1).astype(int)
    assessor_arm = ass is not None

    assert len(out) == n
    out.to_parquet(OUT / f"{city}_conf.parquet")
    return {"city": city, "n": n, "assessor_arm": bool(assessor_arm),
            "demo_missing_pct": round(100 * demo_miss.mean(), 1),
            "list_sqft_missing_pct": round(100 * float(np.isnan(ls).mean()), 1),
            "tracts": int(pd.Series(tract).nunique())}


def main():
    rows = []
    print(f"{'city':13s}{'n':>7s}{'assessor':>10s}{'demo_miss%':>12s}{'sqft_miss%':>12s}{'tracts':>8s}")
    print("-" * 62)
    for c in CITIES:
        r = build(c)
        rows.append(r)
        print(f"{c:13s}{r['n']:7d}{str(r['assessor_arm']):>10s}"
              f"{r['demo_missing_pct']:11.1f}%{r['list_sqft_missing_pct']:11.1f}%{r['tracts']:8d}")
    (OUT / "conf_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}/{{city}}_conf.parquet")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
PROC = REPO / "data" / "processed"
CENSUS = REPO / "results" / "census_bg"
OSM = REPO / "results" / "osm_context"
ASSR = REPO / "results" / "assessor"

FALLBACK = ["dc", "chicago", "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
UTM = {"dc": 32618, "chicago": 32616, "seattle": 32610, "denver": 32613,
       "atlanta": 32616, "portland": 32610, "phoenix": 32612, "dallas": 32614}
CENSUS_COLS = ["median_household_income", "median_home_value", "median_gross_rent",
               "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
               "labor_force_participation", "pct_under_25", "pct_over_60"]
AMENITY = ["amenity_food_dining", "amenity_retail", "amenity_services", "amenity_recreation",
           "amenity_transportation", "amenity_education", "amenity_total", "amenity_diversity"]
MICRO = ["dist_park_m", "dist_transit_m", "dist_school_m", "dist_restaurant_m",
         "dist_retail_m", "dist_medical_m"]


def project(lon, lat, epsg):
    lon = np.nan_to_num(np.asarray(lon, float), nan=np.nanmedian(lon))
    lat = np.nan_to_num(np.asarray(lat, float), nan=np.nanmedian(lat))
    g = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(epsg)
    return np.column_stack([g.x.values, g.y.values])


def census_join(city, lat, lon):
    bg = gpd.read_file(CENSUS / f"{city}_census_bg.gpkg", layer="bg")
    have = [c for c in CENSUS_COLS if c in bg.columns]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    j = gpd.sjoin(pts, bg[have + ["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    out = pd.DataFrame(index=np.arange(len(lat)))
    for c in CENSUS_COLS:
        out[c] = j[c].to_numpy(float) if c in have else np.nan
    return out


def assessor_property(city, lat, lon, list_beds, list_sqft, list_year):
    p = ASSR / f"{city}_assessor.parquet"
    if p.exists():
        a = pd.read_parquet(p)
        good = np.isfinite(a[["lat", "lon"]].to_numpy(float)).all(1)
        a = a[good].reset_index(drop=True)
        src = project(a["lon"].to_numpy(), a["lat"].to_numpy(), UTM[city])
        _, idx = cKDTree(src).query(project(lon, lat, UTM[city]))
        return pd.DataFrame({
            "bedrooms": pd.to_numeric(a["bedrooms"], errors="coerce").to_numpy()[idx],
            "bldg_area_sqft": pd.to_numeric(a["bldg_area_sqft"], errors="coerce").to_numpy()[idx],
            "year_built": pd.to_numeric(a["year_built"], errors="coerce").to_numpy()[idx],
            "lot_area_sqft": np.nan})
    return pd.DataFrame({"bedrooms": list_beds, "bldg_area_sqft": list_sqft,
                         "year_built": list_year, "lot_area_sqft": np.nan})


def build(city):
    emb = pd.read_parquet(PROC / f"{city}_embeddings.parquet",
                          columns=["latitude", "longitude", "beds", "sqft", "year_built"])
    lat = pd.to_numeric(emb.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(emb.longitude, errors="coerce").to_numpy(float)
    n = len(emb)
    df = pd.DataFrame({"latitude": lat, "longitude": lon})

    prop = assessor_property(city, lat, lon,
                             pd.to_numeric(emb.beds, errors="coerce").to_numpy(),
                             pd.to_numeric(emb.sqft, errors="coerce").to_numpy(),
                             pd.to_numeric(emb.year_built, errors="coerce").to_numpy())
    for c in ["bedrooms", "bldg_area_sqft", "lot_area_sqft", "year_built"]:
        df[c] = prop[c].to_numpy()

    cen = census_join(city, lat, lon)
    for c in CENSUS_COLS:
        df[c] = cen[c].to_numpy()

    osm_p = OSM / f"{city}_osm.parquet"
    if osm_p.exists():
        o = pd.read_parquet(osm_p).set_index("row")
        for c in AMENITY + MICRO:
            col = np.full(n, np.nan)
            if c in o.columns:
                col[o.index.to_numpy()] = o[c].to_numpy()
            df[c] = col
        osm_ok = True
    else:
        for c in AMENITY + MICRO:
            df[c] = np.nan
        osm_ok = False

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    dst = PROC / f"{city}_parcels_micro_geo.gpkg"
    if dst.exists():
        dst.unlink()
    gdf.to_file(dst, layer=city, driver="GPKG")
    ncols = sum(df[c].notna().any() for c in df.columns if c not in ("latitude", "longitude"))
    return {"city": city, "n": n, "confounder_cols_nonempty": int(ncols),
            "assessor": (ASSR / f"{city}_assessor.parquet").exists(), "osm": osm_ok}


def main():
    cities = sys.argv[1:] or FALLBACK
    print(f"{'city':11s}{'n':>8s}{'conf_cols':>11s}{'assessor':>10s}{'osm':>6s}")
    for c in cities:
        r = build(c)
        print(f"{c:11s}{r['n']:8d}{r['confounder_cols_nonempty']:11d}"
              f"{str(r['assessor']):>10s}{str(r['osm']):>6s}", flush=True)


if __name__ == "__main__":
    main()

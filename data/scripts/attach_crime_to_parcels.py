"""Attach the canonical CRIME block to rebuilt parcel gpkgs from the new
city_downloaders crime parquets.

build_active_confounders.py writes property + census + amenity + micro but no
crime. This step reads data/raw/{city}/{city}_crime.parquet (produced by
city_downloaders/{city}.py, already carrying a canonical crime_category and the
per-city lat/lon fields in pipeline_dicts.CRIME_LATLON_FIELDS), counts incidents
within 500 m of each listing by haversine BallTree, and writes the four canonical
crime columns back into {city}_parcels_micro_geo.gpkg. crime_total is the sum of
the three classified categories, matching build_canonical_confounders.

Cities whose crime feed has no coordinates (phoenix) are skipped and left on the
no-crime spec, which is the disclosed two-tier fallback.

    python data/scripts/attach_crime_to_parcels.py chicago dallas dc denver ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from pipeline_dicts import CRIME_LATLON_FIELDS

PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
BUFFER_M = 500.0
EARTH_R = 6371008.8
CATS = ("violent", "property", "quality_of_life")


def _radius_counts(lat, lon, plat, plon, radius_m):
    n = len(lat)
    if len(plat) == 0:
        return np.zeros(n)
    tree = BallTree(np.radians(np.column_stack([plat, plon])), metric="haversine")
    q = np.radians(np.column_stack([np.nan_to_num(lat), np.nan_to_num(lon)]))
    counts = tree.query_radius(q, r=radius_m / EARTH_R, count_only=True).astype(float)
    counts[np.isnan(lat) | np.isnan(lon)] = 0.0
    return counts


def attach(city: str) -> str:
    gpkg = PROC / f"{city}_parcels_micro_geo.gpkg"
    crime_path = RAW / city / f"{city}_crime.parquet"
    if not gpkg.exists():
        return f"{city}: no parcel gpkg, skip"
    if not crime_path.exists():
        return f"{city}: no crime parquet, skip (stays no-crime)"

    lat_col, lon_col = CRIME_LATLON_FIELDS.get(city, (None, None))
    if lat_col is None or lon_col is None:
        return f"{city}: crime feed has no coordinates, skip (disclosed fallback)"

    crime = pd.read_parquet(crime_path)
    if lat_col not in crime.columns or lon_col not in crime.columns:
        return f"{city}: crime missing {lat_col}/{lon_col}, skip"
    crime[lat_col] = pd.to_numeric(crime[lat_col], errors="coerce")
    crime[lon_col] = pd.to_numeric(crime[lon_col], errors="coerce")
    crime = crime.dropna(subset=[lat_col, lon_col])
    if "crime_category" not in crime.columns:
        return f"{city}: crime has no crime_category, skip"

    gdf = gpd.read_file(gpkg, layer=city)
    cent = gdf.geometry.centroid
    lat = cent.y.to_numpy(float)
    lon = cent.x.to_numpy(float)

    classified = 0
    total = np.zeros(len(gdf))
    for cat in CATS:
        sub = crime[crime["crime_category"] == cat]
        c = _radius_counts(lat, lon, sub[lat_col].to_numpy(float),
                           sub[lon_col].to_numpy(float), BUFFER_M)
        gdf[f"crime_{cat}"] = c
        total += c
        classified += len(sub)
    gdf["crime_total"] = total

    tmp = gpkg.with_suffix(".gpkg.tmp")
    gdf.to_file(tmp, layer=city, driver="GPKG")
    tmp.replace(gpkg)
    return (f"{city}: attached crime to {len(gdf):,} listings "
            f"({len(crime):,} incidents, {classified:,} classified, "
            f"mean total/listing {total.mean():.1f})")


def main():
    cities = sys.argv[1:] or list(CRIME_LATLON_FIELDS.keys())
    for c in cities:
        print(attach(c), flush=True)


if __name__ == "__main__":
    main()

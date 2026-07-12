"""Rebuild data/processed/philadelphia_parcels_micro_geo.gpkg from source.

Philadelphia's parcel file was never on disk (nine of twelve markets are missing
theirs), so the confounder block silently degraded to latitude/longitude and the
paper's actual specification could not be run there. This rebuilds the PROPERTY
and CENSUS blocks — the bulk of the confounding power — from two live sources:

  OPA property assessments (phl.carto.com)  -> geometry + bedrooms, building area,
    lot area, year built, market value, category, census tract
  ACS 2022 5-year block groups (census API) -> the eleven canonical CENSUS columns

Crime, amenity and micro-geo blocks are not rebuilt here; get_features_and_target
uses whatever canonical columns are present and drops the rest, so the result is a
genuine rich-confounder spec (spatial + property + census, ~17 controls) rather
than the lat/lon-only fallback. Column names match canonical_confounders exactly.

    python data/scripts/build_philadelphia_parcels.py
"""
from __future__ import annotations

import io
import sys
import time
import urllib.parse
import urllib.request
import json as _json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from config import CENSUS_YEAR, CENSUS_VARIABLES, CENSUS_DERIVED  # noqa: E402

PROC = REPO / "data" / "processed"
CITY = "philadelphia"
STATE_FIPS, COUNTY_FIPS = "42", "101"
CARTO = "https://phl.carto.com/api/v2/sql"
CENSUS_API = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5"
TIGER = f"https://www2.census.gov/geo/tiger/TIGER{CENSUS_YEAR}/BG/tl_{CENSUS_YEAR}_{STATE_FIPS}_bg.zip"

# OPA residential category codes: 1 single family, 2 multi family, 3 mixed use
OPA_SQL = (
    "SELECT parcel_number, the_geom, number_of_bedrooms, number_of_bathrooms, "
    "total_livable_area, total_area, year_built, market_value, "
    "category_code, census_tract "
    "FROM opa_properties_public "
    "WHERE category_code IN ('1','2','3') AND the_geom IS NOT NULL"
)


def _get_bytes(url, params=None):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            if i == 4:
                raise
            print(f"  retry {i+1} after {e}")
            time.sleep(2 * 2 ** i)


def fetch_opa() -> gpd.GeoDataFrame:
    print("OPA: downloading residential assessments (GeoJSON)...")
    content = _get_bytes(CARTO, params={"q": OPA_SQL, "format": "GeoJSON"})
    gdf = gpd.read_file(io.BytesIO(content))
    gdf = gdf.set_crs(4326, allow_override=True)
    print(f"  {len(gdf)} parcels")
    ren = {"number_of_bedrooms": "bedrooms",
           "total_livable_area": "bldg_area_sqft",
           "total_area": "lot_area_sqft",
           "number_of_bathrooms": "bathrooms",
           "market_value": "median_home_value_parcel"}
    gdf = gdf.rename(columns=ren)
    for c in ["bedrooms", "bldg_area_sqft", "lot_area_sqft", "bathrooms", "year_built"]:
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
    gdf.loc[(gdf.year_built < 1682) | (gdf.year_built > CENSUS_YEAR + 4), "year_built"] = np.nan
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    return gdf


def fetch_census() -> gpd.GeoDataFrame:
    import os
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "no CENSUS_API_KEY in env; the Census API now requires a key for all "
            "ACS5 requests. Get a free one at https://api.census.gov/data/key_signup.html "
            "then: export CENSUS_API_KEY=... and re-run to add the demographic block.")
    print("ACS: fetching block-group variables...")
    variables = ",".join(CENSUS_VARIABLES.keys())
    url = (f"{CENSUS_API}?get={variables}&for=block%20group:*"
           f"&in=state:{STATE_FIPS}&in=county:{COUNTY_FIPS}&in=tract:*&key={key}")
    data = _json.loads(_get_bytes(url))
    df = pd.DataFrame(data[1:], columns=data[0])
    for v in CENSUS_VARIABLES:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df.loc[df[v] < 0, v] = np.nan
    df = df.rename(columns=CENSUS_VARIABLES)
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]

    for name, (num, den) in CENSUS_DERIVED.items():
        if num in df and den in df:
            df[name] = df[num] / df[den].replace(0, np.nan)
    total = df["age_total"].replace(0, np.nan)
    df["pct_under_25"] = df[["age_5_9", "age_18_19"]].sum(axis=1) / total
    df["pct_over_60"] = df[["age_60_61", "age_85_plus"]].sum(axis=1) / total

    print("TIGER: downloading block-group geometries...")
    bg = gpd.read_file(f"zip+{TIGER}")
    bg = bg[bg.COUNTYFP == COUNTY_FIPS][["GEOID", "geometry"]].to_crs(4326)
    keep = ["median_household_income", "median_home_value", "median_gross_rent",
            "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
            "labor_force_participation", "pct_under_25", "pct_over_60", "GEOID"]
    merged = bg.merge(df[keep], on="GEOID", how="inner")
    print(f"  {len(merged)} block groups with census attached")
    return merged


def main() -> None:
    opa = fetch_opa()
    try:
        census = fetch_census()
        print("joining parcels to block groups (spatial within)...")
        joined = gpd.sjoin(opa, census, how="left", predicate="within").drop(
            columns=["index_right"], errors="ignore")
        matched = joined["GEOID"].notna().mean()
        print(f"  census matched for {100*matched:.1f}% of parcels")
    except RuntimeError as e:
        print(f"\n[census skipped] {e}")
        print("Writing PROPERTY + spatial parcel file now; re-run with a key to add census.")
        joined = opa

    out = PROC / f"{CITY}_parcels_micro_geo.gpkg"
    joined.to_file(out, driver="GPKG", layer=CITY)
    print(f"\nwrote {len(joined)} parcels -> {out}")

    from canonical_confounders import PROPERTY, CENSUS
    have = [c for c in PROPERTY + CENSUS if c in joined.columns]
    print(f"canonical confounder columns present ({len(have)}): {have}")
    print("coverage:")
    for c in have:
        print(f"  {c:28s} {100*joined[c].notna().mean():5.1f}% non-null")


if __name__ == "__main__":
    main()

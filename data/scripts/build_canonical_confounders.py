"""City-agnostic confounder attacher for the 12-city expansion.

Reads:
  data/processed/{slug}_listings.parquet     (from scrape + reparse)
  data/raw/{slug}/{slug}_crime.parquet       (from city_downloaders)
  pipeline_dicts.STATE_FIPS                  (Census ACS targeting)
  pipeline_dicts.COUNTY_FIPS                 (Census ACS targeting)
  pipeline_dicts.CITY_BBOXES                 (Overpass query envelope)
  pipeline_dicts.CRIME_LATLON_FIELDS         (per-city crime lat/lon col names)

Writes:
  data/processed/{slug}_listings_enriched.parquet

Attaches the canonical 35-feature confounder vector per listing:
  CENSUS  (11 ACS variables; tract-level join)
  CRIME   (4 categories within 500m KDE buffer)
  AMENITY (8 categories within 500m; Overpass)
  MICRO_GEO (6 nearest-amenity distances in meters; Overpass)

Usage:
  python3 build_canonical_confounders.py --city dc
  python3 build_canonical_confounders.py --cities new9
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from pipeline_dicts import (  # noqa: E402
    STATE_FIPS, COUNTY_FIPS, CITY_BBOXES,
    CRIME_LATLON_FIELDS,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "_confounder_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

BUFFER_M = 500.0
HAVERSINE_R = 6371008.8

CENSUS_VARS = {
    "median_household_income": "B19013_001E",
    "median_home_value":       "B25077_001E",
    "median_gross_rent":       "B25064_001E",
    "pct_bachelors":           ("B15003_022E", "B15003_001E"),
    "pct_white":               ("B02001_002E", "B02001_001E"),
    "pct_black":               ("B02001_003E", "B02001_001E"),
    "pct_asian":               ("B02001_005E", "B02001_001E"),
    "pct_hispanic":            ("B03003_003E", "B03003_001E"),
    "labor_force_participation": ("B23025_002E", "B23025_001E"),
    "pct_under_25":            ("B01001_003E,B01001_004E,B01001_005E,B01001_006E,B01001_007E,B01001_008E,B01001_009E,B01001_010E,B01001_027E,B01001_028E,B01001_029E,B01001_030E,B01001_031E,B01001_032E,B01001_033E,B01001_034E", "B01003_001E"),
    "pct_over_60":             ("B01001_018E,B01001_019E,B01001_020E,B01001_021E,B01001_022E,B01001_023E,B01001_024E,B01001_025E,B01001_042E,B01001_043E,B01001_044E,B01001_045E,B01001_046E,B01001_047E,B01001_048E,B01001_049E", "B01003_001E"),
}

OSM_AMENITY_TAGS = {
    "amenity_food_dining":   ["amenity=restaurant", "amenity=cafe", "amenity=fast_food", "amenity=bar", "amenity=pub"],
    "amenity_retail":        ["shop=*"],
    "amenity_services":      ["amenity=bank", "amenity=post_office", "amenity=pharmacy", "amenity=library"],
    "amenity_recreation":    ["leisure=park", "leisure=fitness_centre", "leisure=playground", "amenity=community_centre"],
    "amenity_transportation": ["highway=bus_stop", "public_transport=stop_position", "railway=station", "amenity=parking"],
    "amenity_education":     ["amenity=school", "amenity=college", "amenity=university", "amenity=kindergarten"],
}

OSM_MICRO_TAGS = {
    "dist_park_m":       ["leisure=park"],
    "dist_transit_m":    ["highway=bus_stop", "railway=station", "public_transport=stop_position"],
    "dist_school_m":     ["amenity=school"],
    "dist_restaurant_m": ["amenity=restaurant"],
    "dist_retail_m":     ["shop=*"],
    "dist_medical_m":    ["amenity=hospital", "amenity=clinic", "amenity=doctors"],
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CENSUS_API_BASE = "https://api.census.gov/data/2022/acs/acs5"


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    lon1 = np.radians(lon1); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * HAVERSINE_R * np.arcsin(np.sqrt(a))


def fetch_census_tract_table(state_fips: str, county_fips: list[str], var_list: list[str], api_key: Optional[str]) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"census_{state_fips}_{'-'.join(county_fips)}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    rows = []
    for cnty in county_fips:
        params = {
            "get": "NAME," + ",".join(var_list),
            "for": "tract:*",
            "in": f"state:{state_fips} county:{cnty}",
        }
        if api_key:
            params["key"] = api_key
        r = requests.get(CENSUS_API_BASE, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        cols = data[0]
        for row in data[1:]:
            rows.append(dict(zip(cols, row)))
    df = pd.DataFrame(rows)
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df.to_parquet(cache_path, index=False)
    return df


def census_table_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw[["geoid"]].copy()
    for name, spec in CENSUS_VARS.items():
        if isinstance(spec, str):
            out[name] = pd.to_numeric(raw[spec], errors="coerce")
        else:
            num_spec, denom_spec = spec
            num_cols = num_spec.split(",")
            num = sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in num_cols)
            denom = pd.to_numeric(raw[denom_spec], errors="coerce").replace(0, np.nan)
            out[name] = num / denom
    return out


def attach_census(listings: pd.DataFrame, slug: str, api_key: Optional[str]) -> pd.DataFrame:
    state_fips = STATE_FIPS[slug]
    cnty_fips = COUNTY_FIPS[slug]
    all_vars = set()
    for spec in CENSUS_VARS.values():
        if isinstance(spec, str):
            all_vars.add(spec)
        else:
            all_vars.update(spec[0].split(","))
            all_vars.add(spec[1])
    raw = fetch_census_tract_table(state_fips, cnty_fips, sorted(all_vars), api_key)
    can = census_table_to_canonical(raw)
    try:
        from census import Census  # noqa: F401
    except ImportError:
        pass
    tract_geoid = _listings_to_tract_geoid(listings, state_fips, cnty_fips)
    listings = listings.copy()
    listings["geoid"] = tract_geoid
    return listings.merge(can, on="geoid", how="left")


def _listings_to_tract_geoid(listings: pd.DataFrame, state_fips: str, cnty_fips: list[str]) -> pd.Series:
    geoids = []
    for _, row in listings[["lat_centroid", "lon_centroid"]].iterrows():
        lat = row["lat_centroid"]; lon = row["lon_centroid"]
        if pd.isna(lat) or pd.isna(lon):
            geoids.append(None); continue
        params = {
            "x": lon, "y": lat,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "layers": "Census Tracts",
            "format": "json",
        }
        try:
            r = requests.get(
                "https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
                params=params, timeout=30,
            )
            r.raise_for_status()
            tracts = r.json().get("result", {}).get("geographies", {}).get("Census Tracts", [])
            if tracts:
                t = tracts[0]
                geoids.append(t.get("GEOID"))
                continue
        except Exception:
            pass
        geoids.append(None)
    return pd.Series(geoids, index=listings.index)


def attach_crime(listings: pd.DataFrame, slug: str) -> pd.DataFrame:
    crime_path = RAW_DIR / slug / f"{slug}_crime.parquet"
    listings = listings.copy()
    for cat in ("violent", "property", "quality_of_life", "total"):
        listings[f"crime_{cat}"] = 0.0
    if not crime_path.exists():
        print(f"  [{slug}] no crime parquet; crime features = 0", file=sys.stderr)
        return listings
    crime = pd.read_parquet(crime_path)
    lat_col, lon_col = CRIME_LATLON_FIELDS.get(slug, (None, None))
    if lat_col is None or lon_col is None or lat_col not in crime.columns or lon_col not in crime.columns:
        print(f"  [{slug}] crime has no lat/lon ({lat_col=}, {lon_col=}); crime features = 0", file=sys.stderr)
        return listings
    crime[lat_col] = pd.to_numeric(crime[lat_col], errors="coerce")
    crime[lon_col] = pd.to_numeric(crime[lon_col], errors="coerce")
    crime = crime.dropna(subset=[lat_col, lon_col])
    if "crime_category" not in crime.columns:
        crime["crime_category"] = None
    print(f"  [{slug}] {len(crime):,} crime incidents; computing 500m buffer counts...")
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    clats = crime[lat_col].to_numpy()
    clons = crime[lon_col].to_numpy()
    ccats = crime["crime_category"].to_numpy()
    cat_indices = {cat: ccats == cat for cat in ("violent", "property", "quality_of_life")}
    n = len(listings)
    counts = {cat: np.zeros(n) for cat in ("violent", "property", "quality_of_life")}
    for i in range(n):
        if np.isnan(lats[i]) or np.isnan(lons[i]):
            continue
        d = haversine_m(lats[i], lons[i], clats, clons)
        within = d < BUFFER_M
        for cat, mask in cat_indices.items():
            counts[cat][i] = float((within & mask).sum())
    for cat in ("violent", "property", "quality_of_life"):
        listings[f"crime_{cat}"] = counts[cat]
    listings["crime_total"] = listings["crime_violent"] + listings["crime_property"] + listings["crime_quality_of_life"]
    return listings


def overpass_query(slug: str, query_body: str) -> dict:
    cache_path = CACHE_DIR / f"overpass_{slug}_{hash(query_body) & 0xffffffff:x}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path).to_dict(orient="list")
    data = f"[out:json][timeout:120];{query_body}out center;"
    for attempt in range(3):
        try:
            r = requests.post(OVERPASS_URL, data={"data": data}, timeout=180)
            r.raise_for_status()
            payload = r.json()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"    overpass retry {attempt+1}/3 after {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    rows = []
    for el in payload.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        rows.append({"id": el.get("id"), "lat": float(lat), "lon": float(lon)})
    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    return df.to_dict(orient="list")


def amenity_filter_clause(tag_spec: str) -> str:
    key, val = tag_spec.split("=", 1)
    if val == "*":
        return f'[{key}]'
    return f'[{key}={val!r}]'


def attach_amenities(listings: pd.DataFrame, slug: str) -> pd.DataFrame:
    listings = listings.copy()
    bbox = CITY_BBOXES[slug]
    bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
    counts = {cat: np.zeros(len(listings)) for cat in OSM_AMENITY_TAGS}
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    for cat, tags in OSM_AMENITY_TAGS.items():
        query_parts = []
        for tag in tags:
            clause = amenity_filter_clause(tag)
            query_parts.append(f'node{clause}{bbox_str};way{clause}{bbox_str};')
        query_body = "(" + "".join(query_parts) + ");"
        print(f"  [{slug}] overpass: {cat}...")
        poi = overpass_query(slug + "_" + cat, query_body)
        plats = np.array(poi.get("lat", []))
        plons = np.array(poi.get("lon", []))
        if len(plats) == 0:
            continue
        for i in range(len(listings)):
            if np.isnan(lats[i]) or np.isnan(lons[i]):
                continue
            d = haversine_m(lats[i], lons[i], plats, plons)
            counts[cat][i] = float((d < BUFFER_M).sum())
        time.sleep(1.0)
    for cat in OSM_AMENITY_TAGS:
        listings[cat] = counts[cat]
    listings["amenity_total"] = sum(listings[cat] for cat in OSM_AMENITY_TAGS)
    nonzero_categories = sum((listings[cat] > 0).astype(int) for cat in OSM_AMENITY_TAGS)
    listings["amenity_diversity"] = nonzero_categories / float(len(OSM_AMENITY_TAGS))
    return listings


def attach_micro_geo(listings: pd.DataFrame, slug: str) -> pd.DataFrame:
    listings = listings.copy()
    bbox = CITY_BBOXES[slug]
    bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    for col, tags in OSM_MICRO_TAGS.items():
        query_parts = []
        for tag in tags:
            clause = amenity_filter_clause(tag)
            query_parts.append(f'node{clause}{bbox_str};way{clause}{bbox_str};')
        query_body = "(" + "".join(query_parts) + ");"
        print(f"  [{slug}] overpass: {col}...")
        poi = overpass_query(slug + "_micro_" + col, query_body)
        plats = np.array(poi.get("lat", []))
        plons = np.array(poi.get("lon", []))
        dists = np.full(len(listings), np.nan)
        if len(plats) > 0:
            for i in range(len(listings)):
                if np.isnan(lats[i]) or np.isnan(lons[i]):
                    continue
                d = haversine_m(lats[i], lons[i], plats, plons)
                dists[i] = float(d.min())
        listings[col] = dists
        time.sleep(1.0)
    return listings


def build_one(slug: str, api_key: Optional[str]) -> Path:
    listings_path = PROCESSED_DIR / f"{slug}_listings.parquet"
    if not listings_path.exists():
        raise FileNotFoundError(f"no listings parquet at {listings_path}")
    listings = pd.read_parquet(listings_path)
    listings = listings[listings["lat_centroid"].notna() & listings["lon_centroid"].notna()].copy()
    print(f"[{slug}] {len(listings)} listings with lat/lon")
    print(f"[{slug}] attaching census...")
    listings = attach_census(listings, slug, api_key)
    print(f"[{slug}] attaching crime...")
    listings = attach_crime(listings, slug)
    print(f"[{slug}] attaching amenities...")
    listings = attach_amenities(listings, slug)
    print(f"[{slug}] attaching micro-geography...")
    listings = attach_micro_geo(listings, slug)
    out_path = PROCESSED_DIR / f"{slug}_listings_enriched.parquet"
    listings.to_parquet(out_path, index=False)
    print(f"[{slug}] wrote {len(listings)} rows to {out_path}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cities", type=str, default="new9", help="new9 | all | comma-separated slugs")
    p.add_argument("--city", type=str, default=None, help="single city alias")
    args = p.parse_args()
    if args.city:
        slugs = [args.city]
    elif args.cities == "new9":
        slugs = ["dc", "philadelphia", "chicago", "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
    elif args.cities == "all":
        slugs = list(STATE_FIPS.keys())
    else:
        slugs = [s.strip() for s in args.cities.split(",") if s.strip()]
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("WARNING: CENSUS_API_KEY not set; Census API will rate-limit at 500/day", file=sys.stderr)
    t0 = time.time()
    for slug in slugs:
        try:
            build_one(slug, api_key)
        except Exception as e:
            print(f"[{slug}] FAILED: {e}", file=sys.stderr)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

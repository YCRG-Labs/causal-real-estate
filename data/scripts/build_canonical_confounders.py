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
import asyncio
import functools
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

print = functools.partial(print, flush=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "YCRG-Labs JBES-2026 research (jacobcrainic@icloud.com)"})

try:
    import geopandas as _gpd
    _gpd.options.io_engine = "pyogrio"
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from pipeline_dicts import (
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
        r = SESSION.get(CENSUS_API_BASE, params=params, timeout=120)
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


def _load_tract_polygons(state_fips: str) -> "object":
    """Download (cached) the TIGER tract shapefile for a state and return a GeoDataFrame."""
    import geopandas as gpd
    tiger_url = f"https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_{state_fips}_tract.zip"
    cache_path = CACHE_DIR / f"tiger_{state_fips}_tract.parquet"
    if cache_path.exists():
        return gpd.read_parquet(cache_path)
    print(f"  downloading TIGER tracts for state FIPS {state_fips}...")
    gdf = gpd.read_file(tiger_url)
    gdf = gdf[["GEOID", "geometry"]].to_crs(4326)
    gdf.to_parquet(cache_path, index=False)
    return gdf


def _listings_to_tract_geoid(listings: pd.DataFrame, state_fips: str, cnty_fips: list[str]) -> pd.Series:
    import geopandas as gpd
    from shapely.geometry import Point
    tracts = _load_tract_polygons(state_fips)
    if cnty_fips:
        tracts = tracts[tracts["GEOID"].str[:5].isin([state_fips + c for c in cnty_fips])].copy()
    pts = gpd.GeoDataFrame(
        listings.index.to_frame(name="_idx"),
        geometry=[Point(lon, lat) if not (pd.isna(lon) or pd.isna(lat)) else None
                  for lat, lon in zip(listings["lat_centroid"], listings["lon_centroid"])],
        crs=4326,
    )
    joined = gpd.sjoin(pts, tracts, how="left", predicate="within")
    out = pd.Series(joined["GEOID"].values, index=listings.index, name="geoid")
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
    tract_geoid = _listings_to_tract_geoid(listings, state_fips, cnty_fips)
    listings = listings.copy()
    listings["geoid"] = tract_geoid
    return listings.merge(can, on="geoid", how="left")


def _balltree_radius_indices(lats: np.ndarray, lons: np.ndarray, plats: np.ndarray, plons: np.ndarray, radius_m: float) -> list[np.ndarray]:
    """Return per-listing array of POI indices within radius_m. Uses BallTree(haversine)."""
    from sklearn.neighbors import BallTree
    if len(plats) == 0:
        return [np.array([], dtype=int) for _ in range(len(lats))]
    poi_rad = np.radians(np.column_stack([plats, plons]))
    listing_rad = np.radians(np.column_stack([np.nan_to_num(lats), np.nan_to_num(lons)]))
    tree = BallTree(poi_rad, metric="haversine")
    radius_rad = radius_m / HAVERSINE_R
    indices = tree.query_radius(listing_rad, r=radius_rad)
    nan_mask = np.isnan(lats) | np.isnan(lons)
    for i in np.where(nan_mask)[0]:
        indices[i] = np.array([], dtype=int)
    return list(indices)


def _balltree_nearest_distance(lats: np.ndarray, lons: np.ndarray, plats: np.ndarray, plons: np.ndarray) -> np.ndarray:
    """Return per-listing nearest-POI distance in meters via BallTree(haversine)."""
    from sklearn.neighbors import BallTree
    if len(plats) == 0:
        return np.full(len(lats), np.nan)
    poi_rad = np.radians(np.column_stack([plats, plons]))
    listing_rad = np.radians(np.column_stack([np.nan_to_num(lats), np.nan_to_num(lons)]))
    tree = BallTree(poi_rad, metric="haversine")
    d_rad, _ = tree.query(listing_rad, k=1)
    d_m = d_rad[:, 0] * HAVERSINE_R
    nan_mask = np.isnan(lats) | np.isnan(lons)
    d_m[nan_mask] = np.nan
    return d_m


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
    crime = crime.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    if "crime_category" not in crime.columns:
        crime["crime_category"] = None
    print(f"  [{slug}] {len(crime):,} crime incidents; balltree 500m buffer...")
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    clats = crime[lat_col].to_numpy()
    clons = crime[lon_col].to_numpy()
    ccats = crime["crime_category"].to_numpy()
    per_listing_indices = _balltree_radius_indices(lats, lons, clats, clons, BUFFER_M)
    counts = {cat: np.zeros(len(listings)) for cat in ("violent", "property", "quality_of_life")}
    for i, idx in enumerate(per_listing_indices):
        if len(idx) == 0:
            continue
        sub = ccats[idx]
        for cat in counts:
            counts[cat][i] = float((sub == cat).sum())
    for cat in counts:
        listings[f"crime_{cat}"] = counts[cat]
    listings["crime_total"] = listings["crime_violent"] + listings["crime_property"] + listings["crime_quality_of_life"]
    return listings


OVERPASS_HEADERS = {
    "User-Agent": "YCRG-Labs JBES-2026 research (jacobcrainic@icloud.com)",
    "Accept": "application/json",
}


def _overpass_cache_path(slug: str, query_body: str) -> Path:
    return CACHE_DIR / f"overpass_{slug}_{hash(query_body) & 0xffffffff:x}.parquet"


def _overpass_parse(payload: dict) -> pd.DataFrame:
    rows = []
    for el in payload.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        rows.append({"id": el.get("id"), "lat": float(lat), "lon": float(lon)})
    return pd.DataFrame(rows)


def overpass_query(slug: str, query_body: str) -> dict:
    cache_path = _overpass_cache_path(slug, query_body)
    if cache_path.exists():
        return pd.read_parquet(cache_path).to_dict(orient="list")
    data = f"[out:json][timeout:120];{query_body}out center;"
    for attempt in range(3):
        try:
            r = SESSION.post(OVERPASS_URL, data={"data": data}, headers=OVERPASS_HEADERS, timeout=180)
            r.raise_for_status()
            payload = r.json()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"    overpass retry {attempt+1}/3 after {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    df = _overpass_parse(payload)
    df.to_parquet(cache_path, index=False)
    return df.to_dict(orient="list")


async def _overpass_async_one(session, sem, slug, qkey, query_body):
    cache_path = _overpass_cache_path(slug, query_body)
    if cache_path.exists():
        return qkey, pd.read_parquet(cache_path)
    data = f"[out:json][timeout:120];{query_body}out center;"
    async with sem:
        last_err = None
        for attempt in range(3):
            try:
                async with session.post(OVERPASS_URL, data={"data": data}, headers=OVERPASS_HEADERS, timeout=180) as r:
                    r.raise_for_status()
                    payload = await r.json()
                df = _overpass_parse(payload)
                df.to_parquet(cache_path, index=False)
                return qkey, df
            except Exception as e:
                last_err = e
                wait = 5 * (attempt + 1)
                print(f"    overpass[{qkey}] retry {attempt+1}/3 after {wait}s: {e}", file=sys.stderr)
                await asyncio.sleep(wait)
        raise last_err


async def _overpass_async_many(slug: str, queries: dict[str, str]) -> dict[str, pd.DataFrame]:
    try:
        import aiohttp
    except ImportError:
        out = {}
        for qkey, qbody in queries.items():
            out[qkey] = pd.DataFrame(overpass_query(f"{slug}_{qkey}", qbody))
        return out
    sem = asyncio.Semaphore(2)
    async with aiohttp.ClientSession() as session:
        tasks = [_overpass_async_one(session, sem, slug, k, q) for k, q in queries.items()]
        results = await asyncio.gather(*tasks)
    return dict(results)


def overpass_batch(slug: str, queries: dict[str, str]) -> dict[str, pd.DataFrame]:
    return asyncio.run(_overpass_async_many(slug, queries))


def amenity_filter_clause(tag_spec: str) -> str:
    key, val = tag_spec.split("=", 1)
    if val == "*":
        return f'[{key}]'
    return f'[{key}={val!r}]'


def _build_overpass_queries(slug: str, bbox: tuple, tag_dict: dict, prefix: str = "") -> dict[str, str]:
    bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
    queries = {}
    for cat, tags in tag_dict.items():
        parts = []
        for tag in tags:
            clause = amenity_filter_clause(tag)
            parts.append(f'node{clause}{bbox_str};way{clause}{bbox_str};')
        queries[prefix + cat] = "(" + "".join(parts) + ");"
    return queries


def attach_amenities(listings: pd.DataFrame, slug: str) -> pd.DataFrame:
    listings = listings.copy()
    bbox = CITY_BBOXES[slug]
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    print(f"  [{slug}] overpass batch: {len(OSM_AMENITY_TAGS)} amenity categories async (sem=2)...")
    queries = _build_overpass_queries(slug, bbox, OSM_AMENITY_TAGS)
    poi_dfs = overpass_batch(slug + "_amenity", queries)
    counts = {}
    for cat in OSM_AMENITY_TAGS:
        df = poi_dfs.get(cat, pd.DataFrame())
        if df.empty:
            counts[cat] = np.zeros(len(listings))
            continue
        plats = df["lat"].to_numpy(dtype=float)
        plons = df["lon"].to_numpy(dtype=float)
        per_listing = _balltree_radius_indices(lats, lons, plats, plons, BUFFER_M)
        counts[cat] = np.array([len(ix) for ix in per_listing], dtype=float)
    for cat in OSM_AMENITY_TAGS:
        listings[cat] = counts[cat]
    listings["amenity_total"] = sum(listings[cat] for cat in OSM_AMENITY_TAGS)
    nonzero_categories = sum((listings[cat] > 0).astype(int) for cat in OSM_AMENITY_TAGS)
    listings["amenity_diversity"] = nonzero_categories / float(len(OSM_AMENITY_TAGS))
    return listings


def attach_micro_geo(listings: pd.DataFrame, slug: str) -> pd.DataFrame:
    listings = listings.copy()
    bbox = CITY_BBOXES[slug]
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    print(f"  [{slug}] overpass batch: {len(OSM_MICRO_TAGS)} micro-geo categories async (sem=2)...")
    queries = _build_overpass_queries(slug, bbox, OSM_MICRO_TAGS, prefix="micro_")
    poi_dfs = overpass_batch(slug + "_micro", queries)
    for col in OSM_MICRO_TAGS:
        df = poi_dfs.get("micro_" + col, pd.DataFrame())
        if df.empty:
            listings[col] = np.nan
            continue
        plats = df["lat"].to_numpy(dtype=float)
        plons = df["lon"].to_numpy(dtype=float)
        listings[col] = _balltree_nearest_distance(lats, lons, plats, plons)
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

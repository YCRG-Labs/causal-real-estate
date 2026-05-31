"""Replication-packet variant of build_canonical_confounders.py.

Differs from the online-Overpass version in one critical way: OSM data is
served from a frozen per-state Geofabrik PBF snapshot rather than the live
Overpass API. The replication packet ships the PBF hashes alongside the code
so a future researcher can reproduce our amenity counts and micro-geographic
distances bit-identically, which is not possible against Overpass (its
underlying database is updated daily and there is no addressable snapshot).

PBF source: https://download.geofabrik.de/north-america/us/{state}-latest.osm.pbf

Usage:
    python3 build_canonical_confounders_pyrosm.py --city dc
    python3 build_canonical_confounders_pyrosm.py --cities new9 --pbf-dir /path/to/cache

The script reads listings + crime parquets exactly like the v1 version. The
only difference is the source of OSM POIs.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
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

from build_canonical_confounders import (  # noqa: E402
    CACHE_DIR, PROCESSED_DIR, RAW_DIR, BUFFER_M, HAVERSINE_R,
    OSM_AMENITY_TAGS, OSM_MICRO_TAGS,
    attach_census, attach_crime,
    _balltree_radius_indices, _balltree_nearest_distance,
)
from pipeline_dicts import STATE_FIPS, CITY_BBOXES  # noqa: E402

print = functools.partial(print, flush=True)

STATE_SLUG_FOR_FIPS = {
    "11": "district-of-columbia", "42": "pennsylvania", "17": "illinois",
    "53": "washington", "08": "colorado", "13": "georgia", "41": "oregon",
    "04": "arizona", "48": "texas", "25": "massachusetts", "36": "new-york",
    "06": "california",
}
GEOFABRIK_BASE = "https://download.geofabrik.de/north-america/us"
PBF_DIR_DEFAULT = REPO_ROOT / "data" / "_geofabrik"


def state_pbf_path(state_fips: str, pbf_dir: Path) -> Path:
    state_slug = STATE_SLUG_FOR_FIPS.get(state_fips)
    if state_slug is None:
        raise KeyError(f"no geofabrik slug for state FIPS {state_fips}")
    return pbf_dir / f"{state_slug}-latest.osm.pbf"


def download_pbf(state_fips: str, pbf_dir: Path) -> Path:
    pbf_dir.mkdir(parents=True, exist_ok=True)
    path = state_pbf_path(state_fips, pbf_dir)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    state_slug = STATE_SLUG_FOR_FIPS[state_fips]
    url = f"{GEOFABRIK_BASE}/{state_slug}-latest.osm.pbf"
    print(f"  geofabrik: downloading {url}")
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def pbf_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pois_from_pbf(pbf_path: Path, tag_dict: dict, bbox: tuple) -> dict[str, pd.DataFrame]:
    try:
        from pyrosm import OSM
    except ImportError as e:
        raise RuntimeError(
            "pyrosm not installed; run `pip install --user pyrosm` to use the replication packet path"
        ) from e
    osm = OSM(str(pbf_path), bounding_box=[bbox[1], bbox[0], bbox[3], bbox[2]])
    out: dict[str, pd.DataFrame] = {}
    for cat, tag_specs in tag_dict.items():
        keys: dict[str, list[str]] = {}
        for spec in tag_specs:
            k, v = spec.split("=", 1)
            keys.setdefault(k, []).append(v)
        records = []
        for k, vs in keys.items():
            filt = {k: True} if vs == ["*"] else {k: vs}
            try:
                pois = osm.get_pois(custom_filter=filt)
            except Exception as e:
                print(f"    pyrosm filter {filt} failed: {e}", file=sys.stderr)
                continue
            if pois is None or len(pois) == 0:
                continue
            geom = pois.geometry
            cent = geom.representative_point()
            for lon, lat in zip(cent.x, cent.y):
                records.append({"lat": float(lat), "lon": float(lon)})
        out[cat] = pd.DataFrame(records)
    return out


def attach_amenities_pyrosm(listings, slug, pbf_dir):
    listings = listings.copy()
    bbox = CITY_BBOXES[slug]
    state_fips = STATE_FIPS[slug]
    pbf_path = download_pbf(state_fips, pbf_dir)
    pois = load_pois_from_pbf(pbf_path, OSM_AMENITY_TAGS, bbox)
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    for cat in OSM_AMENITY_TAGS:
        df = pois.get(cat, pd.DataFrame())
        if df.empty:
            listings[cat] = 0.0
            continue
        per_listing = _balltree_radius_indices(
            lats, lons, df["lat"].to_numpy(), df["lon"].to_numpy(), BUFFER_M,
        )
        listings[cat] = np.array([len(ix) for ix in per_listing], dtype=float)
    listings["amenity_total"] = sum(listings[cat] for cat in OSM_AMENITY_TAGS)
    nonzero = sum((listings[cat] > 0).astype(int) for cat in OSM_AMENITY_TAGS)
    listings["amenity_diversity"] = nonzero / float(len(OSM_AMENITY_TAGS))
    return listings


def attach_micro_geo_pyrosm(listings, slug, pbf_dir):
    listings = listings.copy()
    bbox = CITY_BBOXES[slug]
    state_fips = STATE_FIPS[slug]
    pbf_path = download_pbf(state_fips, pbf_dir)
    pois = load_pois_from_pbf(pbf_path, OSM_MICRO_TAGS, bbox)
    lats = listings["lat_centroid"].to_numpy()
    lons = listings["lon_centroid"].to_numpy()
    for col in OSM_MICRO_TAGS:
        df = pois.get(col, pd.DataFrame())
        if df.empty:
            listings[col] = np.nan
            continue
        listings[col] = _balltree_nearest_distance(
            lats, lons, df["lat"].to_numpy(), df["lon"].to_numpy(),
        )
    return listings


def build_one_pyrosm(slug, api_key, pbf_dir):
    listings_path = PROCESSED_DIR / f"{slug}_listings.parquet"
    listings = pd.read_parquet(listings_path)
    listings = listings[listings["lat_centroid"].notna() & listings["lon_centroid"].notna()].copy()
    print(f"[{slug}] {len(listings)} listings with lat/lon (pyrosm/geofabrik path)")
    listings = attach_census(listings, slug, api_key)
    listings = attach_crime(listings, slug)
    listings = attach_amenities_pyrosm(listings, slug, pbf_dir)
    listings = attach_micro_geo_pyrosm(listings, slug, pbf_dir)
    state_fips = STATE_FIPS[slug]
    pbf_path = state_pbf_path(state_fips, pbf_dir)
    listings["_pbf_sha256"] = pbf_sha256(pbf_path) if pbf_path.exists() else ""
    out_path = PROCESSED_DIR / f"{slug}_listings_enriched_pyrosm.parquet"
    listings.to_parquet(out_path, index=False)
    print(f"[{slug}] wrote {len(listings)} rows to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default="new9")
    p.add_argument("--city", default=None)
    p.add_argument("--pbf-dir", default=str(PBF_DIR_DEFAULT))
    args = p.parse_args()
    pbf_dir = Path(args.pbf_dir)
    if args.city:
        slugs = [args.city]
    elif args.cities == "new9":
        slugs = ["dc", "philadelphia", "chicago", "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
    elif args.cities == "all":
        slugs = list(STATE_FIPS.keys())
    else:
        slugs = [s.strip() for s in args.cities.split(",") if s.strip()]
    api_key = os.environ.get("CENSUS_API_KEY")
    t0 = time.time()
    for slug in slugs:
        try:
            build_one_pyrosm(slug, api_key, pbf_dir)
        except Exception as e:
            print(f"[{slug}] FAILED: {e}", file=sys.stderr)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

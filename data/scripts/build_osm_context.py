"""Amenity + micro-geo confounders from OpenStreetMap, for all twelve markets.

For each city: compute a bbox from the listing coordinates, pull the six amenity
categories from Overpass, project to the local UTM, then for every listing count
POIs within 500 m (per category, total, Shannon diversity) and measure the
distance to the nearest POI of each micro-geo category. Writes
results/osm_context/<city>_osm.parquet keyed by the listing's positional row.

    python data/scripts/build_osm_context.py            # all 12, skip done
    python data/scripts/build_osm_context.py sf boston  # subset
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from config import OSM_AMENITY_TAGS, AMENITY_RADIUS_M  # noqa: E402

PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "osm_context"
OUT.mkdir(parents=True, exist_ok=True)
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
MICRO_GEO = {"park": "recreation", "transit": "transportation", "school": "education",
             "restaurant": "food_dining", "retail": "retail", "medical": "services"}
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]


def overpass(bbox, tag_key, tag_values):
    s, w, n, e = bbox
    regex = "|".join(tag_values)
    q = (f"[out:json][timeout:180];("
         f'node["{tag_key}"~"^({regex})$"]({s},{w},{n},{e});'
         f'way["{tag_key}"~"^({regex})$"]({s},{w},{n},{e});'
         f");out center;")
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for ep in ENDPOINTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(ep, data=data,
                                             headers={"User-Agent": "causal-re-research/1.0"})
                with urllib.request.urlopen(req, timeout=200) as r:
                    return json.loads(r.read())["elements"]
            except Exception as ex:  # noqa: BLE001
                last = ex
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"overpass failed for {tag_key}={tag_values}: {last}")


def pois_for_city(bbox):
    rows = []
    for category, tagdict in OSM_AMENITY_TAGS.items():
        for tag_key, tag_values in tagdict.items():
            els = overpass(bbox, tag_key, tag_values)
            for el in els:
                lat = el.get("lat") or (el.get("center") or {}).get("lat")
                lon = el.get("lon") or (el.get("center") or {}).get("lon")
                if lat is not None and lon is not None:
                    rows.append((category, float(lat), float(lon)))
            print(f"    {category}/{tag_key}: {len(els)}", flush=True)
            time.sleep(1)
    return pd.DataFrame(rows, columns=["category", "lat", "lon"])


def shannon(counts):
    c = np.asarray([x for x in counts if x > 0], float)
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def build(city):
    dst = OUT / f"{city}_osm.parquet"
    if dst.exists():
        print(f"  {city}: already done, skip")
        return
    lst = pd.read_parquet(PROC / f"{city}_embeddings.parquet", columns=["latitude", "longitude"])
    lat = pd.to_numeric(lst.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(lst.longitude, errors="coerce").to_numpy(float)
    good = np.isfinite(lat) & np.isfinite(lon)
    pad = 0.02
    bbox = (lat[good].min() - pad, lon[good].min() - pad, lat[good].max() + pad, lon[good].max() + pad)
    print(f"  {city}: bbox {tuple(round(x,3) for x in bbox)}", flush=True)

    poi = pois_for_city(bbox)
    print(f"  {city}: {len(poi)} POIs total", flush=True)

    listings = gpd.GeoSeries(gpd.points_from_xy(lon[good], lat[good]), crs=4326)
    utm = listings.estimate_utm_crs()
    lpts = np.column_stack([listings.to_crs(utm).x.values, listings.to_crs(utm).y.values])
    poig = gpd.GeoSeries(gpd.points_from_xy(poi.lon, poi.lat), crs=4326).to_crs(utm)
    poi_xy = np.column_stack([poig.x.values, poig.y.values])

    cats = list(OSM_AMENITY_TAGS.keys())
    out = {f"amenity_{c}": np.zeros(good.sum(), int) for c in cats}
    out["amenity_total"] = np.zeros(good.sum(), int)
    out["amenity_diversity"] = np.zeros(good.sum(), float)
    trees = {c: cKDTree(poi_xy[poi.category.values == c]) if (poi.category.values == c).any() else None
             for c in cats}

    for i, p in enumerate(lpts):
        percat = []
        for c in cats:
            t = trees[c]
            n = len(t.query_ball_point(p, AMENITY_RADIUS_M)) if t is not None else 0
            out[f"amenity_{c}"][i] = n
            percat.append(n)
        out["amenity_total"][i] = sum(percat)
        out["amenity_diversity"][i] = shannon(percat)

    for feat, cat in MICRO_GEO.items():
        t = trees.get(cat)
        if t is None:
            out[f"dist_{feat}_m"] = np.full(good.sum(), np.nan)
        else:
            out[f"dist_{feat}_m"], _ = t.query(lpts)

    df = pd.DataFrame(out)
    df.insert(0, "row", np.where(good)[0])
    df.to_parquet(dst)
    print(f"  {city}: wrote {len(df)} rows -> {dst}", flush=True)


def main():
    for c in (sys.argv[1:] or ALL_12):
        try:
            build(c)
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

"""Portland/Multnomah county-assessor attributes -> results/assessor/portland_assessor.parquet.

Source: Multnomah County Taxlot Parcels ArcGIS FeatureServer (county assessor roll).
Structural fields present: ACTYEARBUILT (year built), MAIN_SQFT/MAINAREA (main building
area). No bedroom count in the taxlot layer -> bedrooms left NaN.
Output columns: lat, lon, bedrooms, bldg_area_sqft, year_built (all float, WGS84).
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "assessor" / "portland_assessor.parquet"
BASE = ("https://services5.arcgis.com/x7DNZL1YqNQVNykA/arcgis/rest/services/"
        "Multnomah_County_Taxlot_Parcels/FeatureServer/0/query")
FIELDS = "MAPTAXLOT,SITUSCITY,ACTYEARBUILT,MAIN_SQFT,MAINAREA"
UA = "YCRG-Labs JBES-2026 research (jacobcrainic@icloud.com)"
PAGE = 2000


def _get(url, params):
    full = url + "?" + urllib.parse.urlencode(params)
    for i in range(5):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == 4:
                raise
            print(f"  retry {i+1} after {e}")
            time.sleep(2 * 2 ** i)


def fetch() -> gpd.GeoDataFrame:
    frames = []
    offset = 0
    while True:
        params = {
            "where": "SITUSCITY='PORTLAND'",
            "outFields": FIELDS,
            "returnGeometry": "true",
            "outSR": 4326,
            "resultRecordCount": PAGE,
            "resultOffset": offset,
            "f": "geojson",
        }
        content = _get(BASE, params)
        gdf = gpd.read_file(io.BytesIO(content))
        n = len(gdf)
        if n == 0:
            break
        frames.append(gdf)
        print(f"  offset {offset}: {n} features")
        if n < PAGE:
            break
        offset += PAGE
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)
    return gdf


def main() -> None:
    gdf = fetch()
    print(f"total taxlots in Portland: {len(gdf)}")
    gdf = gdf.dropna(subset=["geometry"])
    gdf = gdf[~gdf.geometry.is_empty]
    # dedupe by parcel id
    gdf = gdf.drop_duplicates(subset=["MAPTAXLOT"], keep="first")

    rep = gdf.geometry.representative_point()
    out = pd.DataFrame({
        "lat": rep.y.values,
        "lon": rep.x.values,
        "bedrooms": np.nan,
        "bldg_area_sqft": pd.to_numeric(gdf["MAIN_SQFT"], errors="coerce").values,
        "year_built": pd.to_numeric(gdf["ACTYEARBUILT"], errors="coerce").values,
    })
    # fall back to MAINAREA where MAIN_SQFT missing/zero
    mainarea = pd.to_numeric(gdf["MAINAREA"], errors="coerce").values
    m = (out["bldg_area_sqft"].isna()) | (out["bldg_area_sqft"] <= 0)
    out.loc[m, "bldg_area_sqft"] = mainarea[m.values]

    # zeros are "no structure" sentinels, not real measurements
    out.loc[out["bldg_area_sqft"] <= 0, "bldg_area_sqft"] = np.nan
    out.loc[(out["year_built"] <= 0) | (out["year_built"] < 1800) | (out["year_built"] > 2026), "year_built"] = np.nan
    out["bedrooms"] = out["bedrooms"].astype(float)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    report(out)


def report(out: pd.DataFrame) -> None:
    print(f"\nwrote {len(out)} rows -> {OUT}")
    for c in ["lat", "lon", "bedrooms", "bldg_area_sqft", "year_built"]:
        cov = 100 * out[c].notna().mean()
        med = out[c].median()
        print(f"  {c:16s} coverage {cov:5.1f}%  median {med}")


if __name__ == "__main__":
    main()

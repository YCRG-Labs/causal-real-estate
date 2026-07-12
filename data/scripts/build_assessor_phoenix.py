"""Phoenix/Maricopa county-assessor attributes -> results/assessor/phoenix_assessor.parquet.

Structural attributes come from the Maricopa Assessor 'Residential Master' (R116)
bulk file (data-maricopa.opendata.arcgis.com item e22983d41d91490d90965544b718a120,
CSV Collection -> Data/Residential_Master.txt, pipe-delimited, no header). Column
order is fixed by the R116 file spec (extracted from the spec's EMF layout image):
  col0  PARCELNUMBER
  col10 ConstructionYear   -> year_built
  col11 Living_sqft        -> bldg_area_sqft
  col37 SITUSCITY          (filter == 'PHOENIX')
The Residential Master has NO bedroom field (only BathroomFixtures) -> bedrooms NaN.

Geometry (lat/lon) comes from the Maricopa 'Parcel Points' shapefile (arcgis.com item
dbf139379db946e1b10a2f15672c142d, EPSG:2868) joined on APN == PARCELNUMBER.
Output columns: lat, lon, bedrooms, bldg_area_sqft, year_built (float, WGS84).
"""
from __future__ import annotations

import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "phoenix"
OUT = REPO / "results" / "assessor" / "phoenix_assessor.parquet"
RES_ZIP = RAW / "Residential_Master.zip"
POINTS_URL = ("https://www.arcgis.com/sharing/rest/content/items/"
              "dbf139379db946e1b10a2f15672c142d/data")
UA = "YCRG-Labs JBES-2026 research (jacobcrainic@icloud.com)"

# 0-indexed positions from the R116 Residential Master file spec (39 cols)
C_APN, C_YEAR, C_SQFT, C_SITUSCITY = 0, 10, 11, 37


def load_residential() -> pd.DataFrame:
    zf = zipfile.ZipFile(RES_ZIP)
    with zf.open("Data/Residential_Master.txt") as fh:
        df = pd.read_csv(fh, sep="|", header=None, dtype=str,
                         usecols=[C_APN, C_YEAR, C_SQFT, C_SITUSCITY],
                         encoding="latin-1", low_memory=False)
    df.columns = ["APN", "year_built", "bldg_area_sqft", "situscity"]
    df["situscity"] = df["situscity"].astype(str).str.strip().str.upper()
    df = df[df["situscity"] == "PHOENIX"].copy()
    df["APN"] = df["APN"].astype(str).str.strip()
    df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
    df["bldg_area_sqft"] = pd.to_numeric(df["bldg_area_sqft"], errors="coerce")
    print(f"  residential-master rows in Phoenix: {len(df)}")
    return df


def load_points() -> gpd.GeoDataFrame:
    zpath = RAW / "ParcelPoints.zip"
    if not (zpath.exists() and zpath.stat().st_size > 1_000_000):
        print("  downloading Parcel Points shapefile...")
        req = urllib.request.Request(POINTS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            zpath.write_bytes(r.read())
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        shp = next(Path(td).rglob("*.shp"))
        pts = gpd.read_file(shp)
    print(f"  parcel points: {len(pts)} (crs {pts.crs})")
    pts = pts.to_crs(4326)
    apn_col = "APN" if "APN" in pts.columns else next(c for c in pts.columns if c.upper() == "APN")
    pts["APN"] = pts[apn_col].astype(str).str.strip()
    pts["lat"] = pts.geometry.y
    pts["lon"] = pts.geometry.x
    pts = pts.dropna(subset=["lat", "lon"]).drop_duplicates("APN")
    return pts[["APN", "lat", "lon"]]


def main() -> None:
    res = load_residential()
    pts = load_points()
    df = res.merge(pts, on="APN", how="inner")
    print(f"  joined (APN match): {len(df)} of {len(res)} "
          f"({100*len(df)/max(len(res),1):.1f}%)")
    df = df.drop_duplicates("APN")

    df.loc[(df["year_built"] < 1850) | (df["year_built"] > 2026), "year_built"] = np.nan
    df.loc[df["bldg_area_sqft"] <= 0, "bldg_area_sqft"] = np.nan

    out = pd.DataFrame({
        "lat": df["lat"].astype(float).values,
        "lon": df["lon"].astype(float).values,
        "bedrooms": np.full(len(df), np.nan),
        "bldg_area_sqft": df["bldg_area_sqft"].astype(float).values,
        "year_built": df["year_built"].astype(float).values,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT}")
    for c in ["lat", "lon", "bedrooms", "bldg_area_sqft", "year_built"]:
        print(f"  {c:16s} coverage {100*out[c].notna().mean():5.1f}%  median {out[c].median()}")


if __name__ == "__main__":
    main()

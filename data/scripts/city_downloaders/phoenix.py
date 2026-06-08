"""Phoenix parcel (Maricopa County shapefile) + crime (block-level CSV, geocoded).

Phoenix has two structural issues the agent flagged: parcel data is a 270MB
zipped shapefile that expands to 1.7GB and requires pyogrio's where= filter
to keep memory in check, and crime data has no incident-level lat/lon (only
100-block addresses). The crime block addresses get geocoded against the
Census batch endpoint via geocode_batch.py downstream.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import DEFAULT_HEADERS, canonicalize_crime, write_parquet

SLUG = "phoenix"
PARCEL_URL = "https://www.arcgis.com/sharing/rest/content/items/c937f17330f64e64abd41976fc8bb17f/data"
CRIME_URL = "https://www.phoenixopendata.com/dataset/cc08aace-9ca9-467f-b6c1-f0879ab1a358/resource/0ce3411a-2fc6-4302-a33f-167f68608a20/download/crime-data_crime-data_crimestat.csv"


def download_parcels(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{SLUG}_parcels.zip"
    if not zip_path.exists():
        print(f"  [{SLUG}] downloading 272MB parcel shapefile zip...")
        with requests.get(PARCEL_URL, headers=DEFAULT_HEADERS, stream=True, timeout=600) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    shp_dir = out_dir / "shp"
    shp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(shp_dir)
    shp_files = list(shp_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"no .shp in {shp_dir}")
    try:
        import pyogrio
        gdf = pyogrio.read_dataframe(
            shp_files[0],
            where="SITUS_CITY='PHOENIX'",
            columns=["APN", "SITUS_ADDR", "SITUS_CITY", "SITUS_ZIP", "LIVING_AREA", "LOT_SIZE_SQFT", "YEAR_BUILT", "BEDROOMS", "BATHS", "FULL_CASH_VALUE", "LAST_SALE_DATE", "LAST_SALE_AMT"],
        )
    except (ImportError, ValueError):
        import geopandas as gpd
        gdf = gpd.read_file(shp_files[0])
        if "SITUS_CITY" in gdf.columns:
            gdf = gdf[gdf["SITUS_CITY"] == "PHOENIX"].copy()
    gdf = gdf.to_crs(4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
    out_path = out_dir / f"{SLUG}_parcels.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    print(f"  [{SLUG}] downloading crime CSV (block-level addresses, no lat/lon)...")
    r = requests.get(CRIME_URL, headers=DEFAULT_HEADERS, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = canonicalize_crime(df, SLUG, "UCR CRIME CATEGORY")
    if "100 BLOCK ADDR" in df.columns:
        df["block_address_for_geocode"] = (
            df["100 BLOCK ADDR"].astype(str).str.replace("XX", "00", regex=False)
            + ", PHOENIX AZ "
            + df["ZIP"].astype(str)
        )
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

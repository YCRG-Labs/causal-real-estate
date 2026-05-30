"""DC parcel + crime downloader."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import arcgis_rest_query, canonicalize_crime, write_geojson_from_arcgis_chunks, write_parquet  # noqa: E402

SLUG = "dc"
PARCEL_URL = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Property_and_Land_WebMercator/FeatureServer/40"
CRIME_URL = "https://maps2.dcgis.dc.gov/dcgis/rest/services/FEEDS/MPD/FeatureServer/7"
PARCEL_FIELDS = "SSL,PREMISEADD,ASSESSMENT,SALEPRICE,SALEDATE,LANDAREA,USECODE,PROPTYPE,NBHDNAME,PRMSWARD"
CRIME_FIELDS = "CCN,REPORT_DAT,START_DATE,BLOCK,OFFENSE,METHOD,SHIFT,WARD,LATITUDE,LONGITUDE"


def download_parcels(out_dir: Path) -> Path:
    chunks = arcgis_rest_query(
        PARCEL_URL,
        where="ASSESSMENT > 0",
        out_fields=PARCEL_FIELDS,
        extra_params={"outSR": 4326},
    )
    return write_geojson_from_arcgis_chunks(chunks, out_dir / f"{SLUG}_parcels.geojson")


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    frames = []
    for chunk in arcgis_rest_query(
        CRIME_URL,
        where="LATITUDE IS NOT NULL",
        out_fields=CRIME_FIELDS,
        return_geometry=False,
        f="json",
        extra_params={"outSR": 4326},
    ):
        rows = [feat["attributes"] for feat in chunk.get("features", [])]
        if rows:
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        print(f"  [{SLUG}] no crime rows returned", file=sys.stderr)
    df = canonicalize_crime(df, SLUG, "OFFENSE")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    p = download_parcels(out_dir)
    print(f"  -> {p}")
    print(f"[{SLUG}] downloading crime...")
    c = download_crime(out_dir, token=os.environ.get("SOCRATA_APP_TOKEN"))
    print(f"  -> {c}")

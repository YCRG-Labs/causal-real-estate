"""Seattle parcel (ArcGIS REST) + crime (Socrata)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import arcgis_rest_query, canonicalize_crime, socrata_query, write_geojson_from_arcgis_chunks, write_parquet  # noqa: E402

SLUG = "seattle"
PARCEL_URL = "https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0"
PARCEL_FIELDS = "PIN,ADDR_FULL,LAT,LON,CTYNAME,KCTP_TAXYR,LOTSQFT,APPRLNDVAL,APPR_IMPR,PREUSE_CODE,PREUSE_DESC,ZIP5"
CRIME_BASE = "https://data.seattle.gov/resource/tazs-3rd5.json"


def download_parcels(out_dir: Path) -> Path:
    chunks = arcgis_rest_query(
        PARCEL_URL,
        where="CTYNAME='SEATTLE'",
        out_fields=PARCEL_FIELDS,
        extra_params={"outSR": 4326},
    )
    return write_geojson_from_arcgis_chunks(chunks, out_dir / f"{SLUG}_parcels.geojson")


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    df = socrata_query(
        CRIME_BASE,
        where="report_date_time >= '2024-01-01T00:00:00' AND latitude IS NOT NULL",
        select="report_number, offense_id, report_date_time, offense_category, offense_sub_category, latitude, longitude, beat, neighborhood",
        app_token=token or os.environ.get("SOCRATA_APP_TOKEN"),
    )
    df = canonicalize_crime(df, SLUG, "offense_sub_category")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

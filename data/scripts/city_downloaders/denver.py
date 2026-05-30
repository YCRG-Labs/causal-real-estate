"""Denver parcel + crime via ArcGIS REST (layer ids 245 and 324)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import arcgis_rest_query, canonicalize_crime, write_geojson_from_arcgis_chunks, write_parquet  # noqa: E402

SLUG = "denver"
PARCEL_URL = "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_PROP_PARCELS_A/FeatureServer/245"
CRIME_URL = "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_CRIME_OFFENSES_P/FeatureServer/324"
PARCEL_FIELDS = "SCHEDNUM,SITUS_ADDRESS_LINE1,SITUS_ZIP,PROP_CLASS,APPRAISED_TOTAL_VALUE,LAND_AREA,RES_ORIG_YEAR_BUILT,RES_ABOVE_GRADE_AREA,SALE_PRICE,SALE_DATE"
CRIME_FIELDS = "INCIDENT_ID,OFFENSE_TYPE_ID,OFFENSE_CATEGORY_ID,REPORTED_DATE,FIRST_OCCURRENCE_DATE,INCIDENT_ADDRESS,GEO_LAT,GEO_LON,NEIGHBORHOOD_ID,IS_CRIME,IS_TRAFFIC"


def download_parcels(out_dir: Path) -> Path:
    chunks = arcgis_rest_query(
        PARCEL_URL,
        out_fields=PARCEL_FIELDS,
        extra_params={"outSR": 4326},
    )
    return write_geojson_from_arcgis_chunks(chunks, out_dir / f"{SLUG}_parcels.geojson")


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    frames = []
    for chunk in arcgis_rest_query(
        CRIME_URL,
        where="IS_CRIME=1 AND GEO_LAT IS NOT NULL",
        out_fields=CRIME_FIELDS,
        return_geometry=False,
        f="json",
        extra_params={"outSR": 4326},
    ):
        rows = [feat["attributes"] for feat in chunk.get("features", [])]
        if rows:
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = canonicalize_crime(df, SLUG, "OFFENSE_CATEGORY_ID")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

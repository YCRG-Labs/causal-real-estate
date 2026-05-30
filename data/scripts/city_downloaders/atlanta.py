"""Atlanta parcel (pre-merged city layer) + crime (live ArcGIS feature service)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import arcgis_rest_query, canonicalize_crime, write_geojson_from_arcgis_chunks, write_parquet  # noqa: E402

SLUG = "atlanta"
PARCEL_URL = "https://gis.atlantaga.gov/dpcd/rest/services/AdministrativeArea/TaxParcel/MapServer/0"
CRIME_URL = "https://services3.arcgis.com/Et5Qfajgiyosiw4d/arcgis/rest/services/OpenDataWebsite_Crime_view/FeatureServer/0"
PARCEL_FIELDS = "PARCELID,SITEADDRESS,SITECITY,SITEZIP,CLASSCD,CLASSDSCRP,TOT_APPR,IMPR_APPR,LANDAPPR,TAXYEAR,NPU,NEIGHBORHOOD"
CRIME_FIELDS = "IncidentNumber,ReportNumber,ReportDate,Part,Crime_Against,NIBRS_Offense,NIBRS_Bucket,StreetAddress,Latitude,Longitude,Zone,NhoodName"


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
        where="NIBRS_Bucket IS NOT NULL AND Latitude IS NOT NULL",
        out_fields=CRIME_FIELDS,
        return_geometry=False,
        f="json",
        extra_params={"outSR": 4326},
    ):
        rows = [feat["attributes"] for feat in chunk.get("features", [])]
        if rows:
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = canonicalize_crime(df, SLUG, "NIBRS_Bucket")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

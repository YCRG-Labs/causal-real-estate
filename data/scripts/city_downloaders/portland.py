"""Portland parcel (ArcGIS REST) + crime (Tableau Public scrape).

Crime is the friction point: PPB publishes monthly stats via Tableau Public
with no direct CSV API. Falls back to writing an empty crime parquet if the
optional `tableauscraper` package is not installed; the rest of the pipeline
handles a Portland-without-crime gracefully (the structured-confounder set
remains usable, the crime category is documented as missing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import arcgis_rest_query, canonicalize_crime, write_geojson_from_arcgis_chunks, write_parquet

SLUG = "portland"
PARCEL_URL = "https://services5.arcgis.com/x7DNZL1YqNQVNykA/arcgis/rest/services/Multnomah_County_Taxlot_Parcels/FeatureServer/0"
PARCEL_FIELDS = "MAPTAXLOT,SITUSADDR,SITUSCITY,SITUSZIP,PROPCLASS,PROP_CODE,DEED_DATE,SALE_PRICE,SALE_DATE,SIZESQFT,IMPTYPE,ACTYEARBUILT,MAINAREA,UNITS,MAIN_SQFT,ROLLM50"
CRIME_TABLEAU_URL = "https://public.tableau.com/app/profile/portlandpolicebureau/viz/MonthlyReportedCrimeStatistics/MonthlyStatistics"


def download_parcels(out_dir: Path) -> Path:
    chunks = arcgis_rest_query(
        PARCEL_URL,
        where="SITUSCITY='PORTLAND'",
        out_fields=PARCEL_FIELDS,
        extra_params={"outSR": 4326},
    )
    return write_geojson_from_arcgis_chunks(chunks, out_dir / f"{SLUG}_parcels.geojson")


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    out_path = out_dir / f"{SLUG}_crime.parquet"
    try:
        from tableauscraper import TableauScraper
    except ImportError:
        print(f"  [{SLUG}] tableauscraper not installed; writing empty crime parquet (document as limitation)", file=sys.stderr)
        empty = pd.DataFrame(columns=["report_date", "offense_category", "latitude", "longitude", "crime_category"])
        return write_parquet(empty, out_path)
    try:
        ts = TableauScraper()
        ts.loads(CRIME_TABLEAU_URL)
        ws = next((w for w in ts.getWorksheets() if "Crime" in w.name or "Monthly" in w.name), ts.getWorksheets()[0])
        df = ws.data
        if not df.empty:
            rename_map = {}
            for col in df.columns:
                low = col.lower()
                if "lat" in low and "lat" not in rename_map.values():
                    rename_map[col] = "latitude"
                elif "lon" in low and "lon" not in rename_map.values():
                    rename_map[col] = "longitude"
                elif "offense" in low or "category" in low:
                    rename_map[col] = "OffenseCategory"
                elif "date" in low or "report" in low:
                    rename_map[col] = "report_date"
            df = df.rename(columns=rename_map)
        df = canonicalize_crime(df, SLUG, "OffenseCategory")
        return write_parquet(df, out_path)
    except Exception as e:
        print(f"  [{SLUG}] tableau scrape failed: {e}; writing empty crime parquet", file=sys.stderr)
        empty = pd.DataFrame(columns=["report_date", "offense_category", "latitude", "longitude", "crime_category"])
        return write_parquet(empty, out_path)


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

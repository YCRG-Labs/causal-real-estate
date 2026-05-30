"""Philadelphia parcel + crime via Carto SQL."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import canonicalize_crime, write_parquet, DEFAULT_HEADERS  # noqa: E402

SLUG = "philadelphia"
PARCEL_SQL = "SELECT parcel_number, location, zip_code, market_value, sale_date, sale_price, year_built, total_livable_area, number_of_bedrooms, number_of_bathrooms, category_code_description, ST_X(the_geom::geometry) AS lon, ST_Y(the_geom::geometry) AS lat FROM opa_properties_public"
CRIME_SQL = "SELECT dispatch_date, dc_dist, ucr_general, text_general_code, point_x, point_y FROM incidents_part1_part2 WHERE dispatch_date >= '2024-01-01'"
CARTO_BASE = "https://phl.carto.com/api/v2/sql"


def _carto_csv(sql: str, page_size: int = 100_000) -> pd.DataFrame:
    frames = []
    offset = 0
    while True:
        page_sql = f"{sql} LIMIT {page_size} OFFSET {offset}"
        r = requests.get(CARTO_BASE, params={"q": page_sql, "format": "csv"}, headers=DEFAULT_HEADERS, timeout=180)
        r.raise_for_status()
        chunk = pd.read_csv(io.StringIO(r.text))
        if chunk.empty:
            break
        frames.append(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_parcels(out_dir: Path) -> Path:
    df = _carto_csv(PARCEL_SQL)
    return write_parquet(df, out_dir / f"{SLUG}_parcels.parquet")


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    df = _carto_csv(CRIME_SQL)
    df = canonicalize_crime(df, SLUG, "text_general_code")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

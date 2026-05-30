"""Chicago parcel (3-way Socrata join) + crime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import canonicalize_crime, socrata_query, write_parquet  # noqa: E402

SLUG = "chicago"
PARCEL_BASE = "https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.json"
STRUCT_BASE = "https://datacatalog.cookcountyil.gov/resource/x54s-btds.json"
VALUE_BASE = "https://datacatalog.cookcountyil.gov/resource/uzyt-m557.json"
CRIME_BASE = "https://data.cityofchicago.org/resource/6zsd-86xi.json"
TAX_YEAR = "2026.0"


def download_parcels(out_dir: Path) -> Path:
    print(f"  [{SLUG}] pulling parcel universe (filtered to City of Chicago)...")
    parcels = socrata_query(
        PARCEL_BASE,
        where=f"cook_municipality_name='CITY OF CHICAGO' AND year='{TAX_YEAR}'",
        select="pin, pin10, year, class, township_name, zip_code, lon, lat, census_tract_geoid",
        app_token=os.environ.get("SOCRATA_APP_TOKEN"),
    )
    print(f"    {len(parcels):,} parcels")
    if parcels.empty:
        return write_parquet(parcels, out_dir / f"{SLUG}_parcels.parquet")
    print(f"  [{SLUG}] pulling structural attributes...")
    struct = socrata_query(
        STRUCT_BASE,
        where=f"year='{TAX_YEAR}'",
        select="pin, char_yrblt, char_bldg_sf, char_beds, char_rooms, char_fbath, char_hbath",
        app_token=os.environ.get("SOCRATA_APP_TOKEN"),
    )
    print(f"  [{SLUG}] pulling assessed values...")
    vals = socrata_query(
        VALUE_BASE,
        where=f"year='{TAX_YEAR}'",
        select="pin, board_tot, mailed_tot, certified_tot",
        app_token=os.environ.get("SOCRATA_APP_TOKEN"),
    )
    if not vals.empty:
        for col in ("board_tot", "mailed_tot", "certified_tot"):
            if col in vals.columns:
                vals[col] = pd.to_numeric(vals[col], errors="coerce") * 1000
    df = parcels.merge(struct, on="pin", how="left") if not struct.empty else parcels
    if not vals.empty:
        df = df.merge(vals, on="pin", how="left")
    return write_parquet(df, out_dir / f"{SLUG}_parcels.parquet")


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    df = socrata_query(
        CRIME_BASE,
        where="year >= 2024 AND latitude IS NOT NULL",
        select="id, date, block, primary_type, description, latitude, longitude, year",
        app_token=token or os.environ.get("SOCRATA_APP_TOKEN"),
    )
    df = canonicalize_crime(df, SLUG, "primary_type")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

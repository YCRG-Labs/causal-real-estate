"""Fetch county ASSESSOR structural attributes (bedrooms, building sqft, year
built) with parcel coordinates for CHICAGO, SEATTLE, DENVER.

These are exogenous confounders: recorded by the county assessor from
physical inspection/permits, not authored by the listing agent, so they are
clean controls for the causal spec (unlike listing-text-derived features).

Sources (see data/scripts/city_endpoints.py for full notes):
  chicago: Cook County Assessor Socrata, three-way join on pin -- geometry+
    lat/lon from nj4t-kc8j, structural (char_yrblt/char_bldg_sf/char_beds)
    from x54s-btds, filtered to CITY OF CHICAGO + char_use='Single-Family'
    so bedroom counts aren't multi-unit aggregates.
  seattle: King County Assessor EXTR_ResBldg.csv (Bedrooms/SqFtTotLiving/
    YrBuilt, keyed by Major+Minor) joined to the ArcGIS parcel-address layer
    for precomputed WGS84 LAT/LON.
  denver: City and County of Denver's own parcel layer (ODC_PROP_PARCELS_A/
    245), which carries RES_ORIG_YEAR_BUILT + RES_ABOVE_GRADE_AREA directly
    but has NO bedroom field on this layer -- bedrooms is left NaN.

Writes results/assessor/<city>_assessor.parquet with columns exactly:
  lat, lon, bedrooms, bldg_area_sqft, year_built  (float, WGS84/EPSG:4326)

    .venv/bin/python data/scripts/fetch_assessor_structural.py --city chicago
    .venv/bin/python data/scripts/fetch_assessor_structural.py --city seattle
    .venv/bin/python data/scripts/fetch_assessor_structural.py --city denver
    .venv/bin/python data/scripts/fetch_assessor_structural.py --city all
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from download_sales import _socrata_once, _arcgis_query, _download_zip_member  # noqa: E402

OUT_DIR = REPO / "results" / "assessor"
COLS = ["lat", "lon", "bedrooms", "bldg_area_sqft", "year_built"]


def _finalize(df: pd.DataFrame, city: str) -> pd.DataFrame:
    for c in COLS:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df[COLS].dropna(subset=["lat", "lon"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{city}_assessor.parquet"
    df.to_parquet(out, index=False)
    print(f"\n{city}: {len(df):,} rows -> {out}")
    for c in COLS:
        cov = 100 * df[c].notna().mean()
        med = df[c].median()
        print(f"  {c:16s} {cov:5.1f}% non-null   median={med}")
    return df


def fetch_chicago() -> pd.DataFrame:
    print("chicago: pulling parcel universe (nj4t-kc8j, lat/lon)...")
    univ = _socrata_once(
        "https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.csv",
        select="pin,lat,lon",
        where="cook_municipality_name='CITY OF CHICAGO' AND year='2026.0' AND lat IS NOT NULL",
        limit=1_200_000, label="cook-univ")
    univ["pin"] = univ["pin"].astype(str).str.zfill(14)
    univ["lat"] = pd.to_numeric(univ["lat"], errors="coerce")
    univ["lon"] = pd.to_numeric(univ["lon"], errors="coerce")
    univ = univ.dropna(subset=["lat", "lon"]).drop_duplicates("pin")

    print("chicago: pulling structural characteristics (x54s-btds)...")
    struct = _socrata_once(
        "https://datacatalog.cookcountyil.gov/resource/x54s-btds.csv",
        select="pin,char_yrblt,char_bldg_sf,char_beds",
        where="year='2026.0' AND card='1.0' AND char_use='Single-Family'",
        limit=1_500_000, label="cook-struct")
    struct["pin"] = struct["pin"].astype(str).str.zfill(14)
    struct = struct.drop_duplicates("pin")

    df = univ.merge(struct, on="pin", how="inner")
    df = df.rename(columns={"char_yrblt": "year_built", "char_bldg_sf": "bldg_area_sqft",
                             "char_beds": "bedrooms"})
    df.loc[(df["year_built"] < 1850) | (df["year_built"] > 2026), "year_built"] = np.nan
    return df


def fetch_seattle() -> pd.DataFrame:
    print("seattle: downloading King County Residential Building extract...")
    raw = _download_zip_member(
        "https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip",
        "resbldg.csv")
    bldg = pd.read_csv(
        io.BytesIO(raw),
        usecols=["Major", "Minor", "BldgNbr", "Bedrooms", "SqFtTotLiving", "YrBuilt"],
        dtype={"Major": str, "Minor": str}, encoding="latin-1")
    bldg = bldg[bldg["BldgNbr"] == 1].copy()
    bldg["pin"] = bldg["Major"].str.zfill(6) + bldg["Minor"].str.zfill(4)
    bldg = bldg.drop_duplicates("pin")

    print("seattle: pulling parcel-address layer (PIN, LAT, LON)...")
    parcels = _arcgis_query(
        "https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/"
        "PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0",
        out_fields="PIN,LAT,LON,CTYNAME", where="CTYNAME='SEATTLE'",
        max_rows=None, geom=False)
    parcels = parcels.dropna(subset=["LAT", "LON"]).drop_duplicates("PIN")

    df = bldg.merge(parcels, left_on="pin", right_on="PIN", how="inner")
    df = df.rename(columns={"LAT": "lat", "LON": "lon", "Bedrooms": "bedrooms",
                             "SqFtTotLiving": "bldg_area_sqft", "YrBuilt": "year_built"})
    df.loc[(df["bedrooms"] <= 0), "bedrooms"] = np.nan
    df.loc[(df["bldg_area_sqft"] <= 0), "bldg_area_sqft"] = np.nan
    df.loc[(df["year_built"] < 1850) | (df["year_built"] > 2026), "year_built"] = np.nan
    return df


DENVER_RESIDENTIAL_DCLASS = re.compile(r"^(10\d|11\d|12[2-4]|13[2-4]|19\d|10S)$")


def fetch_denver() -> pd.DataFrame:
    print("denver: pulling residential parcels (ODC_PROP_PARCELS_A/245)...")
    df = _arcgis_query(
        "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/"
        "ODC_PROP_PARCELS_A/FeatureServer/245",
        out_fields="SCHEDNUM,D_CLASS,RES_ORIG_YEAR_BUILT,RES_ABOVE_GRADE_AREA",
        where="RES_ORIG_YEAR_BUILT IS NOT NULL AND RES_ABOVE_GRADE_AREA IS NOT NULL",
        max_rows=None, geom=True, page=2000)
    if df.empty:
        raise SystemExit("denver: empty pull -- check field names/where clause")
    df = df[df["D_CLASS"].astype(str).str.match(DENVER_RESIDENTIAL_DCLASS)]
    df = df.dropna(subset=["lat", "lon"]).drop_duplicates("SCHEDNUM")
    df = df.rename(columns={"RES_ORIG_YEAR_BUILT": "year_built",
                             "RES_ABOVE_GRADE_AREA": "bldg_area_sqft"})
    df.loc[(df["year_built"] < 1850) | (df["year_built"] > 2026), "year_built"] = np.nan
    # No bedroom field exists on this layer (verified against the live field
    # list); leave NaN rather than invent one.
    df["bedrooms"] = np.nan
    return df


FETCHERS = {"chicago": fetch_chicago, "seattle": fetch_seattle, "denver": fetch_denver}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(FETCHERS) + ["all"])
    args = ap.parse_args()
    cities = list(FETCHERS) if args.city == "all" else [args.city]
    for city in cities:
        df = FETCHERS[city]()
        _finalize(df, city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

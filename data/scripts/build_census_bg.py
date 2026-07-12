"""Build census block-group demographics for all twelve markets.

Generic: for each city, pull the eleven canonical ACS variables at block-group
level (Census API, key required) and the TIGER block-group geometries for its
state, and write results/census_bg/<city>_census_bg.gpkg. This is the only
external data the clean all-twelve confounder block needs, because the listing's
own beds/baths/sqft/year_built are already exact and the demographics attach by a
lat/lon spatial join.

    CENSUS_API_KEY=... python data/scripts/build_census_bg.py
"""
from __future__ import annotations

import io
import os
import sys
import time
import json as _json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from config import CENSUS_YEAR, CENSUS_VARIABLES, CENSUS_DERIVED  # noqa: E402
from city_endpoints import CITIES  # noqa: E402

OUT = REPO / "results" / "census_bg"
OUT.mkdir(parents=True, exist_ok=True)
TIGER_CACHE = REPO / "data" / "raw" / "census"
TIGER_CACHE.mkdir(parents=True, exist_ok=True)
CENSUS_API = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5"
KEEP = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60", "GEOID"]


def _get(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == 4:
                raise
            print(f"    retry {i+1} after {e}")
            time.sleep(2 * 2 ** i)


def acs(state, county, key):
    variables = ",".join(CENSUS_VARIABLES.keys())
    url = (f"{CENSUS_API}?get={variables}&for=block%20group:*"
           f"&in=state:{state}&in=county:{county}&in=tract:*&key={key}")
    data = _json.loads(_get(url))
    df = pd.DataFrame(data[1:], columns=data[0])
    for v in CENSUS_VARIABLES:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df.loc[df[v] < 0, v] = np.nan
    df = df.rename(columns=CENSUS_VARIABLES)
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    for name, (num, den) in CENSUS_DERIVED.items():
        if num in df and den in df:
            df[name] = df[num] / df[den].replace(0, np.nan)
    total = df["age_total"].replace(0, np.nan)
    df["pct_under_25"] = df[["age_5_9", "age_18_19"]].sum(axis=1) / total
    df["pct_over_60"] = df[["age_60_61", "age_85_plus"]].sum(axis=1) / total
    return df


def tiger(state):
    fn = TIGER_CACHE / f"tl_{CENSUS_YEAR}_{state}_bg.zip"
    if not fn.exists():
        url = f"https://www2.census.gov/geo/tiger/TIGER{CENSUS_YEAR}/BG/tl_{CENSUS_YEAR}_{state}_bg.zip"
        print(f"    downloading TIGER {state}...")
        fn.write_bytes(_get(url))
    return gpd.read_file(f"zip://{fn}")


def build(city, cfg, key):
    dst = OUT / f"{city}_census_bg.gpkg"
    if dst.exists():
        print(f"  {city}: already done, skip")
        return
    frames = []
    for county in cfg.county_fips:
        frames.append(acs(cfg.state_fips, county, key))
    df = pd.concat(frames, ignore_index=True)
    bg = tiger(cfg.state_fips)
    bg = bg[bg.COUNTYFP.isin(cfg.county_fips)][["GEOID", "geometry"]].to_crs(4326)
    merged = bg.merge(df[KEEP], on="GEOID", how="inner")
    merged.to_file(dst, driver="GPKG", layer="bg")
    print(f"  {city}: {len(merged)} block groups -> {dst}")


def main():
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        raise SystemExit("set CENSUS_API_KEY (get one at https://api.census.gov/data/key_signup.html)")
    cities = sys.argv[1:] or list(CITIES.keys())
    for c in cities:
        cfg = CITIES.get(c)
        if not cfg:
            print(f"  {c}: no config, skip")
            continue
        try:
            build(c, cfg, key)
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

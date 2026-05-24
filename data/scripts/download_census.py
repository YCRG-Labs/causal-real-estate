import os
import sys
import time
import requests
import zipfile
import pandas as pd
import geopandas as gpd
from pathlib import Path
from config import (
    RAW_DIR, CENSUS_YEAR, CENSUS_VARIABLES,
    STATE_FIPS, COUNTY_FIPS, CITIES,
)

CENSUS_DIR = RAW_DIR / "census"
CENSUS_API_BASE = f"https://api.census.gov/data/{CENSUS_YEAR}/acs/acs5"
TIGER_BASE = f"https://www2.census.gov/geo/tiger/TIGER{CENSUS_YEAR}/BG"

# Census API: free key from https://api.census.gov/data/key_signup.html
# Without a key, the API rate-limits aggressively and may return HTML error
# pages instead of JSON. Set CENSUS_API_KEY env var to authenticate.
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "").strip()


def _get_with_retry(url: str, attempts: int = 5, base_delay: float = 2.0):
    last_err = None
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            # Validate JSON before returning (Census sometimes returns HTML
            # error pages with HTTP 200 when rate-limited).
            try:
                return resp.json()
            except ValueError:
                last_err = (
                    f"non-JSON response (first 200 chars): "
                    f"{resp.text[:200].replace(chr(10), ' ')!r}"
                )
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(base_delay * (2 ** i))
    raise RuntimeError(
        f"Census API failed after {attempts} attempts. Last error: {last_err}\n"
        f"URL: {url}\n"
        f"FIX: get a free key at https://api.census.gov/data/key_signup.html "
        f"and set CENSUS_API_KEY env var."
    )


def fetch_acs(state_fips, county_fips):
    variables = ",".join(CENSUS_VARIABLES.keys())
    key_suffix = f"&key={CENSUS_API_KEY}" if CENSUS_API_KEY else ""
    rows = []
    for county in county_fips:
        url = (
            f"{CENSUS_API_BASE}?get={variables}"
            f"&for=block%20group:*"
            f"&in=state:{state_fips}&in=county:{county}&in=tract:*"
            f"{key_suffix}"
        )
        data = _get_with_retry(url)
        header = data[0]
        rows.extend(data[1:])

    df = pd.DataFrame(rows, columns=header)

    for var in CENSUS_VARIABLES:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        df.loc[df[var] < 0, var] = pd.NA

    df = df.rename(columns=CENSUS_VARIABLES)
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    return df


def download_tiger_bg(state_fips):
    filename = f"tl_{CENSUS_YEAR}_{state_fips}_bg.zip"
    url = f"{TIGER_BASE}/{filename}"
    dest = CENSUS_DIR / filename

    if dest.exists():
        print(f"  {filename} already exists, skipping")
        return dest

    print(f"  Downloading {filename}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return dest


def load_tiger_bg(zip_path, county_fips):
    gdf = gpd.read_file(f"zip://{zip_path}")
    gdf = gdf[gdf["COUNTYFP"].isin(county_fips)]
    return gdf[["GEOID", "geometry"]]


def process_city(city):
    print(f"\n{city}:")
    state = STATE_FIPS[city]
    counties = COUNTY_FIPS[city]

    print("  Fetching ACS data...")
    acs_df = fetch_acs(state, counties)
    print(f"  Got {len(acs_df)} block groups")

    zip_path = download_tiger_bg(state)
    tiger_gdf = load_tiger_bg(zip_path, counties)
    print(f"  Got {len(tiger_gdf)} block group geometries")

    merged = tiger_gdf.merge(acs_df, on="GEOID", how="inner")
    merged = merged.drop(columns=["state", "county", "tract", "block group"], errors="ignore")

    out_path = CENSUS_DIR / f"{city}_block_groups.gpkg"
    merged.to_file(out_path, driver="GPKG", layer="block_groups")
    print(f"  Saved {len(merged)} block groups → {out_path}")


def main():
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        process_city(city)
    print("\nCensus download complete.")


if __name__ == "__main__":
    main()

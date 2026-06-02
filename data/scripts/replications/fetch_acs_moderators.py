"""Fetch the 8 candidate city-level moderators for the 12 CBSAs via Census ACS 5-year.

Pulls 2018-2022 ACS 5-year estimates at the metropolitan statistical area
(CBSA) level for the 12 metros in our DML sample. Per the meta-regression
research agent's recipe:

  - Population (log)                B01003_001E
  - Median household income         B19013_001E
  - % renter-occupied               B25003_003E / B25003_001E
  - % bachelors+ (age 25+)          (B15003_022E..025E) / B15003_001E
  - Diversity (Gini-Simpson on B03002 race/Hispanic groups)
  - Median home value               B25077_001E
  - Population density              B01003 / Gazetteer ALAND_SQMI
  - Housing-stock median year built B25035_001E

The CENSUS_API_KEY env var is required. Output:
results/replications/acs_moderators_12city.csv with one row per metro.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
OUT_CSV = REPO / "results" / "replications" / "acs_moderators_12city.csv"

CITY_CBSA = {
    "boston": "14460",
    "nyc": "35620",
    "sf": "41860",
    "dc": "47900",
    "philadelphia": "37980",
    "chicago": "16980",
    "seattle": "42660",
    "denver": "19740",
    "atlanta": "12060",
    "portland": "38900",
    "phoenix": "38060",
    "dallas": "19100",
}

ACS_VARS = [
    "B01003_001E",
    "B19013_001E",
    "B25003_001E", "B25003_003E",
    "B15003_001E",
    "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
    "B25077_001E",
    "B25035_001E",
    "B03002_001E",
    "B03002_003E",
    "B03002_004E",
    "B03002_006E",
    "B03002_007E",
    "B03002_008E",
    "B03002_009E",
    "B03002_012E",
]


def fetch_acs(api_key: str, year: int = 2022) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": ",".join(["NAME"] + ACS_VARS),
        "for": "metropolitan statistical area/micropolitan statistical area:*",
        "key": api_key,
    }
    print(f"GET {base}  with {len(ACS_VARS)} vars")
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={"metropolitan statistical area/micropolitan statistical area": "cbsa"})
    df["cbsa"] = df["cbsa"].astype(str)
    for col in ACS_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def diversity_gini_simpson(row) -> float:
    groups = [row["B03002_003E"], row["B03002_004E"], row["B03002_006E"],
              row["B03002_007E"], row["B03002_008E"], row["B03002_009E"],
              row["B03002_012E"]]
    total = row["B03002_001E"]
    if total is None or pd.isna(total) or total <= 0:
        return float("nan")
    p = np.array([(g / total) if (g and not pd.isna(g)) else 0.0 for g in groups], dtype=float)
    return float(1.0 - np.sum(p ** 2))


def fetch_gazetteer_landarea(year: int = 2024) -> pd.DataFrame:
    """Pull CBSA land area from the Census Gazetteer. The Census reorganises
    these annually; try a small list of recent years until one returns 200.
    """
    import io, zipfile
    for y in (year, 2024, 2023, 2021, 2020):
        url = f"https://www2.census.gov/geo/docs/maps-data/data/gazetteer/{y}_Gazetteer/{y}_Gaz_cbsa_national.zip"
        print(f"GET gazetteer {y} land area")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"  {y} failed: {e}")
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        raw = z.read(name).decode("latin-1")
        df = pd.read_csv(io.StringIO(raw), sep="\t")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"GEOID": "cbsa"})
        df["cbsa"] = df["cbsa"].astype(str)
        return df[["cbsa", "ALAND_SQMI"]]
    print("All gazetteer years failed; returning empty (density moderator will be NaN)")
    return pd.DataFrame({"cbsa": [], "ALAND_SQMI": []})


def main():
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("ERROR: set CENSUS_API_KEY", file=sys.stderr); return 1

    acs = fetch_acs(api_key, year=2022)
    gaz = fetch_gazetteer_landarea(year=2022)

    rows = []
    for city, cbsa in CITY_CBSA.items():
        sub = acs[acs["cbsa"] == cbsa]
        if sub.empty:
            print(f"  [warn] no ACS row for {city} ({cbsa})"); continue
        r = sub.iloc[0]
        gz = gaz[gaz["cbsa"] == cbsa]
        land_sqmi = float(gz["ALAND_SQMI"].iloc[0]) if not gz.empty else float("nan")
        pop = float(r["B01003_001E"])
        renter = float(r["B25003_003E"]) / max(float(r["B25003_001E"]), 1.0)
        ba_or_more = sum(float(r[c]) for c in
                          ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"])
        college = ba_or_more / max(float(r["B15003_001E"]), 1.0)
        rows.append({
            "city": city,
            "cbsa": cbsa,
            "name": r["NAME"],
            "population": pop,
            "log_population": float(np.log(max(pop, 1.0))),
            "median_hh_income": float(r["B19013_001E"]),
            "pct_renter": float(renter),
            "pct_college": float(college),
            "diversity_gini_simpson": diversity_gini_simpson(r),
            "median_home_value": float(r["B25077_001E"]),
            "median_year_built": float(r["B25035_001E"]),
            "land_area_sqmi": land_sqmi,
            "pop_density_per_sqmi": float(pop / land_sqmi) if land_sqmi > 0 else float("nan"),
        })

    if not rows:
        print("no rows", file=sys.stderr); return 1
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n=== ACS moderators for {len(df)} metros ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nCSV -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

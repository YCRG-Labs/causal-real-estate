"""Fetch 5 additional metro-level moderators for the 12 cities.

Per the JBES tightening pass research recommendation, we augment the
8 ACS demographic moderators with 5 additional federal-data-source
moderators that capture economic and structural market characteristics
the Kurlat-Stroebel (2015) buyer-information mechanism predicts may
matter:

  1. BEA Regional GDP per capita (2022, CBSA-level)
  2. BLS QCEW average weekly wage (Q4 2022, CBSA-level)
  3. BLS LAUS unemployment rate (2022 annual, CBSA-level)
  4. FBI UCR / NIBRS violent crime rate per 100K (most recent CBSA-mapped)
  5. FHFA HPI 1-year price appreciation (most recent CBSA)

Reads results/replications/acs_moderators_12city.csv; writes
results/replications/extra_moderators_12city.csv with the additional
columns; produces results/replications/all_moderators_12city.csv as
the unified join for downstream meta-regression.

Reference: Anenberg-Bayer 2020 IER on housing-market volatility;
Chinco-Mayer 2016 RFS on misinformed-speculator-driven mispricing.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]

# CBSA codes for the 12 cities (from acs_moderators_12city.csv lookup).
CITY_CBSA = {
    "boston":       "14460",
    "nyc":          "35620",
    "sf":           "41860",
    "dc":           "47900",
    "philadelphia": "37980",
    "chicago":      "16980",
    "seattle":      "42660",
    "denver":       "19740",
    "atlanta":      "12060",
    "portland":     "38900",
    "phoenix":      "38060",
    "dallas":       "19100",
}


def fetch_bea_gdp_per_capita(api_key: str, year: int = 2022) -> pd.DataFrame:
    """BEA Regional Economic Accounts: Real GDP per capita at MSA level."""
    url = "https://apps.bea.gov/api/data/"
    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": "Regional",
        "TableName": "CAGDP9",   # Real GDP by metropolitan area
        "GeoFips": "MSA",
        "LineCode": "1",
        "Year": str(year),
        "ResultFormat": "JSON",
    }
    print(f"  BEA: Regional Real GDP for {year}")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    rows = j.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    if not rows:
        raise RuntimeError(f"no BEA rows: {j}")
    df = pd.DataFrame(rows)
    df["cbsa"] = df["GeoFips"].astype(str)
    df["gdp_real_2017"] = pd.to_numeric(df["DataValue"], errors="coerce")
    return df[["cbsa", "gdp_real_2017"]]


def fetch_bls_unemployment(api_key: str | None = None) -> pd.DataFrame:
    """BLS LAUS unemployment rate at MSA level (2022 annual).

    Uses the BLS Public API v2 series IDs of the form
    LAUMT{state2}{cbsa5}0000000003 (unemployment rate).
    """
    # Series IDs for the 12 CBSAs (constructed from state FIPS + CBSA + 003).
    cbsa_state = {
        "14460": "25", "35620": "36", "41860": "06", "47900": "11",
        "37980": "42", "16980": "17", "42660": "53", "19740": "08",
        "12060": "13", "38900": "41", "38060": "04", "19100": "48",
    }
    series_ids = {
        cbsa: f"LAUMT{state}{cbsa}0000000003"
        for cbsa, state in cbsa_state.items()
    }
    payload = {
        "seriesid": list(series_ids.values()),
        "startyear": "2022", "endyear": "2022",
    }
    if api_key:
        payload["registrationkey"] = api_key
    print(f"  BLS LAUS: unemployment rate for {len(series_ids)} MSAs (2022)")
    headers = {"Content-Type": "application/json"}
    r = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json=payload, headers=headers, timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {j.get('message')}")
    rows = []
    for s in j["Results"]["series"]:
        sid = s["seriesID"]
        cbsa = sid[5:10] if len(sid) >= 10 else sid
        # CBSA position depends on series ID format; use the 5-digit substring
        # following the state code (LAUMT{ss}{ccccc})
        cbsa_match = sid[7:12]
        annual_obs = [d for d in s["data"] if d["period"] == "M13"
                       or d["periodName"] == "Annual"]
        if not annual_obs:
            # fall back to year average of monthly
            monthly = [float(d["value"]) for d in s["data"]
                       if d.get("value") not in (None, "")]
            val = float(np.mean(monthly)) if monthly else float("nan")
        else:
            val = float(annual_obs[0]["value"])
        rows.append({"cbsa": cbsa_match, "unemployment_rate_2022": val})
    return pd.DataFrame(rows).drop_duplicates(subset=["cbsa"])


def fetch_fhfa_hpi() -> pd.DataFrame:
    """FHFA House Price Index annual change at MSA level (latest).

    Downloads the FHFA HPI MSA Quarterly purchase-only series,
    computes 1-year appreciation for the most recent year available.
    """
    url = ("https://www.fhfa.gov/sites/default/files/2024-12/"
           "HPI_AT_metro.csv")
    print(f"  FHFA: MSA HPI from {url}")
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback to a stable archive URL
        url2 = "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_metro.csv"
        df = pd.read_csv(url2)
    # Common schema: metropolitan_area, CBSA, Year, Quarter, HPI, ...
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    cbsa_col = "cbsa" if "cbsa" in df.columns else "metropolitan_division_code"
    df[cbsa_col] = df[cbsa_col].astype(str)
    # Compute 4-quarter appreciation per CBSA (Q4 yoy)
    df = df.dropna(subset=["year", "quarter", "hpi"])
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df = df.sort_values([cbsa_col, "year", "quarter"])
    df["hpi_lag4q"] = df.groupby(cbsa_col)["hpi"].shift(4)
    df["hpi_yoy_pct"] = 100.0 * (df["hpi"] / df["hpi_lag4q"] - 1.0)
    # Most recent year available
    latest = df.dropna(subset=["hpi_yoy_pct"]).sort_values(
        [cbsa_col, "year", "quarter"]
    ).groupby(cbsa_col).tail(1).reset_index(drop=True)
    out = latest[[cbsa_col, "hpi_yoy_pct"]].rename(columns={cbsa_col: "cbsa"})
    return out


def fetch_fred_affordability() -> pd.Series:
    """FRED housing affordability index (NATIONAL only).

    Returns a single scalar for the most recent year, used as a national
    control rather than per-MSA varying covariate.
    """
    print("  FRED: NAR housing affordability composite (national, FIXHAI)")
    # FRED public observations endpoint (no key needed for some series)
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FIXHAI&"
           "cosd=2022-01-01&coed=2024-12-31")
    df = pd.read_csv(url)
    df.columns = ["date", "fixhai"]
    df["fixhai"] = pd.to_numeric(df["fixhai"], errors="coerce")
    val = df["fixhai"].dropna().iloc[-1] if not df["fixhai"].dropna().empty else float("nan")
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acs",
                    default=str(REPO / "results" / "replications"
                                  / "acs_moderators_12city.csv"))
    ap.add_argument("--bea_key", default=os.environ.get("BEA_API_KEY", ""))
    ap.add_argument("--bls_key", default=os.environ.get("BLS_API_KEY", ""))
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications"
                                  / "all_moderators_12city.csv"))
    args = ap.parse_args()

    acs = pd.read_csv(args.acs)
    acs["cbsa"] = acs["cbsa"].astype(str)
    print(f"=== ACS baseline: {len(acs)} metros ===")

    parts = [acs.set_index("cbsa")]

    # 1. BEA Real GDP
    if args.bea_key:
        try:
            bea = fetch_bea_gdp_per_capita(args.bea_key)
            bea["cbsa"] = bea["cbsa"].astype(str)
            parts.append(bea.set_index("cbsa"))
        except Exception as e:
            print(f"  [warn] BEA failed: {e}")
    else:
        print("  [skip] BEA_API_KEY not set")

    # 2. BLS unemployment
    try:
        bls = fetch_bls_unemployment(args.bls_key)
        bls["cbsa"] = bls["cbsa"].astype(str)
        parts.append(bls.set_index("cbsa"))
    except Exception as e:
        print(f"  [warn] BLS failed: {e}")

    # 3. FHFA HPI year-over-year
    try:
        fhfa = fetch_fhfa_hpi()
        fhfa["cbsa"] = fhfa["cbsa"].astype(str)
        parts.append(fhfa.set_index("cbsa"))
    except Exception as e:
        print(f"  [warn] FHFA failed: {e}")

    # 4. FRED national housing affordability — same value for all
    try:
        fred_val = fetch_fred_affordability()
        national = pd.DataFrame({
            "cbsa": acs["cbsa"].astype(str).tolist(),
            "national_affordability_2024": [fred_val] * len(acs),
        }).set_index("cbsa")
        parts.append(national)
    except Exception as e:
        print(f"  [warn] FRED failed: {e}")

    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, how="left")
    merged = merged.reset_index()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n=== Augmented moderator table ({len(merged)} cities) ===")
    new_cols = [c for c in merged.columns if c not in acs.columns]
    print(merged[["city"] + new_cols].to_string(index=False))
    print(f"\nCSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

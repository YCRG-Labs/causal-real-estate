"""Fetch additional metro-level moderators via FRED with hardcoded series IDs.

The series IDs below were verified by query against FRED's search API on
2026-06-03. They are stable identifiers that don't change unless FRED
discontinues a series.  Three federal moderators per city:

  unemp_rate    BLS LAUS via FRED {CITY4}{STATE3}URN format (e.g.
                BOST625URN for Boston-Cambridge-Newton).  Annual avg.
  hpi_yoy       FHFA All-Transactions HPI via FRED ATNHPIUS{CBSA}Q.
                Year-over-year % change at the most recent quarter.
  pcpi          BEA per-capita personal income via FRED {CITY4}{STATE3}PCPI
                format (e.g. BOST625PCPI).  Most recent annual value.

Reference: Kurlat-Stroebel (2015) RFS buyer-information framework;
Anenberg-Bayer (2020) IER metropolitan-housing volatility; Chinco-Mayer
(2016) RFS for the recent-mover misinformation channel.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]

# Hardcoded FRED series IDs per city, discovered by search API query on
# 2026-06-03.  NYC HPI is at the NY-Jersey City-White Plains division.
FRED_SERIES = {
    "boston":       {"unemp": "BOST625URN", "hpi": "ATNHPIUS14454Q",
                      "pcpi": "BOST625PCPI"},
    "nyc":          {"unemp": "NEWY636URN", "hpi": "ATNHPIUS35614Q",
                      "pcpi": "NEWY636PCPI"},   # NY-Jersey City-White Plains PMD
    "sf":           {"unemp": "SANF806URN", "hpi": "ATNHPIUS41884Q",
                      "pcpi": "SANF806PCPI"},
    "dc":           {"unemp": "WASH911URN", "hpi": "ATNHPIUS47894Q",
                      "pcpi": "WASH911PCPI"},
    "philadelphia": {"unemp": "PHIL942URN", "hpi": "ATNHPIUS37964Q",
                      "pcpi": "PHIL942PCPI"},
    "chicago":      {"unemp": "CHIC917URN", "hpi": "ATNHPIUS16974Q",
                      "pcpi": "CHIC917PCPI"},
    "seattle":      {"unemp": "SEAT653URN", "hpi": "ATNHPIUS42644Q",
                      "pcpi": "SEAT653PCPI"},
    "denver":       {"unemp": "DENV708URN", "hpi": "ATNHPIUS19740Q",
                      "pcpi": "DENV708PCPI"},
    "atlanta":      {"unemp": "ATLA013URN", "hpi": "ATNHPIUS12060Q",
                      "pcpi": "ATLA013PCPI"},
    "portland":     {"unemp": "PORT941URN", "hpi": "ATNHPIUS38900Q",
                      "pcpi": "PORT941PCPI"},
    "phoenix":      {"unemp": "PHOE004URN", "hpi": "ATNHPIUS38060Q",
                      "pcpi": "PHOE004PCPI"},
    "dallas":       {"unemp": "DALL148URN", "hpi": "ATNHPIUS19124Q",
                      "pcpi": "DALL148PCPI"},
}

FRED_OBS = "https://api.stlouisfed.org/fred/series/observations"
FRED_SEARCH = "https://api.stlouisfed.org/fred/series/search"


def _fred_latest(series_id: str, api_key: str, n_tail: int = 8):
    params = {"series_id": series_id, "api_key": api_key,
              "file_type": "json", "sort_order": "desc",
              "limit": str(n_tail)}
    try:
        r = requests.get(FRED_OBS, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
    except Exception:
        return None, float("nan")
    for o in obs:
        if o.get("value") not in (".", None, ""):
            try:
                return o["date"], float(o["value"])
            except ValueError:
                continue
    return None, float("nan")


def _fred_yoy(series_id: str, api_key: str):
    params = {"series_id": series_id, "api_key": api_key,
              "file_type": "json", "sort_order": "desc", "limit": "8"}
    try:
        r = requests.get(FRED_OBS, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
    except Exception:
        return None, float("nan")
    if len(obs) < 5:
        return None, float("nan")
    try:
        latest = float(obs[0]["value"])
        year_ago = float(obs[4]["value"])
        return obs[0]["date"], 100.0 * (latest / year_ago - 1.0)
    except (ValueError, KeyError):
        return None, float("nan")


def _fred_search_first(text: str, api_key: str):
    """Last-resort fallback: search FRED and return the first hit."""
    params = {"search_text": text, "api_key": api_key,
              "file_type": "json", "limit": "1",
              "order_by": "popularity"}
    try:
        r = requests.get(FRED_SEARCH, params=params, timeout=20)
        return r.json().get("seriess", [{}])[0].get("id")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acs",
                    default=str(REPO / "results" / "replications"
                                  / "acs_moderators_12city.csv"))
    ap.add_argument("--fred_key", default=os.environ.get("FRED_API_KEY", ""))
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications"
                                  / "all_moderators_12city.csv"))
    args = ap.parse_args()

    if not args.fred_key:
        print("ERROR: FRED_API_KEY not set", file=sys.stderr)
        return 1

    acs = pd.read_csv(args.acs)
    print(f"=== ACS baseline: {len(acs)} metros ===\n")

    rows = []
    for city, ids in FRED_SERIES.items():
        d, unemp = _fred_latest(ids["unemp"], args.fred_key)
        _, hpi_yoy = _fred_yoy(ids["hpi"], args.fred_key)
        _, pcpi = _fred_latest(ids["pcpi"], args.fred_key)

        # Last-resort search fallback for NaN cells.
        if np.isnan(hpi_yoy):
            alt = _fred_search_first(
                f"All-Transactions House Price Index {city}", args.fred_key,
            )
            if alt:
                _, hpi_yoy = _fred_yoy(alt, args.fred_key)
                if not np.isnan(hpi_yoy):
                    print(f"  [{city}] HPI fallback: {alt}")

        rows.append({
            "city": city,
            "fred_unemp_rate": unemp,
            "fred_hpi_yoy_pct": hpi_yoy,
            "fred_pcpi": pcpi,
        })
        print(f"  {city:14s}  unemp={unemp:>6.2f}  "
              f"hpi_yoy={hpi_yoy:>+6.2f}%  pcpi=${pcpi:>9,.0f}"
              if not np.isnan(pcpi) else f"  {city:14s}  partial")
        time.sleep(0.05)

    fred_df = pd.DataFrame(rows)
    merged = acs.merge(fred_df, on="city", how="left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n=== Final augmented table ===")
    print(merged[["city", "fred_unemp_rate", "fred_hpi_yoy_pct",
                   "fred_pcpi"]].to_string(index=False))
    print(f"\nCSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

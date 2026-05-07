"""Re-derive raw Redfin listing descriptions for the embedding subset.

Raw descriptions are NOT redistributed in this dataset because Redfin's Terms
of Service prohibit redistribution and listing remarks carry MLS copyright.
This script lets you regenerate them against your own Redfin access; the
output joins back to the released embedding parquets via lat/lon and zip.

Usage:
    python scripts/fetch_descriptions.py --city nyc --out descriptions.csv

You assume responsibility for compliance with Redfin's Terms of Service and
applicable MLS rules when running this script. The authors of this dataset
do not authorize or endorse scraping in violation of any third party's terms.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

CITY_PARAMS = {
    "boston": {"region_id": 1826, "region_type": 6, "max_pages": 10},
    "nyc": {"region_id": 30749, "region_type": 6, "max_pages": 10},
    "sf": {
        "region_id": 17151,
        "region_type": 6,
        "max_pages": 30,
        "bbox": {"lat_min": 37.70, "lat_max": 37.83, "lon_min": -122.52, "lon_max": -122.36},
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SEARCH_URL = (
    "https://www.redfin.com/stingray/api/gis?al=1&num_homes=350&ord=redfin-recommended-asc"
    "&page=1&sf=1,2,3,5,6,7&status=9&uipt=1,2,3&v=8"
)


def fetch_listings(city: str) -> list[dict]:
    if city not in CITY_PARAMS:
        raise SystemExit(f"unknown city: {city}")
    params = CITY_PARAMS[city]
    out: list[dict] = []
    for page in range(1, params["max_pages"] + 1):
        url = f"{SEARCH_URL}&region_id={params['region_id']}&region_type={params['region_type']}&page={page}"
        if "bbox" in params:
            b = params["bbox"]
            url += f"&lat_min={b['lat_min']}&lat_max={b['lat_max']}&long_min={b['lon_min']}&long_max={b['lon_max']}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  page {page}: {resp.status_code}, stopping")
            break
        text = resp.text[4:] if resp.text.startswith("{}&&") else resp.text
        homes = json.loads(text).get("payload", {}).get("homes", [])
        if not homes:
            break
        for h in homes:
            out.append({
                "address": h.get("streetLine", {}).get("value", ""),
                "zip": h.get("zip", ""),
                "latitude": h.get("latLong", {}).get("latitude"),
                "longitude": h.get("latLong", {}).get("longitude"),
                "price": h.get("price", {}).get("value"),
                "url": h.get("url", ""),
            })
        time.sleep(random.uniform(3, 6))
    return out


def fetch_description(url: str) -> str:
    full = f"https://www.redfin.com{url}" if url.startswith("/") else url
    resp = requests.get(full, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    div = (
        soup.find("div", {"id": "TextContent-TextContent"})
        or soup.find("div", class_=re.compile("remarks"))
        or soup.find("p", class_=re.compile("description"))
    )
    return div.get_text(strip=True) if div else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=list(CITY_PARAMS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    listings = fetch_listings(args.city)
    print(f"fetched {len(listings)} listings, retrieving descriptions...")
    rows = []
    for i, lst in enumerate(listings, 1):
        desc = fetch_description(lst["url"]) if lst.get("url") else ""
        rows.append({**lst, "description": desc})
        if i % 25 == 0:
            print(f"  {i}/{len(listings)}")
        time.sleep(random.uniform(2, 4))
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

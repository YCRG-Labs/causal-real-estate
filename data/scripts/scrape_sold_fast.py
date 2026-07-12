"""Fast Redfin SOLD-listing scraper — descriptions + sale prices in bulk.

Two facts make this ~100x faster than per-page fetching (both verified live):
  * the GIS JSON returns `listingRemarks` (the description) for every home, so no
    per-/home/<id> page fetch is needed (remarks are capped at ~699 chars in GIS,
    which covers ~95% of listings untruncated);
  * SOLD homes come from status=4 + sold_within_days, carrying price.value (the
    SALE price), soldDate (epoch ms), and all structured attributes.

The GIS endpoint hard-caps any single query at ~350 results (page_number does not
paginate), so we tile by PRICE BAND and split any band that hits the cap in half,
recursively, until every leaf returns < cap. Dedupe by propertyId.

    python data/scripts/scrape_sold_fast.py --city boston --years 3
    python data/scripts/scrape_sold_fast.py --all --years 3 --delay 0.4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from city_endpoints import CITIES

OUT = REPO / "results" / "sold_scrape"
OUT.mkdir(parents=True, exist_ok=True)
GIS = "https://www.redfin.com/stingray/api/gis"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.redfin.com/",
}
CAP = 350
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _get(region_id, lo, hi, sold_days, delay):
    p = {"al": 1, "mpt": 99, "num_homes": CAP, "ord": "redfin-recommended-asc",
         "page_number": 1, "region_id": region_id, "region_type": 6,
         "sf": "1,2,3,5,6,7", "sold_within_days": sold_days, "status": 4,
         "uipt": "1,2,3,4,5,6,7,8", "v": 8, "min_price": int(lo), "max_price": int(hi)}
    url = GIS + "?" + urllib.parse.urlencode(p)
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS),
                                        timeout=30) as r:
                t = r.read().decode("utf-8", "ignore")
            if t.startswith("{}&&"):
                t = t[4:]
            time.sleep(delay)
            return json.loads(t).get("payload", {}).get("homes", [])
        except Exception as e:  # noqa: BLE001
            if i == 3:
                print(f"    fail band {lo:.0f}-{hi:.0f}: {e}", file=sys.stderr)
                return []
            time.sleep(1.5 * (i + 1))


def _v(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse(h):
    return {
        "property_id": h.get("propertyId"),
        "url": "https://www.redfin.com" + (h.get("url") or ""),
        "address": _v(h, "streetLine", "value"), "zip": _v(h, "postalCode", "value"),
        "latitude": _v(h, "latLong", "value", "latitude"),
        "longitude": _v(h, "latLong", "value", "longitude"),
        "sale_price": _v(h, "price", "value"),
        "sold_date_ms": h.get("soldDate"),
        "beds": h.get("beds"), "baths": h.get("baths"),
        "sqft": _v(h, "sqFt", "value"), "lot_size": _v(h, "lotSize", "value"),
        "year_built": _v(h, "yearBuilt", "value"),
        "property_type": h.get("uiPropertyType"),
        "mls_status": h.get("mlsStatus"),
        "description": h.get("listingRemarks") or "",
    }


def scrape_city(city, sold_days, delay, max_depth=16):
    region_id = CITIES[city].redfin_city_id
    seen, rows, calls = set(), [], [0]

    def recurse(lo, hi, depth):
        homes = _get(region_id, lo, hi, sold_days, delay)
        calls[0] += 1
        new = 0
        for h in homes:
            pid = h.get("propertyId")
            if pid and pid not in seen:
                seen.add(pid)
                rows.append(parse(h))
                new += 1
        # band saturated the cap and price range still splittable -> subdivide
        if len(homes) >= CAP and depth < max_depth and (hi - lo) > 500:
            mid = (lo + hi) / 2
            recurse(lo, mid, depth + 1)
            recurse(mid, hi, depth + 1)

    for _ in range(3):
        recurse(0, 5_000_000, 0)          # dense range, resolves fast
        recurse(5_000_000, 200_000_000, 18)  # luxury tail, single sparse call
        if rows:
            break
        time.sleep(3.0)
    return pd.DataFrame(rows), calls[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()
    cities = ALL_12 if args.all else [args.city]
    sold_days = args.years * 365

    for city in cities:
        t0 = time.time()
        df, calls = scrape_city(city, sold_days, args.delay)
        if len(df):
            df["sold_date"] = pd.to_datetime(df["sold_date_ms"], unit="ms", errors="coerce")
        dst = OUT / f"{city}_sold.parquet"
        df.to_parquet(dst)
        dt = time.time() - t0
        wd = (df.description.str.len() > 0).mean() if len(df) else 0
        wp = df.sale_price.notna().mean() if len(df) else 0
        drange = (f"{df.sold_date.min().date()}..{df.sold_date.max().date()}"
                  if len(df) and df.sold_date.notna().any() else "n/a")
        print(f"{city}: {len(df)} sold homes in {calls} calls, {dt:.0f}s | "
              f"desc {100*wd:.0f}% price {100*wp:.0f}% | sold {drange} | "
              f"median ${df.sale_price.median():,.0f}" if len(df) else f"{city}: 0")


if __name__ == "__main__":
    main()

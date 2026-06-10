"""Attach a recorded SALE price and date to each scraped listing by matching the
listing to its nearest recorded-sale parcel, producing the sale-price outcome the
text-valuation literature uses in place of the scraped asking price.

Input  : data/processed/<city>_recorded_sales.parquet  (lat, lon, sale_price, sale_date)
         data/processed/<city>_listings.parquet         (lat_centroid, lon_centroid, ...)
Output : data/processed/<city>_listings_saleprice.parquet
         (sha16, list_price, sale_price, sale_date, match_dist_m, lat, lon)

Match is nearest-neighbour in projected meters within --max_dist (default 40 m,
about one parcel), so a listing is paired to the recorded sale of the parcel it
sits on. This is the universal join step; every market reuses it once its
<city>_recorded_sales.parquet exists.

  python3 data/scripts/join_sales_to_listings.py --city philadelphia
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"


def _latlon_to_m(lat, lon, lat0, lon0):
    R = 6_371_000.0
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * R
    y = np.radians(lat - lat0) * R
    return np.column_stack([x, y])


def join_city(city: str, max_dist: float) -> pd.DataFrame:
    sp = PROC / f"{city}_recorded_sales.parquet"
    lp = PROC / f"{city}_listings.parquet"
    if not sp.exists():
        raise SystemExit(f"missing {sp} -- run download_sales.py --city {city} first")
    sales = pd.read_parquet(sp).dropna(subset=["lat", "lon", "sale_price"])
    L = pd.read_parquet(lp).copy()
    L["sha16"] = L["source_html_sha256"].astype(str).str[:16]
    L = L.dropna(subset=["lat_centroid", "lon_centroid"])

    lat0, lon0 = float(L["lat_centroid"].mean()), float(L["lon_centroid"].mean())
    S = _latlon_to_m(sales["lat"].to_numpy(), sales["lon"].to_numpy(), lat0, lon0)
    Q = _latlon_to_m(L["lat_centroid"].to_numpy(), L["lon_centroid"].to_numpy(), lat0, lon0)
    tree = cKDTree(S)
    dist, idx = tree.query(Q)
    matched = dist <= max_dist

    out = pd.DataFrame({
        "sha16": L["sha16"].to_numpy(),
        "list_price": pd.to_numeric(L.get("price"), errors="coerce").to_numpy(),
        "sale_price": sales["sale_price"].to_numpy()[idx],
        "sale_date": sales["sale_date"].to_numpy()[idx],
        "match_dist_m": dist,
        "lat": L["lat_centroid"].to_numpy(),
        "lon": L["lon_centroid"].to_numpy(),
    })
    out = out[matched].reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--max_dist", type=float, default=40.0, help="match radius in meters")
    args = ap.parse_args()

    out = join_city(args.city, args.max_dist)
    n_list = len(pd.read_parquet(PROC / f"{args.city}_listings.parquet"))
    dst = PROC / f"{args.city}_listings_saleprice.parquet"
    out.to_parquet(dst, index=False)

    sp = out["sale_price"]
    lp = out["list_price"]
    ratio = (lp / sp).replace([np.inf, -np.inf], np.nan).dropna()
    yrs = pd.to_datetime(out["sale_date"], errors="coerce", utc=True).dt.year
    print(f"{args.city}: {len(out):,}/{n_list:,} listings matched to a recorded sale "
          f"({100*len(out)/n_list:.0f}%) within {args.max_dist:.0f} m")
    print(f"  median sale ${sp.median():,.0f}  median list ${lp.median():,.0f}  "
          f"list/sale median {ratio.median():.2f}")
    if yrs.notna().any():
        print(f"  sale years {int(yrs.min())}-{int(yrs.max())}  "
              f"2020+ {100*(yrs>=2020).mean():.0f}%  2023+ {100*(yrs>=2023).mean():.0f}%")
    print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

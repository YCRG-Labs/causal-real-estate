from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "results" / "sold_scrape"
OUT = REPO / "results" / "soldprice"
OUT.mkdir(parents=True, exist_ok=True)
CITIES = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]

RESIDENTIAL = {1, 2, 3, 4}
SINGLE_UNIT = {1, 3}
MIN_PRICE = 1000
MIN_DESC = 50


def clean_city(city):
    df = pd.read_parquet(SRC / f"{city}_sold.parquet")
    n0 = len(df)
    code = pd.to_numeric(df["property_type"], errors="coerce")
    price = pd.to_numeric(df["sale_price"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    sold = pd.to_datetime(df["sold_date"], errors="coerce")
    desc_len = df["description"].fillna("").str.len()

    keep = (code.isin(RESIDENTIAL) & (price > MIN_PRICE) & sold.notna()
            & np.isfinite(lat) & np.isfinite(lon) & (desc_len >= MIN_DESC))
    steps = {
        "start": n0,
        "residential(1-4)": int((code.isin(RESIDENTIAL)).sum()),
        "+price>1000": int((code.isin(RESIDENTIAL) & (price > MIN_PRICE)).sum()),
        "+sold_date": int((code.isin(RESIDENTIAL) & (price > MIN_PRICE) & sold.notna()).sum()),
        "+coords+desc": int(keep.sum()),
    }
    d = df[keep].copy()
    d["price"] = price[keep]
    d["log_price"] = np.log(d["price"])
    d["property_code"] = code[keep].astype(int)
    d["is_single_unit"] = d["property_code"].isin(SINGLE_UNIT)
    d["latitude"] = lat[keep]
    d["longitude"] = lon[keep]
    d["sold_date"] = sold[keep]
    d["sold_quarter"] = d["sold_date"].dt.to_period("Q").astype(str)

    before = len(d)
    d = d.drop_duplicates(subset=["address", "price", "description"], keep="first")
    dupes = before - len(d)

    d = d.reset_index(drop=True)
    assert d["price"].gt(MIN_PRICE).all() and d["sold_date"].notna().all()
    assert np.isfinite(d["latitude"]).all() and np.isfinite(d["longitude"]).all()
    return d, steps, dupes


def main():
    rows = []
    print(f"{'city':13s}{'start':>8s}{'resid':>8s}{'final':>8s}{'dupes':>7s}"
          f"{'single%':>9s}{'med$':>10s}{'quarters':>9s}")
    print("-" * 74)
    total = 0
    for c in CITIES:
        d, steps, dupes = clean_city(c)
        d.to_parquet(OUT / f"{c}_panel.parquet")
        total += len(d)
        rows.append({"city": c, **steps, "dupes_dropped": dupes, "final": len(d)})
        print(f"{c:13s}{steps['start']:8d}{steps['residential(1-4)']:8d}{len(d):8d}"
              f"{dupes:7d}{100*d['is_single_unit'].mean():8.0f}%"
              f"{d['price'].median():10,.0f}{d['sold_quarter'].nunique():9d}")
    print("-" * 74)
    print(f"{'TOTAL':13s}{'':8s}{'':8s}{total:8d}")
    pd.DataFrame(rows).to_json(OUT / "panel_summary.json", orient="records", indent=2)
    print(f"\nwrote {OUT}/{{city}}_panel.parquet  (+ panel_summary.json)")


if __name__ == "__main__":
    main()

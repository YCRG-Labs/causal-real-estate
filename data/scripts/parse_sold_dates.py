"""Recover the per-listing sale DATE from cached Redfin HTML, offline.

The scrape retained source HTML (gzip, named by sha256 prefix) but never parsed
the sale date into the listings table, only the sale price. The status=9 scrape
returns SOLD listings whose sale dates span ~1975-2026, so the outcome (nominal
last-sold price) is denominated across five decades with no time control. This
script recovers the date so sale-year / year-quarter fixed effects can enter the
confounder set.

Signal: each listing-detail page renders its primary home first, carrying a
`lastSoldDate` epoch field. Where both `lastSoldDate` and the first `soldDate`
occurrence are present they agree in 100% of audited pages, and the first
`soldDate` is present on every page, so we take the first `soldDate` as the
listing's sale date and use `lastSoldDate` only to validate.

No network. Reads data/raw/redfin/<city>/html/*.html.gz, joins to the listings
table by sha256 prefix, writes data/processed/<city>_sold_dates.parquet with
columns: sha16, sale_date, sale_year, sale_quarter (e.g. '2022Q3').

  python3 data/scripts/parse_sold_dates.py --all_12
  python3 data/scripts/parse_sold_dates.py --city atlanta
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "redfin"
PROC = REPO / "data" / "processed"

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

_FIRST_SOLD = re.compile(rb'soldDate\\?":(\d{12,13})')
_LAST_SOLD = re.compile(rb'lastSoldDate\\?":(\d{12,13})')
_HEAD_BYTES = 262_144


def _epoch_ms_to_date(ms: int) -> dt.date:
    return dt.datetime.utcfromtimestamp(ms / 1000).date()


def _sale_date_from_html(path: Path) -> tuple[str, str] | None:
    """Return (sha16, ISO date) or None. Reads a head slice first; falls back to
    the full member only if the primary soldDate has not yet appeared."""
    sha16 = path.name.replace(".html.gz", "")
    try:
        with gzip.open(path, "rb") as f:
            head = f.read(_HEAD_BYTES)
            m = _FIRST_SOLD.search(head)
            if m is None:
                rest = f.read()
                m = _FIRST_SOLD.search(head + rest)
    except (OSError, EOFError):
        return None
    if m is None:
        return None
    try:
        d = _epoch_ms_to_date(int(m.group(1)))
    except (ValueError, OverflowError, OSError):
        return None
    return sha16, d.isoformat()


def _validate(path: Path) -> bool | None:
    """True/False if first==last when both present, else None (no lastSoldDate)."""
    try:
        with gzip.open(path, "rb") as f:
            html = f.read()
    except (OSError, EOFError):
        return None
    mf, ml = _FIRST_SOLD.search(html), _LAST_SOLD.search(html)
    if mf is None or ml is None:
        return None
    return mf.group(1) == ml.group(1)


def parse_city(city: str, workers: int, audit: int = 400) -> pd.DataFrame:
    html_dir = RAW / city / "html"
    files = sorted(html_dir.glob("*.html.gz"))
    if not files:
        print(f"  {city}: no cached HTML at {html_dir}")
        return pd.DataFrame(columns=["sha16", "sale_date", "sale_year", "sale_quarter"])

    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(_sale_date_from_html, files, chunksize=64) if r]
        agree = [v for v in ex.map(_validate, files[:audit]) if v is not None]

    df = pd.DataFrame(rows, columns=["sha16", "sale_date"])
    d = pd.to_datetime(df["sale_date"])
    df["sale_year"] = d.dt.year
    df["sale_quarter"] = d.dt.year.astype(str) + "Q" + d.dt.quarter.astype(str)
    rate = (sum(agree) / len(agree)) if agree else float("nan")
    yrs = df["sale_year"]
    pre20 = (yrs < 2020).mean() if len(yrs) else float("nan")
    print(f"  {city:13s} files={len(files):6d}  dated={len(df):6d} "
          f"({100*len(df)/max(len(files),1):.0f}%)  "
          f"first==last={rate:.3f} (n={len(agree)})  "
          f"yrs {int(yrs.min())}-{int(yrs.max())} median={int(yrs.median())} "
          f"pre2020={100*pre20:.0f}%")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cities = ALL_12 if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")

    PROC.mkdir(parents=True, exist_ok=True)
    for city in cities:
        df = parse_city(city, workers=args.workers)
        if not df.empty:
            out = PROC / f"{city}_sold_dates.parquet"
            df.to_parquet(out, index=False)
            print(f"    -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

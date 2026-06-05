"""Convert the 12-city ZIP-shard listings parquets into the CSV format the
embedding pipeline expects.

generate_embeddings.py reads `data/raw/descriptions/{city}_descriptions.csv`
with columns including `description`, `latitude`, `longitude`. The fresh
Phase-2 scrape writes `data/processed/{city}_listings.parquet` with columns
`description`, `lat_centroid`, `lon_centroid` and the larger 75,480-listing
corpus. This adapter bridges the two so embeddings re-run against the new
data without touching the encoder script itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "data" / "processed"
DST_DIR = REPO / "data" / "raw" / "descriptions"

CITIES = [
    "sf", "boston", "nyc", "dc", "philadelphia", "chicago",
    "seattle", "denver", "atlanta", "portland", "phoenix", "dallas",
]


def convert_one(city: str) -> int:
    src = SRC_DIR / f"{city}_listings.parquet"
    dst = DST_DIR / f"{city}_descriptions.csv"
    if not src.exists():
        print(f"{city}: missing parquet at {src}, skipping", file=sys.stderr)
        return 0
    df = pd.read_parquet(src)
    df = df.rename(columns={"lat_centroid": "latitude",
                            "lon_centroid": "longitude"})
    if "description" not in df.columns:
        print(f"{city}: no description column, skipping", file=sys.stderr)
        return 0
    # Redfin ships ZIP+4 strings like "11355-4788"; pgeocode and the
    # downstream geocoder treat ZIP as numeric, so we collapse to the
    # 5-digit base ZIP. Anything that already is 5-digit passes through.
    # Drop rows where the ZIP can't be parsed: downstream geocoding does an
    # unconditional df["zip"].astype(float) and would crash on NaN.
    if "zip" in df.columns:
        df["zip"] = (df["zip"].astype(str)
                              .str.extract(r"^(\d{5})", expand=False))
        df = df[df["zip"].notna()]
    keep = df["description"].notna() & (df["description"].str.len() > 20)
    df = df.loc[keep].copy()
    DST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"{city}: {len(df):>6} rows -> {dst.relative_to(REPO)}")
    return len(df)


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else CITIES
    total = 0
    for c in cities:
        total += convert_one(c)
    print(f"\ntotal: {total:,} listings across {len(cities)} cities")


if __name__ == "__main__":
    main()

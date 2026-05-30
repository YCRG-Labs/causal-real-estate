"""Census Bureau batch geocoder for the 12-city expansion.

Replaces per-address serial calls with a single CSV upload to the documented
batch endpoint at geocoding.geo.census.gov/geocoder/locations/addressbatch.
Maximum 10,000 records per call; our 4,200 addresses across 12 cities fit
in one batch. Cuts geocoding from ~1 hour to ~2 minutes.

Usage:
    python geocode_batch.py --cities new9
    python geocode_batch.py --cities all --benchmark Public_AR_Current
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_endpoints import CITIES, list_new, list_ready  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
DEFAULT_BENCHMARK = "Public_AR_Current"
MAX_BATCH_SIZE = 10_000


def load_listings_for_geocoding(slugs: list[str]) -> pd.DataFrame:
    frames = []
    for slug in slugs:
        path = PROCESSED_DIR / f"{slug}_listings.parquet"
        if not path.exists():
            print(f"  [{slug}] no listings parquet at {path}, skipping", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        if "address" not in df.columns or len(df) == 0:
            print(f"  [{slug}] parquet has no address column or is empty, skipping", file=sys.stderr)
            continue
        df = df[df["address"].astype(str).str.strip() != ""].copy()
        if df.empty:
            print(f"  [{slug}] all addresses empty after filter, skipping", file=sys.stderr)
            continue
        df["__city_slug"] = slug
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_batch_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    for _, row in df.iterrows():
        unique_id = f"{row['__city_slug']}__{row['url']}"
        addr = str(row.get("address", "")).strip()
        parts = addr.split(",")
        street = parts[0].strip() if parts else addr
        city = parts[1].strip() if len(parts) > 1 else ""
        state = ""
        zip_code = str(row.get("zip", "")).strip()
        if len(parts) >= 3:
            sz = parts[2].strip().split()
            if len(sz) >= 1:
                state = sz[0]
            if len(sz) >= 2 and not zip_code:
                zip_code = sz[1]
        writer.writerow([unique_id, street, city, state, zip_code])
    return buf.getvalue().encode("utf-8")


def post_batch(csv_bytes: bytes, benchmark: str, timeout_s: int = 600) -> str:
    files = {"addressFile": ("addresses.csv", csv_bytes, "text/csv")}
    data = {"benchmark": benchmark}
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0)) as client:
        r = client.post(BATCH_URL, files=files, data=data)
    r.raise_for_status()
    return r.text


def parse_batch_response(response_text: str) -> pd.DataFrame:
    rows = []
    reader = csv.reader(io.StringIO(response_text))
    for parts in reader:
        if len(parts) < 4:
            continue
        unique_id = parts[0]
        match_status = parts[2] if len(parts) > 2 else ""
        match_type = parts[3] if len(parts) > 3 else ""
        matched_address = parts[4] if len(parts) > 4 else ""
        coords = parts[5] if len(parts) > 5 else ""
        lon, lat = None, None
        if coords and "," in coords:
            try:
                lon_s, lat_s = coords.split(",", 1)
                lon = float(lon_s)
                lat = float(lat_s)
            except (ValueError, TypeError):
                pass
        rows.append({
            "unique_id": unique_id,
            "match_status": match_status,
            "match_type": match_type,
            "matched_address": matched_address,
            "geocoded_lat": lat,
            "geocoded_lon": lon,
        })
    return pd.DataFrame(rows)


def chunk(df: pd.DataFrame, size: int = MAX_BATCH_SIZE):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size].copy()


def attach_geocodes_to_listings(df_listings: pd.DataFrame, df_geocoded: pd.DataFrame) -> dict[str, int]:
    df_geocoded["__city_slug"] = df_geocoded["unique_id"].str.split("__", n=1).str[0]
    df_geocoded["url"] = df_geocoded["unique_id"].str.split("__", n=1).str[1]
    counts: dict[str, int] = {}
    for slug, group in df_geocoded.groupby("__city_slug"):
        path = PROCESSED_DIR / f"{slug}_listings.parquet"
        if not path.exists():
            continue
        live = pd.read_parquet(path)
        merged = live.merge(
            group[["url", "geocoded_lat", "geocoded_lon", "match_status"]],
            on="url", how="left",
        )
        merged["lat_centroid"] = merged["geocoded_lat"].combine_first(merged.get("lat_centroid"))
        merged["lon_centroid"] = merged["geocoded_lon"].combine_first(merged.get("lon_centroid"))
        merged.to_parquet(path, index=False)
        n_matched = int((merged["match_status"] == "Match").sum()) if "match_status" in merged.columns else 0
        counts[slug] = n_matched
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Census Bureau batch geocoder")
    parser.add_argument("--cities", default="new9")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    args = parser.parse_args()
    if args.cities == "new9":
        cities = list_new()
    elif args.cities == "all":
        cities = list_ready()
    else:
        slugs = [s.strip() for s in args.cities.split(",") if s.strip()]
        cities = [CITIES[s] for s in slugs]
    slugs = [c.slug for c in cities]
    print(f"Loading listings for {len(slugs)} cities: {slugs}")
    df = load_listings_for_geocoding(slugs)
    if df.empty:
        print("No listings to geocode", file=sys.stderr)
        return 1
    print(f"  {len(df)} addresses queued for batch geocode")
    results = []
    t0 = time.time()
    for i, batch in enumerate(chunk(df, MAX_BATCH_SIZE), 1):
        n_batches = (len(df) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        print(f"  batch {i}/{n_batches}: uploading {len(batch)} addresses...")
        csv_bytes = build_batch_csv(batch)
        try:
            resp = post_batch(csv_bytes, args.benchmark)
        except httpx.HTTPError as e:
            print(f"    batch {i} failed: {e}", file=sys.stderr)
            continue
        df_batch = parse_batch_response(resp)
        results.append(df_batch)
        print(f"    batch {i}: {df_batch.shape[0]} responses, {int((df_batch['match_status']=='Match').sum())} matches")
    if not results:
        print("No batches succeeded", file=sys.stderr)
        return 1
    df_geocoded = pd.concat(results, ignore_index=True)
    counts = attach_geocodes_to_listings(df, df_geocoded)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min:")
    for slug, n in counts.items():
        print(f"  {slug}: {n} addresses geocoded")
    out_csv = PROCESSED_DIR / "geocode_batch_results.csv"
    df_geocoded.to_csv(out_csv, index=False)
    print(f"\nRaw responses cached at {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

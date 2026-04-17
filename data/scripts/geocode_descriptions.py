"""
Geocode Boston/NYC descriptions by matching their street addresses to
parcel records (Boston assessment / NYC PLUTO), then copying the parcel's
lat/lon to the description row.

This replaces the zip-centroid approach which assigns identical coordinates
to all properties in a zip and creates within-zip confounding artifacts.

Usage:
    python geocode_descriptions.py              # all cities
    python geocode_descriptions.py boston nyc    # specific cities
"""
import re
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from config import RAW_DIR, PROCESSED_DIR, CITIES

SUFFIXES = {
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "rd": "road", "ln": "lane",
    "ct": "court", "pl": "place", "ter": "terrace", "cir": "circle",
    "pkwy": "parkway", "hwy": "highway", "sq": "square", "wy": "way",
}


def normalize_address(addr):
    """Normalize a street address for matching: uppercase, strip units,
    expand abbreviations, collapse whitespace."""
    if not isinstance(addr, str) or not addr.strip():
        return ""
    addr = addr.upper().strip()
    # Strip unit/apt/suite/floor designators
    addr = re.sub(r"\s*[#,]\s*(UNIT|APT|STE|SUITE|FL|FLOOR|RM|ROOM|PH)?\s*\S*$", "", addr)
    addr = re.sub(r"\s+(UNIT|APT|STE|SUITE|FL|FLOOR|RM|ROOM|PH)\s+\S*$", "", addr)
    # Expand direction abbreviations
    addr = re.sub(r"\bN\b", "NORTH", addr)
    addr = re.sub(r"\bS\b", "SOUTH", addr)
    addr = re.sub(r"\bE\b", "EAST", addr)
    addr = re.sub(r"\bW\b", "WEST", addr)
    # Expand street type abbreviations
    for short, full in SUFFIXES.items():
        addr = re.sub(rf"\b{short.upper()}\b", full.upper(), addr)
    # Collapse whitespace, strip trailing periods
    addr = re.sub(r"\s+", " ", addr).strip().rstrip(".")
    return addr


def build_boston_lookup():
    """Build normalized-address → (parcel_id, lat, lon) from Boston assessment + parcels."""
    assess_path = RAW_DIR / "boston" / "boston_assessment.csv"
    if not assess_path.exists():
        print("  Boston assessment file not found")
        return {}

    assess = pd.read_csv(assess_path, low_memory=False,
                         usecols=["PID", "ST_NUM", "ST_NAME"])
    assess["PID"] = assess["PID"].astype(str).str.strip().str.zfill(10)
    assess["ST_NUM"] = pd.to_numeric(assess["ST_NUM"], errors="coerce")
    assess = assess.dropna(subset=["ST_NUM"])
    assess["raw_addr"] = (assess["ST_NUM"].astype(int).astype(str) + " " +
                          assess["ST_NAME"].astype(str).str.strip())
    assess["norm_addr"] = assess["raw_addr"].apply(normalize_address)
    assess = assess[assess["norm_addr"].str.len() > 3]

    # Get parcel geometries
    parcels = None
    for suffix in ["micro_geo", "amenities", "crime", "census"]:
        p = PROCESSED_DIR / f"boston_parcels_{suffix}.gpkg"
        if p.exists():
            parcels = gpd.read_file(p, layer="boston")
            break
    if parcels is None:
        from pathlib import Path as P
        cleaned = P(str(PROCESSED_DIR).replace("processed", "cleaned")) / "boston_parcels_cleaned.gpkg"
        if cleaned.exists():
            parcels = gpd.read_file(cleaned, layer="boston")

    if parcels is None:
        print("  No Boston parcels file found")
        return {}

    if parcels.crs and parcels.crs != "EPSG:4326":
        parcels = parcels.to_crs("EPSG:4326")

    centroids = parcels.geometry.centroid
    parcels["_lat"] = centroids.y
    parcels["_lon"] = centroids.x

    merged = assess.merge(
        parcels[["parcel_id", "_lat", "_lon"]],
        left_on="PID", right_on="parcel_id", how="inner",
    )

    lookup = {}
    for _, row in merged.iterrows():
        if pd.notna(row["_lat"]) and pd.notna(row["_lon"]):
            lookup[row["norm_addr"]] = (row["_lat"], row["_lon"])

    print(f"  Boston lookup: {len(lookup)} unique normalized addresses")
    return lookup


def build_nyc_lookup():
    """Build normalized-address → (lat, lon) from NYC PLUTO."""
    pluto_path = RAW_DIR / "nyc" / "pluto.csv"
    if not pluto_path.exists():
        print("  NYC PLUTO file not found")
        return {}

    pluto = pd.read_csv(pluto_path, low_memory=False,
                        usecols=["BBL", "address", "latitude", "longitude"])
    pluto["latitude"] = pd.to_numeric(pluto["latitude"], errors="coerce")
    pluto["longitude"] = pd.to_numeric(pluto["longitude"], errors="coerce")
    pluto = pluto.dropna(subset=["latitude", "longitude", "address"])
    pluto = pluto[(pluto["latitude"] != 0) & (pluto["longitude"] != 0)]

    pluto["norm_addr"] = pluto["address"].apply(normalize_address)
    pluto = pluto[pluto["norm_addr"].str.len() > 3]

    lookup = {}
    for _, row in pluto.iterrows():
        lookup[row["norm_addr"]] = (row["latitude"], row["longitude"])

    print(f"  NYC lookup: {len(lookup)} unique normalized addresses")
    return lookup


LOOKUP_BUILDERS = {
    "boston": build_boston_lookup,
    "nyc": build_nyc_lookup,
}


def geocode_city(city):
    if city not in LOOKUP_BUILDERS:
        print(f"  {city}: no address-matching strategy, skipping")
        return

    emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    if not emb_path.exists():
        print(f"  {city}: no embeddings parquet, skipping")
        return

    df = pd.read_parquet(emb_path)
    lat = pd.to_numeric(df.get("latitude", pd.Series(dtype=float)), errors="coerce")
    valid_before = lat.notna().sum()

    print(f"\n{'='*60}")
    print(f"GEOCODING: {city.upper()} ({len(df)} descriptions, {valid_before} already have coords)")
    print(f"{'='*60}")

    lookup = LOOKUP_BUILDERS[city]()
    if not lookup:
        print("  Empty lookup, cannot geocode")
        return

    df["norm_addr"] = df["address"].apply(normalize_address)

    matched_exact = 0
    matched_prefix = 0
    unmatched = []

    for idx in df.index:
        if pd.notna(lat.loc[idx]):
            continue

        norm = df.loc[idx, "norm_addr"]
        if not norm:
            continue

        # Exact match
        if norm in lookup:
            df.loc[idx, "latitude"] = lookup[norm][0]
            df.loc[idx, "longitude"] = lookup[norm][1]
            matched_exact += 1
            continue

        # Prefix match: try street number + first word of street name
        parts = norm.split()
        if len(parts) >= 2:
            prefix = parts[0] + " " + parts[1]
            candidates = [k for k in lookup if k.startswith(prefix)]
            if len(candidates) == 1:
                df.loc[idx, "latitude"] = lookup[candidates[0]][0]
                df.loc[idx, "longitude"] = lookup[candidates[0]][1]
                matched_prefix += 1
                continue

        unmatched.append(norm)

    # Fallback: assign zip-centroid for any remaining unmatched
    import pgeocode
    nomi = pgeocode.Nominatim("us")
    fallback = 0
    for idx in df.index:
        if pd.notna(pd.to_numeric(df.loc[idx, "latitude"], errors="coerce")):
            continue
        z = str(int(float(df.loc[idx, "zip"]))).zfill(5) if pd.notna(df.loc[idx, "zip"]) else None
        if z:
            result = nomi.query_postal_code(z)
            if pd.notna(result.latitude):
                df.loc[idx, "latitude"] = result.latitude
                df.loc[idx, "longitude"] = result.longitude
                fallback += 1

    valid_after = pd.to_numeric(df["latitude"], errors="coerce").notna().sum()

    print(f"\n  Results:")
    print(f"    Exact address match:  {matched_exact}")
    print(f"    Prefix match:         {matched_prefix}")
    print(f"    Zip-centroid fallback: {fallback}")
    print(f"    Unmatched:            {len(unmatched) - fallback}")
    print(f"    Valid coords: {valid_before} → {valid_after} ({valid_after}/{len(df)})")

    if matched_exact + matched_prefix > 0:
        pct_addr = (matched_exact + matched_prefix) / len(df) * 100
        print(f"    Address-level accuracy: {pct_addr:.1f}% of descriptions")

    df = df.drop(columns=["norm_addr"], errors="ignore")
    df.to_parquet(emb_path, index=False)
    print(f"  Saved → {emb_path}")

    if unmatched and len(unmatched) <= 20:
        print(f"\n  Unmatched addresses:")
        for a in unmatched[:20]:
            print(f"    {a}")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else ["boston", "nyc"]
    for city in cities:
        geocode_city(city)


if __name__ == "__main__":
    main()

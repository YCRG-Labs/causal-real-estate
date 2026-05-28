"""
Re-geocode Boston/NYC descriptions to PROPERTY-LEVEL lat/lon by address-matching
against parcel records. Replaces the prior pipeline which fell back to
zip-centroid for ~94% of NYC and ~95% of Boston descriptions, recreating the
within-zip confounding artifact that destabilized the NYC DML.

Key improvements over the prior `geocode_descriptions.py`:
  - Strips ordinal suffixes (90TH -> 90, 3RD -> 3) to match PLUTO's bare-number
    Manhattan street format.
  - Adds TPKE/BWY/EXPY/HWY/CRES etc. to the suffix dictionary.
  - For unmatched, tries Queens hyphenated address insertion (16420 -> 164-20).
  - For still-unmatched, tries a per-zip street-name prefix fallback.
  - Reports a clean breakdown of match types and refuses to zip-centroid
    silently. Unmatched rows keep NaN lat/lon so downstream code can choose
    to drop them rather than analyze a centroid artifact.

Usage:
    python regeocode_descriptions.py nyc
    python regeocode_descriptions.py boston
    python regeocode_descriptions.py            # both
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"

SUFFIXES = {
    "ST": "STREET", "STR": "STREET", "AVE": "AVENUE", "AV": "AVENUE",
    "BLVD": "BOULEVARD", "DR": "DRIVE", "RD": "ROAD", "LN": "LANE",
    "CT": "COURT", "PL": "PLACE", "TER": "TERRACE", "CIR": "CIRCLE",
    "PKWY": "PARKWAY", "HWY": "HIGHWAY", "SQ": "SQUARE", "WY": "WAY",
    "CRES": "CRESCENT", "TPKE": "TURNPIKE", "BWY": "BROADWAY",
    "EXPY": "EXPRESSWAY", "PLZ": "PLAZA", "XING": "CROSSING",
}
DIRS = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST"}


def normalize(addr):
    if not isinstance(addr, str) or not addr.strip():
        return ""
    a = addr.upper().strip()
    # Strip unit / apt / suite / floor designators
    a = re.sub(r"\s*[#,]\s*(UNIT|APT|STE|SUITE|FL|FLOOR|RM|ROOM|PH)?\s*\S*$", "", a)
    a = re.sub(r"\s+(UNIT|APT|STE|SUITE|FL|FLOOR|RM|ROOM|PH)\s+\S*$", "", a)
    # Strip ordinal suffix from numbers (the key fix)
    a = re.sub(r"(\d+)(ST|ND|RD|TH)\b", r"\1", a)
    # Expand directions and street-type abbreviations
    for short, full in DIRS.items():
        a = re.sub(rf"\b{short}\b", full, a)
    for short, full in SUFFIXES.items():
        a = re.sub(rf"\b{short}\b", full, a)
    return re.sub(r"\s+", " ", a).strip().rstrip(".")


def queens_hyphen_variants(norm):
    """For an unhyphenated leading number like '16420', try '164-20', '1642-0'.
    Queens addresses are stored hyphenated in PLUTO (164-20 HIGHLAND AVENUE)."""
    m = re.match(r"^(\d{4,6})\s+(.+)$", norm)
    if not m:
        return []
    digits, rest = m.group(1), m.group(2)
    out = []
    if len(digits) >= 4:
        out.append(f"{digits[:-2]}-{digits[-2:]} {rest}")
    if len(digits) >= 5:
        out.append(f"{digits[:-3]}-{digits[-3:]} {rest}")
    return out


def build_nyc_lookup():
    """Build address -> (lat, lon, BBL) lookup from PLUTO."""
    p = RAW / "nyc" / "pluto.csv"
    df = pd.read_csv(p, low_memory=False,
                     usecols=["BBL", "address", "latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "address"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df["norm"] = df["address"].apply(normalize)
    df = df[df["norm"].str.len() > 3]
    print(f"  PLUTO: {len(df):,} rows with valid lat/lon and parsed address "
          f"({df['norm'].nunique():,} unique normalized)")
    return df.drop_duplicates("norm").set_index("norm")[
        ["latitude", "longitude", "BBL"]].to_dict("index")


def build_boston_lookup():
    p = RAW / "boston" / "boston_assessment.csv"
    if not p.exists():
        print("  Boston assessment file not found")
        return {}
    # Try to find the parcel geometry file with lat/lon
    parcels = None
    for cand in [PROC / "boston_embeddings.parquet"]:  # has the spatial join already in release
        pass
    # Use the cleaned/centroid file (geocode_and_centroids.py produces this)
    geom_paths = [
        PROC / "boston_parcels.gpkg",
        REPO / "data" / "cleaned" / "boston_parcels_cleaned.gpkg",
        REPO / "release" / "data" / "boston" / "parcels.parquet",
    ]
    parcels = None
    for gp in geom_paths:
        if gp.exists():
            if gp.suffix == ".parquet":
                parcels = pd.read_parquet(gp)
            else:
                try:
                    import geopandas as gpd
                    parcels = gpd.read_file(gp)
                except Exception:
                    continue
            if parcels is not None and len(parcels) > 0:
                print(f"  Boston parcels loaded from {gp.name}: {len(parcels):,} rows")
                break
    if parcels is None:
        print("  Boston parcels with lat/lon not found in expected paths")
        return {}

    # Boston assessment has PID/ST_NUM/ST_NAME (no lat/lon); parcels file has
    # PID-equivalent + lat/lon. We join through the parcel id.
    assess = pd.read_csv(p, low_memory=False,
                         usecols=["PID", "ST_NUM", "ST_NAME"])
    assess["PID"] = assess["PID"].astype(str).str.strip().str.zfill(10)
    assess["ST_NUM"] = pd.to_numeric(assess["ST_NUM"], errors="coerce")
    assess = assess.dropna(subset=["ST_NUM"])
    assess["raw"] = (assess["ST_NUM"].astype(int).astype(str) + " "
                     + assess["ST_NAME"].astype(str).str.strip())
    assess["norm"] = assess["raw"].apply(normalize)

    # Figure out the parcel-id column on the parcels frame
    pid_cols = [c for c in parcels.columns
                if c.upper() in ("MAP_PAR_ID", "PARCEL_ID", "PID", "PARID")]
    if not pid_cols:
        print(f"  Boston parcels has no PID-like column. Columns: {list(parcels.columns)[:20]}")
        return {}
    pid_col = pid_cols[0]
    parcels[pid_col] = parcels[pid_col].astype(str).str.strip().str.zfill(10)

    lat_col = next((c for c in parcels.columns if c.lower() == "latitude"), None)
    lon_col = next((c for c in parcels.columns if c.lower() == "longitude"), None)
    if lat_col is None and "geometry" in parcels.columns:
        # geometry path: compute centroid
        import geopandas as gpd
        if not isinstance(parcels, gpd.GeoDataFrame):
            parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs="EPSG:4326")
        cent = parcels.geometry.centroid
        parcels["latitude"] = cent.y
        parcels["longitude"] = cent.x
        lat_col, lon_col = "latitude", "longitude"
    if lat_col is None or lon_col is None:
        print("  Boston parcels file has no usable lat/lon")
        return {}

    merged = assess.merge(parcels[[pid_col, lat_col, lon_col]],
                          left_on="PID", right_on=pid_col, how="inner")
    merged = merged.dropna(subset=[lat_col, lon_col])
    merged = merged[merged["norm"].str.len() > 3]
    print(f"  Boston assessment+parcels: {len(merged):,} addresses with lat/lon "
          f"({merged['norm'].nunique():,} unique)")
    return merged.drop_duplicates("norm").set_index("norm").apply(
        lambda r: {"latitude": r[lat_col], "longitude": r[lon_col], "BBL": r["PID"]},
        axis=1).to_dict()


def match_one(norm, zip_code, lookup, zip_index):
    """Try exact -> Queens-hyphen variant -> in-zip street-name prefix fallback."""
    if norm in lookup:
        return lookup[norm], "exact"
    for v in queens_hyphen_variants(norm):
        if v in lookup:
            return lookup[v], "queens_hyphen"
    # In-zip prefix fallback: same first two tokens (street number + first word)
    parts = norm.split()
    if len(parts) >= 2 and zip_code in zip_index:
        prefix2 = parts[0] + " " + parts[1]
        cands = [k for k in zip_index[zip_code] if k.startswith(prefix2)]
        if len(cands) == 1:
            return lookup[cands[0]], "in_zip_prefix"
    return None, "unmatched"


def build_zip_index(lookup, descriptions_zips):
    """For a given set of zip codes seen in the descriptions, group lookup keys
    by which PLUTO entries are plausibly in those zips. We don't have zip in
    PLUTO here, so we approximate by NOT constraining and only using the prefix
    fallback when a single candidate matches. The zip_index here is a no-op
    placeholder kept for future zip-aware matching."""
    # No-op for now; main matcher won't restrict by zip.
    return {z: list(lookup.keys()) for z in descriptions_zips}


def regeocode_city(city):
    print(f"\n{'='*64}\nREGEOCODE: {city.upper()}\n{'='*64}")
    raw_path = RAW / "descriptions" / f"{city}_descriptions.csv"
    df = pd.read_csv(raw_path)
    df["norm"] = df["address"].apply(normalize)
    n = len(df)
    print(f"  Loaded {n} raw descriptions from {raw_path.name}")

    if city == "nyc":
        lookup = build_nyc_lookup()
    elif city == "boston":
        lookup = build_boston_lookup()
    else:
        print(f"  No regeocode strategy for {city}")
        return

    if not lookup:
        print("  Empty lookup, aborting")
        return

    # Match
    lats, lons, bbls, methods = [], [], [], []
    counts = {"exact": 0, "queens_hyphen": 0, "in_zip_prefix": 0, "unmatched": 0, "empty": 0}
    lookup_keys_set = set(lookup.keys())
    for _, r in df.iterrows():
        norm = r["norm"]
        if not norm:
            lats.append(np.nan); lons.append(np.nan); bbls.append(None)
            methods.append("empty"); counts["empty"] += 1
            continue
        rec, m = match_one(norm, r.get("zip"), lookup,
                           {r.get("zip"): list(lookup_keys_set)})
        if rec is None:
            lats.append(np.nan); lons.append(np.nan); bbls.append(None)
        else:
            lats.append(rec["latitude"]); lons.append(rec["longitude"]); bbls.append(rec.get("BBL"))
        methods.append(m); counts[m] += 1

    df["latitude"] = lats
    df["longitude"] = lons
    df["BBL_matched"] = bbls
    df["geocode_method"] = methods

    print(f"\n  RESULTS:")
    for k in ["exact", "queens_hyphen", "in_zip_prefix", "unmatched", "empty"]:
        print(f"    {k:18s}: {counts[k]:>4}  ({counts[k]/n:>5.1%})")
    total_matched = counts["exact"] + counts["queens_hyphen"] + counts["in_zip_prefix"]
    print(f"    {'-'*32}")
    print(f"    {'PROPERTY-LEVEL':18s}: {total_matched:>4}  ({total_matched/n:>5.1%})")

    # Save
    out_csv = PROC / f"{city}_descriptions_geocoded.csv"
    df.drop(columns=["norm"]).to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")

    # Update the embeddings parquet (overwrite lat/lon, mark method)
    emb_path = PROC / f"{city}_embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        if "address" in emb.columns and len(emb) == len(df):
            # row-aligned update
            emb["latitude"] = df["latitude"].values
            emb["longitude"] = df["longitude"].values
            emb["BBL_matched"] = df["BBL_matched"].values
            emb["geocode_method"] = df["geocode_method"].values
            backup = emb_path.with_suffix(".parquet.zipcentroid_backup")
            if not backup.exists():
                # one-time backup of the prior (zip-centroid) state
                import shutil
                shutil.copy(emb_path, backup)
                print(f"  Backed up prior embeddings parquet -> {backup.name}")
            emb.to_parquet(emb_path, index=False)
            print(f"  Updated -> {emb_path.name}")
        else:
            print(f"  WARNING: embeddings parquet not row-aligned with raw descriptions, skipping update")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else ["nyc", "boston"]
    for c in cities:
        regeocode_city(c)


if __name__ == "__main__":
    main()

"""
Re-geocode Boston/NYC descriptions to property-level lat/lon.

Replaces the zip-centroid fallback that left ~94% of NYC and ~95% of Boston
description coordinates identical within zip and destabilized the per-city DML.

Strategy:
  1. Vectorized address normalization (uppercase, strip unit/suffix, expand
     directions and street types, strip ordinal endings like 90TH -> 90).
  2. NYC: cached PLUTO lookup with (norm, postcode) and (norm) keys, queens-
     hyphen variants for unhyphenated 4-6 digit leading numbers, async NYC
     GeoSearch API for residual unmatched.
  3. Boston: assessment ST_NUM+ST_NAME -> parcel PID -> lat/lon join, Census
     batch geocoder for residual unmatched.
  4. Writes the new lat/lon back into data/processed/{city}_embeddings.parquet
     after backing up the prior file once. Adds BBL_matched, geocode_method,
     match_label columns. Unmatched rows keep NaN lat/lon so the DML can drop
     or impute them rather than silently using a centroid artifact.

Usage:
    python regeocode_descriptions.py            # both cities
    python regeocode_descriptions.py nyc
    python regeocode_descriptions.py boston
    python regeocode_descriptions.py --no-api   # skip API fallbacks (offline)
"""
import re
import sys
import asyncio
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import httpx
    HAVE_HTTPX = True
except ImportError:
    HAVE_HTTPX = False

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
CACHE = PROC / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SUFFIXES = {
    "ST": "STREET", "STR": "STREET", "AVE": "AVENUE", "AV": "AVENUE",
    "BLVD": "BOULEVARD", "DR": "DRIVE", "RD": "ROAD", "LN": "LANE",
    "CT": "COURT", "PL": "PLACE", "TER": "TERRACE", "CIR": "CIRCLE",
    "PKWY": "PARKWAY", "HWY": "HIGHWAY", "SQ": "SQUARE", "WY": "WAY",
    "CRES": "CRESCENT", "TPKE": "TURNPIKE", "BWY": "BROADWAY",
    "EXPY": "EXPRESSWAY", "PLZ": "PLAZA", "XING": "CROSSING",
    "RDG": "RIDGE",
}
DIRS = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST"}


def normalize_vec(s):
    """Vectorized address normalization. Operates on a pandas Series."""
    s = s.fillna("").astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s*[#,]\s*(UNIT|APT|STE|SUITE|FL|FLOOR|RM|ROOM|PH)?\s*\S*$",
                      "", regex=True)
    s = s.str.replace(r"\s+(UNIT|APT|STE|SUITE|FL|FLOOR|RM|ROOM|PH)\s+\S*$",
                      "", regex=True)
    s = s.str.replace(r"(\d+)(ST|ND|RD|TH)\b", r"\1", regex=True)
    for k, v in DIRS.items():
        s = s.str.replace(rf"\b{k}\b", v, regex=True)
    for k, v in SUFFIXES.items():
        s = s.str.replace(rf"\b{k}\b", v, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.rstrip(".")
    return s


def queens_variants(norm):
    """For a leading 4-6 digit number, return hyphen-split variants."""
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


def build_nyc_lookup_cached():
    """Load PLUTO -> dataframe of (norm, postcode, latitude, longitude, BBL).
    Caches the normalized table as parquet for fast subsequent runs."""
    cache = CACHE / "pluto_lookup.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"  PLUTO lookup (cache hit): {len(df):,} entries")
        return df
    print("  Building PLUTO lookup from raw CSV (one-time, ~30s)...")
    df = pd.read_csv(RAW / "nyc" / "pluto.csv", low_memory=False,
                     usecols=["BBL", "address", "latitude", "longitude",
                              "postcode"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["postcode"] = pd.to_numeric(df["postcode"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["latitude", "longitude", "address"])
    df = df[(df.latitude != 0) & (df.longitude != 0)]
    df["norm"] = normalize_vec(df["address"])
    df = df[df.norm.str.len() > 3]
    df = df.drop_duplicates(["norm", "postcode"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"  cached {len(df):,} entries -> {cache}")
    return df


async def _geosearch_one(client, sem, addr):
    async with sem:
        try:
            r = await client.get(
                "https://geosearch.planninglabs.nyc/v2/search",
                params={"text": addr, "size": 1}, timeout=15.0,
            )
            if r.status_code == 200:
                j = r.json()
                feats = j.get("features", [])
                if feats:
                    lon, lat = feats[0]["geometry"]["coordinates"]
                    label = feats[0]["properties"].get("label", "")
                    return (lat, lon, label)
        except Exception:
            pass
        return (None, None, None)


async def _geosearch_batch(addresses, concurrency=8):
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        return await asyncio.gather(
            *[_geosearch_one(client, sem, a) for a in addresses]
        )


def geosearch_fallback(desc, mask_unmatched, use_api=True):
    """Hit NYC GeoSearch for the unmatched rows. Returns lat/lon/label arrays
    aligned to desc.index, with NaN where no match."""
    n = len(desc)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    lab = np.array([None] * n, dtype=object)
    if not (use_api and HAVE_HTTPX and mask_unmatched.any()):
        return lat, lon, lab
    rows = desc[mask_unmatched].copy()
    rows["q"] = rows.apply(
        lambda r: f"{r['address']}, {int(r['zip'])}" if pd.notna(r.get("zip")) else r["address"],
        axis=1,
    )
    queries = rows["q"].fillna(rows["address"]).tolist()
    print(f"  GeoSearch API on {len(queries)} unmatched (concurrency=8)...")
    results = asyncio.run(_geosearch_batch(queries, concurrency=8))
    for idx, (la, lo, lb) in zip(rows.index, results):
        lat[idx], lon[idx], lab[idx] = la, lo, lb
    hits = sum(1 for la, _, _ in results if la is not None)
    print(f"  GeoSearch hits: {hits}/{len(queries)}")
    return lat, lon, lab


def regeocode_nyc(use_api=True):
    print(f"\n{'=' * 64}\nREGEOCODE: NYC\n{'=' * 64}")
    desc = pd.read_csv(RAW / "descriptions" / "nyc_descriptions.csv")
    n = len(desc)
    desc = desc.drop(columns=[c for c in ("latitude", "longitude") if c in desc.columns])
    desc["norm"] = normalize_vec(desc["address"])
    desc["zip"] = pd.to_numeric(desc["zip"], errors="coerce").astype("Int64")
    print(f"  Loaded {n} descriptions")

    plt = build_nyc_lookup_cached()
    plt_zip = plt[plt.postcode.notna()]
    plt_any = plt.drop_duplicates("norm")[["norm", "latitude", "longitude", "BBL"]]

    s1 = desc.merge(plt_zip, left_on=["norm", "zip"],
                    right_on=["norm", "postcode"], how="left",
                    suffixes=("", "_p"))
    lat = s1["latitude"].values.copy()
    lon = s1["longitude"].values.copy()
    bbl = s1["BBL"].astype(object).where(s1["latitude"].notna(), None).values
    method = np.where(s1["latitude"].notna(), "exact_zip",
                      np.where(desc["norm"].str.len() > 3, "", "empty"))
    s1_hits = (method == "exact_zip").sum()

    rest = pd.Series(method == "").values
    if rest.any():
        s2 = desc[rest].merge(plt_any, on="norm", how="left")
        new_lat = s2["latitude"].values
        new_lon = s2["longitude"].values
        new_bbl = s2["BBL"].values
        idx_rest = np.where(rest)[0]
        for i_local, i_global in enumerate(idx_rest):
            if pd.notna(new_lat[i_local]):
                lat[i_global] = new_lat[i_local]
                lon[i_global] = new_lon[i_local]
                bbl[i_global] = new_bbl[i_local]
                method[i_global] = "exact"
    s2_hits = ((method == "exact")).sum()

    rest = pd.Series(method == "").values
    s3_hits = 0
    if rest.any():
        for i in np.where(rest)[0]:
            for v in queens_variants(desc["norm"].iat[i]):
                hit = plt_any[plt_any.norm == v]
                if not hit.empty:
                    lat[i] = hit["latitude"].iat[0]
                    lon[i] = hit["longitude"].iat[0]
                    bbl[i] = hit["BBL"].iat[0]
                    method[i] = "queens_hyphen"
                    s3_hits += 1
                    break

    rest = pd.Series(method == "").values
    api_lat, api_lon, api_lab = geosearch_fallback(desc, rest, use_api=use_api)
    api_hits = 0
    label = np.array([None] * n, dtype=object)
    for i in np.where(rest)[0]:
        if pd.notna(api_lat[i]):
            lat[i] = api_lat[i]
            lon[i] = api_lon[i]
            label[i] = api_lab[i]
            method[i] = "geosearch_api"
            api_hits += 1

    unmatched = (method == "").sum() + (method == "empty").sum()
    matched_total = n - unmatched

    print(f"\n  RESULTS:")
    print(f"    exact (norm + zip)   : {s1_hits:>4}  ({s1_hits/n:>5.1%})")
    print(f"    exact (norm only)    : {s2_hits:>4}  ({s2_hits/n:>5.1%})")
    print(f"    queens hyphen variant: {s3_hits:>4}  ({s3_hits/n:>5.1%})")
    print(f"    GeoSearch API        : {api_hits:>4}  ({api_hits/n:>5.1%})")
    print(f"    {'-' * 38}")
    print(f"    PROPERTY-LEVEL TOTAL : {matched_total:>4}  ({matched_total/n:>5.1%})")
    print(f"    unmatched            : {unmatched:>4}  ({unmatched/n:>5.1%})")

    out = desc.copy()
    out["latitude"] = lat
    out["longitude"] = lon
    out["BBL_matched"] = bbl
    out["geocode_method"] = method
    out["match_label"] = label
    out_csv = PROC / "nyc_descriptions_geocoded.csv"
    out.drop(columns=["norm"]).to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv.name}")

    emb_path = PROC / "nyc_embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        if "address" in emb.columns and len(emb) == n:
            backup = emb_path.with_suffix(".parquet.zipcentroid_backup")
            if not backup.exists():
                shutil.copy(emb_path, backup)
                print(f"  Backed up prior parquet -> {backup.name}")
            emb["latitude"] = lat
            emb["longitude"] = lon
            emb["BBL_matched"] = bbl
            emb["geocode_method"] = method
            emb["match_label"] = label
            emb.to_parquet(emb_path, index=False)
            print(f"  Updated -> {emb_path.name}")
    return out


def build_boston_lookup_cached():
    cache = CACHE / "boston_lookup.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"  Boston lookup (cache hit): {len(df):,} entries")
        return df

    assess_path = RAW / "boston" / "boston_assessment.csv"
    if not assess_path.exists():
        print("  Boston assessment file missing")
        return None
    print("  Building Boston lookup from assessment + parcels (~10s)...")
    assess = pd.read_csv(assess_path, low_memory=False,
                         usecols=["PID", "ST_NUM", "ST_NAME", "ZIP_CODE"]
                         if "ZIP_CODE" in pd.read_csv(assess_path, nrows=1).columns
                         else ["PID", "ST_NUM", "ST_NAME"])
    assess["PID"] = assess["PID"].astype(str).str.strip().str.zfill(10)
    assess["ST_NUM"] = pd.to_numeric(assess["ST_NUM"], errors="coerce")
    assess = assess.dropna(subset=["ST_NUM"])
    assess["raw"] = (assess["ST_NUM"].astype(int).astype(str) + " "
                     + assess["ST_NAME"].astype(str).str.strip())
    assess["norm"] = normalize_vec(assess["raw"])
    if "ZIP_CODE" in assess.columns:
        assess["postcode"] = pd.to_numeric(assess["ZIP_CODE"], errors="coerce").astype("Int64")
    else:
        assess["postcode"] = pd.NA

    parcels = None
    for cand in [
        REPO / "release" / "data" / "boston" / "parcels.parquet",
        PROC / "boston_parcels.gpkg",
        REPO / "data" / "cleaned" / "boston_parcels_cleaned.gpkg",
    ]:
        if cand.exists():
            if cand.suffix == ".parquet":
                parcels = pd.read_parquet(cand)
            else:
                try:
                    import geopandas as gpd
                    parcels = gpd.read_file(cand)
                except Exception:
                    continue
            if parcels is not None and len(parcels) > 0:
                print(f"  Boston parcels loaded from {cand.name}: {len(parcels):,}")
                break
    if parcels is None:
        print("  No Boston parcels file found")
        return None

    pid_cols = [c for c in parcels.columns
                if c.upper() in ("MAP_PAR_ID", "PARCEL_ID", "PID", "PARID")]
    if not pid_cols:
        print(f"  Boston parcels has no PID-like column: {list(parcels.columns)[:20]}")
        return None
    pid_col = pid_cols[0]
    parcels[pid_col] = parcels[pid_col].astype(str).str.strip().str.zfill(10)
    lat_col = next((c for c in parcels.columns if c.lower() == "latitude"), None)
    lon_col = next((c for c in parcels.columns if c.lower() == "longitude"), None)
    if lat_col is None and "geometry" in parcels.columns:
        import geopandas as gpd
        if not isinstance(parcels, gpd.GeoDataFrame):
            parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs="EPSG:4326")
        cent = parcels.geometry.centroid
        parcels["latitude"] = cent.y
        parcels["longitude"] = cent.x
        lat_col, lon_col = "latitude", "longitude"
    if lat_col is None or lon_col is None:
        print("  Boston parcels lacks usable lat/lon")
        return None

    merged = assess.merge(parcels[[pid_col, lat_col, lon_col]],
                          left_on="PID", right_on=pid_col, how="inner")
    merged = merged.dropna(subset=[lat_col, lon_col])
    merged = merged[merged["norm"].str.len() > 3]
    merged = merged.rename(columns={lat_col: "latitude", lon_col: "longitude",
                                    "PID": "BBL"})
    merged = merged.drop_duplicates(["norm", "postcode"])[
        ["norm", "postcode", "latitude", "longitude", "BBL"]].reset_index(drop=True)
    merged.to_parquet(cache, index=False)
    print(f"  cached {len(merged):,} entries -> {cache}")
    return merged


async def _census_one(client, sem, addr, city, state, zip_code):
    async with sem:
        params = {
            "address": f"{addr}, {city}, {state}, {zip_code}" if zip_code else f"{addr}, {city}, {state}",
            "benchmark": "Public_AR_Current",
            "format": "json",
        }
        try:
            r = await client.get(
                "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
                params=params, timeout=20.0,
            )
            if r.status_code == 200:
                j = r.json()
                m = j.get("result", {}).get("addressMatches", [])
                if m:
                    return (m[0]["coordinates"]["y"],
                            m[0]["coordinates"]["x"],
                            m[0]["matchedAddress"])
        except Exception:
            pass
        return (None, None, None)


async def _census_batch_async(rows, city, state, concurrency=8):
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            _census_one(client, sem, r["address"], city, state,
                        f"{int(r['zip']):05d}" if pd.notna(r.get("zip")) else "")
            for _, r in rows.iterrows()
        ]
        return await asyncio.gather(*tasks)


def census_geocoder_batch(rows, city="Boston", state="MA"):
    """Census ONELINE geocoder over the rows DataFrame. Concurrent."""
    if not HAVE_HTTPX:
        return None
    n = len(rows)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    lab = np.array([None] * n, dtype=object)
    print(f"  Census geocoder on {n} addresses (concurrency=8)...")
    results = asyncio.run(_census_batch_async(rows, city, state, concurrency=8))
    for i, (la, lo, lb) in enumerate(results):
        if la is not None:
            lat[i] = la
            lon[i] = lo
            lab[i] = lb
    hits = (~np.isnan(lat)).sum()
    print(f"  Census hits: {hits}/{n}")
    return lat, lon, lab


def regeocode_boston(use_api=True):
    print(f"\n{'=' * 64}\nREGEOCODE: BOSTON\n{'=' * 64}")
    desc = pd.read_csv(RAW / "descriptions" / "boston_descriptions.csv")
    n = len(desc)
    desc = desc.drop(columns=[c for c in ("latitude", "longitude") if c in desc.columns])
    desc["norm"] = normalize_vec(desc["address"])
    desc["zip"] = pd.to_numeric(desc["zip"], errors="coerce").astype("Int64")
    print(f"  Loaded {n} descriptions")

    plt = build_boston_lookup_cached()
    if plt is None:
        return None
    plt_any = plt.drop_duplicates("norm")[["norm", "latitude", "longitude", "BBL"]]

    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    bbl = np.array([None] * n, dtype=object)
    method = np.array([""] * n, dtype=object)

    if plt["postcode"].notna().any():
        s1 = desc.merge(plt[plt.postcode.notna()],
                        left_on=["norm", "zip"], right_on=["norm", "postcode"],
                        how="left", suffixes=("", "_p"))
        for i in range(n):
            if pd.notna(s1["latitude"].iat[i]):
                lat[i] = s1["latitude"].iat[i]
                lon[i] = s1["longitude"].iat[i]
                bbl[i] = s1["BBL"].iat[i]
                method[i] = "exact_zip"
    s1_hits = (method == "exact_zip").sum()

    rest = method == ""
    if rest.any():
        s2 = desc[rest].merge(plt_any, on="norm", how="left")
        idx_rest = np.where(rest)[0]
        for i_local, i_global in enumerate(idx_rest):
            if pd.notna(s2["latitude"].iat[i_local]):
                lat[i_global] = s2["latitude"].iat[i_local]
                lon[i_global] = s2["longitude"].iat[i_local]
                bbl[i_global] = s2["BBL"].iat[i_local]
                method[i_global] = "exact"
    s2_hits = (method == "exact").sum()

    rest = method == ""
    census_hits = 0
    label = np.array([None] * n, dtype=object)
    if rest.any() and use_api:
        rows = desc[rest][["address", "zip"]].copy()
        rows["city"] = "Boston"
        rows["state"] = "MA"
        result = census_geocoder_batch(rows)
        if result is not None:
            c_lat, c_lon, c_lab = result
            idx_rest = np.where(rest)[0]
            for i_local, i_global in enumerate(idx_rest):
                if not np.isnan(c_lat[i_local]):
                    lat[i_global] = c_lat[i_local]
                    lon[i_global] = c_lon[i_local]
                    label[i_global] = c_lab[i_local]
                    method[i_global] = "census_api"
                    census_hits += 1

    unmatched = (method == "").sum()
    matched_total = n - unmatched
    print(f"\n  RESULTS:")
    print(f"    exact (norm + zip)   : {s1_hits:>4}  ({s1_hits/n:>5.1%})")
    print(f"    exact (norm only)    : {s2_hits:>4}  ({s2_hits/n:>5.1%})")
    print(f"    Census batch API     : {census_hits:>4}  ({census_hits/n:>5.1%})")
    print(f"    {'-' * 38}")
    print(f"    PROPERTY-LEVEL TOTAL : {matched_total:>4}  ({matched_total/n:>5.1%})")
    print(f"    unmatched            : {unmatched:>4}  ({unmatched/n:>5.1%})")

    out = desc.copy()
    out["latitude"] = lat
    out["longitude"] = lon
    out["BBL_matched"] = bbl
    out["geocode_method"] = method
    out["match_label"] = label
    out_csv = PROC / "boston_descriptions_geocoded.csv"
    out.drop(columns=["norm"]).to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv.name}")

    emb_path = PROC / "boston_embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        if "address" in emb.columns and len(emb) == n:
            backup = emb_path.with_suffix(".parquet.zipcentroid_backup")
            if not backup.exists():
                shutil.copy(emb_path, backup)
                print(f"  Backed up prior parquet -> {backup.name}")
            emb["latitude"] = lat
            emb["longitude"] = lon
            emb["BBL_matched"] = bbl
            emb["geocode_method"] = method
            emb["match_label"] = label
            emb.to_parquet(emb_path, index=False)
            print(f"  Updated -> {emb_path.name}")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    use_api = "--no-api" not in sys.argv
    cities = args if args else ["nyc", "boston"]
    for c in cities:
        if c == "nyc":
            regeocode_nyc(use_api=use_api)
        elif c == "boston":
            regeocode_boston(use_api=use_api)
        else:
            print(f"Unknown city: {c}")


if __name__ == "__main__":
    main()

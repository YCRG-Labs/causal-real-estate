"""Residential-scope filter for the 12-city DML corpus.

Drops listings that violate the single-family / condo hedonic scope:
  - description matches a multi-unit / portfolio / auction / commercial keyword
  - beds > 8 (apartment building)
  - bldg_area_sqft > 8000 (commercial multifamily envelope)
  - price < 75000 (placeholder / allocated portfolio prices)

This is the standard residential-only filter used in Redfin/Zillow hedonic
work (Cabral 2024 RAND, Bishop-Heblich-Iceland 2025 JUE, Tan-Tang-Yu 2025
ReStat) and follows Rosen (1974)'s scope condition that the hedonic
specification is for owner-occupied / single-unit residential transactions.

We filter at the DML input stage (after embeddings have been generated): we
drop rows from {city}_embeddings.parquet and {city}_parcels_micro_geo.gpkg
by index match. Pre-filter copies are saved as {city}_embeddings_prefilter
and {city}_parcels_micro_geo_prefilter. A manifest JSON logs per-city drop
counts and the specific listings removed for the JBES referee.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

KEYWORD_PATTERN = re.compile(
    r"\b("
    r"duplex|triplex|fourplex|four-plex|fiveplex|sixplex|"
    r"multi[-\s]?family|multifamily|multi[-\s]?unit|multiunit|"
    r"portfolio|sealed[-\s]?bid|auction|"
    r"investment\s+opportunity|investment\s+property|"
    r"\d+[-\s]?unit\s+(community|building|portfolio|complex|property|investment)|"
    r"\d+\s+units\b|"
    r"leased|fully\s+leased|cap\s+rate|pro\s+forma|rent\s+roll|"
    r"\bnoi\b|\bgross\s+rents?\b|\bcommercial\b|\bmixed[-\s]?use\b|"
    r"value[-\s]?add\b|opportunity\s+zone|"
    r"development\s+opportunity|tear[-\s]?down|land\s+value"
    r")\b",
    flags=re.IGNORECASE,
)

BEDS_MAX = 8
SQFT_MAX = 8000.0
PRICE_MIN = 75000.0


def filter_one(city: str, dry_run: bool = False, backup: bool = True) -> dict:
    emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    gpkg_path = PROCESSED_DIR / f"{city}_parcels_micro_geo.gpkg"
    if not emb_path.exists():
        return {"city": city, "skipped": True, "reason": "no embeddings parquet"}

    emb = pd.read_parquet(emb_path)
    n0 = len(emb)
    mask = pd.Series(True, index=emb.index)
    reasons: dict[str, list] = {}

    desc_col = "description" if "description" in emb.columns else (
        "clean_description" if "clean_description" in emb.columns else None
    )
    if desc_col is not None:
        keyword_hits = emb[desc_col].fillna("").astype(str).str.contains(
            KEYWORD_PATTERN, regex=True, na=False,
        )
        mask &= ~keyword_hits
        reasons["keyword"] = emb.loc[keyword_hits, "url"].tolist() if "url" in emb.columns else keyword_hits[keyword_hits].index.tolist()

    for col, lo, hi, label in [
        ("beds", None, BEDS_MAX, f"beds>{BEDS_MAX}"),
        ("bedrooms", None, BEDS_MAX, f"bedrooms>{BEDS_MAX}"),
        ("sqft", None, SQFT_MAX, f"sqft>{SQFT_MAX}"),
        ("bldg_area_sqft", None, SQFT_MAX, f"bldg_area_sqft>{SQFT_MAX}"),
        ("price", PRICE_MIN, None, f"price<{PRICE_MIN}"),
    ]:
        if col not in emb.columns:
            continue
        col_vals = pd.to_numeric(emb[col], errors="coerce")
        cond = pd.Series(True, index=emb.index)
        if lo is not None:
            cond &= (col_vals >= lo) | col_vals.isna()
        if hi is not None:
            cond &= (col_vals <= hi) | col_vals.isna()
        violated = ~cond
        if violated.any():
            ids = emb.loc[violated, "url"].tolist() if "url" in emb.columns else violated[violated].index.tolist()
            reasons.setdefault(label, []).extend(ids)
        mask &= cond

    n1 = int(mask.sum())
    dropped = n0 - n1
    print(f"[{city}] {n0} -> {n1} ({dropped} dropped, {dropped/n0*100:.1f}%)")
    for r, ids in reasons.items():
        if ids:
            print(f"    {r}: {len(ids)} listings")
            for u in list(ids)[:3]:
                print(f"       - {u}")
            if len(ids) > 3:
                print(f"       ... +{len(ids)-3} more")

    report = {"city": city, "n_pre": n0, "n_post": n1, "dropped": dropped,
              "by_reason": {k: len(v) for k, v in reasons.items()},
              "dropped_urls": {k: v for k, v in reasons.items() if v}}

    if dry_run:
        return report

    emb_filtered = emb[mask].reset_index(drop=True)
    if backup:
        backup_path = PROCESSED_DIR / f"{city}_embeddings_prefilter.parquet"
        if not backup_path.exists():
            shutil.copy2(emb_path, backup_path)
    emb_filtered.to_parquet(emb_path, index=False)

    if gpkg_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_path, layer=city) if (gpkg_path.exists()) else None
        if gdf is not None:
            if "url" in gdf.columns and "url" in emb_filtered.columns:
                keep_urls = set(emb_filtered["url"].tolist())
                gdf_filt = gdf[gdf["url"].isin(keep_urls)].reset_index(drop=True)
            else:
                lat_keep = emb_filtered["latitude"].round(6).astype(str) + "_" + emb_filtered["longitude"].round(6).astype(str)
                lat_all = gdf["latitude"].round(6).astype(str) + "_" + gdf["longitude"].round(6).astype(str) if "latitude" in gdf.columns else gdf.geometry.y.round(6).astype(str) + "_" + gdf.geometry.x.round(6).astype(str)
                gdf_filt = gdf[lat_all.isin(set(lat_keep))].reset_index(drop=True)
            if backup:
                gpkg_backup = PROCESSED_DIR / f"{city}_parcels_micro_geo_prefilter.gpkg"
                if not gpkg_backup.exists():
                    shutil.copy2(gpkg_path, gpkg_backup)
            gpkg_path.unlink()
            gdf_filt.to_file(gpkg_path, layer=city, driver="GPKG")
            print(f"    gpkg: {len(gdf)} -> {len(gdf_filt)} parcels")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    cities = args.cities or ALL_12
    reports = []
    for c in cities:
        try:
            r = filter_one(c, dry_run=args.dry_run, backup=not args.no_backup)
            reports.append(r)
        except Exception as e:
            import traceback
            print(f"[{c}] FAILED: {e}", file=sys.stderr); traceback.print_exc()
            reports.append({"city": c, "skipped": True, "error": str(e)})

    out_path = REPO_ROOT / "results" / "residential_filter_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=2, default=str))

    pre = sum(r.get("n_pre", 0) for r in reports if "n_pre" in r)
    post = sum(r.get("n_post", 0) for r in reports if "n_post" in r)
    print(f"\nTotal 12-city: {pre} -> {post} ({pre-post} dropped, {(pre-post)/max(pre,1)*100:.1f}%)")
    print(f"Manifest -> {out_path}")


if __name__ == "__main__":
    main()

"""Adapt the new-9-cities output layout to what fast_bootstrap_dml_v2.py expects.

The legacy DML pipeline reads:
  data/processed/{city}_embeddings.parquet         (emb_0..767, price, latitude, longitude)
  data/processed/{city}_parcels_micro_geo.gpkg     (canonical PROPERTY + CENSUS + CRIME +
                                                    AMENITY + MICRO_GEO + _centroid_lat/lon)

The new 9-city pipeline produced:
  data/processed/{slug}_embeddings_mpnet_base.parquet  (emb_0..767, listing cols incl
                                                        lat_centroid, lon_centroid, price)
  data/processed/{slug}_listings_enriched.parquet      (35 canonical confounders attached)

This script joins them on `url`, renames columns to the canonical names the DML script
expects, writes `{slug}_embeddings.parquet`, and writes a per-listing
`{slug}_parcels_micro_geo.gpkg` that the cKDTree spatial join in the DML script can
match each listing to its own "parcel" (since listings already carry their confounders).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from canonical_confounders import PROPERTY, CENSUS, CRIME, AMENITY, MICRO_GEO

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

EMBEDDING_DIM = 768
EMB_COLS = [f"emb_{i}" for i in range(EMBEDDING_DIM)]

RENAME_MAP = {
    "lat_centroid": "latitude",
    "lon_centroid": "longitude",
    "beds": "bedrooms",
    "sqft": "bldg_area_sqft",
}


def prepare_city(slug: str, embedding_suffix: str = "mpnet_base") -> tuple[Path, Path]:
    emb_path = PROCESSED_DIR / f"{slug}_embeddings_{embedding_suffix}.parquet"
    enr_path = PROCESSED_DIR / f"{slug}_listings_enriched.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"no embeddings parquet at {emb_path}")
    if not enr_path.exists():
        raise FileNotFoundError(f"no enriched parquet at {enr_path}")
    emb = pd.read_parquet(emb_path)
    enr = pd.read_parquet(enr_path)
    print(f"[{slug}] embeddings: {len(emb)} rows; enriched: {len(enr)} rows")

    confounder_cols = [c for c in (PROPERTY + CENSUS + CRIME + AMENITY + MICRO_GEO) if c in enr.columns]
    enr_keep = enr[["url"] + [c for c in confounder_cols if c not in emb.columns]].copy()
    merged = emb.merge(enr_keep, on="url", how="left")
    print(f"[{slug}] merged: {len(merged)} rows × {len(merged.columns)} cols")

    for src, dst in RENAME_MAP.items():
        if src in merged.columns and dst not in merged.columns:
            merged[dst] = merged[src]
    if "lot_area_sqft" not in merged.columns:
        merged["lot_area_sqft"] = np.nan

    out_emb = PROCESSED_DIR / f"{slug}_embeddings.parquet"
    merged.to_parquet(out_emb, index=False)
    print(f"[{slug}] wrote {out_emb} ({len(merged)} rows)")

    import geopandas as gpd
    from shapely.geometry import Point

    parcel_cols = list(set(PROPERTY + CENSUS + CRIME + AMENITY + MICRO_GEO + ["price"]) & set(merged.columns))
    pcl = merged[["latitude", "longitude"] + parcel_cols].copy()
    pcl["_centroid_lat"] = pcl["latitude"]
    pcl["_centroid_lon"] = pcl["longitude"]
    if "price" in pcl.columns:
        pcl["sale_price"] = pcl["price"]
    valid = pcl["latitude"].notna() & pcl["longitude"].notna()
    pcl = pcl[valid].reset_index(drop=True)
    gdf = gpd.GeoDataFrame(
        pcl,
        geometry=[Point(lon, lat) for lat, lon in zip(pcl["latitude"], pcl["longitude"])],
        crs=4326,
    )
    out_gpkg = PROCESSED_DIR / f"{slug}_parcels_micro_geo.gpkg"
    if out_gpkg.exists():
        out_gpkg.unlink()
    gdf.to_file(out_gpkg, layer=slug, driver="GPKG")
    print(f"[{slug}] wrote {out_gpkg} ({len(gdf)} parcel-points, {len(parcel_cols)} confounders)")
    return out_emb, out_gpkg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default="new9")
    p.add_argument("--city", default=None)
    p.add_argument("--suffix", default="mpnet_base")
    args = p.parse_args()
    if args.city:
        slugs = [args.city]
    elif args.cities == "new9":
        slugs = ["dc", "philadelphia", "chicago", "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
    else:
        slugs = [s.strip() for s in args.cities.split(",") if s.strip()]
    for slug in slugs:
        try:
            prepare_city(slug, embedding_suffix=args.suffix)
        except Exception as e:
            print(f"[{slug}] FAILED: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

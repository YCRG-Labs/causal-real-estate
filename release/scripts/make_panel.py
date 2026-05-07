"""Flatten per-city GeoPackages and embedding parquets into release-ready parquet files.

Drops geometry blobs, Redfin-licensed text (description, address, url),
keeps lat/lon and parcel IDs. Run from the repo root.
"""
from pathlib import Path
import sys
import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "processed"
OUT = REPO / "release" / "data"

CITIES = ["boston", "nyc", "sf"]

REDFIN_TEXT_COLS = ["description", "clean_description", "address", "url"]
EMBEDDING_KEEP_META = ["zip", "latitude", "longitude", "price"]
# SF assessor roll exposes street addresses; drop them since the paper's
# analysis uses lat/lon and addresses are not needed for replication.
SF_ADDRESS_COLS = [
    "from_address_num",
    "to_address_num",
    "street_name",
    "street_type",
    "odd_even",
    "property_location",
]


def flatten_panel(city: str) -> None:
    src = SRC / f"{city}_parcels_micro_geo.gpkg"
    dst = OUT / city / "parcels.parquet"
    print(f"[{city}] reading {src.name}")
    gdf = gpd.read_file(src)
    if "geometry" in gdf.columns:
        gdf = gdf.drop(columns=["geometry"])
    if "the_geom" in gdf.columns:
        gdf = gdf.drop(columns=["the_geom"])
    if "shape" in gdf.columns:
        gdf = gdf.drop(columns=["shape"])
    if city == "sf":
        gdf = gdf.drop(columns=[c for c in SF_ADDRESS_COLS if c in gdf.columns])
    df = pd.DataFrame(gdf)
    print(f"[{city}] writing {dst.relative_to(REPO)} ({len(df):,} rows, {len(df.columns)} cols)")
    df.to_parquet(dst, index=False)


def flatten_embeddings(city: str) -> None:
    pairs = [
        (f"{city}_embeddings.parquet", "embeddings_mpnet.parquet"),
        (f"{city}_embeddings_all_MiniLM_L6_v2.parquet", "embeddings_minilm.parquet"),
    ]
    for src_name, dst_name in pairs:
        src = SRC / src_name
        dst = OUT / city / dst_name
        if not src.exists():
            print(f"[{city}] skip {src_name} (not found)")
            continue
        df = pd.read_parquet(src)
        drop = [c for c in REDFIN_TEXT_COLS if c in df.columns]
        df = df.drop(columns=drop)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        meta_cols = [c for c in EMBEDDING_KEEP_META if c in df.columns]
        df = df[meta_cols + emb_cols]
        print(
            f"[{city}] writing {dst.relative_to(REPO)} "
            f"({len(df):,} rows, {len(emb_cols)}-dim embedding, dropped {drop})"
        )
        df.to_parquet(dst, index=False)


def main() -> None:
    for city in CITIES:
        (OUT / city).mkdir(parents=True, exist_ok=True)
        flatten_panel(city)
        flatten_embeddings(city)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())

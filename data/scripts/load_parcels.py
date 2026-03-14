import sys
import geopandas as gpd
import pandas as pd
from config import CITIES, CLEANED_DIR
from utils import (
    ensure_dirs,
    load_csv_with_bom,
    wkt_to_geodataframe,
    fix_invalid_geometries,
    standardize_columns,
    save_geopackage,
    load_geopackage,
)


def load_boston(cfg):
    df = load_csv_with_bom(cfg["raw_file"])

    df = df[~df[cfg["parcel_id_col"]].isin(cfg["invalid_parcel_ids"])]
    df = df[df["POLY_TYPE"].isin(cfg["valid_poly_types"])]

    gdf = wkt_to_geodataframe(df, cfg["geometry_col"], cfg["source_crs"])
    gdf = fix_invalid_geometries(gdf)
    gdf = standardize_columns(gdf, cfg["column_map"], cfg["drop_cols"])

    return gdf


def load_nyc(cfg):
    gdf = load_geopackage(cfg["raw_file"])

    if gdf.crs is None or str(gdf.crs) != cfg["source_crs"]:
        gdf = gdf.set_crs(cfg["source_crs"], allow_override=True)

    gdf = gdf[gdf["LandUse"].isin(cfg["residential_land_use"])]
    gdf = gdf.dropna(subset=["geometry"])
    gdf = fix_invalid_geometries(gdf)
    gdf = standardize_columns(gdf, cfg["column_map"])

    return gdf


def load_sf(cfg):
    gdf = load_geopackage(cfg["raw_file"])

    if gdf.crs is None or str(gdf.crs) != cfg["source_crs"]:
        gdf = gdf.set_crs(cfg["source_crs"], allow_override=True)

    gdf = gdf[gdf["active"] == True]
    gdf = gdf.dropna(subset=["geometry"])
    gdf = fix_invalid_geometries(gdf)
    gdf = standardize_columns(gdf, cfg["column_map"])

    assessor = pd.read_csv(cfg["assessor_file"])
    latest_year = assessor["closed_roll_year"].max()
    assessor = assessor[assessor["closed_roll_year"] == latest_year]
    assessor = assessor.rename(columns=cfg["assessor_column_map"])
    assessor = assessor.drop_duplicates(subset=["parcel_id"], keep="last")

    gdf = gdf.merge(assessor, on="parcel_id", how="left")

    return gdf


LOADERS = {
    "boston": load_boston,
    "nyc": load_nyc,
    "sf": load_sf,
}


def main(city):
    ensure_dirs()

    if city not in CITIES:
        print(f"Unknown city: {city}. Options: {list(CITIES.keys())}")
        sys.exit(1)

    cfg = CITIES[city]
    loader = LOADERS[city]
    gdf = loader(cfg)

    out_path = CLEANED_DIR / f"{city}_parcels_loaded.gpkg"
    save_geopackage(gdf, out_path, layer=city)
    print(f"{city}: loaded {len(gdf)} parcels → {out_path}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "boston"
    main(city)

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
    from shapely.geometry import Point

    keep_cols = list(cfg["column_map"].keys()) + ["BBL", "borough"]
    df = pd.read_csv(cfg["raw_file"], low_memory=False, usecols=keep_cols)
    df = df[df["landuse"].isin(cfg["residential_land_use"])]
    if "boroughs" in cfg:
        df = df[df["borough"].isin(cfg["boroughs"])]
        df = df.drop(columns=["borough"])
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=cfg["source_crs"])

    sales = pd.read_csv(cfg["sales_file"], low_memory=False)
    sales.columns = sales.columns.str.strip()
    sales["SALE PRICE"] = pd.to_numeric(
        sales["SALE PRICE"].astype(str).str.replace(",", "").str.replace("$", ""),
        errors="coerce",
    )
    sales = sales.dropna(subset=["SALE PRICE", "BBL"])
    sales = sales[sales["SALE PRICE"] > 0]
    sales["BBL"] = sales["BBL"].astype(float).astype(int).astype(str)
    sales = sales.sort_values("SALE DATE").drop_duplicates(subset=["BBL"], keep="last")
    sales = sales.rename(columns={"SALE PRICE": "sale_price", "SALE DATE": "sale_date"})

    gdf["BBL"] = gdf["BBL"].astype(int).astype(str)
    gdf = gdf.merge(sales[["BBL", "sale_price", "sale_date"]], on="BBL", how="left")
    gdf = standardize_columns(gdf, cfg["column_map"])

    return gdf


def load_sf(cfg):
    from shapely.geometry import Point

    df = pd.read_csv(cfg["raw_file"])

    if "active" in df.columns:
        df = df[df["active"] == True]

    df["centroid_latitude"] = pd.to_numeric(df["centroid_latitude"], errors="coerce")
    df["centroid_longitude"] = pd.to_numeric(df["centroid_longitude"], errors="coerce")
    df = df.dropna(subset=["centroid_latitude", "centroid_longitude"])
    df = df[(df["centroid_latitude"] != 0) & (df["centroid_longitude"] != 0)]

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df["centroid_longitude"], df["centroid_latitude"])
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=cfg["source_crs"])
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

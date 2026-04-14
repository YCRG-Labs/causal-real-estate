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

    if "assessment_file" in cfg and cfg["assessment_file"].exists():
        assess = pd.read_csv(cfg["assessment_file"], low_memory=False)
        assess = assess.rename(columns=cfg["assessment_column_map"])
        assess["parcel_id"] = assess["parcel_id"].astype(str).str.strip().str.zfill(10)
        assess = assess.drop_duplicates(subset=["parcel_id"], keep="last")

        gdf["parcel_id"] = gdf["parcel_id"].astype(str).str.strip().str.zfill(10)

        keep_cols = list(cfg["assessment_column_map"].values())
        available = [c for c in keep_cols if c in assess.columns]
        gdf = gdf.merge(assess[available], on="parcel_id", how="left")

    return gdf


NYC_SQFT_PER_BEDROOM = 350.0


def impute_nyc_bedrooms(gdf):
    """
    PLUTO does not publish bedroom counts. We impute them from
    residential floor area and number of residential units:
        bedrooms ≈ round((res_area_sqft / units_res) / SQFT_PER_BEDROOM)
    capped at [1, 8]. This eliminates the cross-city specification
    asymmetry where NYC was missing one property feature relative
    to BOS and SF. The flag `bedrooms_imputed=True` records that the
    column is derived rather than observed, so downstream sensitivity
    can drop it for NYC if desired.
    """
    if "res_area_sqft" not in gdf.columns or "units_res" not in gdf.columns:
        gdf["bedrooms"] = pd.NA
        gdf["bedrooms_imputed"] = True
        return gdf

    res_area = pd.to_numeric(gdf["res_area_sqft"], errors="coerce")
    units = pd.to_numeric(gdf["units_res"], errors="coerce").replace(0, 1)
    units = units.fillna(1)

    sqft_per_unit = res_area / units
    bedrooms = (sqft_per_unit / NYC_SQFT_PER_BEDROOM).round()
    bedrooms = bedrooms.clip(lower=1, upper=8)

    gdf["bedrooms"] = bedrooms
    gdf["bedrooms_imputed"] = True
    print(f"  Imputed bedrooms for {gdf['bedrooms'].notna().sum()}/{len(gdf)} parcels "
          f"(from res_area/units, capped [1,8])")
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

    gdf = impute_nyc_bedrooms(gdf)

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

    if "inferred_prices_file" in cfg and cfg["inferred_prices_file"].exists():
        prices = pd.read_csv(cfg["inferred_prices_file"])
        prices["parcel_id"] = prices["parcel_id"].astype(str).str.strip()
        prices = prices.drop_duplicates(subset=["parcel_id"], keep="last")
        gdf = gdf.merge(prices[["parcel_id", "sale_price", "sale_date"]], on="parcel_id", how="left")

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

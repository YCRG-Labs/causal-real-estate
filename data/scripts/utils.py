import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.validation import make_valid
from config import CLEANED_DIR, PROCESSED_DIR, MIN_PARCEL_AREA_SQM


def ensure_dirs():
    for d in [CLEANED_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_csv_with_bom(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def wkt_to_geodataframe(df, wkt_col, crs):
    df = df.dropna(subset=[wkt_col]).copy()
    df["geometry"] = df[wkt_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    gdf = gdf.drop(columns=[wkt_col])
    return gdf


def fix_invalid_geometries(gdf):
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
    return gdf


def reproject(gdf, target_crs):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set")
    return gdf.to_crs(target_crs)


def compute_area_sqm(gdf, local_crs):
    projected = reproject(gdf, local_crs)
    return projected.geometry.area


def drop_slivers(gdf, local_crs, min_area=MIN_PARCEL_AREA_SQM):
    areas = compute_area_sqm(gdf, local_crs)
    return gdf[areas >= min_area].copy()


def standardize_columns(df, column_map, drop_cols=None):
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
    df = df.rename(columns=column_map)
    return df


def save_geopackage(gdf, path, layer="default"):
    gdf.to_file(path, driver="GPKG", layer=layer)


def load_geopackage(path, layer=None):
    return gpd.read_file(path, layer=layer)

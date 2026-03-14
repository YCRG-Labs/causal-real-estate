import sys
import numpy as np
import pandas as pd
from config import CITIES, CLEANED_DIR, MIN_PARCEL_AREA_SQM
from utils import load_geopackage, save_geopackage, drop_slivers, compute_area_sqm


def clean_parcel_ids(gdf):
    gdf["parcel_id"] = gdf["parcel_id"].astype(str).str.strip()
    gdf = gdf[gdf["parcel_id"].notna() & (gdf["parcel_id"] != "")]
    gdf = gdf.drop_duplicates(subset=["parcel_id"], keep="first")
    return gdf


def clean_numeric_cols(gdf):
    numeric_candidates = [
        "lot_area_sqft", "bldg_area_sqft", "res_area_sqft",
        "property_area_sqft", "units_res", "units", "num_floors",
        "num_bldgs", "year_built", "bedrooms", "bathrooms",
        "rooms", "stories", "assessed_land", "assessed_total",
        "assessed_improvement",
    ]
    for col in numeric_candidates:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    return gdf


def filter_year_built(gdf):
    if "year_built" in gdf.columns:
        gdf.loc[gdf["year_built"] < 1700, "year_built"] = np.nan
        gdf.loc[gdf["year_built"] > 2026, "year_built"] = np.nan
    return gdf


def add_area_sqm(gdf, local_crs):
    gdf["area_sqm"] = compute_area_sqm(gdf, local_crs)
    return gdf


def main(city):
    if city not in CITIES:
        print(f"Unknown city: {city}. Options: {list(CITIES.keys())}")
        sys.exit(1)

    cfg = CITIES[city]
    in_path = CLEANED_DIR / f"{city}_parcels_loaded.gpkg"
    gdf = load_geopackage(in_path, layer=city)

    gdf = clean_parcel_ids(gdf)
    gdf = clean_numeric_cols(gdf)
    gdf = filter_year_built(gdf)
    gdf = drop_slivers(gdf, cfg["local_crs"])
    gdf = add_area_sqm(gdf, cfg["local_crs"])

    out_path = CLEANED_DIR / f"{city}_parcels_cleaned.gpkg"
    save_geopackage(gdf, out_path, layer=city)
    print(f"{city}: cleaned {len(gdf)} parcels → {out_path}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "boston"
    main(city)

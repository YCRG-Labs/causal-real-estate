import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from config import CITIES, RAW_DIR, PROCESSED_DIR, AMENITY_RADIUS_M
from utils import ensure_dirs, save_geopackage

AMENITY_DIR = RAW_DIR / "amenities"

MICRO_GEO_CATEGORIES = {
    "park": "recreation",
    "transit": "transportation",
    "school": "education",
    "restaurant": "food_dining",
    "retail": "retail",
    "medical": "services",
}


def distance_to_nearest(parcel_coords, amenity_coords):
    if len(amenity_coords) == 0:
        return np.full(len(parcel_coords), np.nan)
    tree = cKDTree(amenity_coords)
    dists, _ = tree.query(parcel_coords, k=1)
    return dists


def attach_micro_geo(city):
    parcels_path = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"
    amenity_path = AMENITY_DIR / f"{city}_amenities.csv"

    if not parcels_path.exists():
        print(f"{city}: no parcels file, skipping")
        return
    if not amenity_path.exists():
        print(f"{city}: no amenities file, skipping")
        return

    parcels = gpd.read_file(parcels_path, layer=city)
    amenities = pd.read_csv(amenity_path)

    cfg = CITIES[city]

    amenity_gdf = gpd.GeoDataFrame(
        amenities,
        geometry=gpd.points_from_xy(amenities["longitude"], amenities["latitude"]),
        crs="EPSG:4326",
    )

    parcels_proj = parcels.to_crs(cfg["local_crs"])
    amenity_proj = amenity_gdf.to_crs(cfg["local_crs"])

    centroids = parcels_proj.geometry.centroid
    parcel_coords = np.column_stack([centroids.x, centroids.y])

    for feat_name, osm_category in MICRO_GEO_CATEGORIES.items():
        subset = amenity_proj[amenity_proj["category"] == osm_category]
        if len(subset) == 0:
            parcels[f"dist_{feat_name}_m"] = np.nan
            print(f"  {feat_name}: no amenities found")
            continue

        am_coords = np.column_stack([subset.geometry.x, subset.geometry.y])
        dists = distance_to_nearest(parcel_coords, am_coords)
        parcels[f"dist_{feat_name}_m"] = dists

        print(f"  {feat_name}: median={np.nanmedian(dists):.0f}m, "
              f"mean={np.nanmean(dists):.0f}m, max={np.nanmax(dists):.0f}m")

    out_path = PROCESSED_DIR / f"{city}_parcels_micro_geo.gpkg"
    save_geopackage(parcels, out_path, layer=city)
    print(f"{city}: attached micro-geo distances to {len(parcels)} parcels → {out_path}")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        attach_micro_geo(city)


if __name__ == "__main__":
    main()

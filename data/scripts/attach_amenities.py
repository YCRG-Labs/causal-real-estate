import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from config import CITIES, RAW_DIR, PROCESSED_DIR, AMENITY_RADIUS_M, OSM_AMENITY_TAGS
from utils import ensure_dirs, save_geopackage

AMENITY_DIR = RAW_DIR / "amenities"


def shannon_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def attach_amenities(city):
    parcels_path = PROCESSED_DIR / f"{city}_parcels_crime.gpkg"
    amenity_path = AMENITY_DIR / f"{city}_amenities.csv"

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

    amenity_coords = np.column_stack([
        amenity_proj.geometry.x, amenity_proj.geometry.y
    ])

    tree = cKDTree(amenity_coords)
    neighbors = tree.query_ball_point(parcel_coords, r=AMENITY_RADIUS_M)

    categories = list(OSM_AMENITY_TAGS.keys())
    amenity_categories = amenity_proj["category"].fillna("").values
    amenity_subcategories = amenity_proj["subcategory"].fillna("").values

    for cat in categories:
        parcels[f"amenity_{cat}"] = 0

    parcels["amenity_total"] = 0
    parcels["amenity_diversity"] = 0.0

    for i, idx_list in enumerate(neighbors):
        if not idx_list:
            continue

        parcels.iloc[i, parcels.columns.get_loc("amenity_total")] = len(idx_list)

        cats = amenity_categories[idx_list]
        subcats = amenity_subcategories[idx_list]

        for cat in categories:
            count = np.sum(cats == cat)
            parcels.iloc[i, parcels.columns.get_loc(f"amenity_{cat}")] = int(count)

        unique_subcats, subcat_counts = np.unique(subcats, return_counts=True)
        parcels.iloc[i, parcels.columns.get_loc("amenity_diversity")] = shannon_entropy(subcat_counts)

    radius_km = AMENITY_RADIUS_M / 1000.0
    area_sqkm = np.pi * radius_km ** 2
    for cat in categories:
        parcels[f"amenity_{cat}_density"] = parcels[f"amenity_{cat}"] / area_sqkm
    parcels["amenity_total_density"] = parcels["amenity_total"] / area_sqkm

    out_path = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"
    save_geopackage(parcels, out_path, layer=city)
    print(f"{city}: attached amenities to {len(parcels)} parcels → {out_path}")
    print(f"  mean total: {parcels['amenity_total'].mean():.1f}, mean density/sqkm: {parcels['amenity_total_density'].mean():.1f}, mean diversity: {parcels['amenity_diversity'].mean():.2f}")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        attach_amenities(city)


if __name__ == "__main__":
    main()

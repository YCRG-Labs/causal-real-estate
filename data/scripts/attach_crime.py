import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from config import CITIES, RAW_DIR, PROCESSED_DIR, CRIME_KDE_BANDWIDTH_M, CRIME_TEMPORAL_WINDOW_DAYS
from utils import ensure_dirs, save_geopackage
from download_crime import CRIME_CROSSWALK

CRIME_DIR = RAW_DIR / "crime"


def classify_crime(offense_desc, city):
    offense_upper = str(offense_desc).upper().strip()
    for category, city_terms in CRIME_CROSSWALK.items():
        terms = city_terms.get(city, [])
        for term in terms:
            if term.upper() in offense_upper:
                return category
    return "other"


def count_within_radius(tree, query_points, radius):
    return np.array(tree.query_ball_point(query_points, r=radius, return_length=True))


def find_sale_date_col(gdf):
    for col in ["sale_date", "last_sale_date", "date"]:
        if col in gdf.columns:
            return col
    return None


def attach_crime(city):
    parcels_path = PROCESSED_DIR / f"{city}_parcels_census.gpkg"
    crime_path = CRIME_DIR / f"{city}_crime.csv"

    parcels = gpd.read_file(parcels_path, layer=city)
    crime = pd.read_csv(crime_path)

    cfg = CITIES[city]

    crime["date"] = pd.to_datetime(crime["date"], errors="coerce")
    crime = crime.dropna(subset=["date"])

    crime_gdf = gpd.GeoDataFrame(
        crime,
        geometry=gpd.points_from_xy(crime["longitude"], crime["latitude"]),
        crs="EPSG:4326",
    )

    parcels_proj = parcels.to_crs(cfg["local_crs"])
    crime_proj = crime_gdf.to_crs(cfg["local_crs"])

    crime_proj["crime_category"] = crime_proj["offense_desc"].apply(
        lambda x: classify_crime(x, city)
    )

    centroids = parcels_proj.geometry.centroid
    parcel_coords = np.column_stack([centroids.x, centroids.y])

    sale_col = find_sale_date_col(parcels)
    use_temporal = sale_col is not None

    if use_temporal:
        parcels["_sale_dt"] = pd.to_datetime(parcels[sale_col], errors="coerce")
        has_date = parcels["_sale_dt"].notna()
        if has_date.sum() < len(parcels) * 0.1:
            use_temporal = False
            print(f"  <10% parcels have sale dates, using all crimes")

    if use_temporal:
        print(f"  Using {CRIME_TEMPORAL_WINDOW_DAYS}-day temporal window from {sale_col}")
        crime_dates = crime_proj["date"].values
        crime_coords_all = np.column_stack([crime_proj.geometry.x, crime_proj.geometry.y])
        crime_cats = crime_proj["crime_category"].values

        for category in ["violent", "property", "quality_of_life", "other"]:
            parcels[f"crime_{category}"] = 0.0
        parcels["crime_total"] = 0.0

        tree = cKDTree(crime_coords_all)
        spatial_neighbors = tree.query_ball_point(parcel_coords, r=CRIME_KDE_BANDWIDTH_M)

        for i in range(len(parcels)):
            sale_dt = parcels["_sale_dt"].iloc[i]
            if pd.isna(sale_dt):
                continue

            idx_list = spatial_neighbors[i]
            if not idx_list:
                continue

            idx_arr = np.array(idx_list)
            dates = crime_dates[idx_arr]
            window_start = sale_dt - pd.Timedelta(days=CRIME_TEMPORAL_WINDOW_DAYS)
            temporal_mask = (dates >= np.datetime64(window_start)) & (dates <= np.datetime64(sale_dt))
            filtered_idx = idx_arr[temporal_mask]

            if len(filtered_idx) == 0:
                continue

            parcels.iat[i, parcels.columns.get_loc("crime_total")] = len(filtered_idx)
            cats = crime_cats[filtered_idx]
            for category in ["violent", "property", "quality_of_life", "other"]:
                parcels.iat[i, parcels.columns.get_loc(f"crime_{category}")] = int(np.sum(cats == category))

        parcels = parcels.drop(columns=["_sale_dt"])
    else:
        for category in ["violent", "property", "quality_of_life", "other"]:
            subset = crime_proj[crime_proj["crime_category"] == category]
            if len(subset) == 0:
                parcels[f"crime_{category}"] = 0.0
                continue
            crime_coords = np.column_stack([subset.geometry.x, subset.geometry.y])
            tree = cKDTree(crime_coords)
            parcels[f"crime_{category}"] = count_within_radius(
                tree, parcel_coords, CRIME_KDE_BANDWIDTH_M
            )

        all_coords = np.column_stack([crime_proj.geometry.x, crime_proj.geometry.y])
        tree_all = cKDTree(all_coords)
        parcels["crime_total"] = count_within_radius(
            tree_all, parcel_coords, CRIME_KDE_BANDWIDTH_M
        )

    out_path = PROCESSED_DIR / f"{city}_parcels_crime.gpkg"
    save_geopackage(parcels, out_path, layer=city)
    print(f"{city}: attached crime to {len(parcels)} parcels → {out_path}")
    print(f"  {len(crime)} total incidents, {len(crime_proj[crime_proj['crime_category'] != 'other'])} classified")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        attach_crime(city)


if __name__ == "__main__":
    main()

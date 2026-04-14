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

    crime_min = crime_proj["date"].min()
    crime_max = crime_proj["date"].max()
    crime_year_range = f"{crime_min.year}-{crime_max.year}" if pd.notna(crime_min) else "unknown"

    parcels["crime_data_year_range"] = crime_year_range
    parcels["crime_temporal_match"] = False

    if use_temporal:
        parcels["_sale_dt"] = pd.to_datetime(parcels[sale_col], errors="coerce")
        has_date = parcels["_sale_dt"].notna()
        if has_date.sum() < len(parcels) * 0.1:
            use_temporal = False
            print(f"  <10% parcels have sale dates, using all crimes")
            print(f"  ⚠ crime_temporal_match=False for all {city} parcels (no sale-date overlap)")

    if use_temporal:
        median_sale = parcels["_sale_dt"].dropna().median()
        window_start = median_sale - pd.Timedelta(days=CRIME_TEMPORAL_WINDOW_DAYS)
        filtered = crime_proj[
            (crime_proj["date"] >= window_start) & (crime_proj["date"] <= median_sale)
        ]
        if len(filtered) > 1000:
            crime_proj = filtered
            parcels.loc[parcels["_sale_dt"].notna(), "crime_temporal_match"] = (
                (parcels.loc[parcels["_sale_dt"].notna(), "_sale_dt"] >= window_start)
                & (parcels.loc[parcels["_sale_dt"].notna(), "_sale_dt"] <= median_sale)
            )
            print(f"  Filtered crimes to {window_start.date()} - {median_sale.date()}: {len(crime_proj)} incidents")
            n_match = parcels["crime_temporal_match"].sum()
            print(f"  {n_match}/{len(parcels)} parcels have sales within the crime window")
        else:
            print(f"  ⚠ Temporal window too sparse ({len(filtered)} incidents in "
                  f"{window_start.date()}-{median_sale.date()}), using all {len(crime_proj)} incidents")
            print(f"  ⚠ crime_temporal_match=False for all parcels — crime data is "
                  f"{crime_year_range} but sales are not in this window")
        parcels = parcels.drop(columns=["_sale_dt"])

    if True:
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

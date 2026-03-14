import sys
import geopandas as gpd
from config import CITIES, CLEANED_DIR, PROCESSED_DIR
from utils import load_geopackage, save_geopackage, reproject, ensure_dirs


def compute_centroids(gdf, local_crs):
    projected = gdf.to_crs(local_crs)
    centroids_proj = projected.geometry.centroid
    centroids_wgs84 = centroids_proj.to_crs("EPSG:4326")
    gdf["latitude"] = centroids_wgs84.y
    gdf["longitude"] = centroids_wgs84.x
    return gdf


def main(city):
    ensure_dirs()

    if city not in CITIES:
        print(f"Unknown city: {city}. Options: {list(CITIES.keys())}")
        sys.exit(1)

    cfg = CITIES[city]
    in_path = CLEANED_DIR / f"{city}_parcels_cleaned.gpkg"
    gdf = load_geopackage(in_path, layer=city)

    gdf = compute_centroids(gdf, cfg["local_crs"])

    out_path = PROCESSED_DIR / f"{city}_parcels.gpkg"
    save_geopackage(gdf, out_path, layer=city)
    print(f"{city}: computed centroids for {len(gdf)} parcels → {out_path}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "boston"
    main(city)

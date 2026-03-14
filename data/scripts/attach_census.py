import sys
import numpy as np
import geopandas as gpd
from config import CITIES, RAW_DIR, PROCESSED_DIR, CENSUS_VARIABLES, CENSUS_DERIVED, CENSUS_DROP_RAW
from utils import ensure_dirs, save_geopackage

CENSUS_DIR = RAW_DIR / "census"


def compute_derived_vars(gdf):
    for derived_name, (numerator, denominator) in CENSUS_DERIVED.items():
        if numerator in gdf.columns and denominator in gdf.columns:
            gdf[derived_name] = (
                gdf[numerator] / gdf[denominator].replace(0, np.nan)
            )

    if "age_total" in gdf.columns:
        young_cols = [c for c in ["age_5_9", "age_18_19"] if c in gdf.columns]
        old_cols = [c for c in ["age_60_61", "age_85_plus"] if c in gdf.columns]
        total = gdf["age_total"].replace(0, np.nan)
        if young_cols:
            gdf["pct_under_25"] = gdf[young_cols].sum(axis=1) / total
        if old_cols:
            gdf["pct_over_60"] = gdf[old_cols].sum(axis=1) / total

    gdf = gdf.drop(columns=[c for c in CENSUS_DROP_RAW if c in gdf.columns])
    return gdf


def attach_census(city):
    parcels_path = PROCESSED_DIR / f"{city}_parcels.gpkg"
    census_path = CENSUS_DIR / f"{city}_block_groups.gpkg"

    parcels = gpd.read_file(parcels_path, layer=city)
    block_groups = gpd.read_file(census_path, layer="block_groups")

    if parcels.crs != block_groups.crs:
        parcels = parcels.to_crs(block_groups.crs)

    census_cols = list(CENSUS_VARIABLES.values()) + ["GEOID"]
    available = [c for c in census_cols if c in block_groups.columns]

    joined = gpd.sjoin(parcels, block_groups[available + ["geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")

    joined = compute_derived_vars(joined)

    out_path = PROCESSED_DIR / f"{city}_parcels_census.gpkg"
    save_geopackage(joined, out_path, layer=city)
    print(f"{city}: attached census to {len(joined)} parcels → {out_path}")

    matched = joined["GEOID"].notna().sum()
    print(f"  {matched}/{len(joined)} matched ({matched/len(joined)*100:.1f}%)")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        attach_census(city)


if __name__ == "__main__":
    main()

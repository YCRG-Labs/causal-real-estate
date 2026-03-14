import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from config import CITIES, PROCESSED_DIR
from utils import ensure_dirs, save_geopackage


def temporal_split(gdf, date_col, train_frac=0.7, val_frac=0.15):
    has_date = gdf[date_col].notna()
    dated = gdf[has_date].copy()
    undated = gdf[~has_date].copy()

    dated[date_col] = pd.to_datetime(dated[date_col], errors="coerce")
    dated = dated.dropna(subset=[date_col])
    dated = dated.sort_values(date_col)

    n = len(dated)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    dated.loc[dated.index[:train_end], "split"] = "train"
    dated.loc[dated.index[train_end:val_end], "split"] = "validation"
    dated.loc[dated.index[val_end:], "split"] = "test"

    undated["split"] = "train"

    result = pd.concat([dated, undated], ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)


def validate_split(gdf, local_crs):
    splits = gdf["split"].unique()
    stats = {}

    projected = gdf.to_crs(local_crs)
    centroids = projected.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    for split in splits:
        mask = gdf["split"] == split
        subset = gdf[mask]
        sub_coords = coords[mask.values]

        if len(sub_coords) < 2:
            continue

        tree = cKDTree(sub_coords)
        dists, _ = tree.query(sub_coords, k=2)
        mean_nn = dists[:, 1].mean()

        income = subset["median_household_income"].dropna()
        median_income = income.median() if len(income) > 0 else np.nan

        stats[split] = {
            "n": len(subset),
            "mean_nn_dist_m": round(mean_nn, 1),
            "median_income": round(median_income, 0) if not np.isnan(median_income) else None,
        }

    demo_cols = ["median_household_income", "median_home_value", "pct_bachelors"]
    available = [c for c in demo_cols if c in gdf.columns]

    if len(splits) >= 2 and available:
        for col in available:
            vals = gdf[col].dropna()
            if len(vals) < 10:
                continue

            bins = np.histogram_bin_edges(vals, bins=20)
            distributions = {}
            for split in splits:
                subset_vals = gdf[gdf["split"] == split][col].dropna()
                hist, _ = np.histogram(subset_vals, bins=bins, density=True)
                hist = hist + 1e-10
                distributions[split] = hist / hist.sum()

            split_list = sorted(distributions.keys())
            for i, s1 in enumerate(split_list):
                for s2 in split_list[i+1:]:
                    js = jensenshannon(distributions[s1], distributions[s2])
                    key = f"js_{col}_{s1}_vs_{s2}"
                    stats.setdefault("divergences", {})[key] = round(js, 4)

    return stats


def find_date_col(gdf):
    candidates = ["sale_date", "last_sale_date", "date"]
    for col in candidates:
        if col in gdf.columns:
            return col
    return None


def split_city(city):
    cfg = CITIES[city]
    parcels_path = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"

    if not parcels_path.exists():
        parcels_path = PROCESSED_DIR / f"{city}_parcels_crime.gpkg"
    if not parcels_path.exists():
        parcels_path = PROCESSED_DIR / f"{city}_parcels_census.gpkg"
    if not parcels_path.exists():
        parcels_path = PROCESSED_DIR / f"{city}_parcels.gpkg"

    gdf = gpd.read_file(parcels_path, layer=city)

    date_col = find_date_col(gdf)
    if date_col is None:
        print(f"{city}: no date column found, assigning all to train")
        gdf["split"] = "train"
    else:
        gdf = temporal_split(gdf, date_col)

    stats = validate_split(gdf, cfg["local_crs"])
    print(f"\n{city} split validation:")
    for split, info in stats.items():
        if split == "divergences":
            continue
        print(f"  {split}: n={info['n']}, mean_nn={info['mean_nn_dist_m']}m, median_income=${info.get('median_income', 'N/A')}")

    if "divergences" in stats:
        print("  JS divergences:")
        for key, val in stats["divergences"].items():
            print(f"    {key}: {val}")

    out_path = PROCESSED_DIR / f"{city}_parcels_split.gpkg"
    save_geopackage(gdf, out_path, layer=city)
    print(f"  Saved → {out_path}")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        split_city(city)


if __name__ == "__main__":
    main()

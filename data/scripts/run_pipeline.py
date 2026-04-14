import argparse
import subprocess
import sys
import time

CITIES = ["boston", "nyc", "sf"]

PER_CITY_DATA_PREP = [
    "load_parcels.py",
    "clean_parcels.py",
    "geocode_and_centroids.py",
    "attach_census.py",
    "attach_crime.py",
    "attach_amenities.py",
    "attach_micro_geo.py",
]

PER_CITY_EMBEDDINGS = [
    "generate_embeddings.py",
]

PER_CITY_ANALYSIS = [
    "causal_inference.py",
]

MULTI_CITY_ANALYSIS = [
    "threshold_sensitivity.py",
    "extended_analysis.py",
    "run_sensitivity.py",
]


def run(script, args, label=None):
    label = label or f"{script} {' '.join(args)}"
    print(f"\n{'='*70}")
    print(f"▶ {label}")
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        capture_output=False,
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n✘ FAILED: {label} (exit {result.returncode}, {dt:.1f}s)")
        sys.exit(result.returncode)
    print(f"✔ {label}  ({dt:.1f}s)")


def run_per_city(scripts, cities):
    for city in cities:
        for script in scripts:
            run(script, [city], label=f"{script} [{city}]")


def run_multi_city(scripts, cities):
    for script in scripts:
        run(script, cities, label=f"{script} [{' '.join(cities)}]")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end causal-real-estate pipeline.",
    )
    parser.add_argument(
        "cities", nargs="*", default=CITIES,
        help=f"Cities to process (default: {' '.join(CITIES)})",
    )
    parser.add_argument(
        "--skip-data-prep", action="store_true",
        help="Skip parcel loading/cleaning/geocoding/census/crime/amenity/micro-geo stages",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip text embedding generation (use existing parquet files)",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip causal inference + threshold sensitivity + extended analysis",
    )
    parser.add_argument(
        "--only", choices=["data", "embeddings", "analysis"],
        help="Run only one stage and exit",
    )
    args = parser.parse_args()

    cities = args.cities

    print(f"\n{'#'*70}")
    print(f"# CAUSAL-REAL-ESTATE PIPELINE")
    print(f"# Cities: {', '.join(cities)}")
    print(f"{'#'*70}")

    t_start = time.time()

    do_data = not args.skip_data_prep and args.only in (None, "data")
    do_emb = not args.skip_embeddings and args.only in (None, "embeddings")
    do_ana = not args.skip_analysis and args.only in (None, "analysis")

    if do_data:
        print(f"\n■ Stage 1/3: Data preparation")
        run_per_city(PER_CITY_DATA_PREP, cities)

    if do_emb:
        print(f"\n■ Stage 2/3: Embeddings")
        run_per_city(PER_CITY_EMBEDDINGS, cities)

    if do_ana:
        print(f"\n■ Stage 3/3: Causal analysis")
        run_per_city(PER_CITY_ANALYSIS, cities)
        run_multi_city(MULTI_CITY_ANALYSIS, cities)

    dt = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"# PIPELINE COMPLETE  ({dt/60:.1f} min)  cities: {', '.join(cities)}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()

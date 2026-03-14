import subprocess
import sys

CITIES = ["boston", "nyc", "sf"]

SCRIPTS = [
    "load_parcels.py",
    "clean_parcels.py",
    "geocode_and_centroids.py",
    "attach_census.py",
]


def run(script, city):
    print(f"\n{'='*60}")
    print(f"Running {script} for {city}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", city],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(f"FAILED: {script} for {city}")
        print(result.stderr)
        sys.exit(1)


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else CITIES

    for city in cities:
        for script in SCRIPTS:
            run(script, city)

    print(f"\nPipeline complete for: {', '.join(cities)}")


if __name__ == "__main__":
    main()

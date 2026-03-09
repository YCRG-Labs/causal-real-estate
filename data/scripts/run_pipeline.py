"""
Causal Real Estate Pipeline

...

This script runs the complete end-to-end pipeline for the Causal Real Estate project.
"""

import subprocess

cities = ["boston", "nyc", "sf"]

scripts = [
"load_parcels.py",
"clean_parcels.py",
"geocode_and_centroids.py",
# process listings
# attach census
# attach crime
# attach amenities
# feature engineering
# final build
]

for city in cities: # Each script receives the city as an argument
    for script in scripts:
        subprocess.run(["python", f"scripts/{script}", city])
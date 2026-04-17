#!/usr/bin/env bash
# Resume the pipeline from wherever it left off.
# Usage: bash scripts/resume_pipeline.sh
# Run from inside data/

set -e
cd "$(dirname "$0")/../"

echo "=== Checking what's already done ==="

# Crime
for city in boston nyc; do
    f="raw/crime/${city}_crime.csv"
    if [ -f "$f" ]; then
        echo "  ✔ $f exists ($(du -h "$f" | cut -f1))"
    else
        echo "  ✘ $f missing — run: python3 scripts/download_crime.py $city"
        exit 1
    fi
done

# Amenities
for city in boston nyc sf; do
    f="raw/amenities/${city}_amenities.csv"
    if [ -f "$f" ]; then
        echo "  ✔ $f exists ($(du -h "$f" | cut -f1))"
    else
        echo "  ✘ $f missing — downloading..."
        python3 scripts/download_amenities.py "$city"
    fi
done

# Descriptions
for city in boston nyc; do
    f="raw/descriptions/${city}_descriptions.csv"
    if [ -f "$f" ]; then
        echo "  ✔ $f exists ($(du -h "$f" | cut -f1))"
    else
        echo "  ✘ $f missing — scraping..."
        python3 scripts/scrape_descriptions.py "$city"
    fi
done

echo ""
echo "=== All downloads present. Running pipeline ==="
python3 scripts/run_pipeline.py boston nyc sf

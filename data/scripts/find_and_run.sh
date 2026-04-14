#!/usr/bin/env bash
#
# Find existing Boston/NYC processed parquet + gpkg files anywhere on this
# Mac, copy them into data/processed/, then run the analysis-only pipeline
# for whichever cities are present.
#
# Usage:
#   bash data/scripts/find_and_run.sh
#
# Run from the repo root (~/causal-real-estate). Searches Spotlight first
# (fast), then falls back to `find` over ~ and /Volumes.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROCESSED_DIR="$REPO_ROOT/data/processed"
mkdir -p "$PROCESSED_DIR"

echo "Repo:      $REPO_ROOT"
echo "Processed: $PROCESSED_DIR"
echo

CITIES_FOUND=()

# Required files per city: embeddings parquet + ANY of the parcels gpkg
# variants (load_analysis_data prefers micro_geo, falls back to amenities).
PARCEL_VARIANTS=(parcels_micro_geo parcels_amenities parcels_census parcels_crime parcels)

find_one() {
  local pattern="$1"
  local hit=""

  if command -v mdfind >/dev/null 2>&1; then
    hit="$(mdfind -name "$pattern" 2>/dev/null | head -1)"
  fi

  if [ -z "$hit" ]; then
    hit="$(find "$HOME" /Volumes -name "$pattern" 2>/dev/null | head -1)"
  fi

  echo "$hit"
}

copy_if_new() {
  local src="$1"
  local dst="$PROCESSED_DIR/$(basename "$src")"
  if [ -f "$dst" ]; then
    echo "  already in place: $dst"
  else
    cp "$src" "$dst"
    echo "  copied: $src → $dst"
  fi
}

search_city() {
  local city="$1"
  echo "=== searching for $city ==="

  local emb
  emb="$(find_one "${city}_embeddings.parquet")"
  if [ -z "$emb" ]; then
    echo "  ✘ ${city}_embeddings.parquet not found anywhere"
    return 1
  fi
  echo "  ✔ embeddings: $emb"
  copy_if_new "$emb"

  local parcels=""
  for variant in "${PARCEL_VARIANTS[@]}"; do
    local hit
    hit="$(find_one "${city}_${variant}.gpkg")"
    if [ -n "$hit" ]; then
      parcels="$hit"
      echo "  ✔ parcels:    $hit"
      copy_if_new "$hit"
      break
    fi
  done
  if [ -z "$parcels" ]; then
    echo "  ⚠ no ${city}_parcels_*.gpkg found; analysis will run without rich confounders"
  fi

  CITIES_FOUND+=("$city")
  return 0
}

search_city boston || true
echo
search_city nyc    || true
echo

# SF is local — confirm it's still present
if [ -f "$PROCESSED_DIR/sf_embeddings.parquet" ]; then
  CITIES_FOUND+=("sf")
  echo "✔ sf already in $PROCESSED_DIR"
else
  echo "⚠ sf_embeddings.parquet missing from $PROCESSED_DIR"
fi
echo

if [ ${#CITIES_FOUND[@]} -eq 0 ]; then
  echo "No cities found. Cannot run pipeline."
  exit 1
fi

echo "=== running analysis pipeline for: ${CITIES_FOUND[*]} ==="
cd "$REPO_ROOT/data"
python3 scripts/run_pipeline.py --skip-data-prep --skip-embeddings "${CITIES_FOUND[@]}"

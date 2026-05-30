"""Per-city data downloaders.

Each module exports two functions:

    download_parcels(out_dir: Path) -> Path
        Download parcel/assessor data for the city to a canonical
        GeoJSON file at out_dir / f"{slug}_parcels.geojson". Returns
        the output path. Reprojects to WGS84 if needed.

    download_crime(out_dir: Path, token: str | None = None) -> Path
        Download incident-level crime data for the city to a canonical
        Parquet file at out_dir / f"{slug}_crime.parquet". Returns the
        output path. Adds canonical crime_category column mapped to one
        of {"violent", "property", "quality_of_life", None}.

The dispatch function `get_downloader(slug)` returns the appropriate
module so the orchestrator can call uniformly across cities.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

_MODULES = {
    "dc": "dc",
    "philadelphia": "philadelphia",
    "chicago": "chicago",
    "seattle": "seattle",
    "denver": "denver",
    "atlanta": "atlanta",
    "portland": "portland",
    "phoenix": "phoenix",
    "dallas": "dallas",
}


def get_downloader(slug: str):
    if slug not in _MODULES:
        raise KeyError(f"no downloader for city slug={slug!r}; available: {list(_MODULES)}")
    return importlib.import_module(f"city_downloaders.{_MODULES[slug]}")


def list_available() -> list[str]:
    return list(_MODULES.keys())

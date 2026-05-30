"""Shared helpers for per-city downloaders.

Three primitives used across cities:
  - arcgis_rest_query: paginate an ArcGIS REST FeatureServer query
  - socrata_query: paginate a Socrata SODA endpoint
  - canonicalize_crime: map per-city offense values to {violent, property, quality_of_life, other}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from pipeline_dicts import CRIME_CROSSWALK  # noqa: E402

USER_AGENT = "YCRG-Labs JBES-2026 research (jacobcrainic@icloud.com)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}


def arcgis_rest_query(
    base_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = True,
    page_size: int = 2000,
    f: str = "geojson",
    extra_params: Optional[dict] = None,
    max_retries: int = 3,
) -> Iterable[dict]:
    """Yield ArcGIS REST query result chunks until exhausted."""
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": str(return_geometry).lower(),
            "resultRecordCount": page_size,
            "resultOffset": offset,
            "f": f,
        }
        if extra_params:
            params.update(extra_params)
        for attempt in range(max_retries):
            try:
                r = requests.get(f"{base_url}/query", params=params, headers=DEFAULT_HEADERS, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  arcgis retry {attempt+1}/{max_retries} after {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
        payload = r.json()
        features = payload.get("features", [])
        if not features:
            break
        yield payload
        if len(features) < page_size:
            break
        offset += page_size


def socrata_query(
    base_url: str,
    where: Optional[str] = None,
    select: str = "*",
    order: Optional[str] = None,
    limit: int = 50000,
    app_token: Optional[str] = None,
    max_pages: int = 200,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Paginate a Socrata SODA JSON endpoint and return concatenated DataFrame."""
    frames = []
    offset = 0
    headers = dict(DEFAULT_HEADERS)
    if app_token:
        headers["X-App-Token"] = app_token
    for page in range(max_pages):
        params = {"$limit": limit, "$offset": offset, "$select": select}
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        for attempt in range(max_retries):
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  socrata retry {attempt+1}/{max_retries} after {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
        rows = r.json()
        if not rows:
            break
        frames.append(pd.json_normalize(rows))
        if len(rows) < limit:
            break
        offset += limit
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def canonicalize_crime(df: pd.DataFrame, slug: str, offense_field: str) -> pd.DataFrame:
    """Add canonical crime_category column based on per-city crosswalk."""
    if offense_field not in df.columns:
        df["crime_category"] = None
        return df
    cw = {}
    for cat in ("violent", "property", "quality_of_life"):
        for v in CRIME_CROSSWALK[cat].get(slug, []):
            cw[v] = cat
    df["crime_category"] = df[offense_field].map(cw)
    return df


def write_parquet(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def write_geojson_from_arcgis_chunks(chunks: Iterable[dict], out_path: Path) -> Path:
    """Write concatenated ArcGIS REST geojson chunks to a single .geojson file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_features = []
    for chunk in chunks:
        all_features.extend(chunk.get("features", []))
    import json
    payload = {
        "type": "FeatureCollection",
        "features": all_features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }
    with out_path.open("w") as f:
        json.dump(payload, f)
    return out_path

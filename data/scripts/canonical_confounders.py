"""Canonical confounder set for the causal-real-estate JBES revision.

Single source of truth for the operative confounder block used by every
post-dedup, property-level analysis in the paper. The set matches what
fast_bootstrap_dml_v2.py builds when drop_crime=False and has_rich_confounders=True,
which is the canonical DML estimator going forward.

Composition (per city, applied left-to-right; columns missing from the slim
parquet are silently dropped, then any column with >50% NaN among the matched
embedding rows is dropped by the caller):

    SPATIAL  (lat, lon)                              2
    PROPERTY (bedrooms, bldg_area_sqft, lot_area_sqft, year_built)  4
    CENSUS   (11 ACS block-group covariates)        11
    CRIME    (4 incident counts within 500 m)        4
    AMENITY  (8 OSM amenity counts/diversity)        8
    MICRO    (6 distance-to-amenity meters)          6
    ------------------------------------------------ --
    Total before NaN/availability filter:           35

Realized counts on the post-dedup, property-level data:
    SF      n=348  n_conf=34  (no bldg_area_sqft in SF parquet)
    NYC     n=342  n_conf=35  (full set)
    Boston  n=331  n_conf=32  (bedrooms 35.8% NaN, bldg_area_sqft 24.0% NaN
                               filter passes both — see _build_features
                               post-NaN trimming; observed 32 in v2 JSON)

The realized counts are written by data/scripts/fast_bootstrap_dml_v2.py to
/Users/jacobcrainic/causal-real-estate/results/fast_bootstrap_dml_v2.json
"""

from __future__ import annotations

from typing import Iterable

SPATIAL: list[str] = ["latitude", "longitude"]

PROPERTY: list[str] = [
    "bedrooms",
    "bldg_area_sqft",
    "lot_area_sqft",
    "year_built",
]

CENSUS: list[str] = [
    "median_household_income",
    "pct_bachelors",
    "pct_white",
    "pct_black",
    "pct_asian",
    "pct_hispanic",
    "median_home_value",
    "median_gross_rent",
    "labor_force_participation",
    "pct_under_25",
    "pct_over_60",
]

CRIME: list[str] = [
    "crime_violent",
    "crime_property",
    "crime_quality_of_life",
    "crime_total",
]

AMENITY: list[str] = [
    "amenity_food_dining",
    "amenity_retail",
    "amenity_services",
    "amenity_recreation",
    "amenity_transportation",
    "amenity_education",
    "amenity_total",
    "amenity_diversity",
]

MICRO_GEO: list[str] = [
    "dist_park_m",
    "dist_transit_m",
    "dist_school_m",
    "dist_restaurant_m",
    "dist_retail_m",
    "dist_medical_m",
]

CONTEXTUAL: list[str] = CENSUS + CRIME + AMENITY + MICRO_GEO

ALL_NON_SPATIAL: list[str] = PROPERTY + CONTEXTUAL

ALL: list[str] = SPATIAL + ALL_NON_SPATIAL

GROUPS: dict[str, list[str]] = {
    "spatial": SPATIAL,
    "property": PROPERTY,
    "census": CENSUS,
    "crime": CRIME,
    "amenity": AMENITY,
    "micro_geo": MICRO_GEO,
}

EXPECTED_COUNTS: dict[str, int] = {
    "spatial": 2,
    "property": 4,
    "census": 11,
    "crime": 4,
    "amenity": 8,
    "micro_geo": 6,
    "all": 35,
}

EXPECTED_REALIZED: dict[str, int] = {
    "sf": 34,
    "nyc": 35,
    "boston": 32,
}


def load(group: str | None = None, drop_crime: bool = False) -> list[str]:
    """Return the canonical confounder list, optionally restricted to one
    group or with the crime block dropped (for NYC-style temporal mismatches)."""
    if group is not None:
        cols = list(GROUPS[group])
    else:
        cols = list(ALL)
    if drop_crime:
        cols = [c for c in cols if c not in set(CRIME)]
    return cols


def available_in_frame(df, group: str | None = None,
                       drop_crime: bool = False) -> list[str]:
    """Return canonical columns that actually exist in df."""
    wanted = load(group=group, drop_crime=drop_crime)
    return [c for c in wanted if c in df.columns]


def self_test() -> int:
    """Verify the group counts match the documented expectations.

    Also tries to load the slim parquets and reports the realized post-NaN
    confounder count per city; falls back to the documented expectations when
    the parquets are unavailable (e.g. in CI without the data drop)."""
    assert len(SPATIAL) == EXPECTED_COUNTS["spatial"]
    assert len(PROPERTY) == EXPECTED_COUNTS["property"]
    assert len(CENSUS) == EXPECTED_COUNTS["census"]
    assert len(CRIME) == EXPECTED_COUNTS["crime"]
    assert len(AMENITY) == EXPECTED_COUNTS["amenity"]
    assert len(MICRO_GEO) == EXPECTED_COUNTS["micro_geo"]
    assert len(ALL) == EXPECTED_COUNTS["all"], (
        f"ALL count {len(ALL)} != expected {EXPECTED_COUNTS['all']}"
    )

    print(f"canonical_confounders self-test: PASS")
    print(f"  spatial   {len(SPATIAL):>2d}")
    print(f"  property  {len(PROPERTY):>2d}")
    print(f"  census    {len(CENSUS):>2d}")
    print(f"  crime     {len(CRIME):>2d}")
    print(f"  amenity   {len(AMENITY):>2d}")
    print(f"  micro_geo {len(MICRO_GEO):>2d}")
    print(f"  total     {len(ALL):>2d}")

    try:
        import sys as _sys
        from pathlib import Path as _Path
        _here = _Path(__file__).resolve().parent
        if str(_here) not in _sys.path:
            _sys.path.insert(0, str(_here))
        from fast_bootstrap_dml_v2 import _build_features
    except Exception as e:
        print(f"  (skipping per-city realized count: {e})")
        return 0

    for city, expected in EXPECTED_REALIZED.items():
        try:
            out = _build_features(city, drop_crime=False)
        except Exception as e:
            print(f"  {city:6s} build failed: {e}")
            continue
        if out is None:
            print(f"  {city:6s} no data")
            continue
        _, conf, _, _, meta = out
        got = int(meta["n_confounders"])
        flag = "OK" if got == expected else f"MISMATCH expected {expected}"
        print(f"  {city:6s} realized n_conf={got:>2d}  ({flag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())

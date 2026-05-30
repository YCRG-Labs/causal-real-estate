"""City endpoints config for the 12-city JBES expansion.

Each city maps to a struct with the canonical data sources required by the
existing pipeline (parcel/assessor, crime, transit GTFS, Redfin city id, FIPS
codes, micro-geography hints). URLs are WebFetch-verified as of 2026-05-30;
see the matching commit message for the verification trail.

Usage from anywhere in the repo:

    from data.scripts.city_endpoints import CITIES, get_city, list_ready
    cfg = get_city("dallas")
    print(cfg.parcel.url)

The pipeline scrapers (scrape_redfin.py, build_confounders.py) iterate over
CITIES in `list_ready()` order, which sorts by data-quality maturity so the
cleanest sources (DC, Philadelphia, Chicago) run first and the noisier ones
(Phoenix, Dallas, Portland) run after the pipeline patterns are debugged.

Verification status meanings:
  - "ready": Socrata / ArcGIS Hub endpoint returns expected metadata
  - "ready_with_friction": works but needs a per-city adapter (Tableau, CRS reproject, etc.)
  - "ready_with_caveat": works but has a documented confounder-quality gap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class ParcelSource:
    url: str
    fmt: str           # "geojson", "shapefile", "csv", "arcgis_rest"
    crs: int           # source CRS as EPSG code; reproject to 4326 in pipeline
    coverage: str      # what the file contains; pipeline filters to city polygon
    note: str = ""


@dataclass(frozen=True)
class CrimeSource:
    url: str
    fmt: str           # "socrata_json", "arcgis_rest", "csv_download", "tableau", "carto_sql"
    has_latlon: bool   # True for incident-level (Chicago, Seattle); False for block-level (Houston)
    has_qol: bool      # False for primary-offense-only feeds (Phoenix)
    note: str = ""


@dataclass(frozen=True)
class CityConfig:
    slug: str
    display_name: str
    state: str
    state_fips: str
    county_fips: tuple[str, ...]      # multiple if city spans counties (Atlanta)
    redfin_city_id: str
    redfin_state_slug: str
    redfin_name_slug: str
    parcel: ParcelSource
    crime: CrimeSource
    transit_gtfs_mdb: Optional[str]   # Mobility Database feed id (mdb-XXX)
    status: str                       # "existing", "ready", "ready_with_friction", "ready_with_caveat"
    tier: str                         # mechanism prediction: "pc1_predicted", "shen_predicted", "null_predicted"
    notes: list[str] = field(default_factory=list)

    @property
    def redfin_url(self) -> str:
        return f"https://www.redfin.com/city/{self.redfin_city_id}/{self.redfin_state_slug}/{self.redfin_name_slug}"


CITIES: dict[str, CityConfig] = {
    # ------------------------------------------------------------------ existing
    "sf": CityConfig(
        slug="sf",
        display_name="San Francisco",
        state="CA", state_fips="06", county_fips=("075",),
        redfin_city_id="17151", redfin_state_slug="CA", redfin_name_slug="San-Francisco",
        parcel=ParcelSource(
            url="https://data.sfgov.org/resource/acdm-wktn.geojson",
            fmt="geojson", crs=4326,
            coverage="City and County of San Francisco (consolidated)",
            note="Existing canonical source; refreshed weekly",
        ),
        crime=CrimeSource(
            url="https://data.sfgov.org/resource/wg3w-h783.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="Existing canonical source",
        ),
        transit_gtfs_mdb="mdb-1066",
        status="existing", tier="shen_predicted",
        notes=["Shen-Ross +0.130 effect confirmed; reference implementation"],
    ),
    "nyc": CityConfig(
        slug="nyc",
        display_name="New York",
        state="NY", state_fips="36",
        county_fips=("061", "047", "005", "081", "085"),  # Manhattan, Brooklyn, Bronx, Queens, Staten Island
        redfin_city_id="30749", redfin_state_slug="NY", redfin_name_slug="New-York",
        parcel=ParcelSource(
            url="https://data.cityofnewyork.us/resource/64uk-42ks.geojson",
            fmt="geojson", crs=4326,
            coverage="NYC PLUTO via DCP / DOF",
            note="Existing canonical source",
        ),
        crime=CrimeSource(
            url="https://data.cityofnewyork.us/resource/qgea-i56i.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
        ),
        transit_gtfs_mdb="mdb-516",
        status="existing", tier="pc1_predicted",
        notes=["PC1 +0.169 effect confirmed; reference implementation"],
    ),
    "boston": CityConfig(
        slug="boston",
        display_name="Boston",
        state="MA", state_fips="25", county_fips=("025",),
        redfin_city_id="1826", redfin_state_slug="MA", redfin_name_slug="Boston",
        parcel=ParcelSource(
            url="https://data.boston.gov/dataset/property-assessment",
            fmt="csv", crs=4326,
            coverage="City of Boston (Suffolk County subset)",
            note="Existing canonical source",
        ),
        crime=CrimeSource(
            url="https://data.boston.gov/dataset/crime-incident-reports",
            fmt="csv_download", has_latlon=True, has_qol=True,
        ),
        transit_gtfs_mdb="mdb-64",
        status="existing", tier="null_predicted",
        notes=["Null on both axes confirmed"],
    ),

    # ------------------------------------------------------------------ new (dense urban tier)
    "dc": CityConfig(
        slug="dc",
        display_name="Washington DC",
        state="DC", state_fips="11", county_fips=("001",),
        redfin_city_id="12839", redfin_state_slug="DC", redfin_name_slug="Washington-DC",
        parcel=ParcelSource(
            url="https://opendata.dc.gov/datasets/common-ownership-lots/about",
            fmt="geojson", crs=4326,
            coverage="District of Columbia (no county subset needed)",
            note="ArcGIS Hub item 515f2c8ff3534302a34bf47f6902ac0d, layer 77; joins to OTR Integrated Tax System for assessment+price",
        ),
        crime=CrimeSource(
            url="https://opendata.dc.gov/api/download/v1/items/74d924ddc3374e3b977e6f002478cb9b/csv?layers=7",
            fmt="csv_download", has_latlon=True, has_qol=True,
            note="MPD Crime Incidents 2025; per-year datasets back to 2008 on the portal",
        ),
        transit_gtfs_mdb="mdb-237",
        status="ready", tier="pc1_predicted",
        notes=[
            "Single jurisdiction, no county filter needed",
            "Federal land lots (Mall, Smithsonian) have NULL assessed values; filter where assessed_value > 0",
        ],
    ),
    "philadelphia": CityConfig(
        slug="philadelphia",
        display_name="Philadelphia",
        state="PA", state_fips="42", county_fips=("101",),
        redfin_city_id="15502", redfin_state_slug="PA", redfin_name_slug="Philadelphia",
        parcel=ParcelSource(
            url="https://opendata-downloads.s3.amazonaws.com/opa_properties_public.csv",
            fmt="csv", crs=4326,
            coverage="City of Philadelphia (city = county, consolidated since 1854)",
            note="OPA properties + DOR parcel polygons on opendataphilly.org; nightly refresh; filter to current assessment year",
        ),
        crime=CrimeSource(
            url="https://phl.carto.com/api/v2/sql?q=SELECT+*+FROM+incidents_part1_part2",
            fmt="carto_sql", has_latlon=True, has_qol=True,
            note="PPD via Carto SQL API; 3.5M+ incidents; Part 1 + Part 2 both included; block-resolution lat/lon",
        ),
        transit_gtfs_mdb="mdb-93",
        status="ready", tier="pc1_predicted",
        notes=["City = county consolidated; one FIPS pair simplifies all downloads"],
    ),
    "chicago": CityConfig(
        slug="chicago",
        display_name="Chicago",
        state="IL", state_fips="17", county_fips=("031",),
        redfin_city_id="29470", redfin_state_slug="IL", redfin_name_slug="Chicago",
        parcel=ParcelSource(
            url="https://datacatalog.cookcountyil.gov/Property-Taxation/Assessor-Parcel-Universe/nj4t-kc8j",
            fmt="csv", crs=3435,
            coverage="ALL of Cook County (1.94M parcels) -- must filter to municipality='Chicago' or by ZIP set",
            note="Socrata SODA API at datacatalog.cookcountyil.gov; native CRS is IL State Plane East ftUS; reproject to 4326 before joins",
        ),
        crime=CrimeSource(
            url="https://data.cityofchicago.org/resource/6zsd-86xi.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="CPD Crimes 2001 to Present; 8.5M+ rows; daily updates; last 7 days redacted",
        ),
        transit_gtfs_mdb="mdb-389",
        status="ready", tier="pc1_predicted",
        notes=[
            "Parcel file is large (1.5+ GB unzipped); plan memory budget",
            "Must filter Cook County parcels to City of Chicago before joining listings",
        ],
    ),

    # ------------------------------------------------------------------ new (mid-tight tier)
    "seattle": CityConfig(
        slug="seattle",
        display_name="Seattle",
        state="WA", state_fips="53", county_fips=("033",),
        redfin_city_id="16163", redfin_state_slug="WA", redfin_name_slug="Seattle",
        parcel=ParcelSource(
            url="https://gis-kingcounty.opendata.arcgis.com/datasets/king-county-parcels/explore",
            fmt="geojson", crs=2926,
            coverage="ALL of King County (638k parcels); filter to City of Seattle polygon",
            note="King County GIS Open Data ArcGIS Hub; native CRS Washington State Plane North (ftUS); reproject required",
        ),
        crime=CrimeSource(
            url="https://data.seattle.gov/resource/tazs-3rd5.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="SPD Crime Data 2008 to Present; 1.5M+ rows; Offense Category field maps directly to violent/property/quality-of-life",
        ),
        transit_gtfs_mdb="mdb-101",
        status="ready_with_friction", tier="shen_predicted",
        notes=["CRS reprojection from EPSG:2926 (WA State Plane North, ftUS) to WGS84 needed before parcel-to-listing join"],
    ),
    "portland": CityConfig(
        slug="portland",
        display_name="Portland",
        state="OR", state_fips="41", county_fips=("051",),
        redfin_city_id="30772", redfin_state_slug="OR", redfin_name_slug="Portland",
        parcel=ParcelSource(
            url="https://gis-multco.opendata.arcgis.com/datasets/10bbca2553634733ba59634bffcfdeb2_0",
            fmt="geojson", crs=2913,
            coverage="ALL of Multnomah County; filter to City of Portland polygon",
            note="Multnomah County Taxlot Parcels; alternative source: Metro RLIS hub at rlisdiscovery.oregonmetro.gov; native CRS OR State Plane North",
        ),
        crime=CrimeSource(
            url="https://www.portland.gov/police/open-data/reported-crime-data",
            fmt="tableau", has_latlon=True, has_qol=True,
            note="PPB monthly reported crime via Tableau Public; download path through https://public.tableau.com/app/profile/portlandpolicebureau; requires Tableau-download wrapper; ~30 day publication lag",
        ),
        transit_gtfs_mdb="mdb-291",
        status="ready_with_friction", tier="shen_predicted",
        notes=[
            "Tableau-only crime data is the slow point; cache monthly CSVs",
            "CRS reprojection from EPSG:2913 (OR State Plane North, ftUS) to WGS84",
        ],
    ),
    "denver": CityConfig(
        slug="denver",
        display_name="Denver",
        state="CO", state_fips="08", county_fips=("031",),
        redfin_city_id="5155", redfin_state_slug="CO", redfin_name_slug="Denver",
        parcel=ParcelSource(
            url="https://opendata-geospatialdenver.hub.arcgis.com/datasets/geospatialDenver::parcels/about",
            fmt="geojson", crs=4326,
            coverage="City and County of Denver (consolidated)",
            note="ArcGIS Hub item 7c53bd0894134e80ae1e478c0789bf49; ~388 MB feature service",
        ),
        crime=CrimeSource(
            url="https://opendata-geospatialdenver.hub.arcgis.com/datasets/geospatialDenver::crime/about",
            fmt="arcgis_rest", has_latlon=True, has_qol=True,
            note="ArcGIS Hub item 1e080d3ce2ae4e2698745a0d02345d4a; rolling 5-year window + current YTD; NIBRS-based; M-F updates",
        ),
        transit_gtfs_mdb="mdb-291",  # RTD; verify before scrape
        status="ready", tier="shen_predicted",
        notes=[
            "Consolidated city-county; single FIPS pair",
            "Crime data is 5-year rolling window; archive snapshots if historical needed (not for our cross-section)",
        ],
    ),

    # ------------------------------------------------------------------ new (sprawling tier)
    "atlanta": CityConfig(
        slug="atlanta",
        display_name="Atlanta",
        state="GA", state_fips="13", county_fips=("121", "089"),  # Fulton, DeKalb
        redfin_city_id="30756", redfin_state_slug="GA", redfin_name_slug="Atlanta",
        parcel=ParcelSource(
            url="https://gisdata.fultoncountyga.gov/",
            fmt="shapefile", crs=2240,
            coverage="Fulton County + DeKalb County; merge then filter to City of Atlanta polygon",
            note="Two-county merge required: Fulton via gisdata.fultoncountyga.gov, DeKalb via dcgis.dekalbcountyga.gov; ~10% of Atlanta lies in DeKalb",
        ),
        crime=CrimeSource(
            url="https://opendata.atlantapd.org/",
            fmt="csv_download", has_latlon=True, has_qol=True,
            note="APD Mark43 RMS launched Dec 2020; pre-2021 vs post-2021 series break; cross-sectional analysis unaffected; CSV downloads only, no live SODA API",
        ),
        transit_gtfs_mdb="mdb-37",  # MARTA
        status="ready_with_friction", tier="null_predicted",
        notes=[
            "City of Atlanta spans Fulton (90%) + DeKalb (10%); merge two parcel layers",
            "APD Mark43 RMS migration in Dec 2020 breaks crime time series; document but cross-sectional analysis is fine",
        ],
    ),
    "phoenix": CityConfig(
        slug="phoenix",
        display_name="Phoenix",
        state="AZ", state_fips="04", county_fips=("013",),
        redfin_city_id="14240", redfin_state_slug="AZ", redfin_name_slug="Phoenix",
        parcel=ParcelSource(
            url="https://data-maricopa.opendata.arcgis.com/",
            fmt="shapefile", crs=2868,
            coverage="ALL of Maricopa County (very large, ~2000 sq mi); filter to City of Phoenix",
            note="ArcGIS Hub item c937f17330f64e64abd41976fc8bb17f; ~270 MB zipped, expands to 1-2 GB; alternative Assessor Book Series for sharded downloads",
        ),
        crime=CrimeSource(
            url="https://www.phoenixopendata.com/dataset/cc08aace-9ca9-467f-b6c1-f0879ab1a358/resource/0ce3411a-2fc6-4302-a33f-167f68608a20/download/crime-data_crime-data_crimestat.csv",
            fmt="csv_download", has_latlon=True, has_qol=False,
            note="Phoenix Open Data Portal crime CSV; PRIMARY-OFFENSE-ONLY (no secondary offenses); quality_of_life category undercounted; document in appendix",
        ),
        transit_gtfs_mdb="mdb-180",  # Valley Metro
        status="ready_with_caveat", tier="null_predicted",
        notes=[
            "Maricopa parcel file is huge; budget RAM for load step",
            "Crime data is primary-offense-only; qol category will be undercounted relative to Chicago/Seattle/Philly; either flag as limitation or impute by ratio",
            "City of Phoenix is ~25% of Maricopa County by area; filter carefully",
        ],
    ),
    "dallas": CityConfig(
        slug="dallas",
        display_name="Dallas",
        state="TX", state_fips="48", county_fips=("113",),
        redfin_city_id="30794", redfin_state_slug="TX", redfin_name_slug="Dallas",
        parcel=ParcelSource(
            url="https://www.dallascad.org/gisdataproducts.aspx",
            fmt="shapefile", crs=2276,
            coverage="ALL of Dallas County; filter to City of Dallas polygon",
            note="DCAD GIS Data Products; ZIP shapefiles per data product; alternative Dallas County Open Data Hub at dallas-county-open-data-hub-dallascountygis.hub.arcgis.com",
        ),
        crime=CrimeSource(
            url="https://www.dallasopendata.com/resource/qv6i-rri7.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="Dallas Police Incidents via Socrata API; incident-level lat/lon; substitute for Houston which has block-level-only data",
        ),
        transit_gtfs_mdb="mdb-152",  # DART
        status="ready", tier="null_predicted",
        notes=[
            "Substituted for Houston after agent verified Houston crime data is block-level only without incident-level lat/lon",
            "DCAD shapefile data products are stable academic source; reproject from EPSG:2276 (TX State Plane North Central, ftUS) to WGS84",
        ],
    ),
}


def get_city(slug: str) -> CityConfig:
    """Fetch a city config by slug; raises KeyError if not present."""
    return CITIES[slug]


def list_ready(include_existing: bool = True) -> list[CityConfig]:
    """Return cities in canonical scrape order: existing first, then new cities ranked by data quality."""
    order = [
        "sf", "nyc", "boston",        # existing
        "dc", "philadelphia", "chicago",  # cleanest open data
        "denver", "seattle",           # next-cleanest
        "atlanta", "portland",         # ready with friction
        "phoenix", "dallas",           # ready with caveats / new substitution
    ]
    cities = [CITIES[s] for s in order]
    if not include_existing:
        cities = [c for c in cities if c.status != "existing"]
    return cities


def list_new() -> list[CityConfig]:
    """The 9 cities being added in the JBES expansion."""
    return list_ready(include_existing=False)


def summary_table() -> str:
    """Produce a human-readable summary of all configured cities."""
    rows = ["| slug | name | state | tier | status |", "|---|---|---|---|---|"]
    for c in list_ready():
        rows.append(f"| {c.slug} | {c.display_name} | {c.state} | {c.tier} | {c.status} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="City endpoints config and verification")
    parser.add_argument("--check", action="store_true", help="Print summary table")
    parser.add_argument("--show", type=str, help="Show config for a specific city")
    args = parser.parse_args()
    if args.show:
        cfg = get_city(args.show)
        print(f"{cfg.display_name} ({cfg.slug})")
        print(f"  state/county FIPS: {cfg.state_fips} / {','.join(cfg.county_fips)}")
        print(f"  redfin URL: {cfg.redfin_url}")
        print(f"  parcel: {cfg.parcel.url} [{cfg.parcel.fmt}, EPSG:{cfg.parcel.crs}]")
        print(f"  crime: {cfg.crime.url} [{cfg.crime.fmt}, latlon={cfg.crime.has_latlon}, qol={cfg.crime.has_qol}]")
        print(f"  GTFS: {cfg.transit_gtfs_mdb}")
        print(f"  status: {cfg.status}, tier: {cfg.tier}")
        for n in cfg.notes:
            print(f"  - {n}")
    else:
        print(summary_table())

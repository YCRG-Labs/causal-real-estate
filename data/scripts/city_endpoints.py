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
            url="https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Property_and_Land_WebMercator/FeatureServer/40",
            fmt="arcgis_rest", crs=3857,
            coverage="District of Columbia (single jurisdiction)",
            note="ArcGIS Hub item 1f6708b1f3774306bef2fa81e612a725 Common Ownership Lots; ~190k features; no year built/beds/baths in this layer, pull those from listings",
        ),
        crime=CrimeSource(
            url="https://maps2.dcgis.dc.gov/dcgis/rest/services/FEEDS/MPD/FeatureServer/7",
            fmt="arcgis_rest", has_latlon=True, has_qol=False,
            note="MPD Crime Incidents 2025 layer 7; per-year layers back to 2008; OFFENSE field has 9 verified categories; DC publishes Part 1 offenses only so quality_of_life is structurally empty",
        ),
        transit_gtfs_mdb="mdb-237",
        status="ready", tier="pc1_predicted",
        notes=[
            "Single jurisdiction, no county filter needed",
            "Federal reservation lots have ASSESSMENT NULL; filter ASSESSMENT > 0",
            "Quality-of-life crime category is structurally empty in MPD data; impute zero and flag in appendix",
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
            url="https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.json",
            fmt="socrata_json", crs=4326,
            coverage="Filter cook_municipality_name='CITY OF CHICAGO' AND year='2026.0' yields 882,479 parcels (server-side)",
            note="Socrata SODA API; pre-computed lat/lon columns avoid reprojection; THREE-WAY JOIN on pin: nj4t-kc8j (geometry+census) + x54s-btds (structural: char_yrblt, char_bldg_sf, char_beds, char_fbath, char_hbath) + uzyt-m557 (assessed values in $thousands, multiply board_tot by 1000)",
        ),
        crime=CrimeSource(
            url="https://data.cityofchicago.org/resource/6zsd-86xi.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="CPD Crimes 2001 to Present; 8.5M+ rows; daily updates; last 7 days redacted; primary_type field has 31 categories",
        ),
        transit_gtfs_mdb="mdb-389",
        status="ready", tier="pc1_predicted",
        notes=[
            "Three Socrata tables join on pin: nj4t-kc8j + x54s-btds + uzyt-m557",
            "Server-side filter cook_municipality_name='CITY OF CHICAGO' avoids the 1.5GB blocker",
            "Assessment values in uzyt-m557 are in $thousands; multiply board_tot * 1000 in column_map",
        ],
    ),

    # ------------------------------------------------------------------ new (mid-tight tier)
    "seattle": CityConfig(
        slug="seattle",
        display_name="Seattle",
        state="WA", state_fips="53", county_fips=("033",),
        redfin_city_id="16163", redfin_state_slug="WA", redfin_name_slug="Seattle",
        parcel=ParcelSource(
            url="https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0",
            fmt="arcgis_rest", crs=2926,
            coverage="Filter CTYNAME='SEATTLE' yields 189,322 parcels (out of 638k King County)",
            note="ArcGIS Hub item 203f285e0db44975abc96940345f1454 PARCEL_ADDRESS_PUB_AREA_3069; has pre-computed LAT/LON columns in WGS84 so reprojection unnecessary; has APPRLNDVAL APPR_IMPR LOTSQFT but no year built/beds/baths in this layer (pull from listings or KC Assessor Residential Building separate file)",
        ),
        crime=CrimeSource(
            url="https://data.seattle.gov/resource/tazs-3rd5.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="SPD Crime Data 2008 to Present; 1.5M+ rows; use offense_sub_category not offense_category (latter has only 3 values); 31 verified subcategories map cleanly to violent/property/qol",
        ),
        transit_gtfs_mdb="mdb-101",
        status="ready", tier="shen_predicted",
        notes=["Use pre-computed LAT/LON columns to skip CRS reprojection from EPSG:2926"],
    ),
    "portland": CityConfig(
        slug="portland",
        display_name="Portland",
        state="OR", state_fips="41", county_fips=("051",),
        redfin_city_id="30772", redfin_state_slug="OR", redfin_name_slug="Portland",
        parcel=ParcelSource(
            url="https://services5.arcgis.com/x7DNZL1YqNQVNykA/arcgis/rest/services/Multnomah_County_Taxlot_Parcels/FeatureServer/0",
            fmt="arcgis_rest", crs=3857,
            coverage="ALL of Multnomah County (211k parcels); filter SITUSCITY='PORTLAND' yields ~140k",
            note="ArcGIS Hub item 13a535596a1f4fe0887b03d14d943229 Multnomah Taxlot Parcels; has ACTYEARBUILT, MAINAREA, MAIN_SQFT, UNITS, SALE_PRICE; ~5% of Portland is in Washington/Clackamas counties",
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
            url="https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_PROP_PARCELS_A/FeatureServer/245",
            fmt="arcgis_rest", crs=2877,
            coverage="City and County of Denver (consolidated, ~225k parcels)",
            note="ArcGIS Hub item 7c53bd0894134e80ae1e478c0789bf49; LAYER ID 245 not 0; CRS EPSG:2877 Colorado State Plane Central NAD83 ftUS; SCHEDNUM is parcel id; has RES_ORIG_YEAR_BUILT, RES_ABOVE_GRADE_AREA but no beds/baths",
        ),
        crime=CrimeSource(
            url="https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_CRIME_OFFENSES_P/FeatureServer/324",
            fmt="arcgis_rest", has_latlon=True, has_qol=True,
            note="ArcGIS Hub item 1e080d3ce2ae4e2698745a0d02345d4a; LAYER ID 324; rolling 5-year window + current YTD; filter IS_CRIME=1 to exclude IS_TRAFFIC=1 rows",
        ),
        transit_gtfs_mdb="mdb-291",
        status="ready", tier="shen_predicted",
        notes=[
            "Consolidated city-county; single FIPS pair",
            "Crime data is 5-year rolling window; archive snapshots if historical needed (not for our cross-section)",
            "Native CRS is EPSG:2877 not 4326; reproject parcels before joins",
        ],
    ),

    # ------------------------------------------------------------------ new (sprawling tier)
    "atlanta": CityConfig(
        slug="atlanta",
        display_name="Atlanta",
        state="GA", state_fips="13", county_fips=("121", "089"),
        redfin_city_id="30756", redfin_state_slug="GA", redfin_name_slug="Atlanta",
        parcel=ParcelSource(
            url="https://gis.atlantaga.gov/dpcd/rest/services/AdministrativeArea/TaxParcel/MapServer/0",
            fmt="arcgis_rest", crs=6447,
            coverage="City of Atlanta pre-merged across Fulton + DeKalb (~130k parcels)",
            note="ArcGIS Hub item ee82525ee33b49778055622c3a3cf534 Tax Parcels 2025 by COA_DCP_GIS; PRE-MERGED so no two-county join needed; no year built or beds/baths in city layer; DeKalb sister layer 7aa40e4967744cb0abadd6cb0dc23c97 has ACTYEARBUILT MAINAREA for DeKalb-side enrichment",
        ),
        crime=CrimeSource(
            url="https://services3.arcgis.com/Et5Qfajgiyosiw4d/arcgis/rest/services/OpenDataWebsite_Crime_view/FeatureServer/0",
            fmt="arcgis_rest", has_latlon=True, has_qol=True,
            note="ArcGIS Hub item 774475034b694ce68b6d2e887aa96544 live feature service (NOT csv-only as previously noted); NIBRS_Bucket field has 28 verified categories; filter NIBRS_Bucket IS NOT NULL; post-Mark43 only 2021+",
        ),
        transit_gtfs_mdb="mdb-37",
        status="ready", tier="null_predicted",
        notes=[
            "Pre-merged city layer avoids the Fulton+DeKalb merge blocker",
            "APD has live ArcGIS feature service via Mark43; CSV-only note in prior config is stale",
            "Native CRS is EPSG:6447 NAD83(2011) Georgia West ftUS; reproject before joins",
        ],
    ),
    "phoenix": CityConfig(
        slug="phoenix",
        display_name="Phoenix",
        state="AZ", state_fips="04", county_fips=("013",),
        redfin_city_id="14240", redfin_state_slug="AZ", redfin_name_slug="Phoenix",
        parcel=ParcelSource(
            url="https://www.arcgis.com/sharing/rest/content/items/c937f17330f64e64abd41976fc8bb17f/data",
            fmt="shapefile_zip", crs=2868,
            coverage="ALL of Maricopa County (~1.7M parcels); filter SITUS_CITY='PHOENIX' yields ~400k",
            note="ArcGIS Hub item c937f17330f64e64abd41976fc8bb17f; download via /data 302-redirect to S3; 272MB zip / 1.7GB unzipped; use pyogrio.read_dataframe with where='SITUS_CITY=PHOENIX' to filter at read-time; has YEAR_BUILT, BEDROOMS, BATHS, LIVING_AREA",
        ),
        crime=CrimeSource(
            url="https://www.phoenixopendata.com/dataset/cc08aace-9ca9-467f-b6c1-f0879ab1a358/resource/0ce3411a-2fc6-4302-a33f-167f68608a20/download/crime-data_crime-data_crimestat.csv",
            fmt="csv_download", has_latlon=False, has_qol=False,
            note="Phoenix Open Data CSV; CRITICAL: no incident-level lat/lon (100 BLOCK ADDR + ZIP only); primary-offense-only; quality_of_life only DRUG OFFENSE; must Census-geocode block addresses to produce usable point dataset; 500m KDE smooths block-resolution",
        ),
        transit_gtfs_mdb="mdb-180",
        status="ready_with_caveat", tier="null_predicted",
        notes=[
            "Maricopa parcel file is 1.7GB unzipped; use pyogrio with where='SITUS_CITY=PHOENIX' to filter at read-time",
            "Crime data has NO incident lat/lon (block address only); geocode 100 BLOCK ADDR + ZIP via Census batch",
            "Crime data is primary-offense-only; quality_of_life undercounted; flag in appendix",
            "City spans 550 sq mi; split amenity Overpass query into 4-6 tiles or use city polygon clip",
        ],
    ),
    "dallas": CityConfig(
        slug="dallas",
        display_name="Dallas",
        state="TX", state_fips="48", county_fips=("113",),
        redfin_city_id="30794", redfin_state_slug="TX", redfin_name_slug="Dallas",
        parcel=ParcelSource(
            url="https://www.dallascad.org/ViewPDFs.aspx?type=3&id=%5C%5CDCAD.ORG%5CWEB%5CWEBDATA%5CWEBFORMS%5CGIS%20PRODUCTS%5CPARCEL_GEOM.zip",
            fmt="shapefile_zip", crs=2276,
            coverage="ALL of Dallas County (~696k parcels); join to DCAD2026_CURRENT.zip on Acct then spatial-filter to City of Dallas polygon",
            note="DCAD download needs Referer + cookie jar from gisdataproducts.aspx first; URL-encoded UNC backslashes; PARCEL_GEOM has only 3 fields (Acct/Type/RecAcs) so must join DCAD2026_CURRENT.zip Account_Info.txt + Res_Detail.txt pipe-delimited files on Acct for YR_BUILT BEDROOMS FULL_BATHS LIVING_AREA_SF MKT_VAL",
        ),
        crime=CrimeSource(
            url="https://www.dallasopendata.com/resource/qv6i-rri7.json",
            fmt="socrata_json", has_latlon=True, has_qol=True,
            note="Dallas Police Incidents via Socrata; lat/lon nested under geocoded_column.latitude/longitude; filter nibrs_crimeagainst IS NOT NULL to drop pre-NIBRS rows; 2014-06 to current",
        ),
        transit_gtfs_mdb="mdb-152",
        status="ready_with_friction", tier="null_predicted",
        notes=[
            "Substituted for Houston after agent verified Houston crime is block-level only",
            "DCAD parcel download requires session cookie + Referer + URL-encoded UNC paths; PARCEL_GEOM is geometry-only with 3 fields, must join DCAD2026_CURRENT pipe-delimited TXT files on Acct",
            "Reproject parcels from EPSG:2276 (TX State Plane North Central ftUS) to WGS84",
            "Socrata geocoded_column is nested JSON object; extract latitude/longitude with .apply",
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

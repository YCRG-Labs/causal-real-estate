import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "raw"
CLEANED_DIR = ROOT_DIR / "cleaned"
PROCESSED_DIR = ROOT_DIR / "processed"

CITIES = {
    "boston": {
        "raw_file": RAW_DIR / "boston" / "boston_parcels_2024_raw.csv",
        "assessment_file": RAW_DIR / "boston" / "boston_assessment.csv",
        "source_crs": "EPSG:4326",
        "local_crs": "EPSG:26986",
        "parcel_id_col": "MAP_PAR_ID",
        "geometry_col": "shape_wkt",
        "column_map": {
            "MAP_PAR_ID": "parcel_id",
            "LOC_ID": "loc_id",
            "POLY_TYPE": "poly_type",
            "Shape_Length": "shape_length_deg",
            "Shape_Area": "shape_area_deg",
        },
        "drop_cols": [
            "OID_", "MAP_NO", "SOURCE", "PLAN_ID", "BND_CHK",
            "NO_MATCH", "TOWN_ID", "last_edited_date",
            "created_user", "created_date", "last_edited_user",
        ],
        "valid_poly_types": ["FEE"],
        "invalid_parcel_ids": ["MASSGIS", "ISLAND"],
        "assessment_column_map": {
            "PID": "parcel_id",
            "BED_RMS": "bedrooms",
            "FULL_BTH": "full_baths",
            "HLF_BTH": "half_baths",
            "TT_RMS": "total_rooms",
            "LIVING_AREA": "living_area_sqft",
            "GROSS_AREA": "bldg_area_sqft",
            "LAND_SF": "lot_area_sqft",
            "YR_BUILT": "year_built",
            "LAND_VALUE": "assessed_land",
            "BLDG_VALUE": "assessed_bldg",
            "TOTAL_VALUE": "assessed_total",
            "OVERALL_COND": "condition",
            "LU_DESC": "land_use_desc",
            "RES_UNITS": "units_res",
        },
    },
    "nyc": {
        "raw_file": RAW_DIR / "nyc" / "pluto.csv",
        "sales_file": RAW_DIR / "nyc" / "nyc_sales.csv",
        "source_crs": "EPSG:4326",
        "local_crs": "EPSG:2263",
        "parcel_id_col": "BBL",
        "column_map": {
            "BBL": "parcel_id",
            "landuse": "land_use",
            "bldgclass": "bldg_class",
            "lotarea": "lot_area_sqft",
            "bldgarea": "bldg_area_sqft",
            "resarea": "res_area_sqft",
            "unitsres": "units_res",
            "numfloors": "num_floors",
            "numbldgs": "num_bldgs",
            "yearbuilt": "year_built",
            "assessland": "assessed_land",
            "assesstot": "assessed_total",
            "latitude": "latitude",
            "longitude": "longitude",
        },
        "residential_land_use": [1, 2, 3],
        "boroughs": ["MN", "BK"],
    },
    "sf": {
        "raw_file": RAW_DIR / "sf" / "sf_parcels.csv",
        "source_crs": "EPSG:4326",
        "local_crs": "EPSG:7131",
        "parcel_id_col": "blklot",
        "geometry_col": None,
        "column_map": {
            "blklot": "parcel_id",
            "block_num": "block",
            "lot_num": "lot",
        },
        "assessor_file": RAW_DIR / "sf" / "sf_assessor_rolls.csv",
        "inferred_prices_file": RAW_DIR / "sf" / "sf_inferred_prices.csv",
        "assessor_join_col": "parcel_number",
        "assessor_column_map": {
            "parcel_number": "parcel_id",
            "year_property_built": "year_built",
            "number_of_bedrooms": "bedrooms",
            "number_of_bathrooms": "bathrooms",
            "number_of_rooms": "rooms",
            "number_of_stories": "stories",
            "number_of_units": "units",
            "property_area": "property_area_sqft",
            "lot_area": "lot_area_sqft",
            "assessed_land_value": "assessed_land",
            "assessed_improvement_value": "assessed_improvement",
            "use_code": "use_code",
            "use_definition": "use_definition",
            "analysis_neighborhood": "neighborhood",
            "current_sales_date": "last_sale_date",
        },
    },
}

MIN_PARCEL_AREA_SQM = 10
MIN_SALE_PRICE = 50_000
MAX_SALE_PRICE = 20_000_000

CENSUS_YEAR = 2022
CENSUS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B15003_022E": "bachelors_degree_count",
    "B15003_001E": "education_total",
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",
    "B23025_002E": "labor_force",
    "B23025_001E": "labor_force_total",
    "B03002_001E": "race_total",
    "B03002_003E": "race_white",
    "B03002_004E": "race_black",
    "B03002_006E": "race_asian",
    "B03002_012E": "race_hispanic",
    "B01001_001E": "age_total",
    "B01001_003E": "age_5_9",
    "B01001_007E": "age_18_19",
    "B01001_011E": "age_25_29",
    "B01001_015E": "age_40_44",
    "B01001_020E": "age_60_61",
    "B01001_025E": "age_85_plus",
}

CENSUS_DERIVED = {
    "pct_white": ("race_white", "race_total"),
    "pct_black": ("race_black", "race_total"),
    "pct_asian": ("race_asian", "race_total"),
    "pct_hispanic": ("race_hispanic", "race_total"),
    "pct_bachelors": ("bachelors_degree_count", "education_total"),
    "labor_force_participation": ("labor_force", "labor_force_total"),
}

CENSUS_DROP_RAW = [
    "bachelors_degree_count", "education_total",
    "labor_force", "labor_force_total",
    "race_total", "race_white", "race_black", "race_asian", "race_hispanic",
    "age_total", "age_5_9", "age_18_19", "age_25_29", "age_40_44", "age_60_61", "age_85_plus",
]

STATE_FIPS = {
    "boston": "25",
    "nyc": "36",
    "sf": "06",
}

COUNTY_FIPS = {
    "boston": ["025"],
    "nyc": ["061", "047"],
    "sf": ["075"],
}

CRIME_KDE_BANDWIDTH_M = 500

AMENITY_RADIUS_M = 500
OSM_AMENITY_TAGS = {
    "food_dining": {"amenity": ["restaurant", "cafe", "bar", "fast_food"]},
    "retail": {"shop": ["supermarket", "convenience", "mall", "clothes"]},
    "services": {"amenity": ["bank", "post_office", "clinic", "hospital", "pharmacy"]},
    "recreation": {"leisure": ["park", "fitness_centre", "playground", "sports_centre"]},
    "transportation": {"public_transport": ["stop_position", "station"]},
    "education": {"amenity": ["school", "library", "university", "college"]},
}

EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

EMBEDDING_ALTERNATIVES = {
    "all-MiniLM-L6-v2": 384,
}

CRIME_TEMPORAL_WINDOW_DAYS = 365

MICRO_GEO_COLS = [
    "dist_park_m", "dist_transit_m", "dist_school_m",
    "dist_restaurant_m", "dist_retail_m", "dist_medical_m",
]

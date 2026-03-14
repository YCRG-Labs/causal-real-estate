import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "raw"
CLEANED_DIR = ROOT_DIR / "cleaned"
PROCESSED_DIR = ROOT_DIR / "processed"

CITIES = {
    "boston": {
        "raw_file": RAW_DIR / "boston" / "boston_parcels_2024_raw.csv",
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
    },
    "nyc": {
        "raw_file": RAW_DIR / "nyc" / "mappluto.gpkg",
        "source_crs": "EPSG:2263",
        "local_crs": "EPSG:2263",
        "parcel_id_col": "BBL",
        "geometry_col": None,
        "column_map": {
            "BBL": "parcel_id",
            "LandUse": "land_use",
            "BldgClass": "bldg_class",
            "LotArea": "lot_area_sqft",
            "BldgArea": "bldg_area_sqft",
            "ResArea": "res_area_sqft",
            "UnitsRes": "units_res",
            "NumFloors": "num_floors",
            "NumBldgs": "num_bldgs",
            "YearBuilt": "year_built",
            "AssessLand": "assessed_land",
            "AssessTot": "assessed_total",
            "Latitude": "latitude",
            "Longitude": "longitude",
        },
        "residential_land_use": ["01", "02", "03"],
    },
    "sf": {
        "raw_file": RAW_DIR / "sf" / "sf_parcels.gpkg",
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
    "B23025_003E": "labor_force",
    "B23025_002E": "labor_force_total",
}

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
    "transportation": {"highway": ["bus_stop"], "railway": ["station", "subway_entrance"]},
    "education": {"amenity": ["school", "library", "university", "college"]},
}

EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

"""Centralised dicts for the legacy download/attach scripts.

Sourced from city_endpoints.py and verified per the per-city research agent
run on 2026-05-30. Drop into download_census.py / download_amenities.py /
download_crime.py / attach_*.py via `from pipeline_dicts import ...`.

Every entry has been URL-verified or curl-probed.
"""

from __future__ import annotations

STATE_FIPS = {
    "boston": "25",
    "nyc": "36",
    "sf": "06",
    "dc": "11",
    "philadelphia": "42",
    "chicago": "17",
    "seattle": "53",
    "denver": "08",
    "atlanta": "13",
    "portland": "41",
    "phoenix": "04",
    "dallas": "48",
}

COUNTY_FIPS = {
    "boston": ["025"],
    "nyc": ["061", "047", "005", "081", "085"],
    "sf": ["075"],
    "dc": ["001"],
    "philadelphia": ["101"],
    "chicago": ["031"],
    "seattle": ["033"],
    "denver": ["031"],
    "atlanta": ["121", "089"],
    "portland": ["051"],
    "phoenix": ["013"],
    "dallas": ["113"],
}

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "boston":       (42.2279, -71.1912, 42.3970, -70.9229),
    "nyc":          (40.4961, -74.2557, 40.9176, -73.7004),
    "sf":           (37.7008, -122.5160, 37.8338, -122.3554),
    "dc":           (38.7916, -77.1198, 38.9960, -76.9094),
    "philadelphia": (39.8670, -75.2803, 40.1380, -74.9558),
    "chicago":      (41.6445, -87.9401, 42.0231, -87.5241),
    "seattle":      (47.4810, -122.4597, 47.7342, -122.2244),
    "denver":       (39.6143, -105.1099, 39.9142, -104.5997),
    "atlanta":      (33.6479, -84.5509, 33.8868, -84.2896),
    "portland":     (45.4325, -122.8367, 45.6529, -122.4720),
    "phoenix":      (33.2905, -112.3240, 33.9184, -111.9255),
    "dallas":       (32.6132, -97.0005, 33.0239, -96.4636),
}


CRIME_CROSSWALK = {
    "violent": {
        "dc":           ["HOMICIDE", "SEX ABUSE", "ASSAULT W/DANGEROUS WEAPON", "ROBBERY"],
        "philadelphia": ["Homicide - Criminal", "Homicide - Justifiable", "Rape", "Aggravated Assault Firearm", "Aggravated Assault No Firearm", "Robbery Firearm", "Robbery No Firearm", "Other Assaults"],
        "chicago":      ["HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE", "ROBBERY", "ASSAULT", "BATTERY", "INTIMIDATION", "KIDNAPPING", "STALKING", "HUMAN TRAFFICKING"],
        "seattle":      ["HOMICIDE", "JUSTIFIABLE HOMICIDE", "RAPE", "ROBBERY", "AGGRAVATED ASSAULT", "ASSAULT OFFENSES", "KIDNAPPING/ABDUCTION", "HUMAN TRAFFICKING", "SEX OFFENSES"],
        "denver":       ["aggravated-assault", "murder", "robbery", "other-crimes-against-persons"],
        "atlanta":      ["Homicide", "Rape", "Robbery", "Aggravated Assault", "Assault Offenses", "Kidnapping/Abduction", "Human Trafficking", "Sex Offenses"],
        "portland":     ["Homicide Offenses", "Sex Offenses", "Assault Offenses", "Robbery", "Kidnapping/Abduction", "Human Trafficking"],
        "phoenix":      ["HOMICIDE", "RAPE", "ROBBERY", "AGGRAVATED ASSAULT"],
        "dallas":       ["HOMICIDE OFFENSES", "ASSAULT OFFENSES", "ROBBERY", "KIDNAPPING/ ABDUCTION", "HUMAN TRAFFICKING", "EXTORTION/ BLACKMAIL"],
    },
    "property": {
        "dc":           ["BURGLARY", "THEFT F/AUTO", "MOTOR VEHICLE THEFT", "THEFT/OTHER", "ARSON"],
        "philadelphia": ["Thefts", "Theft from Vehicle", "Motor Vehicle Theft", "Burglary Residential", "Burglary Non-Residential", "Receiving Stolen Property", "Arson", "Fraud", "Embezzlement"],
        "chicago":      ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE", "ARSON", "DECEPTIVE PRACTICE", "CRIMINAL TRESPASS", "OTHER OFFENSE"],
        "seattle":      ["LARCENY-THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "PROPERTY OFFENSES (INCLUDES STOLEN, DESTRUCTION)", "EXTORTION/FRAUD/FORGERY/BRIBERY (INCLUDES BAD CHECKS)", "ARSON"],
        "denver":       ["auto-theft", "burglary", "larceny", "theft-from-motor-vehicle", "arson", "white-collar-crime"],
        "atlanta":      ["Burglary", "Auto Theft", "Theft From Auto", "Shoplifting", "All Other Larceny", "Damage to Property", "Arson", "Stolen Property Offenses", "Fraud Offenses", "Counterfeiting/Forgery", "Embezzelment", "Extortion/Blackmail"],
        "portland":     ["Larceny Offenses", "Burglary", "Motor Vehicle Theft", "Arson", "Destruction/Damage/Vandalism Of Property", "Counterfeiting/Forgery", "Fraud Offenses", "Embezzlement", "Stolen Property Offenses", "Bribery", "Extortion/Blackmail"],
        "phoenix":      ["BURGLARY", "LARCENY-THEFT", "MOTOR VEHICLE THEFT", "ARSON"],
        "dallas":       ["LARCENY/ THEFT OFFENSES", "MOTOR VEHICLE THEFT", "BURGLARY/ BREAKING & ENTERING", "ARSON", "DESTRUCTION/ DAMAGE/ VANDALISM OF PROPERTY", "FRAUD OFFENSES", "COUNTERFEITING / FORGERY", "EMBEZZELMENT", "STOLEN PROPERTY OFFENSES", "BRIBERY"],
    },
    "quality_of_life": {
        "dc":           [],
        "philadelphia": ["Vandalism/Criminal Mischief", "Narcotic / Drug Law Violations", "Disorderly Conduct", "DRIVING UNDER THE INFLUENCE", "Public Drunkenness", "Liquor Law Violations", "Vagrancy/Loitering", "Weapon Violations", "Prostitution and Commercialized Vice", "Gambling Violations"],
        "chicago":      ["NARCOTICS", "OTHER NARCOTIC VIOLATION", "LIQUOR LAW VIOLATION", "PUBLIC PEACE VIOLATION", "INTERFERENCE WITH PUBLIC OFFICER", "WEAPONS VIOLATION", "CONCEALED CARRY LICENSE VIOLATION", "PROSTITUTION", "OBSCENITY", "PUBLIC INDECENCY", "GAMBLING", "OFFENSE INVOLVING CHILDREN"],
        "seattle":      ["NARCOTIC VIOLATIONS (INCLUDES DRUG EQUIP.)", "DRUG/ALCOHOL VIOLATIONS", "DUI", "WEAPON LAW VIOLATION", "TRESPASS", "DISORDERLY CONDUCT & VAGRANCY VIOLATIONS", "LIQUOR LAW VIOLATIONS & DRUNKENNESS", "PROSTITUTION OFFENSES", "VIOLATION OF NO CONTACT ORDER", "NON-VIOLENT FAMILY OFFENSES", "PORNOGRAPHY", "GAMBLING OFFENSES", "ANIMAL CRUELTY"],
        "denver":       ["drug-alcohol", "public-disorder", "all-other-crimes"],
        "atlanta":      ["Drug/Narcotic Offenses", "Weapon Law Violations", "All Other Offenses", "Pornography/Obscene Material", "Prostitution Offenses", "Gambling Offenses", "Bribery", "Animal Cruelty"],
        "portland":     ["Drug/Narcotic Offenses", "Weapon Law Violations", "Disorderly Conduct", "Liquor Law Violations", "Drunkenness", "Curfew/Loitering/Vagrancy Violations", "Driving Under the Influence", "Prostitution Offenses", "Pornography/Obscene Material", "Gambling Offenses", "Trespass of Real Property", "Animal Cruelty", "All Other Offenses"],
        "phoenix":      ["DRUG OFFENSE"],
        "dallas":       ["DRUG/ NARCOTIC VIOLATIONS", "WEAPON LAW VIOLATIONS", "TRESPASS OF REAL PROPERTY", "DISORDERLY CONDUCT", "PUBLIC INTOXICATION", "LIQUOR LAW VIOLATIONS", "DRIVING UNDER THE INFLUENCE", "FAMILY OFFENSES, NONVIOLENT", "GAMBLING OFFENSES", "PORNOGRAPHY/ OBSCENE MATERIAL", "MISCELLANEOUS", "ALL OTHER OFFENSES", "ANIMAL OFFENSES", "PEEPING TOM", "CURFEW/ LOITERING/ VAGRANCY VIOLATIONS"],
    },
}


CRIME_OFFENSE_FIELD = {
    "dc":           "OFFENSE",
    "philadelphia": "text_general_code",
    "chicago":      "primary_type",
    "seattle":      "offense_sub_category",
    "denver":       "OFFENSE_CATEGORY_ID",
    "atlanta":      "NIBRS_Bucket",
    "portland":     "OffenseCategory",
    "phoenix":      "UCR CRIME CATEGORY",
    "dallas":       "nibrs_crime_category",
}


CRIME_LATLON_FIELDS = {
    "dc":           ("LATITUDE", "LONGITUDE"),
    "philadelphia": ("point_y", "point_x"),
    "chicago":      ("latitude", "longitude"),
    "seattle":      ("latitude", "longitude"),
    "denver":       ("GEO_LAT", "GEO_LON"),
    "atlanta":      ("Latitude", "Longitude"),
    "portland":     ("OpenDataLat", "OpenDataLon"),
    "phoenix":      (None, None),
    "dallas":       ("geocoded_column.latitude", "geocoded_column.longitude"),
}


def all_cities() -> list[str]:
    return list(STATE_FIPS.keys())


def new_cities() -> list[str]:
    return [c for c in STATE_FIPS if c not in ("boston", "nyc", "sf")]


if __name__ == "__main__":
    print(f"{len(all_cities())} cities configured: {all_cities()}")
    print(f"  {len(new_cities())} new: {new_cities()}")
    print(f"\nCITY_BBOXES coverage:")
    for c, (s, w, n, e) in CITY_BBOXES.items():
        print(f"  {c}: ({s:.4f}, {w:.4f}, {n:.4f}, {e:.4f})  span={n-s:.3f}deg x {e-w:.3f}deg")

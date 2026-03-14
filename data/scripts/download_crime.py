import sys
import csv
import requests
import pandas as pd
from pathlib import Path
from config import RAW_DIR, CITIES

CRIME_DIR = RAW_DIR / "crime"
PAGE_SIZE = 50000

BOSTON_RESOURCE_IDS = [
    "b973d8cb-eeb2-4e7e-99da-c92938efc9c0",
    "313e56df-6d77-49d2-9c49-ee411f10cf58",
    "f4495ee9-c42c-4019-82c1-d067f07e45d2",
    "be047094-85fe-4104-a480-4fa3d03f9623",
]

CRIME_CROSSWALK = {
    "violent": {
        "boston": [
            "HOMICIDE", "ASSAULT", "ROBBERY", "AGGRAVATED ASSAULT",
            "ASSAULT - AGGRAVATED", "MANSLAUGHTER",
        ],
        "nyc": ["FELONY ASSAULT", "ROBBERY", "MURDER & NON-NEGL. MANSLAUGHTER", "RAPE"],
        "sf": ["Homicide", "Robbery", "Assault"],
    },
    "property": {
        "boston": [
            "BURGLARY", "LARCENY", "AUTO THEFT", "LARCENY FROM MOTOR VEHICLE",
            "RESIDENTIAL BURGLARY", "COMMERCIAL BURGLARY",
        ],
        "nyc": ["BURGLARY", "GRAND LARCENY", "GRAND LARCENY OF MOTOR VEHICLE", "PETIT LARCENY"],
        "sf": ["Burglary", "Larceny Theft", "Motor Vehicle Theft"],
    },
    "quality_of_life": {
        "boston": ["VANDALISM", "DISORDERLY CONDUCT", "DRUG VIOLATION", "LIQUOR VIOLATION"],
        "nyc": ["CRIMINAL MISCHIEF & RELATED OF", "OFFENSES AGAINST PUBLIC ADMINI"],
        "sf": ["Vandalism", "Drug Offense", "Disorderly Conduct"],
    },
}


def download_boston():
    print("boston:")
    frames = []
    for rid in BOSTON_RESOURCE_IDS:
        offset = 0
        while True:
            url = (
                f"https://data.boston.gov/api/3/action/datastore_search"
                f"?resource_id={rid}&limit={PAGE_SIZE}&offset={offset}"
            )
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            records = data["result"]["records"]
            if not records:
                break
            frames.append(pd.DataFrame(records))
            offset += PAGE_SIZE
            print(f"\r  {rid[:8]}... {offset} rows", end="", flush=True)
        print()

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "INCIDENT_NUMBER": "incident_id",
        "OFFENSE_DESCRIPTION": "offense_desc",
        "OCCURRED_ON_DATE": "date",
        "Lat": "latitude",
        "Long": "longitude",
    })
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df["city"] = "boston"

    out = CRIME_DIR / "boston_crime.csv"
    df[["incident_id", "offense_desc", "date", "latitude", "longitude", "city"]].to_csv(out, index=False)
    print(f"  Saved {len(df)} incidents → {out}")


def download_soda_crime(endpoint, dest, city, id_col, desc_col, date_col, lat_col, lon_col):
    print(f"{city}:")
    offset = 0
    total = 0
    header_written = False

    with open(dest, "w", newline="") as f:
        writer = None
        while True:
            resp = requests.get(
                endpoint,
                params={"$limit": PAGE_SIZE, "$offset": offset, "$order": ":id"},
                timeout=120,
            )
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            reader = csv.reader(lines)
            header = next(reader)
            rows = list(reader)
            if not rows:
                break

            temp = pd.DataFrame(rows, columns=header)
            temp = temp.rename(columns={
                id_col: "incident_id",
                desc_col: "offense_desc",
                date_col: "date",
                lat_col: "latitude",
                lon_col: "longitude",
            })
            temp["city"] = city
            temp["latitude"] = pd.to_numeric(temp["latitude"], errors="coerce")
            temp["longitude"] = pd.to_numeric(temp["longitude"], errors="coerce")
            temp = temp.dropna(subset=["latitude", "longitude"])
            temp = temp[(temp["latitude"] != 0) & (temp["longitude"] != 0)]

            out_df = temp[["incident_id", "offense_desc", "date", "latitude", "longitude", "city"]]

            if not header_written:
                out_df.to_csv(f, index=False)
                header_written = True
            else:
                out_df.to_csv(f, index=False, header=False)

            total += len(out_df)
            print(f"\r  {total:,} rows", end="", flush=True)

            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    print(f"\n  Saved {total:,} incidents → {dest}")


def download_nyc():
    download_soda_crime(
        "https://data.cityofnewyork.us/resource/qgea-i56i.csv",
        CRIME_DIR / "nyc_crime_historic.csv",
        "nyc (historic)",
        id_col="cmplnt_num",
        desc_col="ofns_desc",
        date_col="cmplnt_fr_dt",
        lat_col="latitude",
        lon_col="longitude",
    )
    download_soda_crime(
        "https://data.cityofnewyork.us/resource/5uac-w243.csv",
        CRIME_DIR / "nyc_crime_ytd.csv",
        "nyc (ytd)",
        id_col="cmplnt_num",
        desc_col="ofns_desc",
        date_col="cmplnt_fr_dt",
        lat_col="latitude",
        lon_col="longitude",
    )

    import pandas as pd
    hist = pd.read_csv(CRIME_DIR / "nyc_crime_historic.csv")
    ytd = pd.read_csv(CRIME_DIR / "nyc_crime_ytd.csv")
    combined = pd.concat([hist, ytd], ignore_index=True)
    combined = combined.drop_duplicates(subset=["incident_id"])
    combined.to_csv(CRIME_DIR / "nyc_crime.csv", index=False)
    print(f"  Combined: {len(combined)} total incidents")


def download_sf():
    download_soda_crime(
        "https://data.sfgov.org/resource/wg3w-h783.csv",
        CRIME_DIR / "sf_crime.csv",
        "sf",
        id_col="incident_number",
        desc_col="incident_category",
        date_col="incident_date",
        lat_col="latitude",
        lon_col="longitude",
    )


DOWNLOADERS = {
    "boston": download_boston,
    "nyc": download_nyc,
    "sf": download_sf,
}


def main():
    CRIME_DIR.mkdir(parents=True, exist_ok=True)
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        DOWNLOADERS[city]()
    print("\nCrime download complete.")


if __name__ == "__main__":
    main()

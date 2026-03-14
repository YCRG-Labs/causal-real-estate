import sys
import csv
import requests
from pathlib import Path
from config import RAW_DIR

SF_DIR = RAW_DIR / "sf"

PARCELS_ENDPOINT = "https://data.sfgov.org/resource/acdm-wktn.csv"
ASSESSOR_ENDPOINT = "https://data.sfgov.org/resource/wv5m-vpq2.csv"

PAGE_SIZE = 50000


def download_soda_csv(endpoint, dest):
    print(f"Downloading {dest.name} via SODA API...")
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

            if not header_written:
                writer = csv.writer(f)
                writer.writerow(header)
                header_written = True

            rows = list(reader)
            if not rows:
                break

            writer.writerows(rows)
            total += len(rows)
            print(f"\r  {total:,} rows", end="", flush=True)

            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    print(f"\n  Saved {total:,} rows to {dest}")


def main():
    SF_DIR.mkdir(parents=True, exist_ok=True)
    download_soda_csv(PARCELS_ENDPOINT, SF_DIR / "sf_parcels.csv")
    download_soda_csv(ASSESSOR_ENDPOINT, SF_DIR / "sf_assessor_rolls.csv")
    print("\nSF downloads complete.")


if __name__ == "__main__":
    main()

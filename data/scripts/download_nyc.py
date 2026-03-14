import sys
import zipfile
import requests
from pathlib import Path
from config import RAW_DIR

NYC_DIR = RAW_DIR / "nyc"

PLUTO_URL = "https://data.cityofnewyork.us/api/views/64uk-42ks/rows.csv?accessType=DOWNLOAD"
SALES_URL = "https://data.cityofnewyork.us/api/views/w2pb-icbu/rows.csv?accessType=DOWNLOAD"


def download_file(url, dest, chunk_size=8192):
    print(f"Downloading {dest.name}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)

    print(f"\n  Saved to {dest}")


def download_pluto():
    NYC_DIR.mkdir(parents=True, exist_ok=True)
    download_file(PLUTO_URL, NYC_DIR / "pluto.csv")


def download_sales():
    NYC_DIR.mkdir(parents=True, exist_ok=True)
    download_file(SALES_URL, NYC_DIR / "nyc_sales.csv")


def main():
    download_pluto()
    download_sales()
    print("\nNYC downloads complete.")


if __name__ == "__main__":
    main()

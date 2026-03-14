import sys
import requests
from pathlib import Path
from config import RAW_DIR

BOSTON_DIR = RAW_DIR / "boston"
ASSESSMENT_URL = "https://data.boston.gov/dataset/e02c44d2-3c64-459c-8fe2-e1ce5f38a035/resource/ee73430d-96c0-423e-ad21-c4cfb54c8961/download/fy2026-property-assessment-data_12_23_2025.csv"


def download_file(url, dest, chunk_size=8192):
    print(f"Downloading {dest.name}...")
    resp = requests.get(url, stream=True, timeout=120)
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


def main():
    BOSTON_DIR.mkdir(parents=True, exist_ok=True)
    download_file(ASSESSMENT_URL, BOSTON_DIR / "boston_assessment.csv")
    print("\nBoston assessment download complete.")


if __name__ == "__main__":
    main()

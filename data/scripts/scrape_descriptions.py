import sys
import time
import re
import json
import random
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from bs4 import BeautifulSoup
from config import RAW_DIR, PROCESSED_DIR, CITIES

DESC_DIR = RAW_DIR / "descriptions"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

REDFIN_SEARCH_URL = "https://www.redfin.com/stingray/api/gis?al=1&num_homes=350&ord=redfin-recommended-asc&page=1&sf=1,2,3,5,6,7&status=9&uipt=1,2,3&v=8"

CITY_PARAMS = {
    "boston": {"region_id": 1826, "region_type": 6},
    "nyc": {"region_id": 30749, "region_type": 6},
    "sf": {"region_id": 20330, "region_type": 6},
}


def fetch_redfin_listings(city, max_pages=10):
    if city not in CITY_PARAMS:
        print(f"  No Redfin params for {city}, skipping")
        return []

    params = CITY_PARAMS[city]
    all_listings = []

    for page in range(1, max_pages + 1):
        url = REDFIN_SEARCH_URL + f"&region_id={params['region_id']}&region_type={params['region_type']}&page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"  Page {page}: status {resp.status_code}, stopping")
                break

            text = resp.text
            if text.startswith("{}&&"):
                text = text[4:]

            data = json.loads(text)
            homes = data.get("payload", {}).get("homes", [])
            if not homes:
                break

            for home in homes:
                listing = {
                    "address": home.get("streetLine", {}).get("value", ""),
                    "city_name": home.get("city", ""),
                    "state": home.get("state", ""),
                    "zip": home.get("zip", ""),
                    "latitude": home.get("latLong", {}).get("latitude"),
                    "longitude": home.get("latLong", {}).get("longitude"),
                    "price": home.get("price", {}).get("value"),
                    "url": home.get("url", ""),
                }
                all_listings.append(listing)

            print(f"\r  Page {page}: {len(all_listings)} listings", end="", flush=True)
            time.sleep(random.uniform(3, 6))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n  Page {page} parse error: {e}")
            break

    print()
    return all_listings


def fetch_description(url):
    try:
        full_url = f"https://www.redfin.com{url}" if url.startswith("/") else url
        resp = requests.get(full_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        desc_div = soup.find("div", {"id": "TextContent-TextContent"})
        if not desc_div:
            desc_div = soup.find("div", class_=re.compile("remarks"))
        if not desc_div:
            desc_div = soup.find("p", class_=re.compile("description"))

        if desc_div:
            text = desc_div.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            return text

        return ""
    except Exception:
        return ""


def scrape_city(city, max_listings=200):
    print(f"\n{city}:")

    listings = fetch_redfin_listings(city)
    if not listings:
        print("  No listings found")
        return

    listings = listings[:max_listings]
    print(f"  Fetching descriptions for {len(listings)} listings...")

    out_path = DESC_DIR / f"{city}_descriptions.csv"
    saved = 0

    for i, listing in enumerate(listings):
        if listing.get("url"):
            listing["description"] = fetch_description(listing["url"])
            if listing["description"] and len(listing["description"]) > 50:
                row = pd.DataFrame([listing])
                row.to_csv(out_path, mode="a", header=(saved == 0), index=False)
                saved += 1
            if (i + 1) % 10 == 0:
                print(f"\r  {i + 1}/{len(listings)} fetched, {saved} saved", end="", flush=True)
            time.sleep(random.uniform(2, 4))

    print(f"\n  Saved {saved} descriptions → {out_path}")


def main():
    DESC_DIR.mkdir(parents=True, exist_ok=True)
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        scrape_city(city)
    print("\nDescription scraping complete.")


if __name__ == "__main__":
    main()

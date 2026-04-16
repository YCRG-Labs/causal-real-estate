import sys
import socket
import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
from config import RAW_DIR, CITIES, OSM_AMENITY_TAGS

socket.setdefaulttimeout(420)

AMENITY_DIR = RAW_DIR / "amenities"
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

CITY_BBOXES = {
    "boston": (42.23, -71.19, 42.40, -70.92),
    "nyc": (40.49, -74.26, 40.92, -73.70),
    "sf": (37.70, -122.52, 37.84, -122.35),
}


def build_overpass_query(bbox, tag_key, tag_values):
    s, w, n, e = bbox
    value_str = "|".join(tag_values)
    return f"""
    [out:json][timeout:300];
    (
      node["{tag_key}"~"{value_str}"]({s},{w},{n},{e});
      way["{tag_key}"~"{value_str}"]({s},{w},{n},{e});
    );
    out center;
    """


def parse_overpass_response(data, category):
    rows = []
    for elem in data.get("elements", []):
        if elem["type"] == "node":
            lat, lon = elem["lat"], elem["lon"]
        elif "center" in elem:
            lat, lon = elem["center"]["lat"], elem["center"]["lon"]
        else:
            continue

        tags = elem.get("tags", {})
        rows.append({
            "osm_id": elem["id"],
            "category": category,
            "subcategory": tags.get("amenity") or tags.get("shop") or tags.get("leisure") or tags.get("highway") or tags.get("railway", ""),
            "name": tags.get("name", ""),
            "latitude": lat,
            "longitude": lon,
        })
    return rows


def download_city(city):
    print(f"\n{city}:")
    bbox = CITY_BBOXES[city]
    all_rows = []

    for category, tag_dict in OSM_AMENITY_TAGS.items():
        for tag_key, tag_values in tag_dict.items():
            query = build_overpass_query(bbox, tag_key, tag_values)
            print(f"  {category}/{tag_key}...", end=" ", flush=True)

            resp = None
            for attempt in range(len(OVERPASS_ENDPOINTS) * 3):
                endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
                try:
                    sess = requests.Session()
                    resp = sess.post(
                        endpoint,
                        data={"data": query},
                        timeout=(30, 360),
                        headers={"Connection": "close"},
                    )
                    if resp.status_code in (429, 504):
                        mirror = endpoint.split("/")[2]
                        wait = min(30 + 15 * attempt, 120)
                        print(f"\n    {resp.status_code} from {mirror}, "
                              f"trying next mirror in {wait}s...",
                              end="", flush=True)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                except (requests.exceptions.ConnectionError,
                        requests.exceptions.ReadTimeout,
                        requests.exceptions.ConnectTimeout,
                        socket.timeout) as e:
                    mirror = endpoint.split("/")[2]
                    wait = min(10 * (2 ** (attempt // len(OVERPASS_ENDPOINTS))), 300)
                    print(f"\n    ⚠ {type(e).__name__} from {mirror}; "
                          f"sleeping {wait}s",
                          end="", flush=True)
                    time.sleep(wait)
                finally:
                    sess.close()
            else:
                print(f" FAILED after {len(OVERPASS_ENDPOINTS) * 3} attempts across all mirrors, skipping")
                continue
            if resp is None or resp.status_code != 200:
                print(f" no successful response, skipping")
                continue
            data = resp.json()

            rows = parse_overpass_response(data, category)
            all_rows.extend(rows)
            print(f"{len(rows)} features")

            time.sleep(10)

    df = pd.DataFrame(all_rows)
    out_path = AMENITY_DIR / f"{city}_amenities.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} amenities → {out_path}")


def main():
    AMENITY_DIR.mkdir(parents=True, exist_ok=True)
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        download_city(city)
    print("\nAmenities download complete.")


if __name__ == "__main__":
    main()

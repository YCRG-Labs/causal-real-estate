"""Async Redfin scraper for the 12-city JBES expansion.

Polite scraping at one request per second per host, asyncio-driven concurrency
across the nine new cities. Resumable via per-city `_seen.jsonl` checkpoint;
crash at city 7 of 9 does not require rerunning cities 1-6.

Output schema (one parquet per city at data/processed/{city}_listings.parquet):
    url, address, description, price, beds, baths, sqft, year_built,
    lot_size, property_type, listing_status, lat_centroid, lon_centroid,
    zip, scrape_date, source_html_sha256
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_endpoints import CITIES, CityConfig, list_new  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "redfin"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
USER_AGENT = "YCRG-Labs-2026/1.0 (jacobcrainic@icloud.com; research, polite)"
PER_HOST_DELAY_S = 2.0
LISTING_PRICE_RE = re.compile(r'"price":\s*\{\s*"value":\s*(\d+)')
LISTING_DATA_RE = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>([^<]+)</script>')


@dataclass
class Listing:
    url: str
    address: str = ""
    description: str = ""
    price: float | None = None
    beds: float | None = None
    baths: float | None = None
    sqft: float | None = None
    year_built: int | None = None
    lot_size: float | None = None
    property_type: str = ""
    listing_status: str = ""
    lat_centroid: float | None = None
    lon_centroid: float | None = None
    zip: str = ""
    scrape_date: str = ""
    source_html_sha256: str = ""


def city_raw_dir(slug: str) -> Path:
    out = RAW_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def seen_path(slug: str) -> Path:
    return city_raw_dir(slug) / "_seen.jsonl"


def load_seen(slug: str) -> set[str]:
    p = seen_path(slug)
    if not p.exists():
        return set()
    seen = set()
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") in (200, 304):
                    seen.add(row["url"])
            except json.JSONDecodeError:
                continue
    return seen


def append_seen(slug: str, url: str, status: int, html_sha: str) -> None:
    with seen_path(slug).open("a") as f:
        f.write(json.dumps({
            "url": url, "status": status, "sha": html_sha,
            "ts": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


def html_sha256(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()


async def fetch_city_index(client: httpx.AsyncClient, cfg: CityConfig, max_pages: int = 30) -> list[str]:
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        index_url = f"{cfg.redfin_url}/page-{page}" if page > 1 else cfg.redfin_url
        try:
            r = await client.get(index_url)
        except httpx.HTTPError as e:
            print(f"  [{cfg.slug}] index page {page} error: {e}", file=sys.stderr)
            break
        if r.status_code != 200:
            print(f"  [{cfg.slug}] index page {page} status {r.status_code}, stopping", file=sys.stderr)
            break
        page_urls = re.findall(r'href="(/[A-Z]{2}/[^"]+/home/\d+)"', r.text)
        if not page_urls:
            break
        urls.extend(f"https://www.redfin.com{u}" for u in page_urls)
        await asyncio.sleep(PER_HOST_DELAY_S)
    return list(dict.fromkeys(urls))


def parse_listing_html(url: str, html: str) -> Listing:
    listing = Listing(
        url=url,
        scrape_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        source_html_sha256=html_sha256(html),
    )
    ldjson_matches = LISTING_DATA_RE.findall(html)
    for raw in ldjson_matches:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            payload = next((p for p in payload if isinstance(p, dict) and p.get("@type") in ("Residence", "House", "SingleFamilyResidence", "Apartment")), payload[0] if payload else {})
        if not isinstance(payload, dict):
            continue
        if payload.get("@type") in ("Residence", "House", "SingleFamilyResidence", "Apartment"):
            addr = payload.get("address", {})
            if isinstance(addr, dict):
                listing.address = " ".join(filter(None, [
                    addr.get("streetAddress", ""), addr.get("addressLocality", ""),
                    addr.get("addressRegion", ""), addr.get("postalCode", ""),
                ])).strip()
                listing.zip = addr.get("postalCode", "") or listing.zip
            geo = payload.get("geo", {})
            if isinstance(geo, dict):
                try:
                    listing.lat_centroid = float(geo.get("latitude")) if geo.get("latitude") else None
                    listing.lon_centroid = float(geo.get("longitude")) if geo.get("longitude") else None
                except (TypeError, ValueError):
                    pass
            if "numberOfRooms" in payload:
                try:
                    listing.beds = float(payload["numberOfRooms"])
                except (TypeError, ValueError):
                    pass
            if "floorSize" in payload:
                fs = payload["floorSize"]
                if isinstance(fs, dict) and fs.get("value"):
                    try:
                        listing.sqft = float(fs["value"])
                    except (TypeError, ValueError):
                        pass
    price_match = LISTING_PRICE_RE.search(html)
    if price_match:
        try:
            listing.price = float(price_match.group(1))
        except (TypeError, ValueError):
            pass
    desc_match = re.search(r'"marketingRemarks":\s*"([^"]+)"', html)
    if desc_match:
        listing.description = desc_match.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
    beds_match = re.search(r'"beds":\s*(\d+(?:\.\d+)?)', html)
    if beds_match and listing.beds is None:
        try:
            listing.beds = float(beds_match.group(1))
        except (TypeError, ValueError):
            pass
    baths_match = re.search(r'"baths":\s*(\d+(?:\.\d+)?)', html)
    if baths_match:
        try:
            listing.baths = float(baths_match.group(1))
        except (TypeError, ValueError):
            pass
    yr_match = re.search(r'"yearBuilt":\s*(\d{4})', html)
    if yr_match:
        try:
            listing.year_built = int(yr_match.group(1))
        except (TypeError, ValueError):
            pass
    ptype_match = re.search(r'"propertyType":\s*"([^"]+)"', html)
    if ptype_match:
        listing.property_type = ptype_match.group(1)
    return listing


async def fetch_listing(client: httpx.AsyncClient, cfg: CityConfig, url: str) -> Listing | None:
    try:
        r = await client.get(url)
    except httpx.HTTPError as e:
        append_seen(cfg.slug, url, -1, "")
        print(f"  [{cfg.slug}] {url} error: {e}", file=sys.stderr)
        return None
    sha = html_sha256(r.text)
    append_seen(cfg.slug, url, r.status_code, sha)
    if r.status_code != 200:
        return None
    raw_path = city_raw_dir(cfg.slug) / "html" / f"{sha[:16]}.html.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        import gzip
        with gzip.open(raw_path, "wt", encoding="utf-8") as f:
            f.write(r.text)
    return parse_listing_html(url, r.text)


async def scrape_city(cfg: CityConfig, resume: bool, max_listings: int) -> int:
    seen = load_seen(cfg.slug) if resume else set()
    parquet_out = PROCESSED_DIR / f"{cfg.slug}_listings.parquet"
    existing: list[Listing] = []
    if parquet_out.exists() and resume:
        try:
            df = pd.read_parquet(parquet_out)
            existing = [Listing(**row) for row in df.to_dict(orient="records")]
        except Exception as e:
            print(f"  [{cfg.slug}] existing parquet load failed: {e}", file=sys.stderr)
    transport = httpx.AsyncHTTPTransport(retries=2)
    limits = httpx.Limits(max_keepalive_connections=4, max_connections=8)
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        follow_redirects=True, transport=transport, limits=limits, timeout=timeout,
    ) as client:
        print(f"[{cfg.slug}] discovering listing URLs...")
        listing_urls = await fetch_city_index(client, cfg)
        if max_listings > 0:
            listing_urls = listing_urls[:max_listings]
        new_urls = [u for u in listing_urls if u not in seen]
        print(f"[{cfg.slug}] {len(listing_urls)} URLs discovered, {len(new_urls)} new")
        listings = list(existing)
        for i, url in enumerate(new_urls, 1):
            if i % 25 == 0:
                print(f"  [{cfg.slug}] {i}/{len(new_urls)}")
            listing = await fetch_listing(client, cfg, url)
            if listing is not None:
                listings.append(listing)
            await asyncio.sleep(PER_HOST_DELAY_S)
    df = pd.DataFrame([asdict(l) for l in listings])
    df = df.drop_duplicates(subset=["url"], keep="last")
    df.to_parquet(parquet_out, index=False)
    print(f"[{cfg.slug}] wrote {len(df)} rows to {parquet_out}")
    return len(df)


async def scrape_all(cities: list[CityConfig], resume: bool, max_listings: int) -> dict[str, int]:
    results = await asyncio.gather(*[scrape_city(c, resume, max_listings) for c in cities], return_exceptions=True)
    out: dict[str, int] = {}
    for cfg, res in zip(cities, results):
        if isinstance(res, Exception):
            print(f"[{cfg.slug}] FAILED: {res}", file=sys.stderr)
            out[cfg.slug] = -1
        else:
            out[cfg.slug] = res
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Async Redfin scraper for the 12-city expansion")
    parser.add_argument("--cities", type=str, default="new9", help="comma-separated slugs or 'new9' or 'all'")
    parser.add_argument("--max-listings", type=int, default=350, help="cap per city; 0 for no cap")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.cities == "new9":
        cities = list_new()
    elif args.cities == "all":
        from city_endpoints import list_ready
        cities = list_ready()
    else:
        slugs = [s.strip() for s in args.cities.split(",") if s.strip()]
        cities = [CITIES[s] for s in slugs]
    print(f"Scraping {len(cities)} cities: {[c.slug for c in cities]}")
    t0 = time.time()
    results = asyncio.run(scrape_all(cities, resume=not args.no_resume, max_listings=args.max_listings))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min:")
    for slug, count in results.items():
        print(f"  {slug}: {count} listings")
    return 0 if all(c >= 0 for c in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

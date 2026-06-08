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

import pandas as pd

try:
    from curl_cffi.requests import AsyncSession
    _BACKEND = "curl_cffi"
except ImportError:
    import httpx
    _BACKEND = "httpx"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_endpoints import CITIES, CityConfig, list_new

import dataclasses as _dataclasses
_zip_shards_path = Path(__file__).parent / "zip_shards.json"
if _zip_shards_path.exists():
    _zip_shards = json.loads(_zip_shards_path.read_text())
    for _slug, _cfg in list(CITIES.items()):
        if _slug in _zip_shards and not _cfg.zip_codes:
            CITIES[_slug] = _dataclasses.replace(
                _cfg, zip_codes=tuple(_zip_shards[_slug]),
            )

RAW_DIR = REPO_ROOT / "data" / "raw" / "redfin"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Chromium";v="121", "Not A(Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}
PER_HOST_DELAY_S = 2.0
LISTING_PRICE_RE = re.compile(r'"price":\s*\{\s*"value":\s*(\d+)')
LISTING_DATA_RE = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.+?)</script>', re.DOTALL)
META_DESCRIPTION_RE = re.compile(r'<meta\s+name="description"\s+content="([^"]+)"')
RESIDENCE_TYPES = {
    "Residence", "House", "SingleFamilyResidence", "Apartment",
    "Townhouse", "Condominium", "ApartmentComplex", "MultiUnit",
    "Place", "Accommodation",
}


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


async def _fetch_index_pages(client, base_url: str, slug: str, max_pages: int = 9,
                              tag: str = "") -> list[str]:
    """Drain up to max_pages of a Redfin search URL.  Redfin caps any
    single search at 9 pages × ~40 listings ≈ 350 unique URLs.
    """
    urls: list[str] = []
    for page in range(1, max_pages + 1):
        index_url = f"{base_url}/page-{page}" if page > 1 else base_url
        try:
            r = await client.get(index_url)
        except Exception as e:
            print(f"  [{slug}{tag}] page {page} error: {e}", file=sys.stderr)
            break
        if r.status_code != 200:
            break
        page_urls = re.findall(r'href="(/[A-Z]{2}/[^"]+/home/\d+)"', r.text)
        if not page_urls:
            break
        urls.extend(f"https://www.redfin.com{u}" for u in page_urls)
        await asyncio.sleep(PER_HOST_DELAY_S)
    return urls


async def fetch_city_index(client, cfg: CityConfig, max_pages: int = 9,
                            target_n: int = 0) -> list[str]:
    """Discover Redfin listing URLs for `cfg`.

    Legacy mode (cfg.zip_codes empty): scrape the city search endpoint
    up to max_pages (9 by default, matching Redfin's pagination cap).
    Yields the ~350-listing-per-city ceiling.

    ZIP-shard mode (cfg.zip_codes non-empty): iterate per-ZIP search
    endpoints `https://www.redfin.com/zipcode/{zip}` and union the
    deduplicated URLs.  Stops early once `target_n` unique URLs are
    discovered (target_n=0 means exhaust all ZIPs).  Bypasses the
    350-cap by sharding the search space; typical per-ZIP yield is
    30-150 listings, so a 30-ZIP city reaches 1,500-4,500 unique
    URLs before any single ZIP hits its own page cap.
    """
    expected_state = cfg.redfin_state_slug
    if not cfg.zip_codes:
        urls = await _fetch_index_pages(client, cfg.redfin_url, cfg.slug,
                                          max_pages=max_pages)
        if urls and f"/{expected_state}/" not in urls[0]:
            print(f"  [{cfg.slug}] WARNING: first URL state mismatch", file=sys.stderr)
            return []
        return list(dict.fromkeys(urls))

    seen: set[str] = set()
    out: list[str] = []
    for zip5 in cfg.zip_codes:
        if target_n and len(seen) >= target_n:
            break
        base = f"https://www.redfin.com/zipcode/{zip5}"
        zip_urls = await _fetch_index_pages(client, base, cfg.slug,
                                              max_pages=max_pages,
                                              tag=f"/{zip5}")
        new = 0
        for u in zip_urls:
            if u not in seen:
                seen.add(u); out.append(u); new += 1
        if new:
            print(f"  [{cfg.slug}/{zip5}] +{new} new (total {len(seen)})",
                  file=sys.stderr)
    return out


def _type_tokens(item: dict) -> set[str]:
    t = item.get("@type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    return set()


def _ingest_residence_item(item: dict, listing: Listing) -> None:
    addr = item.get("address")
    if isinstance(addr, dict):
        parts = [
            addr.get("streetAddress", ""),
            addr.get("addressLocality", ""),
            addr.get("addressRegion", ""),
            addr.get("postalCode", ""),
        ]
        candidate = " ".join(p for p in parts if p).strip()
        if candidate and not listing.address:
            listing.address = candidate
        if not listing.zip:
            listing.zip = addr.get("postalCode", "") or ""
    geo = item.get("geo")
    if isinstance(geo, dict):
        try:
            lat = geo.get("latitude")
            lon = geo.get("longitude")
            if lat is not None and listing.lat_centroid is None:
                listing.lat_centroid = float(lat)
            if lon is not None and listing.lon_centroid is None:
                listing.lon_centroid = float(lon)
        except (TypeError, ValueError):
            pass
    for beds_field in ("numberOfBedrooms", "numberOfRooms"):
        if beds_field in item and listing.beds is None:
            try:
                listing.beds = float(item[beds_field])
                break
            except (TypeError, ValueError):
                pass
    for baths_field in ("numberOfBathroomsTotal", "numberOfFullBathrooms"):
        if baths_field in item and listing.baths is None:
            try:
                listing.baths = float(item[baths_field])
                break
            except (TypeError, ValueError):
                pass
    fs = item.get("floorSize")
    if isinstance(fs, dict) and fs.get("value") and listing.sqft is None:
        try:
            listing.sqft = float(fs["value"])
        except (TypeError, ValueError):
            pass
    yr = item.get("yearBuilt")
    if yr is not None and listing.year_built is None:
        try:
            listing.year_built = int(yr)
        except (TypeError, ValueError):
            pass
    cat = item.get("accommodationCategory") or item.get("@type")
    if isinstance(cat, list):
        cat = next((c for c in cat if c not in ("Product", "RealEstateListing")), None)
    if isinstance(cat, str) and cat and not listing.property_type:
        listing.property_type = cat


def _ingest_product_item(item: dict, listing: Listing) -> None:
    offers = item.get("offers")
    if isinstance(offers, dict):
        price = offers.get("price")
        if price is not None and listing.price is None:
            try:
                listing.price = float(str(price).replace(",", ""))
            except (TypeError, ValueError):
                pass
    desc = item.get("description")
    if isinstance(desc, str) and desc and not listing.description:
        listing.description = desc


def parse_listing_html(url: str, html: str) -> Listing:
    listing = Listing(
        url=url,
        scrape_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        source_html_sha256=html_sha256(html),
    )
    ldjson_matches = LISTING_DATA_RE.findall(html)
    for raw in ldjson_matches:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            types = _type_tokens(item)
            if types & RESIDENCE_TYPES:
                _ingest_residence_item(item, listing)
            if "Product" in types or "RealEstateListing" in types:
                _ingest_product_item(item, listing)
                main = item.get("mainEntity")
                if isinstance(main, dict):
                    _ingest_residence_item(main, listing)
    if not listing.description:
        m = META_DESCRIPTION_RE.search(html)
        if m:
            listing.description = m.group(1)
    if listing.price is None:
        price_match = LISTING_PRICE_RE.search(html)
        if price_match:
            try:
                listing.price = float(price_match.group(1))
            except (TypeError, ValueError):
                pass
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


async def fetch_listing(client, cfg: CityConfig, url: str) -> Listing | None:
    try:
        r = await client.get(url)
    except Exception as e:
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
    if _BACKEND == "curl_cffi":
        client_ctx = AsyncSession(impersonate="chrome120", timeout=30.0)
    else:
        transport = httpx.AsyncHTTPTransport(retries=2)
        limits = httpx.Limits(max_keepalive_connections=4, max_connections=8)
        timeout = httpx.Timeout(30.0, connect=10.0)
        client_ctx = httpx.AsyncClient(
            headers=BROWSER_HEADERS,
            follow_redirects=True, transport=transport, limits=limits, timeout=timeout,
        )
    async with client_ctx as client:
        print(f"[{cfg.slug}] discovering listing URLs (backend={_BACKEND})...")
        listing_urls = await fetch_city_index(client, cfg, target_n=max_listings)
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
    if not listings:
        print(f"[{cfg.slug}] no listings collected; skipping parquet write")
        return 0
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


def reparse_city(cfg: CityConfig) -> int:
    import gzip
    parquet_out = PROCESSED_DIR / f"{cfg.slug}_listings.parquet"
    seen_file = seen_path(cfg.slug)
    if not seen_file.exists():
        print(f"[{cfg.slug}] no _seen.jsonl at {seen_file}; skipping", file=sys.stderr)
        return 0
    url_to_sha: dict[str, str] = {}
    with seen_file.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == 200 and row.get("sha"):
                url_to_sha[row["url"]] = row["sha"]
    html_dir = city_raw_dir(cfg.slug) / "html"
    listings: list[Listing] = []
    n_missing = 0
    for url, sha in url_to_sha.items():
        cache_path = html_dir / f"{sha[:16]}.html.gz"
        if not cache_path.exists():
            n_missing += 1
            continue
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            print(f"  [{cfg.slug}] {cache_path.name} read error: {e}", file=sys.stderr)
            continue
        listings.append(parse_listing_html(url, html))
    if not listings:
        print(f"[{cfg.slug}] no cached HTML to re-parse")
        return 0
    df = pd.DataFrame([asdict(l) for l in listings])
    df = df.drop_duplicates(subset=["url"], keep="last")
    df.to_parquet(parquet_out, index=False)
    n_addr = int((df["address"].astype(str).str.strip() != "").sum()) if "address" in df.columns else 0
    n_desc = int((df["description"].astype(str).str.strip() != "").sum()) if "description" in df.columns else 0
    n_latlon = int(df["lat_centroid"].notna().sum()) if "lat_centroid" in df.columns else 0
    print(f"[{cfg.slug}] reparsed {len(df)} listings (addresses={n_addr}, descriptions={n_desc}, latlon={n_latlon}, missing_html={n_missing})")
    return len(df)


def reparse_all(cities: list[CityConfig]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cfg in cities:
        try:
            out[cfg.slug] = reparse_city(cfg)
        except Exception as e:
            print(f"[{cfg.slug}] reparse FAILED: {e}", file=sys.stderr)
            out[cfg.slug] = -1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Async Redfin scraper for the 12-city expansion")
    parser.add_argument("--cities", type=str, default="new9", help="comma-separated slugs or 'new9' or 'all'")
    parser.add_argument("--max-listings", type=int, default=0,
                         help="cap per city (0 = no cap; in ZIP-shard mode "
                              "this is the early-stop target). Default 0 "
                              "now that fetch_city_index supports per-ZIP "
                              "discovery for bypassing the legacy 350-cap.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--reparse", action="store_true", help="re-parse cached HTML without any network fetch")
    args = parser.parse_args()
    if args.cities == "new9":
        cities = list_new()
    elif args.cities == "all":
        from city_endpoints import list_ready
        cities = list_ready()
    else:
        slugs = [s.strip() for s in args.cities.split(",") if s.strip()]
        cities = [CITIES[s] for s in slugs]
    t0 = time.time()
    if args.reparse:
        print(f"Re-parsing cached HTML for {len(cities)} cities: {[c.slug for c in cities]}")
        results = reparse_all(cities)
    else:
        print(f"Scraping {len(cities)} cities: {[c.slug for c in cities]}")
        results = asyncio.run(scrape_all(cities, resume=not args.no_resume, max_listings=args.max_listings))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min:")
    for slug, count in results.items():
        print(f"  {slug}: {count} listings")
    return 0 if all(c >= 0 for c in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

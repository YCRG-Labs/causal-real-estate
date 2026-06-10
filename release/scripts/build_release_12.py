"""Assemble the twelve-market release, ToS-compliant.

INCLUDES (redistributable): sentence-transformer embeddings (derived from the
descriptions), structured property attributes and parcel lat/lon, the recovered
sale dates, and the public-record recorded sale prices for the four markets that
publish them (Philadelphia, Chicago, DC, NYC).

EXCLUDES (scraped commercial-platform content, not redistributed): the raw and
cleaned listing descriptions, the asking (listing) price, and the address/url.
These are reconstructable with the released collection code. The census, crime,
and amenity confounders are not shipped here either; they are regenerated from
their public sources by the pipeline scripts (attach_census/crime/amenity).

Run where the full embeddings live (i.e. on the compute box):
  python3 release/scripts/build_release_12.py

Writes release/data_12/<city>.parquet (one anonymized id per listing) and
release/data_12/MANIFEST.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
OUT = REPO / "release" / "data_12"

MARKETS = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
           "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
SALE_MARKETS = {"philadelphia", "chicago", "dc", "nyc"}

KEEP_GEO = ["latitude", "longitude", "zip"]
KEEP_PROP = ["beds", "baths", "sqft", "year_built", "lot_size", "property_type"]
DROP = ["description", "clean_description", "price", "address", "url",
        "city_name", "state", "BBL_matched", "geocode_method", "match_label"]


def build_city(city: str) -> dict:
    ep = PROC / f"{city}_embeddings.parquet"
    if not ep.exists():
        return {"city": city, "status": "MISSING embeddings"}
    e = pd.read_parquet(ep)
    if "source_html_sha256" in e.columns:
        e["id"] = e["source_html_sha256"].astype(str).str[:16]
    elif "url" in e.columns:
        e["id"] = e["url"].astype(str).map(
            lambda u: hashlib.sha256(u.encode()).hexdigest()[:16])
    else:
        e["id"] = [f"{city}{i:06d}" for i in range(len(e))]
    emb_cols = [c for c in e.columns if c.startswith("emb_")]
    keep = ["id"] + [c for c in KEEP_GEO + KEEP_PROP if c in e.columns] + emb_cols
    out = e[keep].copy()

    sd = PROC / f"{city}_sold_dates.parquet"
    if sd.exists():
        S = pd.read_parquet(sd).rename(columns={"sha16": "id"})
        out = out.merge(S[["id", "sale_year", "sale_quarter"]].drop_duplicates("id"),
                        on="id", how="left")

    if city in SALE_MARKETS:
        sp = PROC / f"{city}_listings_saleprice.parquet"
        if sp.exists():
            P = pd.read_parquet(sp).rename(columns={"sha16": "id"})
            out = out.merge(P[["id", "sale_price", "sale_date"]].drop_duplicates("id"),
                            on="id", how="left")

    out = out[[c for c in out.columns if c not in DROP]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / f"{city}.parquet", index=False)
    return {"city": city, "status": "ok", "n": len(out), "emb_dim": len(emb_cols),
            "has_sale_price": city in SALE_MARKETS and "sale_price" in out.columns,
            "sale_price_cov": (f"{100*out['sale_price'].notna().mean():.0f}%"
                               if "sale_price" in out.columns else "-")}


def main() -> int:
    rows = [build_city(c) for c in MARKETS]
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        "# Twelve-market release manifest", "",
        "ToS-compliant package: derived embeddings, structured/geo features, sale",
        "dates, and public-record sale prices for the four markets that publish",
        "them. Raw descriptions and asking prices are NOT included (scraped content,",
        "reconstructable via the collection code). Census/crime/amenity confounders",
        "are regenerated from public sources by the pipeline.", "",
        "| Market | n | emb dim | recorded sale price |", "|---|---|---|---|",
    ]
    for r in ok:
        lines.append(f"| {r['city']} | {r['n']:,} | {r['emb_dim']} | "
                     f"{r['sale_price_cov'] if r['has_sale_price'] else 'reconstruct via scraper'} |")
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n")

    for r in rows:
        print(f"  {r['city']:13s} {r['status']}" +
              (f"  n={r['n']:,}  sale_price={r['sale_price_cov']}" if r["status"] == "ok" else ""))
    print(f"\n{len(ok)}/{len(MARKETS)} markets packaged -> {OUT}")
    return 0 if len(ok) == len(MARKETS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

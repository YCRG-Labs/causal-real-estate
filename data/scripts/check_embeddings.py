"""Preflight check: are the full-panel embeddings present and joinable to the
listings table and the recovered sale dates, so the sale-time-FE robustness and
the spatial/meta fixes can run on real data?

For each market it reports, with a READY / PARTIAL / MISSING verdict:
  - embeddings file found, row count, embedding dim (768=mpnet, 384=MiniLM)
  - the join key it carries (url / source_html_sha256 / row-aligned fallback)
  - join coverage to listings (price, lat, lon)
  - join coverage to <city>_sold_dates.parquet (sale_year)

No network, no compute. Read-only.

  python3 data/scripts/check_embeddings.py --all_12
  python3 data/scripts/check_embeddings.py --city nyc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _emb_path(city: str) -> Path | None:
    for name in (f"{city}_embeddings.parquet",
                 f"{city}_embeddings_all_mpnet_base_v2.parquet"):
        p = PROC / name
        if p.exists():
            return p
    return None


def _frac(x) -> str:
    return f"{100*float(np.mean(x)):.0f}%"


def check_city(city: str) -> dict:
    r = {"city": city, "verdict": "MISSING", "notes": []}
    ep = _emb_path(city)
    if ep is None:
        r["notes"].append("no embeddings parquet in data/processed/")
        return r

    e = pd.read_parquet(ep)
    emb_cols = [c for c in e.columns if c.startswith("emb_")]
    r["file"] = ep.name
    r["n"] = len(e)
    r["emb_dim"] = len(emb_cols)
    r["encoder"] = {768: "mpnet", 384: "MiniLM"}.get(len(emb_cols), f"dim={len(emb_cols)}")
    if not emb_cols:
        r["notes"].append("no emb_* columns")
        return r

    has_price = "price" in e.columns and pd.to_numeric(e["price"], errors="coerce").notna().any()
    has_geo = {"latitude", "longitude"}.issubset(e.columns)
    r["price"] = _frac(pd.to_numeric(e.get("price", pd.Series(np.nan, index=e.index)),
                                     errors="coerce").notna()) if has_price else "—"
    r["latlon"] = _frac(pd.to_numeric(e.get("latitude", pd.Series(np.nan, index=e.index)),
                                      errors="coerce").notna()) if has_geo else "—"

    lst_path = PROC / f"{city}_listings.parquet"
    sd_path = PROC / f"{city}_sold_dates.parquet"
    if not lst_path.exists() or not sd_path.exists():
        r["notes"].append("missing listings or sold_dates parquet; run parse_sold_dates.py")
        r["verdict"] = "PARTIAL"
        return r

    L = pd.read_parquet(lst_path)
    L["sha16"] = L["source_html_sha256"].astype(str).str[:16]
    SD = pd.read_parquet(sd_path)[["sha16", "sale_year", "sale_quarter"]]
    LS = L.merge(SD, on="sha16", how="left")

    key = None
    if "source_html_sha256" in e.columns:
        key = "sha"
        e2 = e.assign(sha16=e["source_html_sha256"].astype(str).str[:16])
        m = e2.merge(LS[["sha16", "sale_year"]], on="sha16", how="left")
        cov = m["sale_year"].notna()
    elif "url" in e.columns:
        key = "url"
        m = e.merge(LS[["url", "sale_year"]].drop_duplicates("url"), on="url", how="left")
        cov = m["sale_year"].notna()
    elif len(e) == len(LS):
        key = "row-aligned (assumed)"
        cov = LS["sale_year"].notna().to_numpy()
        r["notes"].append("no url/sha key; assuming row alignment with listings — verify")
    else:
        key = "NONE"
        cov = np.zeros(len(e), dtype=bool)
        r["notes"].append("no join key and row counts differ; cannot attach sale_year")

    r["key"] = key
    r["sale_year_cov"] = _frac(cov)

    ready = (len(emb_cols) in (768, 384) and has_price and has_geo
             and key not in ("NONE",) and float(np.mean(cov)) >= 0.80)
    if ready and len(emb_cols) != 768:
        r["notes"].append("encoder is not mpnet(768); headline uses mpnet")
    r["verdict"] = "READY" if ready else "PARTIAL"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    args = ap.parse_args()
    cities = ALL_12 if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")

    rows = [check_city(c) for c in cities]
    hdr = f"{'market':13s} {'verdict':8s} {'n':>6s} {'enc':>6s} {'key':>10s} {'price':>6s} {'latlon':>7s} {'saleyr':>7s}"
    print(hdr)
    print("-" * len(hdr))
    nready = 0
    for r in rows:
        nready += r["verdict"] == "READY"
        print(f"{r['city']:13s} {r['verdict']:8s} {r.get('n','—'):>6} {r.get('encoder','—'):>6} "
              f"{str(r.get('key','—')):>10} {r.get('price','—'):>6} {r.get('latlon','—'):>7} "
              f"{r.get('sale_year_cov','—'):>7}")
        for note in r["notes"]:
            print(f"               ↳ {note}")
    print(f"\n{nready}/{len(rows)} markets READY for the sale-time-FE robustness run.")
    return 0 if nready == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

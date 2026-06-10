"""Acquire real RECORDED SALE prices (deed/assessor transactions) per market and
join them to the scraped listings, so the outcome can be the realized sale price
the text-valuation literature uses (Shen & Ross, Nowak & Smith, Baur) rather than
the agent's asking price.

Why this over the scraped price: the scraped `price` is the listing/asking price,
which stochastically dominates sale price by ~26-28% and carries different hedonic
coefficients (Empirical Economics 2021), so it is not a substitute for transaction
data. Recorded sales from the county assessor/recorder cover the full parcel
universe (not the resale-selected subset the Redfin lastSoldPrice field exposes),
are free and public in disclosure states, and key to listings by parcel/lat-lon.

DALLAS is intentionally absent: Texas is a non-disclosure state (Gov Code
Sec. 552.149), so sale prices are not public record. Dallas must be dropped, kept
on list price with a flag, or sourced from a paid vendor.

Per-market source registry (endpoints/methods) is in SOURCES below; Philadelphia
is implemented and verified against the live API. The others document the access
path so each can be filled in and run the same normalize/join.

  python3 data/scripts/download_sales.py --city philadelphia
  python3 data/scripts/download_sales.py --city philadelphia --limit 5000   # smoke

Writes data/processed/<city>_recorded_sales.parquet with columns:
  parcel_id, address, lat, lon, sale_price, sale_date.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"

# Disclosure-state markets with free public recorded-sale data. Dallas (TX) is
# excluded by law. Each entry documents the access path; `fetch` is implemented
# where verified.
SOURCES = {
    "philadelphia": {
        "method": "carto",
        "endpoint": "https://phl.carto.com/api/v2/sql",
        "table": "opa_properties_public",
        "cols": {"parcel_id": "parcel_number", "address": "location",
                 "lat": "lat", "lon": "lng",
                 "sale_price": "sale_price", "sale_date": "sale_date"},
    },
    "chicago": {"method": "socrata", "verified": True,
                "endpoint": "https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.json",
                "note": "Cook Assessor Parcel Sales (pin, sale_price, sale_date); 518k sales since 2020. "
                        "No lat/lon here: join pin -> Parcel Universe nj4t-kc8j for coordinates."},
    "dc": {"method": "arcgis", "verified": "saleprice field exists",
           "endpoint": "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Property_and_Land/MapServer",
           "note": "CAMA property sales / Owner Points carry SALEPRICE, SALEDATE, SSL. "
                   "Pin the live sales layer id (Owner Points 26 is retired); query /<id>/query."},
    "phoenix": {"method": "download_zip",
                "endpoint": "https://www.mcassessor.maricopa.gov/page/data_sales/",
                "note": "Maricopa Sales Affidavits ZIP (pipe-delimited): APN, SalePrice, SaleDate, grantor/grantee. "
                        "Join APN -> Maricopa parcels (api.mcassessor.maricopa.gov) for lat/lon."},
    "seattle": {"method": "download", "verified": "rpsale_extr is the table",
                "endpoint": "https://info.kingcounty.gov/assessor/datadownload/default.aspx",
                "note": "King EXTR Real Property Sales (rpsale_extr): key=Major+Minor, SalePrice, DocumentDate. "
                        "Join Major+Minor -> parcel_extr for geometry/lat-lon."},
    "denver": {"method": "BLOCKED_likely",
               "endpoint": "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_PROP_PARCELS_A/FeatureServer",
               "note": "Denver open-data parcels carry geometry but NO sale price (verified). "
                       "(An earlier endpoint on org ioennV6PpG5Xodq0 was Fairfax County VA, not Denver -- "
                       "misattributed.) Denver sale prices live at property.spatialest.com (search tool, not bulk). "
                       "Likely needs the assessor's non-open route or a vendor."},
    "atlanta": {"method": "arcgis_or_qpublic",
                "endpoint": "https://gisdata.fultoncountyga.gov/",
                "note": "Fulton GIS property layer carries recorded sale date+price (GA is disclosure); "
                        "find the parcels FeatureServer with sale fields + geometry."},
    "portland": {"method": "download_shapefile",
                 "endpoint": "https://rlisdiscovery.oregonmetro.gov/datasets/9d3c396ffad44649bc7451465aa300f0",
                 "note": "RLIS Taxlots is a SHAPEFILE download (not a queryable API); typically carries "
                         "SALEPRICE/SALEDATE compiled from county assessors. Download + parse + use geometry."},
    "boston": {"method": "BLOCKED_easy",
               "endpoint": "https://www.mass.gov/info-details/massgis-data-property-tax-parcels",
               "note": "Analyze Boston assessment has NO sale price (verified). MassGIS L3 standardized "
                       "parcels carry LS_PRICE/LS_DATE but as a statewide GDB download, not a clean API. "
                       "Like SF: no clean free sale-price feed -- drop or use MassGIS GDB with effort."},
    "sf": {"method": "BLOCKED",
           "endpoint": "https://data.sfgov.org/resource/wv5m-vpq2.json",
           "note": "DataSF open tax roll carries ONLY assessed values, no transaction price "
                   "(verified). Real SF sale prices need the Recorder (not open data) or a vendor. "
                   "Like Dallas, no clean free sale-price feed -- drop or keep assessed-flagged."},
    # dallas: Texas non-disclosure (Gov Code 552.149); no public sale price. Excluded by law.
    "nyc": {"method": "have",
            "note": "DOF Rolling Sales already integrated in load_parcels.py (sale_price, sale_date by BBL)"},
}


def _carto(endpoint: str, q: str) -> pd.DataFrame:
    url = endpoint + "?" + urllib.parse.urlencode({"q": q, "format": "csv"})
    with urllib.request.urlopen(url, timeout=120) as r:
        return pd.read_csv(io.StringIO(r.read().decode("utf-8")))


def _socrata(resource: str, select: str, where: str, max_rows: int | None,
             order: str = ":id", page: int = 50000) -> pd.DataFrame:
    """Paginated Socrata pull. Large pulls (Cook universe ~1.8M) are best run on
    Brev for the faster network; pass max_rows to bound a local smoke test."""
    frames, offset = [], 0
    while True:
        params = {"$select": select, "$where": where, "$order": order,
                  "$limit": page, "$offset": offset}
        url = resource + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=180) as r:
            chunk = pd.read_csv(io.StringIO(r.read().decode("utf-8"))) \
                if resource.endswith(".csv") else pd.read_json(io.BytesIO(r.read()))
        if len(chunk) == 0:
            break
        frames.append(chunk)
        offset += len(chunk)
        if len(chunk) < page or (max_rows and offset >= max_rows):
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_chicago(limit: int | None) -> pd.DataFrame:
    sales_url = "https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.csv"
    univ_url = "https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.csv"
    sales = _socrata(
        sales_url, select="pin,sale_price,sale_date",
        where="sale_price > 10000 AND sale_date >= '2015-01-01T00:00:00' "
              "AND sale_filter_less_than_10k = false AND is_multisale = false",
        max_rows=limit)
    sales["pin"] = sales["pin"].astype(str).str.zfill(14)
    sales = sales.sort_values("sale_date").drop_duplicates("pin", keep="last")

    univ = _socrata(
        univ_url, select="pin,lat,lon",
        where="lat IS NOT NULL AND lon IS NOT NULL",
        max_rows=(limit * 4 if limit else None))
    univ["pin"] = univ["pin"].astype(str).str.zfill(14)
    univ = univ.dropna(subset=["lat", "lon"]).drop_duplicates("pin")

    df = sales.merge(univ, on="pin", how="inner")
    df = df.rename(columns={"pin": "parcel_id"})
    df["address"] = ""
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce", utc=True)
    return df[["parcel_id", "address", "lat", "lon", "sale_price", "sale_date"]]


def fetch_philadelphia(limit: int | None) -> pd.DataFrame:
    s = SOURCES["philadelphia"]
    sel = ("parcel_number AS parcel_id, location AS address, "
           "ST_Y(the_geom) AS lat, ST_X(the_geom) AS lon, sale_price, sale_date")
    where = "sale_price > 10000 AND sale_date IS NOT NULL AND the_geom IS NOT NULL"
    cap = f" LIMIT {limit}" if limit else ""
    df = _carto(s["endpoint"], f"SELECT {sel} FROM {s['table']} WHERE {where}{cap}")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["sale_price", "lat", "lon"])
    return df[["parcel_id", "address", "lat", "lon", "sale_price", "sale_date"]]


def _arcgis_query(layer_url: str, out_fields: str, where: str,
                  max_rows: int | None, geom: bool, page: int = 1000) -> pd.DataFrame:
    """Paginated ArcGIS FeatureServer/MapServer query. With geom=True, requests
    geometry in WGS84 (outSR=4326) and returns lat/lon (point x/y or polygon-ring
    centroid)."""
    rows, offset = [], 0
    while True:
        params = {"where": where, "outFields": out_fields, "f": "json",
                  "resultOffset": offset, "resultRecordCount": page,
                  "returnGeometry": "true" if geom else "false"}
        if geom:
            params["outSR"] = "4326"
        url = layer_url.rstrip("/") + "/query?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=180) as r:
            d = json.loads(r.read().decode("utf-8"))
        feats = d.get("features", [])
        if not feats:
            break
        for ft in feats:
            rec = dict(ft.get("attributes", {}))
            if geom:
                g = ft.get("geometry", {}) or {}
                if "x" in g and "y" in g:
                    rec["lon"], rec["lat"] = g["x"], g["y"]
                elif g.get("rings"):
                    pts = [p for ring in g["rings"] for p in ring]
                    if pts:
                        rec["lon"] = sum(p[0] for p in pts) / len(pts)
                        rec["lat"] = sum(p[1] for p in pts) / len(pts)
            rows.append(rec)
        offset += len(feats)
        if not d.get("exceededTransferLimit") or (max_rows and offset >= max_rows):
            break
    return pd.DataFrame(rows)


def _normkey(s):
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.upper()


def _arcgis_two_table(sales_url, sales_fields, geom_url, geom_field, key,
                      price, date, where_sales, limit, geom_where="1=1") -> pd.DataFrame:
    """Generic county pattern: a sales table keyed by parcel id joined to a
    geometry layer keyed by the same id (whitespace-normalized). Used by DC."""
    sales = _arcgis_query(sales_url, sales_fields, where_sales, limit, geom=False)
    geo = _arcgis_query(geom_url, geom_field, geom_where,
                        (limit * 6 if limit else None), geom=True)
    if sales.empty or geo.empty or "lat" not in geo:
        raise SystemExit(f"two-table fetch incomplete: sales={len(sales)} geo={len(geo)}; "
                         f"verify the geometry layer and that its {geom_field} format "
                         f"matches the sales {key} format.")
    sales["_k"] = _normkey(sales[key])
    geo["_k"] = _normkey(geo[geom_field])
    sales = sales.sort_values(date).drop_duplicates("_k", keep="last")
    geo = geo.dropna(subset=["lat", "lon"]).drop_duplicates("_k")
    df = sales.merge(geo[["_k", "lat", "lon"]], on="_k", how="inner")
    df = df.rename(columns={key: "parcel_id", price: "sale_price", date: "sale_date"})
    df["address"] = ""
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce", utc=True, unit="ms")
    df = df[df["sale_price"] > 10000]
    return df[["parcel_id", "address", "lat", "lon", "sale_price", "sale_date"]]


def fetch_dc(limit: int | None) -> pd.DataFrame:
    base = "https://maps2.dcgis.dc.gov/dcgis/rest/services"
    return _arcgis_two_table(
        sales_url=f"{base}/DCGIS_APPS/Real_Property_Application/MapServer/4",
        sales_fields="SSL,PRICE,SALEDATE", key="SSL", price="PRICE", date="SALEDATE",
        where_sales="PRICE > 10000 AND SALEDATE IS NOT NULL",
        geom_url=f"{base}/DCGIS_DATA/Property_and_Land/MapServer/40",
        geom_field="SSL", geom_where="SSL NOT LIKE 'PAR%'", limit=limit)


FETCHERS = {"philadelphia": fetch_philadelphia, "chicago": fetch_chicago,
            "dc": fetch_dc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.city not in SOURCES:
        raise SystemExit(f"{args.city}: no recorded-sale source registered "
                         f"(Dallas is non-disclosure and has none).")
    if args.city not in FETCHERS:
        s = SOURCES[args.city]
        print(f"{args.city}: source registered but fetch not yet implemented.")
        print(f"  method={s.get('method')}  endpoint={s.get('endpoint','')}")
        print(f"  {s.get('note','')}")
        return 2

    df = FETCHERS[args.city](args.limit)
    out = PROC / f"{args.city}_recorded_sales.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    yrs = df["sale_date"].dt.year
    print(f"{args.city}: {len(df):,} recorded sales  "
          f"median ${df['sale_price'].median():,.0f}  "
          f"years {int(yrs.min())}-{int(yrs.max())}  "
          f"2020+ {100*(yrs>=2020).mean():.0f}%  2023+ {100*(yrs>=2023).mean():.0f}%")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

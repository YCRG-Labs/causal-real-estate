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
import tempfile
import urllib.parse
import urllib.request
import zipfile
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
    "phoenix": {"method": "agol_zip", "verified": True,
                "endpoint": "https://www.arcgis.com/sharing/rest/content/items/f3484c72a938497286adc4e5de7e9963/data",
                "note": "Maricopa Assessor 'Sales Affidavits' (pipe-delimited Sales_Affidavits.txt inside "
                        "Sales_Affidavits.zip; the mcassessor.maricopa.gov/page/data_sales/ page's download "
                        "button now points at this ArcGIS-Online item, NOT a raw FTP path). Cols: PARCELNUMBER, "
                        "SALEDATE_MMYYYY (month/year only), SALEPRICE, DEEDDATE_MMDDYYYY (day-precision, use this "
                        "for sale_date), DEEDTYPE, PROPERTYTYPECODE, SITUSADDRESS/CITY/ZIP, ASSESSORCODE (arms- "
                        "length flag: 'X' = clean, 'R##' = ratio-study reject, 'W##' = warning, 'A##'/'B##' = "
                        "exempt/other -- keep ASSESSORCODE=='X'). No lat/lon in this file; join PARCELNUMBER -> "
                        "APN in the 'Parcel Points' shapefile (item dbf139379db946e1b10a2f15672c142d, same "
                        "arcgis.com/.../data download pattern, EPSG:2868, reproject to 4326)."},
    "seattle": {"method": "kingcounty_zip", "verified": True,
                "endpoint": "https://aqua.kingcounty.gov/extranet/assessor/Real%20Property%20Sales.zip",
                "note": "King County Assessor EXTR_RPSale.csv (comma-delimited, ~1.38M rows, weekly refresh). "
                        "Key=Major+Minor->10-char PIN. Cols: ExciseTaxNbr,Major,Minor,DocumentDate,SalePrice,"
                        "RecordingNbr,...,PropertyType,PrincipalUse,SaleInstrument,AFForestLand,AFCurrentUseLand,"
                        "AFNonProfitUse,AFHistoricProperty,SaleReason,PropertyClass,SaleWarning. Code lookup is "
                        "Lookup.zip -> EXTR_LookUp.csv (LUType 1=PropertyType, 2=PrincipalUse, 4=PropertyClass, "
                        "5=SaleReason, 6=SaleInstrument, 7=SaleWarning). No lat/lon in EXTR_Parcel.csv; join "
                        "Major+Minor -> PIN in the ArcGIS PARCEL_ADDRESS_PUB_AREA_3069 layer (has precomputed "
                        "LAT/LON in WGS84, already used for the parcel confounders in city_endpoints.py)."},
    "denver": {"method": "arcgis", "verified": True,
               "endpoint": "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_PROP_PARCELS_A/FeatureServer/245",
               "note": "CORRECTED vs the prior note in this file: Denver's OWN parcel layer (245, the same one "
                       "city_endpoints.py already uses for confounders) carries SALE_DATE and SALE_PRICE directly "
                       "-- no separate sales table or join needed. (A different WebSearch hit, a 'Sales' table on "
                       "org hQZvdtFxRzJpMtdS, turned out to be Grant County on the same GIS-vendor template, NOT "
                       "Denver -- verified and discarded, same misattribution trap as the earlier Fairfax note.) "
                       "Fields: SCHEDNUM (parcel id), SALE_DATE, SALE_PRICE, ASAL_INSTR (deed/instrument type, "
                       "the only arms-length proxy available -- there is no Washington/Colorado-style numeric "
                       "qualification code on this layer), D_CLASS (property use, e.g. 110-119 SFR, 100s condo). "
                       "Geometry is polygon in EPSG:2877 native but the /query endpoint returns clean WGS84 "
                       "rings when outSR=4326 is requested (verified); centroid the ring for lat/lon."},
    "atlanta": {"method": "BLOCKED_paid",
                "endpoint": "https://www.gsccca.org/learn/search-systems/pt-61-index",
                "note": "VERIFIED BLOCKED. Fulton's own open-data parcel layers (Tax_Parcels_2025 and the "
                        "ARC-hosted FULTON_COUNTY_TAX_PARCELS mirror) carry only assessed/appraised values and "
                        "ownership -- NO sale price or sale date field exists on either (checked the live field "
                        "list). Georgia's actual disclosure source is the PT-61 real-estate transfer-tax "
                        "affidavit, but that is indexed statewide by GSCCCA (Georgia Superior Court Clerks' "
                        "Cooperative Authority), a paid subscription search portal with no free bulk export or "
                        "API. qPublic (Schneider Corp) is a single-parcel lookup UI, not a bulk source, and "
                        "blocks scripted fetches (403). Atlanta must be dropped, scraped from qPublic per-parcel "
                        "(slow, ToS-risk), or sourced via a paid GSCCCA subscription / data vendor."},
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


import os
import time


def _socrata(resource: str, select: str, where: str, max_rows: int | None,
             order: str = ":id", page: int = 50000, label: str = "") -> pd.DataFrame:
    """Paginated Socrata pull. Unauthenticated requests are throttled from a
    shared pool (dev.socrata.com/docs/app-tokens), which stalls large pulls, so
    we send an app token when SOCRATA_APP_TOKEN is set, log progress, and retry
    with backoff on transient errors."""
    token = os.environ.get("SOCRATA_APP_TOKEN")
    frames, offset = [], 0
    while True:
        params = {"$select": select, "$where": where, "$order": order,
                  "$limit": page, "$offset": offset}
        url = resource + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        if token:
            req.add_header("X-App-Token", token)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    chunk = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"  [{label}] page@{offset} error ({e}); retry in {wait}s", flush=True)
                time.sleep(wait)
        if len(chunk) == 0:
            break
        frames.append(chunk)
        offset += len(chunk)
        print(f"  [{label}] pulled {offset:,} rows", flush=True)
        if len(chunk) < page or (max_rows and offset >= max_rows):
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _socrata_once(resource: str, select: str, where: str, limit: int,
                  label: str = "") -> pd.DataFrame:
    """Single streaming Socrata request with a large $limit. One connection the
    server streams is far more robust than many paginated pages, which throttling
    kills mid-run without an app token. Honors SOCRATA_APP_TOKEN; retries."""
    token = os.environ.get("SOCRATA_APP_TOKEN")
    params = {"$select": select, "$where": where, "$limit": limit}
    url = resource + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-App-Token", token)
    for attempt in range(5):
        try:
            print(f"  [{label}] streaming up to {limit:,} rows...", flush=True)
            with urllib.request.urlopen(req, timeout=900) as r:
                df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
            print(f"  [{label}] got {len(df):,} rows", flush=True)
            return df
        except Exception as e:
            if attempt == 4:
                raise
            wait = 5 * 2 ** attempt
            print(f"  [{label}] error ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)


def fetch_chicago(limit: int | None) -> pd.DataFrame:
    sales_url = "https://datacatalog.cookcountyil.gov/resource/wvhk-k5uv.csv"
    univ_url = "https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.csv"
    sales = _socrata_once(
        sales_url, select="pin,sale_price,sale_date",
        where="sale_price > 10000 AND sale_date >= '2015-01-01T00:00:00' "
              "AND sale_filter_less_than_10k = false AND is_multisale = false",
        limit=(limit or 1_500_000), label="cook-sales")
    sales["pin"] = sales["pin"].astype(str).str.zfill(14)
    sales = sales.sort_values("sale_date").drop_duplicates("pin", keep="last")

    # The parcel universe is historic (every parcel x every year); pull one
    # streaming request of the latest year's pin/lat/lon rather than paginating
    # ~2.3M rows, which Socrata throttling makes fragile without a token.
    univ = _socrata_once(
        univ_url, select="pin,lat,lon", where="year='2026.0' AND lat IS NOT NULL",
        limit=(limit * 4 if limit else 2_000_000), label="cook-universe")
    univ["pin"] = univ["pin"].astype(str).str.zfill(14)
    univ["lat"] = pd.to_numeric(univ["lat"], errors="coerce")
    univ["lon"] = pd.to_numeric(univ["lon"], errors="coerce")
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


def fetch_nyc(limit: int | None) -> pd.DataFrame:
    """NYC DOF rolling sales are already in the parcels file (sale_price,
    sale_date, lat/lon by BBL); reshape to the recorded-sales schema."""
    for p in (REPO / "release" / "data" / "nyc" / "parcels.parquet",
              PROC / "nyc_parcels_micro_geo.gpkg"):
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else None
            break
    else:
        raise SystemExit("nyc parcels not found")
    df = df.rename(columns={"latitude": "lat", "longitude": "lon", "parcel_id": "parcel_id"})
    df = df[pd.to_numeric(df["sale_price"], errors="coerce") > 10000].copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce", utc=True)
    df["address"] = ""
    if limit:
        df = df.head(limit)
    return df[["parcel_id", "address", "lat", "lon", "sale_price", "sale_date"]].dropna(subset=["lat", "lon"])


def _download_zip_member(url: str, member_suffix: str, timeout: int = 300) -> bytes:
    """Download a zip and return the bytes of the (single) member ending in
    `member_suffix`. Used for both King County's aqua.kingcounty.gov extracts
    and Maricopa's ArcGIS-Online 'CSV Collection' items, which are both plain
    zips with one data file inside."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        blob = r.read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = [n for n in z.namelist() if n.lower().endswith(member_suffix)]
    if not names:
        raise SystemExit(f"no *{member_suffix} member in {url}: {z.namelist()}")
    return z.read(names[0])


def fetch_seattle(limit: int | None) -> pd.DataFrame:
    """King County Assessor EXTR_RPSale.csv joined to the ArcGIS parcel-address
    layer (which carries precomputed WGS84 LAT/LON; EXTR_Parcel.csv itself has
    no coordinates) on the 10-char Major+Minor PIN.

    Arms-length filter: PropertyClass=='8' (Res-Improved property, per
    EXTR_LookUp LUType=4) and SaleWarning blank (no ratio-study warning code,
    per EXTR_LookUp LUType=7 -- IAAO's "generally invalid" categories, e.g.
    corporate affiliates(11), estate/guardian(12), sheriff/tax sale(14),
    gov't-to-gov't(16), quit claim(18), partial interest(22), forced sale(23),
    related party(51), all map onto specific SaleWarning codes here).
    """
    s = SOURCES["seattle"]
    raw = _download_zip_member(s["endpoint"], "extr_rpsale.csv")
    sales = pd.read_csv(
        io.BytesIO(raw),
        usecols=["Major", "Minor", "DocumentDate", "SalePrice",
                 "PropertyClass", "SaleWarning"],
        dtype={"Major": str, "Minor": str, "PropertyClass": str, "SaleWarning": str},
        encoding="latin-1",
    )
    sales["SaleWarning"] = sales["SaleWarning"].fillna("").str.strip()
    sales["PropertyClass"] = sales["PropertyClass"].str.strip()
    sales = sales[(sales["PropertyClass"] == "8") & (sales["SaleWarning"] == "")
                  & (sales["SalePrice"] > 10000)].copy()
    sales["pin"] = sales["Major"].str.zfill(6) + sales["Minor"].str.zfill(4)
    sales["DocumentDate"] = pd.to_datetime(sales["DocumentDate"], errors="coerce", utc=True)
    sales = sales.dropna(subset=["DocumentDate"]).sort_values("DocumentDate")
    sales = sales.drop_duplicates("pin", keep="last")
    if limit:
        sales = sales.tail(limit)

    parcels = _arcgis_query(
        "https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/"
        "PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0",
        out_fields="PIN,ADDR_FULL,LAT,LON,CTYNAME",
        where="CTYNAME='SEATTLE'", max_rows=None, geom=False)
    parcels = parcels.dropna(subset=["LAT", "LON"]).drop_duplicates("PIN")

    df = sales.merge(parcels, left_on="pin", right_on="PIN", how="inner")
    df = df.rename(columns={"SalePrice": "sale_price", "DocumentDate": "sale_date",
                             "ADDR_FULL": "address", "LAT": "lat", "LON": "lon"})
    return df[["pin", "address", "lat", "lon", "sale_price", "sale_date"]].rename(
        columns={"pin": "parcel_id"})


def fetch_denver(limit: int | None) -> pd.DataFrame:
    """Denver's own open-data parcel layer (ODC_PROP_PARCELS_A/245) carries
    SALE_DATE/SALE_PRICE directly, so no join is needed. There is no numeric
    qualification/validity code on this layer (unlike King County or
    Maricopa); ASAL_INSTR (deed/instrument type) is the only arms-length
    proxy available, so we keep warranty-deed-family instruments and drop
    quit claims, trustee/sheriff/treasurer deeds, gifts, parcel splits, etc.
    D_CLASS restricts to single-family/condo/duplex/triplex/rowhouse (100s-
    190s), excluding larger apartment buildings (200s) and all commercial
    classes.
    """
    ARMS_LENGTH_INSTR = {"WD: WARRANTY", "SW: SPECIAL WARRANTY",
                          "BG: BARGAIN & SALE", "BS: BARGAIN & SALE"}
    layer = SOURCES["denver"]["endpoint"]
    df = _arcgis_query(
        layer, out_fields="SCHEDNUM,SALE_DATE,SALE_PRICE,ASAL_INSTR,D_CLASS,"
                           "SITUS_ADDR_NBR,SITUS_STR_NAME,SITUS_CITY",
        where="SALE_PRICE > 10000 AND SALE_DATE IS NOT NULL", max_rows=limit, geom=True)
    if df.empty:
        raise SystemExit("denver: empty pull -- check field names/where clause")
    df = df[df["ASAL_INSTR"].isin(ARMS_LENGTH_INSTR)]
    df = df[df["D_CLASS"].astype(str).str.match(r"^(10\d|11\d|12[2-4]|13[2-4]|19\d|10S)$")]
    df = df.dropna(subset=["lat", "lon"]).sort_values("SALE_DATE").drop_duplicates("SCHEDNUM", keep="last")
    df["address"] = (df["SITUS_ADDR_NBR"].fillna("").astype(str) + " "
                      + df["SITUS_STR_NAME"].fillna("").astype(str)).str.strip()
    df["SALE_DATE"] = pd.to_datetime(df["SALE_DATE"], errors="coerce", utc=True, unit="ms")
    df = df.rename(columns={"SCHEDNUM": "parcel_id", "SALE_PRICE": "sale_price", "SALE_DATE": "sale_date"})
    return df[["parcel_id", "address", "lat", "lon", "sale_price", "sale_date"]]


def fetch_phoenix(limit: int | None) -> pd.DataFrame:
    """Maricopa Assessor 'Sales Affidavits' (pipe-delimited, no lat/lon) joined
    to the 'Parcel Points' shapefile (has APN + point geometry in EPSG:2868)
    on parcel number.

    Arms-length filter: ASSESSORCODE=='X' (ABSENCE OF REJECT/WARNING/EXEMPT
    CODE -- the R##/W##/A##/B## prefixes are ratio-study reject/warning/
    exempt codes, e.g. R97 foreclosure, R3 related-party, R6 duress, R17
    exchange, A3 gov't, A7 gift, B9 straw man). PROPERTYTYPECODE=='B' is
    Single Family Residence; DEEDDATE has day precision, unlike SALEDATE
    which is month/year only, so DEEDDATE is used for sale_date -- EXCEPT
    that DEEDDATE has occasional single-digit-year typos at the source (e.g.
    parcel 12922025B's actual recorded row is DEEDNUMBER=990808008 (a 1999
    document number) with DEEDDATE_MMDDYYYY='08262099', a straight 1999->2099
    transposition). We cross-check DEEDDATE's year against SALEDATE_MMYYYY's
    year and fall back to the first of the SALEDATE month whenever they
    disagree by more than a year, rather than trusting DEEDDATE blindly.
    """
    import geopandas as gpd

    s = SOURCES["phoenix"]
    raw = _download_zip_member(s["endpoint"], "sales_affidavits.txt")
    sales = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str, low_memory=False, encoding="latin-1")
    sales["SALEPRICE"] = pd.to_numeric(sales["SALEPRICE"], errors="coerce")
    sales = sales[(sales["ASSESSORCODE"] == "X")
                  & (sales["PROPERTYTYPECODE"] == "B")
                  & (sales["SALEPRICE"] > 10000)].copy()

    deed_date = pd.to_datetime(sales["DEEDDATE_MMDDYYYY"], format="%m%d%Y", errors="coerce", utc=True)
    sale_month = pd.to_datetime(sales["SALEDATE_MMYYYY"], format="%m%Y", errors="coerce", utc=True)
    year_disagrees = (deed_date.dt.year - sale_month.dt.year).abs() > 1
    sales["sale_date"] = deed_date.where(~year_disagrees, sale_month)
    sales = sales.dropna(subset=["sale_date"]).sort_values("sale_date")
    sales = sales.drop_duplicates("PARCELNUMBER", keep="last")
    if limit:
        sales = sales.tail(limit)

    points_url = ("https://www.arcgis.com/sharing/rest/content/items/"
                  "dbf139379db946e1b10a2f15672c142d/data")
    raw_zip = urllib.request.urlopen(
        urllib.request.Request(points_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "ParcelPoints.zip"
        tmp.write_bytes(raw_zip)
        pts = gpd.read_file(f"zip://{tmp}!ParcelPoints.shp", columns=["APN"]).to_crs(4326)
    pts["lon"] = pts.geometry.x
    pts["lat"] = pts.geometry.y
    pts = pts.dropna(subset=["lat", "lon"]).drop_duplicates("APN")

    df = sales.merge(pts[["APN", "lat", "lon"]], left_on="PARCELNUMBER", right_on="APN", how="inner")
    df["address"] = (df["SITUSADDRESS"].fillna("").astype(str) + ", "
                      + df["SITUSCITY"].fillna("").astype(str)).str.strip(", ")
    df = df.rename(columns={"PARCELNUMBER": "parcel_id", "SALEPRICE": "sale_price"})
    return df[["parcel_id", "address", "lat", "lon", "sale_price", "sale_date"]]


FETCHERS = {"philadelphia": fetch_philadelphia, "chicago": fetch_chicago,
            "dc": fetch_dc, "nyc": fetch_nyc, "seattle": fetch_seattle,
            "denver": fetch_denver, "phoenix": fetch_phoenix}


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

"""Dallas parcel (DCAD cookie-jar + ZIP + pipe-delim join) + crime (Socrata)."""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from city_downloaders._helpers import DEFAULT_HEADERS, canonicalize_crime, socrata_query, write_parquet  # noqa: E402

SLUG = "dallas"
DCAD_HOME = "https://www.dallascad.org/"
DCAD_LISTING = "https://www.dallascad.org/gisdataproducts.aspx"
DCAD_GEOM = "https://www.dallascad.org/ViewPDFs.aspx?type=3&id=%5C%5CDCAD.ORG%5CWEB%5CWEBDATA%5CWEBFORMS%5CGIS%20PRODUCTS%5CPARCEL_GEOM.zip"
DCAD_DATA = "https://www.dallascad.org/ViewPDFs.aspx?type=3&id=%5C%5CDCAD.ORG%5CWEB%5CWEBDATA%5CWEBFORMS%5CDATA%20PRODUCTS%5CDCAD2026_CURRENT.ZIP"
CRIME_BASE = "https://www.dallasopendata.com/resource/qv6i-rri7.json"


def _dcad_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["Referer"] = DCAD_LISTING
    s.get(DCAD_HOME, timeout=60)
    s.get(DCAD_LISTING, timeout=60)
    return s


def _download_zip(session: requests.Session, url: str, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    print(f"  downloading {url[-60:]}...")
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return out_path


def download_parcels(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _dcad_session()
    geom_zip = _download_zip(s, DCAD_GEOM, out_dir / "PARCEL_GEOM.zip")
    data_zip = _download_zip(s, DCAD_DATA, out_dir / "DCAD2026_CURRENT.zip")
    shp_dir = out_dir / "shp"
    shp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(geom_zip) as zf:
        zf.extractall(shp_dir)
    txt_dir = out_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(data_zip) as zf:
        zf.extractall(txt_dir)
    import geopandas as gpd
    shp_files = list(shp_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"no .shp in {shp_dir}")
    gdf = gpd.read_file(shp_files[0])
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    def _read_pipe(name):
        p = next(iter(txt_dir.rglob(name)), None)
        if p is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(p, sep="|", encoding="latin-1", low_memory=False)
        except Exception as e:
            print(f"    warning: failed to read {name}: {e}", file=sys.stderr)
            return pd.DataFrame()
    info = _read_pipe("Account_Info.txt")
    res = _read_pipe("Res_Detail.txt")
    gdf["Acct"] = gdf["Acct"].astype(str).str.strip()
    if not info.empty:
        info["ACCOUNT_NUM"] = info["ACCOUNT_NUM"].astype(str).str.strip()
        gdf = gdf.merge(info, left_on="Acct", right_on="ACCOUNT_NUM", how="left", suffixes=("", "_info"))
    if not res.empty:
        res["ACCOUNT_NUM"] = res["ACCOUNT_NUM"].astype(str).str.strip()
        gdf = gdf.merge(res, left_on="Acct", right_on="ACCOUNT_NUM", how="left", suffixes=("", "_res"))
    out_path = out_dir / f"{SLUG}_parcels.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def download_crime(out_dir: Path, token: str | None = None) -> Path:
    df = socrata_query(
        CRIME_BASE,
        where="nibrs_crimeagainst IS NOT NULL AND date1 >= '2024-01-01T00:00:00'",
        select="incidentnum, date1, nibrs_crime_category, nibrs_crime, nibrs_crimeagainst, incident_address, zip_code, geocoded_column",
        app_token=token or os.environ.get("SOCRATA_APP_TOKEN"),
    )
    if not df.empty and "geocoded_column" in df.columns:
        df["latitude"] = df["geocoded_column"].apply(lambda x: x.get("latitude") if isinstance(x, dict) else None)
        df["longitude"] = df["geocoded_column"].apply(lambda x: x.get("longitude") if isinstance(x, dict) else None)
    df = canonicalize_crime(df, SLUG, "nibrs_crime_category")
    return write_parquet(df, out_dir / f"{SLUG}_crime.parquet")


if __name__ == "__main__":
    out_dir = REPO_ROOT / "data" / "raw" / SLUG
    print(f"[{SLUG}] downloading parcels...")
    print(f"  -> {download_parcels(out_dir)}")
    print(f"[{SLUG}] downloading crime...")
    print(f"  -> {download_crime(out_dir, token=os.environ.get('SOCRATA_APP_TOKEN'))}")

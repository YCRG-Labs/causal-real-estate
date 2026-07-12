"""Dallas/DCAD county-assessor attributes -> results/assessor/dallas_assessor.parquet.

Texas is a non-disclosure state for sale PRICE, but DCAD structural attributes are
public. Sources (DCAD data products, session-cookie download):
  RES_DETAIL.CSV  (quoted CSV) -> ACCOUNT_NUM, YR_BUILT, TOT_LIVING_AREA_SF,
                                   TOT_MAIN_SF, NUM_BEDROOMS
  ACCOUNT_INFO.CSV             -> ACCOUNT_NUM, PROPERTY_CITY (filter == 'DALLAS')
  PARCEL_GEOM.shp (EPSG:2276)  -> Acct + polygon geometry
Join building attrs to geometry on Acct == ACCOUNT_NUM; reproject to WGS84.
Output columns: lat, lon, bedrooms, bldg_area_sqft, year_built (float, WGS84).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "dallas"
OUT = REPO / "results" / "assessor" / "dallas_assessor.parquet"
GEOM = RAW / "geom" / "PARCEL_GEOM" / "PARCEL_GEOM.shp"
RES = RAW / "data" / "RES_DETAIL.CSV"
INFO = RAW / "data" / "ACCOUNT_INFO.CSV"


def load_geom() -> gpd.GeoDataFrame:
    g = gpd.read_file(GEOM, columns=["Acct"])
    print(f"  parcel geom: {len(g)} (crs {g.crs})")
    g = g.to_crs(4326)
    g = g.dropna(subset=["geometry"])
    g = g[~g.geometry.is_empty]
    rep = g.geometry.representative_point()
    out = pd.DataFrame({"Acct": g["Acct"].astype(str).str.strip(),
                        "lat": rep.y.values, "lon": rep.x.values})
    return out.drop_duplicates("Acct")


def load_res() -> pd.DataFrame:
    info = pd.read_csv(INFO, dtype=str, usecols=["ACCOUNT_NUM", "PROPERTY_CITY"])
    info["PROPERTY_CITY"] = info["PROPERTY_CITY"].astype(str).str.strip().str.upper()
    dallas = set(info.loc[info["PROPERTY_CITY"] == "DALLAS", "ACCOUNT_NUM"].astype(str).str.strip())
    print(f"  City-of-Dallas accounts: {len(dallas)}")

    rd = pd.read_csv(RES, dtype=str,
                     usecols=["ACCOUNT_NUM", "YR_BUILT", "TOT_MAIN_SF",
                              "TOT_LIVING_AREA_SF", "NUM_BEDROOMS"])
    rd["ACCOUNT_NUM"] = rd["ACCOUNT_NUM"].astype(str).str.strip()
    rd = rd[rd["ACCOUNT_NUM"].isin(dallas)].copy()
    for c in ["YR_BUILT", "TOT_MAIN_SF", "TOT_LIVING_AREA_SF", "NUM_BEDROOMS"]:
        rd[c] = pd.to_numeric(rd[c], errors="coerce")
    # keep the primary structure per account (largest living area)
    rd["_area"] = rd["TOT_LIVING_AREA_SF"].fillna(0)
    rd = rd.sort_values("_area").drop_duplicates("ACCOUNT_NUM", keep="last")
    print(f"  residential rows in Dallas: {len(rd)}")

    rd["bldg_area_sqft"] = rd["TOT_LIVING_AREA_SF"]
    m = (rd["bldg_area_sqft"].isna()) | (rd["bldg_area_sqft"] <= 0)
    rd.loc[m, "bldg_area_sqft"] = rd.loc[m, "TOT_MAIN_SF"]
    return rd[["ACCOUNT_NUM", "YR_BUILT", "bldg_area_sqft", "NUM_BEDROOMS"]]


def main() -> None:
    geom = load_geom()
    res = load_res()
    df = res.merge(geom, left_on="ACCOUNT_NUM", right_on="Acct", how="inner")
    print(f"  joined to geometry: {len(df)} of {len(res)} "
          f"({100*len(df)/max(len(res),1):.1f}%)")

    beds = df["NUM_BEDROOMS"].astype(float)
    year = df["YR_BUILT"].astype(float)
    sqft = df["bldg_area_sqft"].astype(float)
    beds[(beds <= 0) | (beds > 30)] = np.nan
    year[(year < 1850) | (year > 2026)] = np.nan
    sqft[sqft <= 0] = np.nan

    out = pd.DataFrame({
        "lat": df["lat"].astype(float).values,
        "lon": df["lon"].astype(float).values,
        "bedrooms": beds.values,
        "bldg_area_sqft": sqft.values,
        "year_built": year.values,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT}")
    for c in ["lat", "lon", "bedrooms", "bldg_area_sqft", "year_built"]:
        print(f"  {c:16s} coverage {100*out[c].notna().mean():5.1f}%  median {out[c].median()}")


if __name__ == "__main__":
    main()

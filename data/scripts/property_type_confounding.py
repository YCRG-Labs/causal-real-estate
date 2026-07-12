"""Does the text direction price semantics, or unmodelled property type?

The published confounder block takes bedrooms, bldg_area_sqft, lot_area_sqft and
year_built from the nearest assessor parcel, never encodes property_type, and
zero-fills any remaining NaN without a missingness indicator. Nine of the twelve
markets have no parcel file at all, so that block silently collapses to latitude
and longitude.

Every market does carry the listing's OWN structured attributes. This script uses
those instead, and estimates the same partially-linear DML effect of the pooled
text PC1 on log list price under three nested confounder sets:

    A  lat, lon, beds, baths, sqft, year_built     NaN -> 0, as the pipeline does
    B  A, but median-imputed with missingness indicators
    C  B + property_type dummies

No rows are dropped, so n and sd(T) are identical across A, B and C and the
comparison is free of the rescaling artifact that a drop-the-land test carries.
If the text direction is priced for semantic content, theta should be stable from
A to C. If it is standing in for property type, theta should collapse at C.

    python data/scripts/property_type_confounding.py --all_12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from replications.compare_to_dml import run_dml

POOLED_CSV = REPO / "results" / "replications" / "pooled_pca_treatment.csv"
OUT_JSON = REPO / "results" / "property_type_confounding.json"

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

STRUCT = ["beds", "baths", "sqft", "year_built"]


def _load(city: str):
    emb = pd.read_parquet(REPO / "data" / "processed" / f"{city}_embeddings.parquet")
    emb = emb.assign(listing_id=np.arange(len(emb)))

    pool = pd.read_csv(POOLED_CSV)
    pc = pool[pool.city == city][["listing_id", "treatment_z"]]
    df = emb.merge(pc, on="listing_id", how="inner")

    price = pd.to_numeric(df["price"], errors="coerce")
    ok = price.notna() & (price > 0) & df.latitude.notna() & df.longitude.notna()
    df, price = df[ok].reset_index(drop=True), price[ok].reset_index(drop=True)
    return df, np.log(price.to_numpy(float))


def _designs(df: pd.DataFrame):
    lat = pd.to_numeric(df.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df.longitude, errors="coerce").to_numpy(float)
    S = df[STRUCT].apply(pd.to_numeric, errors="coerce")

    # A: zero-fill, no indicators, no property type (what the pipeline does)
    A = np.column_stack([lat, lon, np.nan_to_num(S.to_numpy(float), nan=0.0)])

    # B: median impute + missingness indicators
    miss = S.isna().to_numpy(float)
    Simp = S.fillna(S.median()).to_numpy(float)
    B = np.column_stack([lat, lon, Simp, miss])

    # C: + property_type dummies
    pt = pd.get_dummies(df.property_type.astype(str), prefix="pt", drop_first=True)
    C = np.column_stack([B, pt.to_numpy(float)])
    return {"A_zerofill": A, "B_impute_plus_missing": B, "C_plus_property_type": C}, pt.shape[1]


def _theta(T, X, Y):
    X = StandardScaler().fit_transform(X)
    r = run_dml(T.reshape(-1, 1), X, Y, label="x", ci_method="if", n_boot=None,
                use_ridge=True, seed=42, n_pca=1)
    if r is None:
        return None
    return {"theta": float(r.theta), "abs_theta": abs(float(r.theta)),
            "se": float(r.se), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high),
            "covers_zero": bool(r.ci_low < 0 < r.ci_high)}


def run_city(city: str) -> dict:
    df, Y = _load(city)
    T = df.treatment_z.to_numpy(float)
    designs, n_pt = _designs(df)
    pt = df.property_type.astype(str)
    nonres = pt.str.contains(r"land|lot|vacant|parking", case=False, na=False)

    out = {"city": city, "n": len(Y), "n_property_type_dummies": n_pt,
           "pct_non_residential": float(100 * nonres.mean())}
    for name, X in designs.items():
        out[name] = _theta(T, X, Y)
    a = out["A_zerofill"]["abs_theta"]
    c = out["C_plus_property_type"]["abs_theta"]
    out["pct_attenuation_A_to_C"] = float(100 * (1 - c / a)) if a else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    args = ap.parse_args()
    cities = ALL_12 if args.all_12 else [args.city]

    rows = []
    for c in cities:
        try:
            rows.append(run_city(c))
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: FAILED {type(e).__name__}: {e}")
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    hdr = f"{'city':14s}{'n':>7s}{'%nonres':>9s}{'A zerofill':>12s}{'B +missing':>12s}{'C +proptype':>13s}{'A->C':>8s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["pct_non_residential"]):
        star = "*" if r["C_plus_property_type"]["covers_zero"] else " "
        print(f"{r['city']:14s}{r['n']:7d}{r['pct_non_residential']:8.1f}%"
              f"{r['A_zerofill']['abs_theta']:12.4f}{r['B_impute_plus_missing']['abs_theta']:12.4f}"
              f"{r['C_plus_property_type']['abs_theta']:12.4f}{star}{r['pct_attenuation_A_to_C']:7.1f}%")
    print("\n* = 95% CI covers zero under spec C")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

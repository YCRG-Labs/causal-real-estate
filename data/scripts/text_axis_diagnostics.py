"""What is the pooled text direction actually measuring?

Three checks on the treatment itself, before any causal machinery runs.

  1. Raw HTML entities. The descriptions were embedded without html.unescape, so
     `&amp;`, `&mdash;` and `&rsquo;` are in the token stream. The rate varies from
     20% of listings (Chicago) to 75% (Seattle), which is a scrape artifact, not a
     property of the housing market.

  2. Description length. PC1 correlates with raw character count at r up to +0.58.
     Across the twelve markets the strength of that correlation predicts the
     published effect size at r = +0.67.

  3. Property type acts through length. Non-residential listings carry
     descriptions 2.5 to 3.6 times shorter than residential ones, so the
     land-versus-house contrast is largely a long-versus-short-text contrast.

The estimation check runs the same partially-linear DML under nested controls, on
listing-level confounders, dropping no rows, so n and sd(T) are constant.

    python data/scripts/text_axis_diagnostics.py --all_12
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
from property_type_confounding import _load, STRUCT, ALL_12

OUT_JSON = REPO / "results" / "text_axis_diagnostics.json"
ENTITY = r"&[a-z]+;|&#\d+;"
NON_RES = r"land|lot|vacant|parking"


def _design(df, length: bool, ptype: bool) -> np.ndarray:
    lat = pd.to_numeric(df.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df.longitude, errors="coerce").to_numpy(float)
    S = df[STRUCT].apply(pd.to_numeric, errors="coerce")
    X = np.column_stack([lat, lon,
                         S.fillna(S.median()).to_numpy(float),
                         S.isna().to_numpy(float)])
    if length:
        L = np.log1p(df.description.astype(str).str.len().to_numpy(float))
        X = np.column_stack([X, L, L ** 2])
    if ptype:
        d = pd.get_dummies(df.property_type.astype(str), drop_first=True)
        X = np.column_stack([X, d.to_numpy(float)])
    return X


def _theta(T, X, Y):
    r = run_dml(T.reshape(-1, 1), StandardScaler().fit_transform(X), Y, label="x",
                ci_method="if", n_boot=None, use_ridge=True, seed=42, n_pca=1)
    return {"abs_theta": abs(float(r.theta)), "se": float(r.se),
            "covers_zero": bool(r.ci_low < 0 < r.ci_high)}


def run_city(city: str) -> dict:
    df, Y = _load(city)
    T = df.treatment_z.to_numpy(float)
    desc = df.description.astype(str)
    L = desc.str.len().to_numpy(float)
    ent = desc.str.contains(ENTITY, regex=True, na=False).to_numpy(float)
    nr = df.property_type.astype(str).str.contains(NON_RES, case=False, na=False).to_numpy()

    out = {
        "city": city, "n": len(Y),
        "entity_rate": float(ent.mean()),
        "corr_entity_pc1": float(np.corrcoef(ent, T)[0, 1]),
        "corr_length_pc1": float(np.corrcoef(L, T)[0, 1]),
        "median_len_non_residential": float(np.median(L[nr])) if nr.any() else None,
        "median_len_residential": float(np.median(L[~nr])),
        "pct_non_residential": float(100 * nr.mean()),
    }
    for name, (l, p) in {"base": (0, 0), "plus_length": (1, 0),
                         "plus_property_type": (0, 1), "plus_both": (1, 1)}.items():
        out[name] = _theta(T, _design(df, bool(l), bool(p)), Y)
    a, b = out["base"]["abs_theta"], out["plus_length"]["abs_theta"]
    out["pct_attenuation_from_length"] = float(100 * (1 - b / a)) if a else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    args = ap.parse_args()
    cities = ALL_12 if args.all_12 else [args.city]

    rows = [run_city(c) for c in cities]
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    hdr = (f"{'city':14s}{'ent%':>7s}{'r(len,PC1)':>12s}{'len n-res':>11s}"
           f"{'len res':>9s}{'base':>9s}{'+len':>9s}{'+both':>9s}{'len drop':>10s}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['city']:14s}{100*r['entity_rate']:6.0f}%{r['corr_length_pc1']:+12.3f}"
              f"{r['median_len_non_residential'] or 0:11.0f}{r['median_len_residential']:9.0f}"
              f"{r['base']['abs_theta']:9.4f}{r['plus_length']['abs_theta']:9.4f}"
              f"{r['plus_both']['abs_theta']:9.4f}{r['pct_attenuation_from_length']:9.1f}%")

    x = np.array([r["corr_length_pc1"] for r in rows])
    y = np.array([r["base"]["abs_theta"] for r in rows])
    if len(rows) > 2:
        print(f"\ncorr( corr(length,PC1) , |theta| ) across {len(rows)} markets = "
              f"{np.corrcoef(x, y)[0, 1]:+.3f}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

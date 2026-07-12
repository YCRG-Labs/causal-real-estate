"""Spatial cluster-bootstrap inference for the clean assessor-arm effect.

Primary fix for the too-narrow IID influence-function SEs. For each market, on the
single-unit clean spec (lat/lon + exogenous assessor property + census demographics
+ length, winsorized), we:

  1. assign every listing to its census tract,
  2. tract cluster bootstrap (B): resample whole tracts with replacement, refit both
     ridge nuisances with cross-fitting, recompute theta -> percentile CI,
  3. report the Moulton factor (boot SE / IID IF SE) and a Cinelli-Hazlett robustness
     value computed on the HONEST (bootstrap) SE.

The pooled-12 PC1 treatment is held fixed across resamples: it is estimated on the
full ~69k-row pool and is empirically stable (cos 0.9999 across resamples), so its
per-market variance contribution is negligible and the generated-regressor concern
is addressed by that stability rather than by re-pooling. A specification curve over
the property-attribute source (assessor / listing / both) is reported alongside.

    python data/scripts/fork_bootstrap.py --B 500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from fork_clean import (imp, winsor, sanitize, pooled_treatment, assessor_property,
                        census_demo, ALL_12, FULL)  # reuse machinery
from causal_inference import load_analysis_data, get_features_and_target

CENSUS = REPO / "results" / "census_bg"
OUT = REPO / "results" / "fork_bootstrap.json"
ALPHAS = np.logspace(-3, 3, 25)


def dml_theta(T, X, Y, seed=0):
    """Ridge partialling-out DML theta on a fixed treatment T (PC1 held fixed)."""
    n = len(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Yr, Tr = np.empty(n), np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        Yr[te] = Y[te] - RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]).predict(X[te])
        Tr[te] = T[te] - RidgeCV(alphas=ALPHAS).fit(X[tr], T[tr]).predict(X[te])
    denom = float(np.mean(Tr * Tr))
    if denom < 1e-12:
        return np.nan, np.nan
    theta = float(np.mean(Tr * Yr)) / denom
    psi = (Yr - theta * Tr) * Tr / denom
    se_if = float(np.sqrt(np.var(psi, ddof=1) / n))
    return theta, se_if


def rv(theta, se, n, k_x, q=1.0):
    if se == 0 or np.isnan(se):
        return float("nan")
    t = abs(theta) / se
    df = max(n - k_x - 1, 1)
    f2 = (q * t) ** 2 / df
    return 0.5 * (math.sqrt(f2 ** 2 + 4 * f2) - f2)


def tracts_for(city, lat, lon):
    bg = gpd.read_file(CENSUS / f"{city}_census_bg.gpkg", layer="bg")
    latf = np.nan_to_num(lat, nan=np.nanmedian(lat))
    lonf = np.nan_to_num(lon, nan=np.nanmedian(lon))
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lonf, latf), crs=4326)
    j = gpd.sjoin(pts, bg[["GEOID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    tr = j["GEOID"].astype(str).str.slice(0, 11)          # 11-digit tract
    # unmatched -> nearest block-group centroid so every row has a cluster
    miss = tr.isin(["", "nan", "None"]) | tr.isna()
    if miss.any():
        cent = bg.to_crs(3857).geometry.centroid
        tree = cKDTree(np.column_stack([cent.x.values, cent.y.values]))
        p3 = gpd.GeoSeries(gpd.points_from_xy(lonf, latf), crs=4326).to_crs(3857)
        _, idx = tree.query(np.column_stack([p3.x.values, p3.y.values]))
        near = bg["GEOID"].astype(str).str.slice(0, 11).to_numpy()[idx]
        tr = tr.mask(miss.values, pd.Series(near, index=tr.index))
    return tr.to_numpy()


def build(city, T):
    emb, parc = load_analysis_data(city)
    emb = sanitize(emb)
    _t, _cf, Y, meta = get_features_and_target(emb, parc, drop_mismatched_crime=True)
    v = np.asarray(meta["valid_mask"], bool)
    lst = emb.iloc[np.where(v)[0]]
    Tv = T[city][v]
    L = pd.read_parquet(FULL / f"{city}_reembed.parquet").sort_values("row")["log_len"].to_numpy()[v]
    lat = pd.to_numeric(lst.latitude, errors="coerce").to_numpy()
    lon = pd.to_numeric(lst.longitude, errors="coerce").to_numpy()
    pt = lst.property_type.astype(str).str.lower()
    single = (pt.str.contains("single") | pt.str.contains("town")).to_numpy()

    lb = pd.to_numeric(lst.beds, errors="coerce").to_numpy()
    ls = pd.to_numeric(lst.sqft, errors="coerce").to_numpy()
    ly = pd.to_numeric(lst.year_built, errors="coerce").to_numpy()
    ab, asq, ay = assessor_property(city, lat, lon)
    demo = winsor(census_demo(city, lat, lon))
    latlon = np.column_stack([lat, lon])
    props = {
        "assessor": winsor(imp(np.column_stack([ab, asq, ay]))),
        "listing": winsor(imp(np.column_stack([lb, ls, ly]))),
        "both": winsor(imp(np.column_stack([ab, asq, ay, lb, ls, ly]))),
    }
    tract = tracts_for(city, lat, lon)
    fin = np.isfinite(Y) & np.isfinite(lat) & np.isfinite(lon)
    m = fin & single
    return dict(Tv=Tv, Y=Y, L=L, latlon=latlon, demo=demo, props=props,
                tract=tract, m=m)


def design(d, prop_key):
    m = d["m"]
    X = np.column_stack([d["latlon"][m], d["props"][prop_key][m], d["demo"][m],
                         d["L"][m], d["L"][m] ** 2])
    return np.nan_to_num(StandardScaler().fit_transform(X), nan=0.0), d["Tv"][m], d["Y"][m], d["tract"][m]


def cluster_boot(T, X, Y, tract, B, seed=0):
    rng = np.random.default_rng(seed)
    tr_ids = np.unique(tract)
    G = len(tr_ids)
    idx_by = {t: np.where(tract == t)[0] for t in tr_ids}
    thetas = np.empty(B)
    for b in range(B):
        samp = rng.choice(tr_ids, size=G, replace=True)
        idx = np.concatenate([idx_by[t] for t in samp])
        thetas[b], _ = dml_theta(T[idx], X[idx], Y[idx], seed=b + 1)
    return thetas[np.isfinite(thetas)], G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--cities", nargs="*")
    args = ap.parse_args()
    T = pooled_treatment()
    from fork_clean import PARCEL_COLS, ASSR
    have = [c for c in (args.cities or ALL_12)
            if (ASSR / f"{c}_assessor.parquet").exists() or c in PARCEL_COLS]

    rows = []
    print(f"{'city':13s}{'n':>6s}{'G':>5s}{'theta':>8s}{'IF se':>8s}{'boot se':>8s}"
          f"{'Moult':>7s}{'95% CI (cluster)':>20s}{'RV':>7s}")
    print("-" * 84)
    for c in have:
        d = build(c, T)
        X, Tt, Yy, tract = design(d, "assessor")
        theta, se_if = dml_theta(Tt, X, Yy, seed=0)
        boots, G = cluster_boot(Tt, X, Yy, tract, args.B)
        se_boot = float(boots.std(ddof=1))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        covers0 = bool(lo < 0 < hi)
        moult = se_boot / se_if if se_if else float("nan")
        rv_boot = rv(theta, se_boot, len(Yy), X.shape[1])
        # specification curve across property source
        spec = {}
        for k in ("assessor", "listing", "both"):
            Xk, Tk, Yk, trk = design(d, k)
            th_k, _ = dml_theta(Tk, Xk, Yk, seed=0)
            bk, _ = cluster_boot(Tk, Xk, Yk, trk, max(args.B // 2, 200))
            spec[k] = {"theta": abs(th_k),
                       "ci": [float(np.quantile(bk, 0.025)), float(np.quantile(bk, 0.975))]}
        rows.append({"city": c, "n": int(len(Yy)), "G": G,
                     "theta": abs(theta), "se_if": se_if, "se_boot": se_boot,
                     "moulton": moult, "ci_low": float(lo), "ci_high": float(hi),
                     "covers_zero": covers0, "rv_boot": rv_boot, "spec_curve": spec})
        print(f"{c:13s}{len(Yy):6d}{G:5d}{abs(theta):8.3f}{se_if:8.3f}{se_boot:8.3f}"
              f"{moult:7.2f}   [{lo:+.3f},{hi:+.3f}]{'*' if covers0 else ' '}{rv_boot:7.3f}")
    OUT.write_text(json.dumps(rows, indent=2))
    print("\n* = cluster CI covers zero. Moulton = boot_se/IF_se. RV on the honest (boot) SE.")
    sig = [r["city"] for r in rows if not r["covers_zero"]]
    print(f"significant after spatial clustering: {len(sig)}/{len(rows)}  ({', '.join(sig)})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

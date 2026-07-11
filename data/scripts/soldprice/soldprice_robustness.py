from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from numpy.linalg import svd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))
sys.path.insert(0, str(REPO / "data" / "scripts" / "soldprice"))
from soldprice_dml import design, DEMO, ALPHAS, PANEL, EMB, CONF, CITIES
from spatial_se import (spatial_hac_se, salerno_jackknife_hac,
                        ibragimov_muller_self_normalized_ci)

OUT = PANEL / "soldprice_robustness.json"
DIM = 768
UTM = {"boston": 32619, "sf": 32610, "dc": 32618, "philadelphia": 32618,
       "chicago": 32616, "seattle": 32610, "denver": 32613, "atlanta": 32616,
       "portland": 32610, "phoenix": 32612}


def dml_residuals(T, X, Y, seed=0):
    n = len(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Yr, Tr, fold = np.empty(n), np.empty(n), np.empty(n, int)
    for f, (tr, te) in enumerate(kf.split(np.arange(n))):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        Yr[te] = Y[te] - RidgeCV(alphas=ALPHAS).fit(Xtr, Y[tr]).predict(Xte)
        Tr[te] = T[te] - RidgeCV(alphas=ALPHAS).fit(Xtr, T[tr]).predict(Xte)
        fold[te] = f
    d = float(np.mean(Tr * Tr))
    theta = float(np.mean(Tr * Yr)) / d
    psi = (Yr - theta * Tr) * Tr / d
    return theta, psi, fold


def to_utm(city, coords):
    g = gpd.GeoSeries(gpd.points_from_xy(coords[:, 1], coords[:, 0]),
                      crs=4326).to_crs(UTM[city])
    return np.column_stack([g.x.values, g.y.values])


def spatial_ses(arm="assessor"):
    rows = []
    for city in CITIES:
        got = design(city, arm)
        if got is None:
            continue
        X, T, Y, tract, coords = got
        theta, psi, fold = dml_residuals(T, X, Y)
        cu = to_utm(city, coords)
        hac = float(spatial_hac_se(psi, cu, bandwidth_quantile=0.10))
        sjk = salerno_jackknife_hac(psi, fold, cu, bandwidth_quantile=0.10)
        im = ibragimov_muller_self_normalized_ci(psi + theta, fold)
        rows.append({"city": city, "theta": theta, "n": int(len(Y)),
                     "se_conley_hac": hac, "se_salerno_jk": float(sjk["se"]),
                     "im_ci": [float(im["ci_low"]), float(im["ci_high"])],
                     "im_se": float(im["se"]),
                     "conley_excludes_0": bool(abs(theta) > 1.96 * hac),
                     "im_excludes_0": bool(im["ci_low"] * im["ci_high"] > 0)})
        print(f"  spatial {city:12s} theta={theta:+.3f} conleyHAC={hac:.3f} "
              f"salernoJK={float(sjk['se']):.3f} IM=[{im['ci_low']:+.3f},{im['ci_high']:+.3f}]",
              flush=True)
    return rows


def egami_split(arm="assessor", seed=0):
    rng = np.random.default_rng(seed)
    train_mask, test_mask, city_mean = {}, {}, {}
    train_blocks = []
    for c in CITIES:
        e = pd.read_parquet(EMB / f"{c}_emb.parquet",
                            columns=[f"emb_{i}" for i in range(DIM)]).to_numpy(np.float64)
        m = rng.random(len(e)) < 0.5
        train_mask[c], test_mask[c] = m, ~m
        mu = e[m].mean(0, keepdims=True)
        city_mean[c] = mu
        train_blocks.append(e[m] - mu)
    direction = svd(np.vstack(train_blocks), full_matrices=False)[2][0]
    del train_blocks
    if direction.sum() < 0:
        direction = -direction

    rows = []
    for c in CITIES:
        e = pd.read_parquet(EMB / f"{c}_emb.parquet",
                            columns=[f"emb_{i}" for i in range(DIM)]).to_numpy(np.float64)
        score = (e - city_mean[c]) @ direction
        te = test_mask[c]
        z = np.zeros(len(e))
        s = score[te]
        z[te] = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        got = design(c, arm, T_override=z, extra_keep=te)
        if got is None:
            continue
        X, T, Y, tract, _ = got
        from soldprice_dml import dml_theta
        theta, se = dml_theta(T, X, Y, seed=0)
        rows.append({"city": c, "n_test": int(len(Y)), "theta_egami": theta, "se_egami": se})
        print(f"  egami   {c:12s} n_test={len(Y):6d} theta_egami={theta:+.3f} (se {se:.3f})",
              flush=True)
    return rows


def pc1_time_diagnostic():
    rows = []
    for c in CITIES:
        T = pd.read_parquet(EMB / f"{c}_treatment.parquet")["treatment"].to_numpy()
        q = pd.read_parquet(PANEL / f"{c}_panel.parquet", columns=["sold_quarter"])
        D = pd.get_dummies(q["sold_quarter"], drop_first=True).to_numpy(float)
        r2 = LinearRegression().fit(D, T).score(D, T)
        rows.append({"city": c, "pc1_on_quarter_r2": float(r2)})
        print(f"  pc1~quarter {c:12s} R2={r2:.4f}", flush=True)
    return rows


def main():
    print("=== spatial-robust SEs (assessor arm) ===", flush=True)
    spatial = spatial_ses("assessor")
    print("\n=== Egami 50/50 split-sample (assessor arm) ===", flush=True)
    egami = egami_split("assessor")
    print("\n=== PC1 ~ sale-quarter calendar diagnostic ===", flush=True)
    diag = pc1_time_diagnostic()
    OUT.write_text(json.dumps({"spatial": spatial, "egami": egami,
                               "pc1_time": diag}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

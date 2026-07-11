from __future__ import annotations

import argparse
import gc
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
PROC = REPO / "data" / "processed"
CENSUS = REPO / "results" / "census_bg"
OUT = REPO / "results" / "truncation_calibration.json"
CITIES = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]
STRUCT = ["beds", "baths", "sqft", "year_built"]
DEMO = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60"]
ALPHAS = np.logspace(-3, 3, 25)
TRUNC = 699


def city_frame(c, per_city, rng):
    lst = pd.read_parquet(PROC / f"{c}_embeddings.parquet",
                          columns=["latitude", "longitude", "price", "property_type",
                                   "description"] + STRUCT)
    lat = pd.to_numeric(lst.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(lst.longitude, errors="coerce").to_numpy(float)
    Y = np.log(pd.to_numeric(lst.price, errors="coerce").to_numpy(float))
    desc = lst.description.fillna("").astype(str)
    ok = (np.isfinite(Y) & np.isfinite(lat) & np.isfinite(lon)
          & (desc.str.len().to_numpy() >= 50))
    idx = np.where(ok)[0]
    if len(idx) > per_city:
        idx = np.sort(rng.choice(idx, per_city, replace=False))
    lst = lst.iloc[idx]
    lat, lon, Y = lat[idx], lon[idx], Y[idx]
    desc = desc.iloc[idx].tolist()
    pts = gpd.GeoDataFrame(lst.assign(_r=np.arange(len(lst))),
                           geometry=gpd.points_from_xy(lon, lat), crs=4326)
    bg = gpd.read_file(CENSUS / f"{c}_census_bg.gpkg", layer="bg")
    j = gpd.sjoin(pts, bg[DEMO + ["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_values("_r")
    S = j[STRUCT].apply(pd.to_numeric, errors="coerce")
    D = j[DEMO].apply(pd.to_numeric, errors="coerce")
    X = np.column_stack([lat, lon, S.fillna(S.median()).to_numpy(float),
                         S.isna().to_numpy(float), D.fillna(D.median()).to_numpy(float)])
    return desc, np.nan_to_num(X), Y


def apply_truncation_correction(theta_trunc, se_trunc, calib_path=None):
    c = json.loads(Path(calib_path or OUT).read_text())
    lam, lam_se = c["lambda_pooled"], c["lambda_pooled_se"]
    theta_c = theta_trunc / lam
    var_c = (se_trunc / lam) ** 2 + (theta_trunc / lam ** 2) ** 2 * lam_se ** 2
    return theta_c, float(np.sqrt(var_c)), 1 / lam


def dml_theta(T, X, Y, seed=0):
    n = len(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Yr, Tr = np.empty(n), np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        Yr[te] = Y[te] - RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]).predict(X[te])
        Tr[te] = T[te] - RidgeCV(alphas=ALPHAS).fit(X[tr], T[tr]).predict(X[te])
    d = float(np.mean(Tr * Tr))
    return float(np.mean(Tr * Yr)) / d if d > 1e-12 else np.nan


def pooled_pc1(embd, sizes):
    Xc, i = [], 0
    for _, n in sizes:
        b = embd[i:i + n]
        Xc.append(b - b.mean(0, keepdims=True))
        i += n
    Xc = np.vstack(Xc)
    d = svd(Xc, full_matrices=False)[2][0]
    if d.sum() < 0:
        d = -d
    sc = Xc @ d
    out, i = {}, 0
    for c, n in sizes:
        s = sc[i:i + n]
        out[c] = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        i += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_city", type=int, default=3000)
    ap.add_argument("--B", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    data, Efull, Etrunc, sizes = {}, [], [], []
    for c in CITIES:
        desc, X, Y = city_frame(c, args.per_city, rng)
        Ef = m.encode(desc, batch_size=32, show_progress_bar=False).astype(np.float64)
        Et = m.encode([d[:TRUNC] for d in desc], batch_size=32,
                      show_progress_bar=False).astype(np.float64)
        data[c] = {"X": StandardScaler().fit_transform(X), "Y": Y, "n": len(Y)}
        Efull.append(Ef)
        Etrunc.append(Et)
        sizes.append((c, len(Y)))
        print(f"embedded {c} (n={len(Y)})", flush=True)
    del m
    gc.collect()
    Efull = np.vstack(Efull)
    Etrunc = np.vstack(Etrunc)
    Tf = pooled_pc1(Efull, sizes)
    Tt = pooled_pc1(Etrunc, sizes)
    for c in CITIES:
        data[c]["Tf"] = Tf[c]
        data[c]["Tt"] = Tt[c]

    rows = []
    for c in CITIES:
        d = data[c]
        tf = dml_theta(d["Tf"], d["X"], d["Y"], seed=0)
        tt = dml_theta(d["Tt"], d["X"], d["Y"], seed=0)
        lam = tt / tf if tf != 0 else np.nan
        rc = LinearRegression().fit(np.column_stack([d["Tt"], d["X"]]), d["Tf"])
        That = rc.predict(np.column_stack([d["Tt"], d["X"]]))
        t_cal = dml_theta(That, d["X"], d["Y"], seed=0)
        n = d["n"]
        lam_b = []
        for b in range(args.B):
            bi = np.random.default_rng(b + 1).integers(0, n, n)
            tfb = dml_theta(d["Tf"][bi], d["X"][bi], d["Y"][bi], seed=b + 1)
            ttb = dml_theta(d["Tt"][bi], d["X"][bi], d["Y"][bi], seed=b + 1)
            if tfb and abs(tfb) > 1e-6:
                lam_b.append(ttb / tfb)
        lam_b = np.array([x for x in lam_b if np.isfinite(x)])
        rows.append({"city": c, "n": n, "theta_full": tf, "theta_trunc": tt,
                     "theta_calibrated": t_cal, "lambda": lam,
                     "lambda_se": float(lam_b.std(ddof=1)),
                     "lambda_ci": [float(np.quantile(lam_b, .025)),
                                   float(np.quantile(lam_b, .975))],
                     "attenuation_pct": 100 * (1 - lam)})
        print(f"  {c:13s} theta_full={tf:+.4f} theta_trunc={tt:+.4f} "
              f"lambda={lam:.3f} atten={100*(1-lam):4.1f}% cal={t_cal:+.4f}", flush=True)

    lams = np.array([r["lambda"] for r in rows])
    ses = np.array([r["lambda_se"] for r in rows])
    w = 1 / ses ** 2
    lam_pool = float(np.sum(w * lams) / np.sum(w))
    lam_pool_se = float(np.sqrt(1 / np.sum(w)))

    loco = []
    for i, r in enumerate(rows):
        mask = np.arange(len(rows)) != i
        lp = float(np.sum(w[mask] * lams[mask]) / np.sum(w[mask]))
        pred_full = r["theta_trunc"] / lp
        loco.append({"city": r["city"], "actual_full": r["theta_full"],
                     "corrected": pred_full, "lambda_minus_c": lp,
                     "rel_err_pct": 100 * (pred_full - r["theta_full"]) / abs(r["theta_full"])})
    loco_mae = float(np.mean([abs(x["rel_err_pct"]) for x in loco]))

    tf = np.array([r["theta_full"] for r in rows])
    tt = np.array([r["theta_trunc"] for r in rows])
    sig = np.abs(tf) >= 0.10
    lam_sig = lams[sig]
    sig_cities = [rows[i]["city"] for i in range(len(rows)) if sig[i]]
    loco_sig_abs, loco_sig_rel = [], []
    sidx = np.where(sig)[0]
    for i in sidx:
        others = [j for j in sidx if j != i]
        lp = float(lams[others].mean())
        corr = tt[i] / lp
        loco_sig_abs.append(abs(corr - tf[i]))
        loco_sig_rel.append(abs(100 * (corr - tf[i]) / tf[i]))

    out = {"per_city_n": args.per_city, "B": args.B,
           "lambda_pooled": lam_pool, "lambda_pooled_se": lam_pool_se,
           "attenuation_pooled_pct": 100 * (1 - lam_pool),
           "correction_factor": 1 / lam_pool,
           "signal_cities": sig_cities,
           "lambda_signal_mean": float(lam_sig.mean()),
           "lambda_signal_sd": float(lam_sig.std(ddof=1)),
           "lambda_signal_range": [float(lam_sig.min()), float(lam_sig.max())],
           "loco_signal_mae_abs": float(np.mean(loco_sig_abs)),
           "loco_signal_mae_rel_pct": float(np.mean(loco_sig_rel)),
           "loco_all_mae_pct": loco_mae,
           "lambda_range_all": [float(lams.min()), float(lams.max())],
           "markets": rows, "loco": loco}
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"signal markets ({len(sig_cities)}): lambda {lam_sig.mean():.3f}"
          f"+/-{lam_sig.std(ddof=1):.3f}, LOCO MAE {np.mean(loco_sig_abs):.4f} theta-units"
          f" / {np.mean(loco_sig_rel):.1f}%", flush=True)
    print(f"\npooled lambda = {lam_pool:.3f} (se {lam_pool_se:.3f}) -> "
          f"attenuation {100*(1-lam_pool):.1f}%, correction factor 1/lambda = {1/lam_pool:.3f}")
    print(f"lambda range across cities: {lams.min():.3f}-{lams.max():.3f}")
    print(f"LOCO transportability MAE: {loco_mae:.1f}% (corrected theta_trunc vs actual theta_full)")
    print(f"regression-calibration cross-check: theta_calibrated should ~= theta_full")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from truncation_calibration import apply_truncation_correction

PANEL = REPO / "results" / "soldprice"
EMB = PANEL / "emb"
CONF = PANEL / "confounders"
OUT = PANEL / "soldprice_dml.json"
CITIES = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]
DEMO = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60"]
ALPHAS = np.logspace(-3, 3, 25)


def dml_theta(T, Xraw, Y, seed=0):
    n = len(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Yr, Tr = np.empty(n), np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        sc = StandardScaler().fit(Xraw[tr])
        Xtr, Xte = sc.transform(Xraw[tr]), sc.transform(Xraw[te])
        Yr[te] = Y[te] - RidgeCV(alphas=ALPHAS).fit(Xtr, Y[tr]).predict(Xte)
        Tr[te] = T[te] - RidgeCV(alphas=ALPHAS).fit(Xtr, T[tr]).predict(Xte)
    d = float(np.mean(Tr * Tr))
    if d < 1e-12:
        return np.nan, np.nan
    theta = float(np.mean(Tr * Yr)) / d
    psi = (Yr - theta * Tr) * Tr / d
    se_if = float(np.sqrt(np.var(psi, ddof=1) / n))
    return theta, se_if


def rv(theta, se, n, k, q=1.0):
    if se == 0 or not np.isfinite(se):
        return float("nan")
    t = abs(theta) / se
    f2 = (q * t) ** 2 / max(n - k - 1, 1)
    return 0.5 * (math.sqrt(f2 ** 2 + 4 * f2) - f2)


def cluster_boot(T, X, Y, tract, B, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(tract)
    idx_by = {t: np.where(tract == t)[0] for t in ids}
    out = np.empty(B)
    for b in range(B):
        samp = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_by[t] for t in samp])
        out[b], _ = dml_theta(T[idx], X[idx], Y[idx], seed=b + 1)
    return out[np.isfinite(out)]


def _winsor(X, k=5.0):
    X = np.array(X, float, copy=True)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        mad = np.nanmedian(np.abs(col - med))
        if mad == 0 or not np.isfinite(mad):
            continue
        lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
        X[:, j] = np.clip(col, lo, hi)
    return X


def design(city, arm, T_override=None, extra_keep=None):
    panel = pd.read_parquet(PANEL / f"{city}_panel.parquet",
                            columns=["log_price", "is_single_unit", "sold_quarter"])
    if T_override is not None:
        T = np.asarray(T_override, float)
    else:
        T = pd.read_parquet(EMB / f"{city}_treatment.parquet")["treatment"].to_numpy()
    L = pd.read_parquet(EMB / f"{city}_emb.parquet", columns=["log_len"])["log_len"].to_numpy()
    c = pd.read_parquet(CONF / f"{city}_conf.parquet")
    n = len(panel)
    assert len(T) == len(L) == len(c) == n, f"{city} misaligned: {len(T)},{len(L)},{len(c)},{n}"

    Y = panel["log_price"].to_numpy(float)
    lat, lon = c["lat"].to_numpy(), c["lon"].to_numpy()
    spatial = np.column_stack([lat, lon, lat ** 2, lon ** 2, lat * lon])
    demo = c[[f"demo_{d}" for d in DEMO]].to_numpy(float)
    demo = np.column_stack([demo, c["demo_missing"].to_numpy()])
    quarter = pd.get_dummies(panel["sold_quarter"], drop_first=True).to_numpy(float)

    if arm == "assessor":
        if "ass_beds" not in c.columns:
            return None
        prop = c[["ass_beds", "ass_sqft", "ass_year", "ass_missing"]].to_numpy(float)
    else:
        prop = c[["list_beds", "list_baths", "list_sqft", "list_year",
                  "list_beds_missing", "list_sqft_missing", "list_year_missing"]].to_numpy(float)

    X = np.column_stack([spatial, demo, prop, quarter, L, L ** 2])
    med = np.nanmedian(X, axis=0)
    ix = np.where(~np.isfinite(X))
    X[ix] = np.take(np.nan_to_num(med), ix[1])
    X = _winsor(X)
    tract = c["tract"].astype(str).to_numpy()
    coords = np.column_stack([lat, lon])
    fin = np.isfinite(Y) & panel["is_single_unit"].to_numpy()
    if extra_keep is not None:
        fin = fin & np.asarray(extra_keep, bool)
    return X[fin], T[fin], Y[fin], tract[fin], coords[fin]


def run_city(city, arm, B):
    got = design(city, arm)
    if got is None:
        return None
    X, T, Y, tract, _ = got
    theta, se_if = dml_theta(T, X, Y, seed=0)
    boots = cluster_boot(T, X, Y, tract, B)
    se_boot = float(boots.std(ddof=1))
    lo, hi = (float(np.quantile(boots, .025)), float(np.quantile(boots, .975)))
    tc, tc_se, factor = apply_truncation_correction(theta, se_boot)
    return {"city": city, "arm": arm, "n": int(len(Y)), "tracts": int(len(np.unique(tract))),
            "theta": theta, "se_if": se_if, "se_boot": se_boot,
            "ci_boot": [lo, hi], "covers_zero": bool(lo < 0 < hi),
            "moulton": se_boot / se_if if se_if else float("nan"),
            "rv": rv(theta, se_boot, len(Y), X.shape[1]),
            "theta_corrected": tc, "se_corrected": tc_se,
            "ci_corrected": [tc - 1.96 * tc_se, tc + 1.96 * tc_se],
            "corr_factor": factor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["assessor", "listing"], default="assessor")
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--cities", nargs="*")
    args = ap.parse_args()
    rows = []
    print(f"{'city':13s}{'arm':>9s}{'n':>7s}{'G':>5s}{'theta':>8s}{'se_boot':>9s}"
          f"{'Moult':>7s}{'95% CI':>18s}{'RV':>7s}{'theta_c':>9s}")
    print("-" * 92)
    for c in (args.cities or CITIES):
        r = run_city(c, args.arm, args.B)
        if r is None:
            print(f"{c:13s}{args.arm:>9s}   no assessor arm -> skip")
            continue
        rows.append(r)
        lo, hi = r["ci_boot"]
        print(f"{c:13s}{r['arm']:>9s}{r['n']:7d}{r['tracts']:5d}{r['theta']:8.3f}{r['se_boot']:9.3f}"
              f"{r['moulton']:7.2f}   [{lo:+.3f},{hi:+.3f}]{'*' if r['covers_zero'] else ' '}"
              f"{r['rv']:7.3f}{r['theta_corrected']:9.3f}")
    (PANEL / f"soldprice_dml_{args.arm}.json").write_text(json.dumps(rows, indent=2))
    sig = [r["city"] for r in rows if not r["covers_zero"]]
    print(f"\n* = cluster CI covers zero. theta_c = truncation-corrected (÷{rows[0]['corr_factor']:.3f} if any).")
    print(f"significant (cluster-robust): {len(sig)}/{len(rows)}  ({', '.join(sig)})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

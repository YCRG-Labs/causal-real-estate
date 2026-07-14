"""Conley spatial-HAC standard errors for the identified-direction DML estimator.

Empirical realization of Theorem (estimated-direction spatial DML): the per-market
Robinson influence-function sum is spatially dependent, so the iid influence-function
interval understates the sampling spread. We form the identified direction on the
pooled residualized embedding (leave-market-out, the out-of-fold construction the
theorem requires), residualize treatment and outcome on the full Baur confounders by
cross-fitted ridge, and compute the Conley (1999) spatial-HAC variance of the score
g_i = Ttilde_i * (Ytilde_i - theta * Ttilde_i) with a Bartlett distance kernel at
several cutoffs, using each listing's latitude and longitude.

Run: source .venv/bin/activate && OMP_NUM_THREADS=1 \
     python data/scripts/replications/spatial_hac_identified.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import lstsq, svd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from replications.baur_2023 import get_features_and_target, load_analysis_data

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
OUT = REPO / "results" / "identified_direction"
CUTOFFS_KM = [1.0, 5.0, 10.0]
ALPHAS = np.logspace(-3, 3, 13)


def loc_basis(lat, lon, zlab):
    lat = np.nan_to_num(lat, nan=float(np.nanmedian(lat)))
    lon = np.nan_to_num(lon, nan=float(np.nanmedian(lon)))
    poly = np.column_stack([np.ones(len(lat)), lat, lon, lat**2, lon**2, lat*lon])
    zdum = pd.get_dummies(pd.Series(zlab).astype(str)).to_numpy(float)
    return np.column_stack([poly, zdum])


def dml_residuals(T1d, conf, Y, seed=42, k=5):
    """Cross-fitted ridge Robinson residualization; returns (Ttilde, Ytilde)."""
    n = len(Y)
    conf_s = StandardScaler().fit_transform(conf)
    Yr = np.empty(n); Tr = np.empty(n)
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(conf_s):
        my = RidgeCV(alphas=ALPHAS).fit(conf_s[tr], Y[tr])
        mt = RidgeCV(alphas=ALPHAS).fit(conf_s[tr], T1d[tr])
        Yr[te] = Y[te] - my.predict(conf_s[te])
        Tr[te] = T1d[te] - mt.predict(conf_s[te])
    return Tr, Yr


def latlon_to_km(lat, lon):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    lat0 = float(np.nanmedian(lat)); lon0 = float(np.nanmedian(lon))
    x = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 110.57
    return np.column_stack([np.nan_to_num(x), np.nan_to_num(y)])


def conley_se(Ttil, Ytil, theta, xy, cutoff):
    """Conley spatial-HAC SE with a Bartlett kernel at the given cutoff (km)."""
    g = Ttil * (Ytil - theta * Ttil)          # score contributions
    D = float(np.sum(Ttil * Ttil))            # Robinson denominator (unnormalized)
    tree = cKDTree(xy)
    S = tree.sparse_distance_matrix(tree, max_distance=cutoff, output_type="coo_matrix")
    w = 1.0 - S.data / cutoff                  # Bartlett weight; diagonal (d=0) -> 1
    num = float(np.sum(w * g[S.row] * g[S.col]))
    num = max(num, 0.0)
    return float(np.sqrt(num) / D) if D > 0 else float("nan")


def iid_se(Ttil, Ytil, theta):
    g = Ttil * (Ytil - theta * Ttil)
    D = float(np.sum(Ttil * Ttil))
    return float(np.sqrt(np.sum(g * g)) / D) if D > 0 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = {}
    print("[1/3] loading cities + full confounders...", flush=True)
    for c in ALL_12:
        loaded = load_analysis_data(c)
        if loaded is None:
            print(f"  {c}: no data", flush=True); continue
        feats = get_features_and_target(*loaded, drop_mismatched_crime=True)
        if feats is None:
            print(f"  {c}: no features", flush=True); continue
        T, conf, Y, meta = feats
        B = loc_basis(meta["lat"], meta["lon"], meta["zip_labels"])
        beta, *_ = lstsq(B, T, rcond=None)
        R = T - B @ beta
        store[c] = {"R": R, "conf": conf, "Y": Y, "n": len(Y),
                    "xy": latlon_to_km(meta["lat"], meta["lon"])}
        print(f"  {c:12} N={len(Y):6} conf={conf.shape[1]}", flush=True)

    print("\n[2/3] pooled identified direction (leave-market-out)...", flush=True)
    R_all = {c: s["R"] for c, s in store.items()}

    print("\n[3/3] DML + Conley spatial-HAC per market...", flush=True)
    rows = []
    for c, s in store.items():
        # out-of-fold direction: pooled SVD excluding market c
        others = np.vstack([R_all[k] for k in store if k != c])
        _, _, VtR = svd(others - others.mean(0), full_matrices=False)
        v_id = VtR[0]
        ti = s["R"] @ v_id
        ti = (ti - ti.mean()) / (ti.std() or 1.0)
        Ttil, Ytil = dml_residuals(ti, s["conf"], s["Y"])
        denom = float(np.mean(Ttil * Ttil))
        if denom <= 0:
            print(f"  {c}: denom collapse", flush=True); continue
        theta = float(np.mean(Ttil * Ytil)) / denom
        se_if = iid_se(Ttil, Ytil, theta)
        rec = {"market": c, "n": s["n"], "theta_identified": theta,
               "se_if": se_if, "t_if": theta / se_if}
        for cut in CUTOFFS_KM:
            se_c = conley_se(Ttil, Ytil, theta, s["xy"], cut)
            rec[f"se_conley_{int(cut)}km"] = se_c
            rec[f"t_conley_{int(cut)}km"] = theta / se_c if se_c == se_c and se_c > 0 else float("nan")
        rec["ratio_conley5_if"] = rec["se_conley_5km"] / se_if if se_if > 0 else float("nan")
        rows.append(rec)
        print(f"  {c:12} theta={theta:+.4f}  se_if={se_if:.4f} "
              f"se_conley5={rec['se_conley_5km']:.4f} (x{rec['ratio_conley5_if']:.2f})  "
              f"t_conley5={rec['t_conley_5km']:+.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "spatial_hac_identified.csv", index=False)

    # pooled inverse-variance under the Conley-5km SEs
    def rep(secol):
        th = df["theta_identified"].to_numpy(float); se = df[secol].to_numpy(float)
        w = 1 / se**2
        return float((th * w).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
    pth_if, pse_if = rep("se_if")
    pth_c, pse_c = rep("se_conley_5km")
    sig_if = int((df["t_if"].abs() > 1.96).sum())
    sig_c5 = int((df["t_conley_5km"].abs() > 1.96).sum())
    med_ratio = float(df["ratio_conley5_if"].median())
    summ = {"cutoffs_km": CUTOFFS_KM,
            "median_conley5_over_if": med_ratio,
            "sig_markets_if": sig_if, "sig_markets_conley5": sig_c5,
            "pooled_if": {"theta": pth_if, "se": pse_if},
            "pooled_conley5": {"theta": pth_c, "se": pse_c}}
    (OUT / "spatial_hac_identified_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\nmedian Conley-5km / IF SE ratio = {med_ratio:.2f}")
    print(f"significant markets: IF {sig_if}/12, Conley-5km {sig_c5}/12")
    print(f"pooled identified: IF se={pse_if:.4f}, Conley-5km se={pse_c:.4f}")
    print(f"wrote {OUT}/spatial_hac_identified.csv")


if __name__ == "__main__":
    main()

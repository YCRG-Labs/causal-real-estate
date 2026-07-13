"""Theorem 1' on the Baur pipeline: full confounder set, identified vs leading direction.

Mirrors baur_pooled_pca.py but compares two treatment directions under the SAME
full Baur confounder block:
  leading     = pooled leading PC of the within-city-centered embedding (the paper's
                Baur channel).
  identified  = pooled leading PC of the within-city embedding after residualizing on
                a location basis (lat/lon quadratic + zip dummies), the direction
                Theorem 1' identifies.

Reports per-market and pooled DML effects on both, the ratio, and the alignment.
GBM nuisances, influence-function SE (point estimates are what the theorem
realization needs; the paper's bootstrap/spatial-robust CIs layer on top).
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import lstsq, svd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from replications.baur_2023 import get_features_and_target, load_analysis_data
from replications.compare_to_dml import run_dml, result_to_dict

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
OUT = REPO / "results" / "identified_direction"


def loc_basis(lat, lon, zlab):
    lat = np.nan_to_num(lat, nan=float(np.nanmedian(lat)))
    lon = np.nan_to_num(lon, nan=float(np.nanmedian(lon)))
    poly = np.column_stack([np.ones(len(lat)), lat, lon, lat**2, lon**2, lat*lon])
    zdum = pd.get_dummies(pd.Series(zlab).astype(str)).to_numpy(float)
    return np.column_stack([poly, zdum])


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
        Tc = T - T.mean(0)  # within-city centered (leading channel)
        B = loc_basis(meta["lat"], meta["lon"], meta["zip_labels"])
        beta, *_ = lstsq(B, T, rcond=None)
        R = T - B @ beta  # within-city location-residualized (identified channel)
        store[c] = {"Tc": Tc, "R": R, "conf": conf, "Y": Y, "n": len(Y),
                    "nconf": conf.shape[1], "nloc": B.shape[1]}
        print(f"  {c:12} N={len(Y):6} conf={conf.shape[1]} loc_feat={B.shape[1]}", flush=True)

    print("\n[2/3] pooled directions...", flush=True)
    Tc_all = np.vstack([s["Tc"] for s in store.values()])
    R_all = np.vstack([s["R"] for s in store.values()])
    _, _, VtC = svd(Tc_all - Tc_all.mean(0), full_matrices=False); v_lead = VtC[0]
    _, _, VtR = svd(R_all - R_all.mean(0), full_matrices=False); v_id = VtR[0]
    if float(v_id @ v_lead) < 0: v_id = -v_id
    cos_il = float(abs(v_id @ v_lead))
    print(f"  |cos(v_identified, v_leading)| = {cos_il:.4f}", flush=True)

    print("\n[3/3] DML per market (full Baur confounders, GBM, IF SE)...", flush=True)
    rows = []
    for c, s in store.items():
        tl = s["Tc"] @ v_lead; tl = (tl - tl.mean()) / (tl.std() or 1)
        ti = s["R"] @ v_id;     ti = (ti - ti.mean()) / (ti.std() or 1)
        dl = run_dml(tl[:, None], s["conf"], s["Y"], label=f"{c}-lead",
                     ci_method="if", use_ridge=True, n_pca=1)
        di = run_dml(ti[:, None], s["conf"], s["Y"], label=f"{c}-id",
                     ci_method="if", use_ridge=True, n_pca=1)
        if dl is None or di is None:
            print(f"  {c}: DML failed", flush=True); continue
        rows.append({"market": c, "n": s["n"], "nconf": s["nconf"], "cos_il": cos_il,
                     "theta_leading": dl.theta, "se_leading": dl.se, "t_leading": dl.theta/dl.se,
                     "theta_identified": di.theta, "se_identified": di.se, "t_identified": di.theta/di.se,
                     "sig_lead": bool(abs(dl.theta/dl.se) > 1.96), "sig_id": bool(abs(di.theta/di.se) > 1.96)})
        r = rows[-1]
        print(f"  {c:12} lead {dl.theta:+.4f}(t={dl.theta/dl.se:+.1f})  "
              f"identified {di.theta:+.4f}(t={di.theta/di.se:+.1f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baur_identified_vs_leading.csv", index=False)

    def rep(col, secol):  # DerSimonian-Laird-ish inverse-variance
        th = df[col].to_numpy(float); se = df[secol].to_numpy(float)
        w = 1/se**2; return float((th*w).sum()/w.sum()), float(np.sqrt(1/w.sum()))
    pl, sl = rep("theta_leading", "se_leading"); pi, si = rep("theta_identified", "se_identified")
    summ = {"cos_il": cos_il,
            "leading": {"theta": pl, "se": sl, "t": pl/sl, "sig_markets": int(df.sig_lead.sum())},
            "identified": {"theta": pi, "se": si, "t": pi/si, "sig_markets": int(df.sig_id.sum())},
            "ratio": pi/pl, "strong_market_median_ratio":
            float((df[df.t_leading.abs()>4].theta_identified/df[df.t_leading.abs()>4].theta_leading).median())}
    (OUT / "baur_identified_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\nPOOLED (inverse-variance, full Baur confounders):")
    print(f"  leading    theta={pl:+.4f} se={sl:.4f} (t={pl/sl:+.1f})  sig {int(df.sig_lead.sum())}/12")
    print(f"  identified theta={pi:+.4f} se={si:.4f} (t={pi/si:+.1f})  sig {int(df.sig_id.sum())}/12")
    print(f"  ratio identified/leading = {pi/pl:.2f}   strong-market median ratio = {summ['strong_market_median_ratio']:.2f}")
    print(f"\nwrote {OUT}/baur_identified_vs_leading.csv")


if __name__ == "__main__":
    main()

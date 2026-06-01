"""Leverage / influence diagnostic for the Shen ridge-DML pipeline.

For each requested city, reproduce the same Doc2Vec uniqueness + ridge-DML
fit used by shen_2021.py --doc2vec --fast --n_boot 500, and answer:

  - what are the top-5 highest-|IC| listings (influence-curve contribution
    psi_i / E[T_resid^2])?
  - how does theta_hat change when we drop the k highest-|IC| rows for
    k in {1, 2, 5, 10}?
  - what is the per-fold theta (5-fold jackknife) and its sd?
  - what is the Huber-trimmed theta (c=1.345) and the count of clipped IF
    contributions?

These are the standard diagnostics from diagnose_all_12.py adapted to the
Shen ridge path. The point is to surface whether a per-city result is
driven by 1-2 listings (Dallas +0.048 sanity) or whether a null result
masks 1-2 extreme contributions (Atlanta -0.025 sanity).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))
sys.path.insert(0, str(REPO / "data" / "scripts" / "replications"))

from causal_inference import (  # noqa: E402
    get_features_and_target,
    load_analysis_data,
)
from shen_2021 import (  # noqa: E402
    _vectorize_doc2vec,
    _uniqueness_from_vectors,
    _knn_peers,
)


def _ridge_resid(T_z, conf, Y, k_folds=5, seed=42):
    n = len(Y)
    conf_s = StandardScaler().fit_transform(np.asarray(conf))
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_res = np.empty(n); T_res = np.empty(n); fold = np.empty(n, dtype=int)
    for f, (tr, te) in enumerate(kf.split(np.arange(n))):
        m_y = RidgeCV(alphas=alphas).fit(conf_s[tr], Y[tr])
        m_t = RidgeCV(alphas=alphas).fit(conf_s[tr], T_z[tr])
        Y_res[te] = Y[te] - m_y.predict(conf_s[te])
        T_res[te] = T_z[te] - m_t.predict(conf_s[te])
        fold[te] = f
    return Y_res, T_res, fold


def _theta_se_from_resids(Y_res, T_res):
    n = len(Y_res)
    denom = float(np.mean(T_res * T_res))
    if denom < 1e-12:
        return float("nan"), float("nan"), float("nan")
    theta = float(np.mean(T_res * Y_res)) / denom
    psi = (Y_res - theta * T_res) * T_res / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))
    return theta, se, denom


def diagnose_one(city, k=5, k_folds=5, seed=42):
    print(f"\n========================  {city}  ========================")
    loaded = load_analysis_data(city)
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        print(f"  [{city}] no features"); return None
    _, confounders, Y, meta = feats
    if len(emb_df) != confounders.shape[0]:
        emb_df = emb_df.iloc[: confounders.shape[0]].reset_index(drop=True)

    desc_col = "clean_description" if "clean_description" in emb_df.columns else "description"
    if "clean_description" in emb_df.columns:
        descriptions = emb_df["clean_description"].fillna(emb_df["description"]).astype(str).tolist()
    else:
        descriptions = emb_df[desc_col].astype(str).tolist()
    lat = pd.to_numeric(emb_df["latitude"], errors="coerce").values.astype(float)
    lon = pd.to_numeric(emb_df["longitude"], errors="coerce").values.astype(float)
    mask = np.isfinite(lat) & np.isfinite(lon)
    if (~mask).any():
        descriptions = [d for d, ok in zip(descriptions, mask) if ok]
        lat = lat[mask]; lon = lon[mask]
        confounders = confounders[mask]; Y = Y[mask]
        emb_df = emb_df.iloc[np.flatnonzero(mask)].reset_index(drop=True)

    vectors = _vectorize_doc2vec(descriptions, seed=seed)
    peers = _knn_peers(lat, lon, k=5)
    uniqueness = _uniqueness_from_vectors(vectors, peers)
    n = len(Y)
    print(f"  n={n}  uniq mean={uniqueness.mean():.3f}  sd={uniqueness.std(ddof=1):.3f}")

    t_sd = float(np.std(uniqueness, ddof=1))
    if t_sd < 1e-12:
        print("  T is constant"); return None
    T_z = (uniqueness - float(np.mean(uniqueness))) / t_sd
    Y_res, T_res, fold = _ridge_resid(T_z, confounders, Y, k_folds=k_folds, seed=seed)
    theta, se, denom = _theta_se_from_resids(Y_res, T_res)
    print(f"  theta_hat={theta:+.4f}  se_IF={se:.4f}  E[T_res^2]={denom:.3f}")

    psi = (Y_res - theta * T_res) * T_res / max(denom, 1e-12)
    IC = psi
    total_var = float((IC ** 2).sum())
    order = np.argsort(-np.abs(IC))

    print("\n  Top-5 |IC| rows:")
    cols_show = ["url", "address", "price", "beds", "bedrooms",
                 "sqft", "year_built", "latitude", "longitude"]
    cols_show = [c for c in cols_show if c in emb_df.columns]
    for r, i in enumerate(order[:5]):
        share = float(IC[i] ** 2) / max(total_var, 1e-30)
        print(f"    rank {r+1}  row={i}  IC={IC[i]:+.3f}  share={share:.3f}  "
              f"Y_res={Y_res[i]:+.3f}  T_res={T_res[i]:+.3f}  fold={fold[i]}  "
              f"uniq={uniqueness[i]:.3f}")
        row = emb_df.iloc[i]
        for c in cols_show[:4]:
            print(f"      {c}: {row[c]}")
        desc = str(row.get(desc_col, ""))[:220].replace("\n", " ")
        print(f"      desc[:220]: {desc}")

    print("\n  theta_{-k} sensitivity (drop the k highest-|IC| rows):")
    sens = []
    for kk in (0, 1, 2, 5, 10):
        if kk == 0:
            sens.append(("none", theta, se, n))
            continue
        drop = order[:kk]
        keep = np.setdiff1d(np.arange(n), drop)
        th_k, se_k, _ = _theta_se_from_resids(Y_res[keep], T_res[keep])
        sens.append((f"-{kk}", th_k, se_k, len(keep)))
        print(f"    drop top-{kk:<2}  theta={th_k:+.4f}  se={se_k:.4f}  "
              f"n={len(keep)}  shift={(th_k - theta):+.4f}")

    print("\n  Per-fold (5-fold jackknife):")
    fold_thetas = []
    for f in sorted(set(fold.tolist())):
        m = fold == f
        denom_f = float(np.mean(T_res[m] * T_res[m]))
        if denom_f < 1e-12:
            fold_thetas.append(float("nan")); continue
        th_f = float(np.mean(T_res[m] * Y_res[m]) / denom_f)
        fold_thetas.append(th_f)
        print(f"    fold {f}  n={int(m.sum())}  theta={th_f:+.4f}  denom={denom_f:.4f}")
    print(f"    fold theta sd={np.nanstd(fold_thetas, ddof=1):.4f}  "
          f"min={np.nanmin(fold_thetas):+.4f}  max={np.nanmax(fold_thetas):+.4f}")

    abs_psi = np.abs(psi)
    sigma = float(np.median(abs_psi) / 0.6745)
    c_hub = 1.345 * max(sigma, 1e-12)
    psi_hub = np.where(abs_psi > c_hub, np.sign(psi) * c_hub, psi)
    theta_hub_adj = float(np.mean(psi_hub - psi))
    theta_huber = theta + theta_hub_adj
    n_clip = int((abs_psi > c_hub).sum())
    print(f"\n  Huber-trimmed (c=1.345)  theta_huber={theta_huber:+.4f}  "
          f"shift from OLS={theta_hub_adj:+.4f}  clipped={n_clip}/{n}")

    return {
        "city": city, "n": int(n), "theta": float(theta), "se_IF": float(se),
        "denom_T_res_sq": float(denom),
        "theta_drop1": float(sens[1][1]), "theta_drop5": float(sens[3][1]),
        "theta_drop10": float(sens[4][1]),
        "fold_theta_sd": float(np.nanstd(fold_thetas, ddof=1)),
        "theta_huber": float(theta_huber), "n_huber_clipped": int(n_clip),
        "top1_IC_share": float(np.max(IC ** 2) / max(total_var, 1e-30)),
    }


ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--all_12", action="store_true",
                    help="run all 12 cities (default: atlanta + dallas pilot pair)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.all_12:
        args.cities = list(ALL_12)
    elif not args.cities:
        args.cities = ["atlanta", "dallas"]
    rows = []
    for c in args.cities:
        try:
            r = diagnose_one(c, seed=args.seed)
            if r is not None:
                rows.append(r)
        except Exception as e:
            import traceback
            print(f"[{c}] FAILED: {e}", file=sys.stderr); traceback.print_exc()
    out = REPO / "results" / "replications" / "shen_leverage_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

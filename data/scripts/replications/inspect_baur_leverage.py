"""Leverage / influence diagnostic for the Baur ridge-DML pipeline (BERT PC1).

For each requested city, reproduce the same BERT PC1 + ridge-DML fit used
by baur_2023.py --fast --n_boot 500, and answer:
  - top-5 highest-|IC| listings (influence-curve contributions)
  - theta_hat sensitivity when k highest-|IC| rows are dropped, k in {1, 2, 5, 10}
  - per-fold (5-fold) theta jackknife
  - Huber-trimmed theta and clipping count

Mirrors inspect_shen_leverage.py but on multivariate BERT treatment via
PC1 extraction inside the ridge path (same code path as
compare_to_dml.run_dml's use_ridge=True branch). Output written to
results/replications/baur_leverage_diag.json with one entry per city.
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

from causal_inference import (  # noqa: E402
    get_features_and_target,
    load_analysis_data,
)


ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _extract_pc1(T_mat, seed=42):
    """PC1 of standardised T, z-scored. Mirrors compare_to_dml ridge path."""
    T_std = StandardScaler().fit_transform(np.asarray(T_mat, dtype=np.float64))
    _, _, vt = np.linalg.svd(T_std, full_matrices=False)
    pc1 = (T_std @ vt[0]).astype(np.float64)
    sd = float(np.std(pc1, ddof=1))
    if sd < 1e-12:
        return None
    return (pc1 - float(np.mean(pc1))) / sd


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


def _theta_se(Y_res, T_res):
    n = len(Y_res)
    denom = float(np.mean(T_res * T_res))
    if denom < 1e-12:
        return float("nan"), float("nan"), float("nan")
    theta = float(np.mean(T_res * Y_res)) / denom
    psi = (Y_res - theta * T_res) * T_res / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))
    return theta, se, denom


def diagnose_one(city, k_folds=5, seed=42):
    print(f"\n========================  {city} (Baur BERT-PC1)  ========================")
    loaded = load_analysis_data(city)
    if loaded is None:
        print(f"  [{city}] no data"); return None
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        print(f"  [{city}] no features"); return None
    T_emb, confounders, Y, meta = feats
    n = len(Y)
    print(f"  N={n}  text_dim={T_emb.shape[1]}  conf_dim={confounders.shape[1]}")

    T_z = _extract_pc1(T_emb, seed=seed)
    if T_z is None:
        print("  [{city}] PC1 SD ~ 0"); return None
    Y_res, T_res, fold = _ridge_resid(T_z, confounders, Y, k_folds=k_folds, seed=seed)
    theta, se, denom = _theta_se(Y_res, T_res)
    print(f"  theta_hat={theta:+.4f}  se_IF={se:.4f}  E[T_res^2]={denom:.3f}")

    psi = (Y_res - theta * T_res) * T_res / max(denom, 1e-12)
    IC = psi
    total_var = float((IC ** 2).sum())
    order = np.argsort(-np.abs(IC))
    top1_share = float(np.max(IC ** 2) / max(total_var, 1e-30))
    print(f"  top-1 |IC| variance share = {top1_share:.3f}")

    sens = []
    for kk in (0, 1, 2, 5, 10):
        if kk == 0:
            sens.append(("none", theta, se, n))
            continue
        drop = order[:kk]
        keep = np.setdiff1d(np.arange(n), drop)
        th_k, se_k, _ = _theta_se(Y_res[keep], T_res[keep])
        sens.append((f"-{kk}", th_k, se_k, len(keep)))
        print(f"    drop top-{kk:<2}  theta={th_k:+.4f}  se={se_k:.4f}  "
              f"shift={(th_k - theta):+.4f}")

    fold_thetas = []
    for f in sorted(set(fold.tolist())):
        m = fold == f
        denom_f = float(np.mean(T_res[m] * T_res[m]))
        if denom_f < 1e-12:
            fold_thetas.append(float("nan")); continue
        th_f = float(np.mean(T_res[m] * Y_res[m]) / denom_f)
        fold_thetas.append(th_f)
    print(f"  fold thetas: "
          f"{[f'{x:+.3f}' if not np.isnan(x) else 'nan' for x in fold_thetas]}  "
          f"sd={float(np.nanstd(fold_thetas, ddof=1)):.4f}")

    abs_psi = np.abs(psi)
    sigma = float(np.median(abs_psi) / 0.6745)
    c_hub = 1.345 * max(sigma, 1e-12)
    psi_hub = np.where(abs_psi > c_hub, np.sign(psi) * c_hub, psi)
    theta_hub_adj = float(np.mean(psi_hub - psi))
    theta_huber = theta + theta_hub_adj
    n_clip = int((abs_psi > c_hub).sum())
    print(f"  Huber theta={theta_huber:+.4f}  shift={theta_hub_adj:+.4f}  "
          f"clipped={n_clip}/{n}")

    return {
        "city": city, "n": int(n), "theta": float(theta), "se_IF": float(se),
        "denom_T_res_sq": float(denom),
        "theta_drop1": float(sens[1][1]), "theta_drop5": float(sens[3][1]),
        "theta_drop10": float(sens[4][1]),
        "fold_theta_sd": float(np.nanstd(fold_thetas, ddof=1)),
        "theta_huber": float(theta_huber), "n_huber_clipped": int(n_clip),
        "top1_IC_share": float(top1_share),
        "fold_thetas": fold_thetas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--all_12", action="store_true")
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

    out = REPO / "results" / "replications" / "baur_leverage_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))
    csv_out = REPO / "results" / "replications" / "baur_leverage_diag.csv"
    if rows:
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"\nSaved -> {out} and {csv_out}")


if __name__ == "__main__":
    main()

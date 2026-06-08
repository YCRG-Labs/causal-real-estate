"""PC-component sensitivity for the Baur 2023 BERT-DML.

The headline Baur theta uses PC1 of the 768-dim BERT embedding. A referee
will ask: is the result driven by PC1 specifically, or do other principal
components also carry text-treatment signal? This script reports
theta_hat + IF SE for k in {1, 2, 3, 5, 10} as the scalar PC index used
in the ridge-DML cross-fit. PC1 reproduces the headline; PC2-5 are noise
that should produce theta ~= 0 under the proper-DML hypothesis. If PC2-5
also exclude zero, the text-treatment representation is multi-dimensional
and the PC1-only headline understates the channel.

Mirrors compare_to_dml._ridge_dml_core but with explicit PC-index control.
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

from causal_inference import (
    get_features_and_target,
    load_analysis_data,
)


ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _pc_scores(T_mat, k_top=10):
    """Top-k_top PC scores of standardised T. Returns (n, k_top) array
    where column j is the (j+1)-th PC score, z-scored to unit variance."""
    T_std = StandardScaler().fit_transform(np.asarray(T_mat, dtype=np.float64))
    u, s, vt = np.linalg.svd(T_std, full_matrices=False)
    pcs = T_std @ vt[:k_top].T
    pcs = pcs / np.maximum(pcs.std(axis=0, ddof=1), 1e-12)
    return pcs


def _ridge_dml_one(T_z, conf, Y, k_folds=5, seed=42):
    n = len(Y)
    conf_s = StandardScaler().fit_transform(np.asarray(conf))
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_res = np.empty(n); T_res = np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = RidgeCV(alphas=alphas).fit(conf_s[tr], Y[tr])
        m_t = RidgeCV(alphas=alphas).fit(conf_s[tr], T_z[tr])
        Y_res[te] = Y[te] - m_y.predict(conf_s[te])
        T_res[te] = T_z[te] - m_t.predict(conf_s[te])
    denom = float(np.mean(T_res ** 2))
    if denom < 1e-12:
        return float("nan"), float("nan")
    theta = float(np.mean(T_res * Y_res)) / denom
    psi = (Y_res - theta * T_res) * T_res / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))
    return theta, se


def sweep_one(city, pc_indices, seed=42):
    loaded = load_analysis_data(city)
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    T_emb, conf, Y, meta = feats
    n = len(Y)
    pcs = _pc_scores(T_emb, k_top=max(pc_indices))
    print(f"\n=== {city}  N={n}  text_dim={T_emb.shape[1]}  conf_dim={conf.shape[1]} ===")
    rows = []
    for k in pc_indices:
        theta, se = _ridge_dml_one(pcs[:, k - 1], conf, Y, seed=seed)
        rows.append({
            "city": city, "pc_index": k, "n": int(n),
            "theta": theta, "se_IF": se,
            "ci_low": theta - 1.96 * se, "ci_high": theta + 1.96 * se,
            "excludes_zero": (theta - 1.96 * se > 0) or (theta + 1.96 * se < 0),
        })
        flag = "Y" if rows[-1]["excludes_zero"] else "n"
        print(f"  PC{k:>2}  theta={theta:+.4f}  se={se:.4f}  "
              f"CI=[{rows[-1]['ci_low']:+.3f}, {rows[-1]['ci_high']:+.3f}]  ≠0={flag}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--pc_indices", type=int, nargs="*",
                    default=[1, 2, 3, 5, 10],
                    help="which PC components to test as scalar treatment "
                         "(default: 1 2 3 5 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv",
                    default=str(REPO / "results" / "replications"
                                / "baur_pc_sensitivity.csv"))
    args = ap.parse_args()
    if args.all_12:
        args.cities = list(ALL_12)
    elif not args.cities:
        args.cities = ["sf"]

    all_rows = []
    for c in args.cities:
        try:
            all_rows.extend(sweep_one(c, args.pc_indices, seed=args.seed))
        except Exception as e:
            import traceback
            print(f"[{c}] FAILED: {e}", file=sys.stderr); traceback.print_exc()

    if not all_rows:
        return 1
    df = pd.DataFrame(all_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nCSV -> {args.out_csv}")
    print("\nSummary: count of cities where each PC excludes zero")
    print(df.groupby("pc_index")["excludes_zero"].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

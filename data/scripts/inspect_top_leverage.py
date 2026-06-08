"""Show the top-5 highest-|IC| listings per outlier-pinned city.

Reproduces the diagnose_all_12 residualization (ridge, k_folds=5, seed=0)
and maps the top-|IC| indices back to the embeddings parquet so we can see
each row's URL, address, price, beds, sqft, year_built, and a description
preview. The point is to verify these are real listings (e.g., a $50M
penthouse with a unique narrative) rather than scrape artifacts (truncated
description, mis-parsed price, address geocoded to wrong city).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from fast_bootstrap_dml_v2 import (
    _build_features, _dml_core, _ridge_oof_predict,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DEFAULT_CITIES = ["philadelphia", "seattle", "portland"]
DISPLAY_COLS = ["url", "address", "price", "beds", "bedrooms", "baths",
                "sqft", "bldg_area_sqft", "year_built", "latitude", "longitude"]


def _emb_df_for(city: str) -> pd.DataFrame:
    p = PROCESSED_DIR / f"{city}_embeddings.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def inspect_city(city: str, k: int = 5, k_folds: int = 5, seed: int = 0):
    print(f"\n{'='*88}\n{city}\n{'='*88}")
    data = _build_features(city, use_cache=True, drop_crime=False)
    if data is None:
        print(f"  [{city}] _build_features returned None"); return
    T, conf, Y, coords, meta = data
    n = len(Y)
    scaler = StandardScaler().fit(conf)
    conf_s = scaler.transform(conf).astype(np.float64, copy=False)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(n)))

    res = _dml_core(T, conf_s, Y, learner_fn=_ridge_oof_predict, seed=seed,
                   k_folds=k_folds, splits=splits, return_residuals=True)
    theta, se_if, Y_resid, T_resid, fold_ids = res
    denom = float(np.mean(T_resid ** 2))
    IC = (Y_resid - theta * T_resid) * T_resid / max(denom, 1e-12)
    order = np.argsort(-np.abs(IC))
    print(f"  theta_hat={theta:+.4f}  se_IF={se_if:.4f}  n={n}  "
          f"max|IC|={np.max(np.abs(IC)):.3f}  total IC var={float((IC**2).sum()):.3f}")

    emb_df = _emb_df_for(city)
    if len(emb_df) != n:
        print(f"  WARNING: emb parquet has {len(emb_df)} rows but _build_features kept {n}; "
              "row alignment uses emb_df.iloc[index] which may be off if rows dropped")

    show_cols = [c for c in DISPLAY_COLS if c in emb_df.columns]
    for rank, i in enumerate(order[:k]):
        print(f"\n  --- rank {rank+1}  row_idx={i}  IC={IC[i]:+.3f}  "
              f"|IC|_share_var={(IC[i]**2)/max((IC**2).sum(),1e-30):.3f} ---")
        print(f"    Y_resid={Y_resid[i]:+.3f}  T_resid={T_resid[i]:+.3f}  fold={fold_ids[i]}")
        row = emb_df.iloc[i]
        for c in show_cols:
            v = row[c]
            print(f"    {c:24s}: {v}")
        if "description" in emb_df.columns:
            desc = str(row["description"])[:300].replace("\n", " ")
            print(f"    description[:300]      : {desc}")
        elif "clean_description" in emb_df.columns:
            desc = str(row["clean_description"])[:300].replace("\n", " ")
            print(f"    clean_description[:300]: {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    cities = args.cities or DEFAULT_CITIES
    for c in cities:
        try:
            inspect_city(c, k=args.k)
        except Exception as e:
            import traceback
            print(f"\n[{c}] FAILED: {e}", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    main()

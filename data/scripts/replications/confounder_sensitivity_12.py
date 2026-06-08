"""Confounder-block sensitivity analysis for the 12-city DML.

Implements Cinelli & Hazlett (2020, JRSSB) Robustness Value (RV) plus
per-block ablation, following the Chernozhukov-Cinelli-Newey-Sharma-
Syrgkanis (2024, arXiv 2112.13398) generalisation to DML. The
methodology research agent flagged that naively dropping lat/lon and
reporting a theta-shift is the WRONG move per Gilbert et al. (2024,
arXiv 2112.14946) "Causal inference framework for spatial confounding."
The right move is to keep all confounders in the long regression and
report:

  1. RV: smallest partial-R² (in both Y and T residual variance) that
     an unobserved confounder would need to drive theta to zero.
  2. RV at α=0.05: same but for statistical significance.
  3. Per-block benchmark: cf_y_block, cf_d_block — the partial R² each
     observed confounder block explains, computed from cross-fit
     residuals. Comparison to RV tells you whether an unobserved
     confounder of comparable strength to that observed block would
     overturn the result.
  4. Naive ablation alongside: theta_short (drop block) - theta_long
     (full set). Mostly for reader intuition; the formal claim lives in
     the RV / benchmark pair.

The treatment is the Shen Doc2Vec uniqueness score (scalar) — the
ridge-DML cross-fit is the same as compare_to_dml's use_ridge=True
branch. Output: results/replications/confounder_sensitivity_12.csv with
one row per (city, block) plus a "full" row per city.
"""
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
sys.path.insert(0, str(REPO / "data" / "scripts" / "replications"))

from causal_inference import (
    load_analysis_data, get_features_and_target, _spatial_join_parcels,
)
from canonical_confounders import (
    SPATIAL, PROPERTY, CENSUS, CRIME, AMENITY, MICRO_GEO, ALL,
)
from shen_2021 import (
    _vectorize_doc2vec, _uniqueness_from_vectors, _knn_peers,
)

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

BLOCKS = {
    "spatial": SPATIAL,
    "property": PROPERTY,
    "census": CENSUS,
    "crime": CRIME,
    "amenity": AMENITY,
    "micro_geo": MICRO_GEO,
}


def _ridge_resid(T_z, conf, Y, k_folds=5, seed=42):
    n = len(Y)
    conf_s = StandardScaler().fit_transform(np.asarray(conf, dtype=np.float64))
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_res = np.empty(n); T_res = np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = RidgeCV(alphas=alphas).fit(conf_s[tr], Y[tr])
        m_t = RidgeCV(alphas=alphas).fit(conf_s[tr], T_z[tr])
        Y_res[te] = Y[te] - m_y.predict(conf_s[te])
        T_res[te] = T_z[te] - m_t.predict(conf_s[te])
    return Y_res, T_res


def _theta_se_from_resids(Y_res, T_res):
    n = len(Y_res)
    denom = float(np.mean(T_res * T_res))
    if denom < 1e-12:
        return float("nan"), float("nan")
    theta = float(np.mean(T_res * Y_res)) / denom
    psi = (Y_res - theta * T_res) * T_res / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))
    return theta, se


def _rv(theta, se, n, k_x, q_alpha=1.96):
    """Cinelli-Hazlett Robustness Value.

    f² = (theta/SE)² × dof_factor, but for DML we collapse dof to (n - k_x).
    RV = 1/2 * (sqrt(f²² + 4f²) - f²) — solves the bias-bound = |theta|.
    RV_alpha = same but with f²_q = q² / t² shifted.
    Returns (RV, RV_alpha, f²) all on the partial-R² scale (0 to 1).
    """
    if se == 0 or math.isnan(se) or math.isnan(theta):
        return float("nan"), float("nan"), float("nan")
    t = abs(theta) / se
    df = max(n - k_x - 1, 1)
    f2 = (t ** 2) / df
    rv = 0.5 * (math.sqrt(f2 ** 2 + 4 * f2) - f2)
    f2_q = max((t ** 2 - q_alpha ** 2), 0) / df
    rv_alpha = 0.5 * (math.sqrt(f2_q ** 2 + 4 * f2_q) - f2_q) if f2_q > 0 else 0.0
    return float(rv), float(rv_alpha), float(f2)


def _partial_r2_block(block_cols, others_cols, X_df, y_array):
    """Partial R² of y on the block given the other columns. Robust to
    empty columns and 0-feature StandardScaler edge cases; returns 0.0
    on any internal failure so the per-city loop does not crash on one
    pathological city.
    """
    try:
        block_cols = [c for c in block_cols if c in X_df.columns]
        others_cols = [c for c in others_cols if c in X_df.columns]
        if not block_cols:
            return 0.0
        X_oth = (X_df[others_cols].to_numpy(dtype=np.float64) if others_cols
                 else np.zeros((len(y_array), 0)))
        X_blk = X_df[block_cols].to_numpy(dtype=np.float64)
        if X_blk.ndim == 1:
            X_blk = X_blk.reshape(-1, 1)
        if X_blk.shape[1] == 0:
            return 0.0
        alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
        if X_oth.shape[1] > 0:
            scaler_oth = StandardScaler().fit(X_oth)
            X_oth_s = scaler_oth.transform(X_oth)
            m = RidgeCV(alphas=alphas).fit(X_oth_s, y_array)
            y_res = y_array - m.predict(X_oth_s)
            m_b = RidgeCV(alphas=alphas).fit(X_oth_s, X_blk)
            X_blk_res = X_blk - m_b.predict(X_oth_s)
        else:
            y_res = y_array - y_array.mean()
            X_blk_res = X_blk - X_blk.mean(axis=0)
        if X_blk_res.shape[1] == 0 or np.allclose(X_blk_res.std(axis=0), 0):
            return 0.0
        scaler_b = StandardScaler().fit(X_blk_res)
        X_blk_res_s = scaler_b.transform(X_blk_res)
        m_y_on_blk = RidgeCV(alphas=alphas).fit(X_blk_res_s, y_res)
        y_hat = m_y_on_blk.predict(X_blk_res_s)
        ss_explained = float(np.sum(y_hat ** 2))
        ss_total = float(np.sum(y_res ** 2))
        if ss_total <= 0:
            return 0.0
        r2 = ss_explained / ss_total
        return float(max(0.0, min(1.0, r2)))
    except Exception:
        return 0.0


def _confounder_block_names(emb_df, parcels, crime_dropped):
    """Reconstruct the ordered column names of the joined confounder array
    that get_features_and_target returns, so each canonical block can be
    ablated and the long-regression RV uses the full ~35-confounder set.

    Neither get_features_and_target nor its meta expose column names, so we
    mirror its assembly order exactly: [latitude, longitude] + PROPERTY +
    CONTEXTUAL (CENSUS + CRIME + AMENITY + MICRO_GEO), with the crime block
    dropped when the city failed the crime temporal-match guard, followed by
    the identical >50%-NaN column filter (col_nan_rate < 0.5 over all rows,
    computed before the row-validity mask). The surviving names line up
    one-to-one with the columns of the returned `confounders`; the caller
    asserts that alignment. Returns None when the estimator fell back to the
    zip one-hot design (no spatial block, no spatial join), where canonical
    block membership is undefined.
    """
    lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)),
                        errors="coerce").values
    lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)),
                        errors="coerce").values
    names: list[str] = []
    parts: list[np.ndarray] = []
    if not np.all(np.isnan(lat)) and not np.all(np.isnan(lon)):
        names += list(SPATIAL)
        parts.append(np.column_stack([lat, lon]))
    joined = None
    if parcels is not None:
        try:
            joined = _spatial_join_parcels(emb_df, parcels)
        except Exception:
            joined = None
    if joined and crime_dropped and "contextual" in joined:
        ctx_cols = joined["contextual_cols"]
        keep = [i for i, c in enumerate(ctx_cols) if not c.startswith("crime_")]
        joined["contextual"] = joined["contextual"][:, keep]
        joined["contextual_cols"] = [ctx_cols[i] for i in keep]
    if joined and "property" in joined:
        names += list(joined["property_cols"])
        parts.append(joined["property"])
    if joined and "contextual" in joined:
        names += list(joined["contextual_cols"])
        parts.append(joined["contextual"])
    if not parts:
        return None
    full = np.hstack(parts)
    good = np.isnan(full).mean(axis=0) < 0.5
    return [nm for nm, keep in zip(names, good) if keep]


def diagnose_one(city, seed=42, k_folds=5):
    print(f"\n========================  {city}  ========================")
    loaded = load_analysis_data(city)
    if loaded is None:
        return None
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return None
    _, confounders, Y, meta = feats
    conf_names = _confounder_block_names(
        emb_df, parcels, bool(meta.get("crime_dropped", False)))
    blocks_recovered = (conf_names is not None
                        and len(conf_names) == confounders.shape[1])
    if not blocks_recovered:
        got = "None" if conf_names is None else str(len(conf_names))
        print(f"  [{city}] WARN: confounder columns could not be mapped to "
              f"blocks (recovered {got} names vs {confounders.shape[1]} "
              f"columns); reporting full-set DML + RV only, skipping per-block "
              f"ablation.")
        conf_names = [f"conf_{i}" for i in range(confounders.shape[1])]
    if len(emb_df) != confounders.shape[0]:
        emb_df = emb_df.iloc[: confounders.shape[0]].reset_index(drop=True)

    desc_col = "clean_description" if "clean_description" in emb_df.columns else "description"
    if "clean_description" in emb_df.columns:
        descriptions = emb_df["clean_description"].fillna(emb_df.get("description", "")).astype(str).tolist()
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
    t_sd = float(np.std(uniqueness, ddof=1))
    if t_sd < 1e-12:
        print(f"  [{city}] uniqueness SD ~ 0; skip"); return None
    T_z = (uniqueness - float(np.mean(uniqueness))) / t_sd

    X_df = pd.DataFrame(np.asarray(confounders, dtype=np.float64),
                        columns=conf_names).reset_index(drop=True).fillna(0.0)

    blocks_present = {bn: [c for c in conf_names if c in set(cols)]
                      for bn, cols in BLOCKS.items()}
    blocks_present = {bn: cols for bn, cols in blocks_present.items() if cols}
    print(f"  N={n}  n_conf={X_df.shape[1]}  blocks present: "
          + ", ".join(f"{bn}({len(c)})" for bn, c in blocks_present.items()))

    full_cols = list(X_df.columns)
    Y_res_full, T_res_full = _ridge_resid(T_z, X_df[full_cols].to_numpy(), Y, k_folds, seed)
    theta_long, se_long = _theta_se_from_resids(Y_res_full, T_res_full)
    rv, rv_a, f2 = _rv(theta_long, se_long, n, len(full_cols))
    print(f"  long DML  theta={theta_long:+.4f}  se={se_long:.4f}  "
          f"RV={rv*100:.1f}%  RV_α=0.05={rv_a*100:.1f}%  f²={f2:.3f}")

    rows = [{
        "city": city, "block_dropped": "none",
        "n": int(n), "n_x": len(full_cols),
        "theta": theta_long, "se": se_long,
        "rv": rv, "rv_alpha": rv_a, "f2": f2,
        "cf_y": float("nan"), "cf_d": float("nan"),
        "delta_theta_short_long": float("nan"),
    }]
    for bn, cols in blocks_present.items():
        cf_y = _partial_r2_block(cols, [c for c in full_cols if c not in cols],
                                  X_df, Y)
        cf_d = _partial_r2_block(cols, [c for c in full_cols if c not in cols],
                                  X_df, T_z)
        short_cols = [c for c in full_cols if c not in cols]
        if not short_cols:
            print(f"    drop {bn:<10}  (skipped: 0 features remain)")
            rows.append({
                "city": city, "block_dropped": bn,
                "n": int(n), "n_x": 0,
                "theta": float("nan"), "se": float("nan"),
                "rv": rv, "rv_alpha": rv_a, "f2": f2,
                "cf_y": cf_y, "cf_d": cf_d,
                "delta_theta_short_long": float("nan"),
            })
            continue
        Y_res_s, T_res_s = _ridge_resid(T_z, X_df[short_cols].to_numpy(),
                                         Y, k_folds, seed)
        theta_short, se_short = _theta_se_from_resids(Y_res_s, T_res_s)
        delta = theta_short - theta_long
        verdict = "ROBUST" if (cf_y < rv and cf_d < rv) else "FRAGILE"
        print(f"    drop {bn:<10}  theta_short={theta_short:+.4f}  "
              f"Δ={delta:+.4f}  cf_y={cf_y*100:>5.1f}%  cf_d={cf_d*100:>5.1f}%  "
              f"vs RV={rv*100:>5.1f}%  {verdict}")
        rows.append({
            "city": city, "block_dropped": bn,
            "n": int(n), "n_x": len(short_cols),
            "theta": theta_short, "se": se_short,
            "rv": rv, "rv_alpha": rv_a, "f2": f2,
            "cf_y": cf_y, "cf_d": cf_d,
            "delta_theta_short_long": delta,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.all_12:
        args.cities = list(ALL_12)
    elif not args.cities:
        args.cities = ["sf"]

    all_rows = []
    for c in args.cities:
        try:
            rows = diagnose_one(c, seed=args.seed)
            if rows:
                all_rows.extend(rows)
        except Exception as e:
            import traceback
            print(f"[{c}] FAILED: {e}", file=sys.stderr); traceback.print_exc()

    if not all_rows:
        return 1
    df = pd.DataFrame(all_rows)
    out = REPO / "results" / "replications" / "confounder_sensitivity_12.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV -> {out}")
    print("\n=== Summary: which blocks have cf > RV (i.e., could overturn theta) ===")
    flag = df[df["block_dropped"] != "none"].copy()
    flag["block_above_rv"] = (flag["cf_y"] > flag["rv"]) | (flag["cf_d"] > flag["rv"])
    print(flag.groupby("block_dropped")["block_above_rv"].sum().to_string())
    print(f"\nTotal city-block pairs FRAGILE: {int(flag['block_above_rv'].sum())} / {len(flag)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

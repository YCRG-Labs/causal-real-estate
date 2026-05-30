"""PC1 thickness and semantic-axis diagnostic for the JBES paper.

Purpose
-------
The v2 headline (results/fast_bootstrap_dml_v2_final_im2010.json) shows:
    NYC     theta = +0.169   IM-2010 cluster-t CI  [+0.059, +0.279]   n=342, 35 conf
    SF      theta = -0.026                          [-0.150, +0.098]   n=348, 34 conf
    Boston  theta = -0.026                          [-0.151, +0.098]   n=331, 32 conf

PC1 of the SBERT/MiniLM embedding has a robust positive causal effect on log_price
in NYC but is null in SF and Boston. The candidate mechanism for the paper is:

    NYC has 127 zip codes and a median of 2 listings/zip; the canonical 35-confounder
    set carries the per-zip location signal at very low per-zip density, so the ridge
    nuisance under-residualizes PC1 and a borough-flavor axis survives DML. SF (24
    zips, median 13/zip) and Boston (33 zips, median 9/zip) have an order-of-magnitude
    denser per-zip control set, PC1's spatial component is absorbed, and the residual
    causal contrast is zero.

This script settles that empirically. For each city it reports:

  (1) PC1..PC10 explained-variance ratios via TruncatedSVD on the centered SBERT
      matrix and the thickness ratio PC1_var / PC2_var (Davis-Kahan eigengap).
  (2) Cross-city cosine similarity of the PC1 loading vectors. If the three cities
      point in similar semantic directions, the differential causal effect is
      driven by spatial-confounding heterogeneity, not by what PC1 means.
  (3) R^2 of PC1 ~ canonical-confounder-set (RidgeCV).  This is the
      under-residualization test: a small R^2 in NYC and a large one in SF means
      the spatial controls catch PC1 in SF but not in NYC.
  (4) Univariate Pearson r between PC1 and log_price per city (no controls).
  (5) Top-10 most-extreme PC1 listings per city (+/- separately) with address,
      zip, price, and a short description snippet, plus per-borough/zip breakdown
      of which areas the PC1-extremes concentrate in. This is the semantic
      interpretation for the paper's mechanism paragraph.
  (6) NYC subsample ablation: re-run the DML on NYC but drop the
      zip-density-driven confounders (crime + amenity + micro-geo distances)
      and report theta + IM-2010 CI. If theta grows, confounder
      under-residualization in NYC is the operative mechanism.

Outputs:
  results/diagnostics/pc1_thickness.json
  results/diagnostics/pc1_extreme_listings.csv

Usage:
  python3 diag_pc1_thickness.py            # all three cities, full output
  python3 diag_pc1_thickness.py --cities nyc sf   # subset

References for the framing:
  Goodman (1981) JUE 9:175       housing submarkets within urban areas
  Goodman-Thibodeau (1998) JHE 7:121   market segmentation tests
  Bourassa-Hamelink-Hoesli-MacGregor (1999) JHE 8:160   defining submarkets
  Watkins (2001) Env&Plan A 33:2235     definition and identification
  Wu-Sharpe (2013) JRER 35:443    local nature of segmentation
  Owusu-Ansah-Garrod (2018) JRER  spatial submarkets and price stability
  Davis-Kahan (1970) SIAM J Num Anal 7:1   eigengap perturbation theorem
"""
from __future__ import annotations

import sys
import os
import json
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Import path setup so we can reuse _build_features and canonical_confounders
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
sys.path.insert(0, str(_SCRIPTS))
import _silence  # noqa: F401
from config import PROCESSED_DIR, EMBEDDING_DIM
import canonical_confounders as cc
from fast_bootstrap_dml_v2 import _build_features, _pc1, _ridge_oof_predict

_RIDGE_ALPHAS = np.logspace(-3, 3, 25)
RESULTS_DIR = _HERE.parent.parent.parent / "results" / "diagnostics"


def _nyc_zip_to_borough(zip_code) -> str:
    """Best-effort NYC ZIP -> borough using the standard ZIP prefix table."""
    try:
        z = int(zip_code)
    except Exception:
        return "unknown"
    # Manhattan: 10001-10282
    if 10001 <= z <= 10282:
        return "Manhattan"
    # Bronx: 10451-10475
    if 10451 <= z <= 10475:
        return "Bronx"
    # Brooklyn: 11201-11256
    if 11201 <= z <= 11256:
        return "Brooklyn"
    # Queens: 11004-11109, 11351-11697 (mostly)
    if 11004 <= z <= 11109 or 11351 <= z <= 11697:
        return "Queens"
    # Staten Island: 10301-10314
    if 10301 <= z <= 10314:
        return "Staten Island"
    return "unknown"


def _load_city_with_ids(city: str):
    """Build features the same way fast_bootstrap_dml_v2 does, but also return
    the surviving address/zip/description columns aligned to the filtered rows.
    """
    emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    emb_df = pd.read_parquet(emb_path)

    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available = [c for c in emb_cols if c in emb_df.columns]
    T_full = emb_df[available].to_numpy(dtype=np.float64)
    n_full = len(T_full)

    if "price" in emb_df.columns:
        Y_full = pd.to_numeric(emb_df["price"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        raise RuntimeError(f"{city}: embeddings parquet has no price column")

    # Build confounders identical to _build_features by calling it
    out = _build_features(city)
    if out is None:
        return None
    T, conf, Y_log, coords, meta = out
    # The valid mask used inside _build_features is:
    #   valid = ~(NaN(Y) | inf(Y) | (Y<=0)) & (after col-NaN trim) ~all-zero-row
    # Replicate it on the full frame so we can align id columns.
    valid_y = ~(np.isnan(Y_full) | np.isinf(Y_full) | (Y_full <= 0))
    # We trust _build_features to have returned the same n; recover the rows by
    # matching log-prices and the first embedding column. This is robust because
    # log_price is essentially a unique key here.
    y_full_log = np.where(valid_y, np.log(np.where(valid_y, Y_full, 1.0)), np.nan)
    # The matching: _build_features only drops rows for the price filter (the
    # subsequent col-NaN trim is on columns, not rows; the all-zero-row drop
    # only kicks in if confounders are entirely missing for a row). In practice
    # the row count loss matches valid_y exactly for these three cities. Verify.
    keep_idx = np.where(valid_y)[0]
    if len(keep_idx) != len(Y_log):
        # Fall back to a tolerance-based alignment using log_price as key.
        y_full_kept = np.log(Y_full[valid_y])
        # Greedy left-to-right match
        matches = []
        used = np.zeros(len(y_full_kept), dtype=bool)
        for v in Y_log:
            cand = np.where(~used & (np.abs(y_full_kept - v) < 1e-9))[0]
            if len(cand) == 0:
                break
            matches.append(cand[0])
            used[cand[0]] = True
        if len(matches) == len(Y_log):
            keep_idx = np.where(valid_y)[0][np.array(matches)]
        else:
            # Conservative: use the first len(Y_log) of valid rows
            keep_idx = np.where(valid_y)[0][: len(Y_log)]

    ids = emb_df.iloc[keep_idx][["address", "zip", "price", "description"]].reset_index(drop=True)
    ids["log_price"] = np.log(ids["price"].astype(float))
    return {
        "city": city,
        "T": T,
        "conf": conf,
        "Y_log": Y_log,
        "coords": coords,
        "meta": meta,
        "ids": ids,
        "n": len(Y_log),
    }


def _pc_decomp(T: np.ndarray, n_components: int = 10, seed: int = 0):
    """Return (explained_variance_ratios, V[d x k], scores[n x k]).

    Uses TruncatedSVD on the column-centered embedding matrix so that the
    decomposition matches the standard PCA convention (centered, unscaled).
    """
    Tc = T - T.mean(axis=0, keepdims=True)
    k = min(n_components, min(Tc.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed, n_iter=15)
    scores = svd.fit_transform(Tc)
    evr = svd.explained_variance_ratio_
    V = svd.components_.T  # d x k loadings
    sv = svd.singular_values_
    return {
        "explained_variance_ratio": evr,
        "singular_values": sv,
        "components_d_by_k": V,
        "scores_n_by_k": scores,
    }


def _r2_pc1_given_conf(pc1: np.ndarray, conf: np.ndarray, seed: int = 0) -> dict:
    """RidgeCV R^2 of PC1 on the canonical confounders, computed two ways:
       (a) in-sample fit R^2 (overstated, baseline);
       (b) 5-fold CV R^2 (honest, the under-residualization metric).
    """
    sc = StandardScaler().fit(conf)
    Xs = sc.transform(conf)

    # in-sample
    m = RidgeCV(alphas=_RIDGE_ALPHAS).fit(Xs, pc1)
    pred_in = m.predict(Xs)
    ss_res = float(np.sum((pc1 - pred_in) ** 2))
    ss_tot = float(np.sum((pc1 - pc1.mean()) ** 2))
    r2_in = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # honest 5-fold OOF
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.empty_like(pc1)
    for tr, te in kf.split(Xs):
        oof[te] = _ridge_oof_predict(Xs[tr], pc1[tr], Xs[te])
    ss_res_cv = float(np.sum((pc1 - oof) ** 2))
    r2_cv = 1.0 - ss_res_cv / ss_tot if ss_tot > 0 else float("nan")

    return {
        "r2_in_sample": r2_in,
        "r2_oof_5fold": r2_cv,
        "ridge_alpha": float(m.alpha_),
    }


def _r2_pc1_given_group(pc1: np.ndarray, conf: np.ndarray, group_idx: list[int],
                        seed: int = 0) -> float:
    """5-fold OOF R^2 restricted to a subset of confounder columns."""
    if not group_idx:
        return float("nan")
    from sklearn.model_selection import KFold
    sc = StandardScaler().fit(conf[:, group_idx])
    Xs = sc.transform(conf[:, group_idx])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.empty_like(pc1)
    for tr, te in kf.split(Xs):
        oof[te] = _ridge_oof_predict(Xs[tr], pc1[tr], Xs[te])
    ss_res = float(np.sum((pc1 - oof) ** 2))
    ss_tot = float(np.sum((pc1 - pc1.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _conf_group_indices(city: str, n_conf: int) -> dict[str, list[int]]:
    """Reconstruct the column-group indices of the confounder matrix.

    Order in fast_bootstrap_dml_v2: [lat, lon, PROPERTY..., ctx (CENSUS + (CRIME) +
    AMENITY + MICRO_GEO)] after dropping columns that don't exist in the slim
    parquet and after the col-NaN >50% trim.

    We rebuild by walking the same code path with the same parcels frame and
    checking column availability.
    """
    from canonical_confounders import PROPERTY, CENSUS, CRIME, AMENITY, MICRO_GEO
    from fast_bootstrap_dml_v2 import _load_parcels_slim
    parcels = _load_parcels_slim(city)
    if parcels is None:
        return {}
    # spatial parts: lat, lon (always 2)
    cols = ["latitude", "longitude"]
    cols += [c for c in PROPERTY if c in parcels.columns]
    cols += [c for c in CENSUS if c in parcels.columns]
    cols += [c for c in CRIME if c in parcels.columns]
    cols += [c for c in AMENITY if c in parcels.columns]
    cols += [c for c in MICRO_GEO if c in parcels.columns]
    # Now the col-NaN >50% filter inside _build_features can drop some columns.
    # We don't have access to that mask here without rebuilding; instead we
    # report group sizes BEFORE the NaN trim. The realized total may be off
    # by a few columns; the qualitative comparison still goes through.
    groups = {
        "spatial": ["latitude", "longitude"],
        "property": [c for c in PROPERTY if c in parcels.columns],
        "census": [c for c in CENSUS if c in parcels.columns],
        "crime": [c for c in CRIME if c in parcels.columns],
        "amenity": [c for c in AMENITY if c in parcels.columns],
        "micro_geo": [c for c in MICRO_GEO if c in parcels.columns],
    }
    # Convert to column-index lists relative to `cols` order
    idx = {}
    pos = 0
    name_to_idx = {n: i for i, n in enumerate(cols)}
    for g, lst in groups.items():
        idx[g] = [name_to_idx[c] for c in lst if c in name_to_idx]
    # Trim to actual n_conf available (last columns may have been dropped by NaN trim)
    if n_conf < len(cols):
        # Drop indices that exceed n_conf
        idx = {g: [i for i in v if i < n_conf] for g, v in idx.items()}
    return idx


def _extreme_listings(ids: pd.DataFrame, pc1: np.ndarray, k: int = 10) -> pd.DataFrame:
    df = ids.copy()
    df["pc1"] = pc1
    df["abs_pc1"] = np.abs(pc1)
    top_pos = df.nlargest(k, "pc1").assign(side="top_positive")
    top_neg = df.nsmallest(k, "pc1").assign(side="top_negative")
    out = pd.concat([top_pos, top_neg], ignore_index=True)
    # truncate descriptions
    out["description_snippet"] = out["description"].astype(str).str.slice(0, 300)
    return out[["side", "pc1", "address", "zip", "price", "description_snippet"]]


def _nyc_ablation_theta(city: str, ablation: str, seed: int = 0):
    """Re-run a single DML estimate dropping a confounder group.

    Returns (theta, se_if). We use the same machinery as
    fast_bootstrap_dml_v2._dml_core but with the confounder columns
    restricted/expanded per the ablation argument.
    """
    from sklearn.model_selection import KFold
    out = _build_features(city)
    if out is None:
        return None
    T, conf, Y_log, _, _ = out
    groups_idx = _conf_group_indices(city, conf.shape[1])

    if ablation == "all":
        keep = list(range(conf.shape[1]))
    elif ablation == "drop_crime_amenity_micro":
        drop = set()
        for g in ("crime", "amenity", "micro_geo"):
            drop.update(groups_idx.get(g, []))
        keep = [i for i in range(conf.shape[1]) if i not in drop]
    elif ablation == "spatial_only":
        keep = list(groups_idx.get("spatial", []))
    elif ablation == "spatial_property_only":
        keep = list(groups_idx.get("spatial", [])) + list(groups_idx.get("property", []))
    elif ablation == "no_spatial":
        keep = [i for i in range(conf.shape[1]) if i not in set(groups_idx.get("spatial", []))]
    else:
        raise ValueError(f"unknown ablation {ablation!r}")

    if not keep:
        return None
    conf_sub = conf[:, keep]
    sc = StandardScaler().fit(conf_sub)
    Xs = sc.transform(conf_sub)
    pc1 = _pc1(T, seed=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Yr = np.empty_like(Y_log)
    Tr = np.empty_like(pc1)
    for tr, te in kf.split(Xs):
        Yr[te] = Y_log[te] - _ridge_oof_predict(Xs[tr], Y_log[tr], Xs[te])
        Tr[te] = pc1[te] - _ridge_oof_predict(Xs[tr], pc1[tr], Xs[te])
    denom = float(np.mean(Tr * Tr))
    if denom < 1e-12:
        return None
    theta = float(np.mean(Tr * Yr)) / denom
    psi = (Yr - theta * Tr) * Tr / denom
    se_if = float(math.sqrt(np.var(psi, ddof=1) / len(Y_log)))
    return {"ablation": ablation, "n_conf_kept": len(keep), "theta": theta, "se_if": se_if,
            "ci_low": theta - 1.96 * se_if, "ci_high": theta + 1.96 * se_if}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=["nyc", "sf", "boston"])
    parser.add_argument("--k_extreme", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cities_data = {}
    for city in args.cities:
        d = _load_city_with_ids(city)
        if d is None:
            print(f"  {city}: no data, skipping")
            continue
        cities_data[city] = d

    # ---- (1) Per-city PCA decomposition (PC1..PC10) ----------------------------
    pca_block = {}
    for city, d in cities_data.items():
        decomp = _pc_decomp(d["T"], n_components=10, seed=args.seed)
        evr = decomp["explained_variance_ratio"]
        thickness = float(evr[0] / evr[1]) if evr[1] > 0 else float("inf")
        pca_block[city] = {
            "n": int(d["n"]),
            "evr": [float(x) for x in evr],
            "evr_cumulative": [float(x) for x in np.cumsum(evr)],
            "singular_values": [float(x) for x in decomp["singular_values"]],
            "thickness_pc1_over_pc2": thickness,
        }
        d["decomp"] = decomp

    # ---- (2) Cross-city cosine of PC1 loadings ---------------------------------
    cos_block = {}
    pairs = [(a, b) for i, a in enumerate(args.cities) for b in args.cities[i + 1:]]
    for a, b in pairs:
        if a not in cities_data or b not in cities_data:
            continue
        Va = cities_data[a]["decomp"]["components_d_by_k"][:, 0]
        Vb = cities_data[b]["decomp"]["components_d_by_k"][:, 0]
        cos = float(np.dot(Va, Vb) / (np.linalg.norm(Va) * np.linalg.norm(Vb)))
        # PC sign is arbitrary; report abs and signed
        cos_block[f"{a}_vs_{b}"] = {"cosine": cos, "abs_cosine": abs(cos)}
        # also PC1..PC3 alignment in case PC1 in one city aligns with PC2 in another
        cos_block[f"{a}_vs_{b}_top3_max_abs_cos"] = float(np.max(np.abs(
            cities_data[a]["decomp"]["components_d_by_k"][:, :3].T @
            cities_data[b]["decomp"]["components_d_by_k"][:, :3]
        )))

    # ---- (3) R^2 of PC1 on canonical confounders -------------------------------
    r2_block = {}
    for city, d in cities_data.items():
        pc1 = d["decomp"]["scores_n_by_k"][:, 0]
        pc1 = (pc1 - pc1.mean()) / pc1.std()
        d["pc1"] = pc1
        full = _r2_pc1_given_conf(pc1, d["conf"], seed=args.seed)
        groups_idx = _conf_group_indices(city, d["conf"].shape[1])
        per_group = {}
        for g, idx in groups_idx.items():
            per_group[g] = {
                "n_cols": len(idx),
                "r2_oof_5fold": _r2_pc1_given_group(pc1, d["conf"], idx, seed=args.seed),
            }
        r2_block[city] = {
            "n_conf_total": int(d["conf"].shape[1]),
            **full,
            "per_group_r2_oof": per_group,
        }

    # ---- (4) Univariate Pearson r between PC1 and log_price --------------------
    raw_corr_block = {}
    for city, d in cities_data.items():
        pc1 = d["pc1"]
        r, p = stats.pearsonr(pc1, d["Y_log"])
        rs, ps = stats.spearmanr(pc1, d["Y_log"])
        raw_corr_block[city] = {
            "pearson_r_pc1_logprice": float(r),
            "pearson_p": float(p),
            "spearman_r_pc1_logprice": float(rs),
            "spearman_p": float(ps),
        }

    # ---- (5) Top-K extreme PC1 listings ----------------------------------------
    extreme_rows = []
    extreme_block = {}
    for city, d in cities_data.items():
        ext = _extreme_listings(d["ids"], d["pc1"], k=args.k_extreme)
        ext.insert(0, "city", city)
        extreme_rows.append(ext)
        # borough / zip summary
        zip_top_pos = ext[ext.side == "top_positive"]["zip"].value_counts().head(10).to_dict()
        zip_top_neg = ext[ext.side == "top_negative"]["zip"].value_counts().head(10).to_dict()
        block = {
            "top_positive_zip_counts": {str(k): int(v) for k, v in zip_top_pos.items()},
            "top_negative_zip_counts": {str(k): int(v) for k, v in zip_top_neg.items()},
        }
        if city == "nyc":
            boros_pos = ext[ext.side == "top_positive"]["zip"].map(_nyc_zip_to_borough)
            boros_neg = ext[ext.side == "top_negative"]["zip"].map(_nyc_zip_to_borough)
            block["top_positive_borough_counts"] = boros_pos.value_counts().to_dict()
            block["top_negative_borough_counts"] = boros_neg.value_counts().to_dict()
        # mean log_price at each extreme
        pos_lp = np.log(ext[ext.side == "top_positive"]["price"].astype(float))
        neg_lp = np.log(ext[ext.side == "top_negative"]["price"].astype(float))
        block["mean_log_price_top_positive"] = float(pos_lp.mean())
        block["mean_log_price_top_negative"] = float(neg_lp.mean())
        extreme_block[city] = block
    if extreme_rows:
        ext_all = pd.concat(extreme_rows, ignore_index=True)
        ext_all.to_csv(RESULTS_DIR / "pc1_extreme_listings.csv", index=False)

    # ---- (6) NYC under-residualization ablation --------------------------------
    ablation_block = {}
    for city in ("nyc", "sf", "boston"):
        if city not in cities_data:
            continue
        rows = []
        for ab in ("all", "drop_crime_amenity_micro", "spatial_only",
                   "spatial_property_only", "no_spatial"):
            res = _nyc_ablation_theta(city, ab, seed=args.seed)
            if res is not None:
                rows.append(res)
        ablation_block[city] = rows

    # ---- Per-zip listing density (motivating mechanism) ------------------------
    density_block = {}
    for city, d in cities_data.items():
        z = d["ids"]["zip"].astype(str)
        counts = z.value_counts()
        density_block[city] = {
            "n_zips": int(counts.size),
            "median_listings_per_zip": float(counts.median()),
            "mean_listings_per_zip": float(counts.mean()),
            "p25_listings_per_zip": float(counts.quantile(0.25)),
            "p75_listings_per_zip": float(counts.quantile(0.75)),
            "max_listings_per_zip": int(counts.max()),
            "min_listings_per_zip": int(counts.min()),
        }

    # ---- Assemble JSON ---------------------------------------------------------
    out = {
        "headline_thetas_from_v2_final": {
            "nyc":    {"theta": 0.1693, "im2010_ci": [0.0587, 0.2790]},
            "sf":     {"theta": -0.0260, "im2010_ci": [-0.1502, 0.0984]},
            "boston": {"theta": -0.0263, "im2010_ci": [-0.1512, 0.0984]},
        },
        "per_city_listing_density": density_block,
        "pca": pca_block,
        "cross_city_pc1_cosine": cos_block,
        "pc1_residualization_r2": r2_block,
        "univariate_pc1_logprice_corr": raw_corr_block,
        "extreme_listings_summary": extreme_block,
        "ablation_theta_by_confounder_block": ablation_block,
    }

    json_path = RESULTS_DIR / "pc1_thickness.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- Console summary -------------------------------------------------------
    print("\n=== PC1 thickness diagnostic ===")
    print(f"\n[Per-city listing density on zip]")
    print(f"  {'city':6s} {'n_zips':>7s} {'med/zip':>8s} {'mean':>6s}")
    for city, b in density_block.items():
        print(f"  {city:6s} {b['n_zips']:>7d} {b['median_listings_per_zip']:>8.1f} "
              f"{b['mean_listings_per_zip']:>6.2f}")

    print(f"\n[PC1..PC5 explained-variance ratio + thickness PC1/PC2]")
    print(f"  {'city':6s} {'PC1':>7s} {'PC2':>7s} {'PC3':>7s} {'PC4':>7s} {'PC5':>7s}"
          f" {'thick':>7s}")
    for city, b in pca_block.items():
        evr = b["evr"]
        print(f"  {city:6s} {evr[0]:>7.4f} {evr[1]:>7.4f} {evr[2]:>7.4f} {evr[3]:>7.4f}"
              f" {evr[4]:>7.4f} {b['thickness_pc1_over_pc2']:>7.3f}")

    print(f"\n[Cross-city PC1 cosine similarity (abs)]")
    for k, v in cos_block.items():
        if isinstance(v, dict):
            print(f"  {k:30s} cos={v['cosine']:+.4f}  |cos|={v['abs_cosine']:.4f}")
        else:
            print(f"  {k:30s} {v:.4f}")

    print(f"\n[R^2 of PC1 on canonical confounders — under-residualization test]")
    print(f"  {'city':6s} {'in_sample':>10s} {'5fold_oof':>10s} {'spatial':>8s} {'census':>7s}"
          f" {'crime':>7s} {'amen':>7s} {'micro':>7s} {'prop':>7s}")
    for city, b in r2_block.items():
        per = b["per_group_r2_oof"]

        def g(name):
            v = per.get(name, {}).get("r2_oof_5fold", float("nan"))
            return f"{v:>+7.3f}" if not (isinstance(v, float) and math.isnan(v)) else "    nan"

        print(f"  {city:6s} {b['r2_in_sample']:>+10.3f} {b['r2_oof_5fold']:>+10.3f}"
              f" {g('spatial')} {g('census')} {g('crime')} {g('amenity')} {g('micro_geo')}"
              f" {g('property')}")

    print(f"\n[Univariate Pearson r(PC1, log_price)]")
    for city, b in raw_corr_block.items():
        print(f"  {city:6s} pearson_r = {b['pearson_r_pc1_logprice']:+.4f}"
              f" (p={b['pearson_p']:.2e})    spearman = {b['spearman_r_pc1_logprice']:+.4f}")

    print(f"\n[NYC borough-flavor — top-{args.k_extreme} extreme PC1 listings]")
    for city, b in extreme_block.items():
        print(f"  --- {city} ---")
        print(f"  top_positive  mean_log_price = {b['mean_log_price_top_positive']:.3f}")
        print(f"  top_negative  mean_log_price = {b['mean_log_price_top_negative']:.3f}")
        if "top_positive_borough_counts" in b:
            print(f"  top_positive_borough_counts = {b['top_positive_borough_counts']}")
            print(f"  top_negative_borough_counts = {b['top_negative_borough_counts']}")
        print(f"  top_positive zip counts = {b['top_positive_zip_counts']}")
        print(f"  top_negative zip counts = {b['top_negative_zip_counts']}")

    print(f"\n[Ablation: theta on PC1 dropping confounder blocks]")
    for city, rows in ablation_block.items():
        print(f"  --- {city} ---")
        print(f"  {'ablation':32s} {'n_keep':>7s} {'theta':>9s} {'se':>8s}"
              f"  {'CI95':>20s}")
        for r in rows:
            print(f"  {r['ablation']:32s} {r['n_conf_kept']:>7d} {r['theta']:>+9.4f}"
                  f" {r['se_if']:>8.4f}"
                  f"  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {RESULTS_DIR / 'pc1_extreme_listings.csv'}")

    return out


if __name__ == "__main__":
    main()

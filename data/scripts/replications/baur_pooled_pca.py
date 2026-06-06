"""Baur, Rosenfelder & Lutz (2023) BERT-PC1 DML, with pooled cross-market PC1.

Re-runs the 12-city Baur replication, replacing the per-market PC1 of the
sentence-BERT embedding with a single global axis defined by the leading
eigenvector of the pooled within-city-centered embedding matrix
(see `pooled_pca_treatment.py`).  This eliminates the sign-flip across
markets that the per-market PC1 specification produces, because the
treatment is now a stable, encoder-defined function of the embedding that
does not depend on the local listing distribution.

The DML estimand on this treatment is "the effect of a per-σ within-city
move along the global semantic axis on log sale price", comparable
across the twelve markets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from replications.baur_2023 import (  # noqa: E402
    _try_import_lightgbm, cv_metrics, get_features_and_target,
    load_analysis_data,
)
from replications.compare_to_dml import result_to_dict, run_dml  # noqa: E402

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _attach_pooled_treatment(city: str, emb_df: pd.DataFrame,
                              pooled_csv: Path) -> np.ndarray | None:
    """Merge the pre-computed pooled PCA scalar treatment onto each
    listing's row in emb_df.  Returns a (n, 1) array aligned with emb_df.
    """
    pool = pd.read_csv(pooled_csv)
    pool_city = pool[pool["city"] == city].copy()
    if len(pool_city) == 0:
        print(f"  [warn] no pooled-PCA rows for {city}")
        return None
    if "listing_id" not in emb_df.columns:
        emb_df = emb_df.assign(listing_id=np.arange(len(emb_df)))
    merged = emb_df[["listing_id"]].merge(
        pool_city[["listing_id", "treatment_z"]],
        on="listing_id", how="left",
    )
    n_missing = merged["treatment_z"].isna().sum()
    if n_missing > 0:
        print(f"  [warn] {n_missing} listings without pooled-PCA score; "
              "filling with city mean (0 after z-score)")
        merged["treatment_z"] = merged["treatment_z"].fillna(0.0)
    return merged["treatment_z"].to_numpy().reshape(-1, 1)


def run_baur_pooled(
    city: str = "sf",
    pooled_csv: Path = REPO / "results" / "replications"
                          / "pooled_pca_treatment.csv",
    n_subset: int | None = None,
    seed: int = 42,
    k_folds: int = 5,
    fast: bool = False,
    n_boot: int | None = None,
) -> dict:
    print(f"\n=== Baur (2023) pooled-PCA replication: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    _T_emb, confounders, Y_log, meta = feats

    T_pooled = _attach_pooled_treatment(city, emb_df, pooled_csv)
    if T_pooled is None:
        return {"city": city, "error": "no pooled treatment"}
    # `_attach_pooled_treatment` builds T_pooled from the UNFILTERED emb_df, so
    # it has one row per original listing. `get_features_and_target` may have
    # dropped INTERIOR rows (non-positive/NaN/Inf price or all-zero confounders)
    # from confounders/Y_log. Subset T_pooled by the same meta["valid_mask"] so
    # the treatment pairs with the surviving listings before it is passed
    # positionally to the DML.
    valid_mask = np.asarray(meta["valid_mask"], dtype=bool)
    T_pooled = T_pooled[valid_mask]
    assert len(T_pooled) == len(confounders) == len(Y_log), (
        f"row-alignment failure: T_pooled={len(T_pooled)} "
        f"confounders={len(confounders)} Y_log={len(Y_log)}")

    if n_subset is not None and n_subset < len(Y_log):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Y_log), size=n_subset, replace=False)
        idx.sort()
        T_pooled = T_pooled[idx]
        confounders = confounders[idx]
        Y_log = Y_log[idx]

    n = len(Y_log)
    print(f"  N={n:,}, structured features={confounders.shape[1]}, "
          f"pooled treatment dim={T_pooled.shape[1]}")
    print(f"  treatment: mean={T_pooled.mean():+.4f}, "
          f"sd={T_pooled.std():+.4f}, range=[{T_pooled.min():+.4f}, "
          f"{T_pooled.max():+.4f}]")

    if fast:
        dml_use_ridge = True
        if n_boot is not None and n_boot > 0:
            dml_ci_method, dml_n_boot = "bootstrap", n_boot
        else:
            dml_ci_method, dml_n_boot = "if", None
    else:
        dml_use_ridge = False
        dml_ci_method = "bootstrap"
        dml_n_boot = n_boot if n_boot is not None else 500
    backend = "ridge" if dml_use_ridge else "gbm"
    print(f"  DML on pooled BERT PC1 (backend={backend}, "
          f"ci_method={dml_ci_method}, n_boot={dml_n_boot})...")
    dml = run_dml(T_pooled, confounders, Y_log,
                   label="DML on pooled BERT PC1",
                   ci_method=dml_ci_method, n_boot=dml_n_boot,
                   use_ridge=dml_use_ridge, seed=seed, n_pca=1)
    if dml is None:
        print("    DML failed")
        return {"city": city, "error": "DML failed"}
    flag = "contains 0" if dml.contains_zero else "EXCLUDES 0"
    print(f"    DML θ={dml.theta:+.4f}  se={dml.se:.4f}  "
          f"95%CI=[{dml.ci_low:+.4f}, {dml.ci_high:+.4f}]  {flag}")

    return {
        "city": city,
        "n": int(n),
        "engine": backend,
        "treatment": "pooled_within_city_centered_pca_pc1",
        "dml": result_to_dict(dml),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n_boot", type=int, default=None)
    ap.add_argument("--pooled_csv",
                    default=str(REPO / "results" / "replications"
                                  / "pooled_pca_treatment.csv"))
    ap.add_argument("--out_dir", type=Path,
                    default=REPO / "results" / "replications"
                              / "baur_pooled_pca")
    args = ap.parse_args()

    cities = list(ALL_12) if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in cities:
        r = run_baur_pooled(c, pooled_csv=Path(args.pooled_csv),
                             n_subset=args.n, seed=args.seed,
                             k_folds=args.k_folds, fast=args.fast,
                             n_boot=args.n_boot)
        out_path = args.out_dir / f"{c}.json"
        out_path.write_text(json.dumps(r, indent=2, default=float))
        rows.append(r)

    # Table rollup
    tbl_rows = []
    for r in rows:
        if "error" in r:
            tbl_rows.append({"city": r["city"], "error": r["error"]})
            continue
        d = r["dml"]
        tbl_rows.append({
            "city": r["city"], "n": r["n"],
            "dml_theta": d["theta"], "dml_se": d["se"],
            "dml_ci_low": d["ci_low"], "dml_ci_high": d["ci_high"],
            "dml_excludes_zero": (not d["contains_zero"]),
        })
    df = pd.DataFrame(tbl_rows)
    tbl_csv = args.out_dir / "baur_pooled_pca_table.csv"
    df.to_csv(tbl_csv, index=False)
    print(f"\n=== Table ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nTable -> {tbl_csv}")
    print(f"Per-city JSONs -> {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())

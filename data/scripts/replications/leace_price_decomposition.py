"""Decompose each market's text-price premium into a spatially-confounded share
(removed by erasing latitude-longitude from the embedding) and a residual share.

For each city we estimate the Baur PC1 DML effect on log price twice:
  raw_theta     = effect on PC1 of the un-erased embedding
  erased_theta  = effect on PC1 of the LEACE-erased embedding (lat-lon removed)
The confounded share is (raw_theta - erased_theta); the residual semantic share
is erased_theta. Both PC1s are oriented to the within-sample price gradient so
the two estimates are sign-comparable. This produces the data for the headline
Gelbach-style decomposition figure (fig:decomposition), which does not yet exist
because LEACE was previously used only as a representation diagnostic.

Run on Brev (needs torch + the 12 embedding parquets):
  python3 data/scripts/replications/leace_price_decomposition.py --all_12 --fast
Writes results/replications/leace_price_decomposition.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from replications.baur_2023 import get_features_and_target, load_analysis_data
from replications.compare_to_dml import run_dml
from leace_deconfound import leace_erase

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _oriented_pc1(emb: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Leading PC of `emb`, sign-oriented so it co-varies positively with y."""
    pc = PCA(n_components=1, random_state=seed).fit_transform(emb)[:, 0]
    if np.corrcoef(pc, y)[0, 1] < 0:
        pc = -pc
    pc = (pc - pc.mean()) / (pc.std(ddof=1) or 1.0)
    return pc.reshape(-1, 1)


def _latlon(emb_df: pd.DataFrame, valid_mask: np.ndarray) -> np.ndarray | None:
    cols = [c for c in ("latitude", "longitude") if c in emb_df.columns]
    if len(cols) != 2:
        return None
    z = emb_df[["latitude", "longitude"]].to_numpy(dtype=float)[valid_mask]
    return StandardScaler().fit_transform(z)


def decompose_city(city: str, fast: bool, seed: int = 42) -> dict:
    print(f"\n=== price decomposition: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    T_emb, confounders, Y_log, meta = feats
    valid = np.asarray(meta["valid_mask"], dtype=bool)
    Z = _latlon(emb_df, valid)
    if Z is None:
        return {"city": city, "error": "no lat-lon"}
    T_emb = np.asarray(T_emb, dtype=float)
    assert len(T_emb) == len(confounders) == len(Y_log) == len(Z)

    # erase the lat-lon concept from the embedding (LEACE, linear guardedness)
    T_erased, _, _, resid = leace_erase(T_emb, Z, shrinkage=True)
    T_erased = np.asarray(T_erased, dtype=float)
    print(f"  n={len(Y_log):,}  guardedness residual={float(resid):.2e}")

    pc_raw = _oriented_pc1(T_emb, Y_log, seed)
    pc_era = _oriented_pc1(T_erased, Y_log, seed)
    kw = dict(label="decomp", ci_method="if", n_boot=None,
              use_ridge=fast, seed=seed, n_pca=1)
    dml_raw = run_dml(pc_raw, confounders, Y_log, **kw)
    dml_era = run_dml(pc_era, confounders, Y_log, **kw)
    if dml_raw is None or dml_era is None:
        return {"city": city, "error": "DML failed"}

    confounded = float(dml_raw.theta - dml_era.theta)
    print(f"  raw theta={dml_raw.theta:+.4f}  erased theta={dml_era.theta:+.4f}"
          f"  confounded share={confounded:+.4f}")
    return {
        "city": city, "n": int(len(Y_log)),
        "raw_theta": float(dml_raw.theta), "raw_se": float(dml_raw.se),
        "erased_theta": float(dml_era.theta), "erased_se": float(dml_era.se),
        "confounded_share": confounded,
        "confounded_frac": (confounded / dml_raw.theta) if dml_raw.theta else float("nan"),
        "guardedness_residual": float(resid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="ridge nuisances + IF intervals (fast); else GBM")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "replications"
                              / "leace_price_decomposition.csv")
    args = ap.parse_args()

    cities = list(ALL_12) if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")

    rows = [decompose_city(c, fast=args.fast, seed=args.seed) for c in cities]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n=== decomposition (per market) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nCSV -> {args.out}")
    errs = [r for r in rows if "error" in r]
    if errs:
        print("errors:", errs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

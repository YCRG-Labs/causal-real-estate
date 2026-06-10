"""Decompose each market's text-price premium into a spatially-confounded share
and a residual share, via a fixed-treatment Gelbach confounder toggle.

WHY NOT LEACE HERE: an earlier version erased lat-lon from the embedding and
re-PCA'd, then compared the DML effect of raw-PC1 vs erased-PC1. That was wrong
twice (an independent code review caught it): (a) lat-lon are ALREADY in the DML
confounder set, so both arms net out geography and the difference is structurally
~0 regardless of the truth; (b) raw-PC1 and erased-PC1 are different directions,
so subtracting their effects is not a decomposition (it manufactured the New York
"sign flip"). The representation-level LEACE erasure remains a valid diagnostic
(the embedding encodes geography), but it is not how you decompose the PRICE
effect.

CORRECT TEST (Gelbach 2016 order-invariant decomposition): hold the treatment
fixed (the oriented leading PC of the embedding) and toggle geography on the
confounder side.
  naive_theta = effect controlling for the NON-geographic confounders only
  geo_theta   = effect also controlling for location (lat, lon AND a nonlinear
                lat-lon basis: lat^2, lon^2, lat*lon)
  confounded_share = naive_theta - geo_theta   (how much location explains)
  residual_share   = geo_theta                 (survives spatial adjustment)
This holds the estimand fixed, removes the double-control, and tests nonlinear
geography, so the share actually measures spatial confounding of the text effect.

Run on Brev (needs the 12 parquets):
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
from spatial_basis import thin_plate_basis

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _oriented_pc1(emb: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Leading PC of `emb`, sign-oriented to co-vary positively with y. Held
    fixed across both DML arms, so its sign is a convention only."""
    pc = PCA(n_components=1, random_state=seed).fit_transform(emb)[:, 0]
    if np.corrcoef(pc, y)[0, 1] < 0:
        pc = -pc
    return ((pc - pc.mean()) / (pc.std(ddof=1) or 1.0)).reshape(-1, 1)


def _geo_basis(lat, lon, basis: str = "quad", k: int = 30, seed: int = 42) -> np.ndarray:
    """Location confounder basis. basis='quad' is the legacy standardized
    [lat, lon, lat*lon, lat^2, lon^2]; basis='tprs' is a rank-k thin-plate
    regression spline (mgcv s(lat,lon,bs='tp')), the flexible-scale control that
    a spatial-statistics reading prefers to the metropolitan-scale quadratic."""
    if basis == "tprs":
        return thin_plate_basis(lat, lon, k=k, seed=seed)
    la = (lat - lat.mean()) / (lat.std() or 1.0)
    lo = (lon - lon.mean()) / (lon.std() or 1.0)
    B = np.column_stack([la, lo, la * lo, la ** 2, lo ** 2])
    return StandardScaler().fit_transform(B)


def decompose_city(city: str, fast: bool, seed: int = 42,
                   basis: str = "quad", k: int = 30) -> dict:
    print(f"\n=== price-geo toggle: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    T_emb, confounders, Y_log, meta = feats
    lat = np.asarray(meta["lat"], dtype=float)
    lon = np.asarray(meta["lon"], dtype=float)
    if not np.isfinite(lat).any() or not np.isfinite(lon).any():
        return {"city": city, "error": "no lat-lon"}
    conf = np.asarray(confounders, dtype=float)
    assert len(conf) == len(Y_log) == len(lat) == len(T_emb)

    pc_raw = _oriented_pc1(np.asarray(T_emb, dtype=float), Y_log, seed)

    cor = np.array([max(abs(np.corrcoef(conf[:, j], lat)[0, 1]),
                        abs(np.corrcoef(conf[:, j], lon)[0, 1]))
                    for j in range(conf.shape[1])])
    geo_cols = cor > 0.99
    conf_naive = conf[:, ~geo_cols]
    geo_b = _geo_basis(lat, lon, basis=basis, k=k, seed=seed)
    conf_geo = np.column_stack([conf_naive, geo_b])
    print(f"  n={len(Y_log):,}  stripped {int(geo_cols.sum())} geo confounder col(s); "
          f"naive p={conf_naive.shape[1]}, geo p={conf_geo.shape[1]}")

    kw = dict(label="toggle", ci_method="if", n_boot=None,
              use_ridge=fast, seed=seed, n_pca=1)
    dml_naive = run_dml(pc_raw, conf_naive, Y_log, **kw)
    dml_geo = run_dml(pc_raw, conf_geo, Y_log, **kw)
    if dml_naive is None or dml_geo is None:
        return {"city": city, "error": "DML failed"}

    confounded = float(dml_naive.theta - dml_geo.theta)
    A = np.column_stack([np.ones(len(Y_log)), geo_b])
    coef, *_ = np.linalg.lstsq(A, pc_raw[:, 0], rcond=None)
    pc1_geo_r2 = float(1.0 - np.var(pc_raw[:, 0] - A @ coef) / (np.var(pc_raw[:, 0]) or 1.0))
    print(f"  naive theta={dml_naive.theta:+.4f}  geo theta={dml_geo.theta:+.4f}"
          f"  confounded={confounded:+.4f}  PC1-geo R^2={pc1_geo_r2:.3f}")
    return {
        "city": city, "n": int(len(Y_log)),
        "basis": basis, "k": (k if basis == "tprs" else 0),
        "geo_basis_cols": int(geo_b.shape[1]),
        "naive_theta": float(dml_naive.theta), "naive_se": float(dml_naive.se),
        "geo_theta": float(dml_geo.theta), "geo_se": float(dml_geo.se),
        "confounded_share": confounded,
        "confounded_frac": (confounded / dml_naive.theta) if dml_naive.theta else float("nan"),
        "pc1_geo_r2": pc1_geo_r2,
        "n_geo_cols_stripped": int(geo_cols.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--basis", choices=["quad", "tprs"], default="quad",
                    help="location control: quad (legacy) or tprs (thin-plate)")
    ap.add_argument("--k", type=int, default=30, help="TPRS basis rank")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cities = list(ALL_12) if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")

    if args.out is None:
        tag = args.basis + (f"{args.k}" if args.basis == "tprs" else "")
        args.out = (REPO / "results" / "replications"
                    / f"leace_price_decomposition_{tag}.csv")

    rows = [decompose_city(c, fast=args.fast, seed=args.seed,
                           basis=args.basis, k=args.k) for c in cities]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n=== price-geo decomposition (per market) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nCSV -> {args.out}")
    errs = [r for r in rows if "error" in r]
    if errs:
        print("errors:", errs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

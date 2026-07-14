"""Spectral-gap diagnostic for Assumption A1 of the estimated-direction theorem.

The estimated direction is the leading eigenvector of the pooled residualized
embedding covariance. The Koltchinskii-Lounici normal approximation for the
spectral projector requires (i) a nonzero eigen-gap separating the leading
eigenvalue from the rest, and (ii) an effective rank r(Sigma)=tr(Sigma)/||Sigma||
small relative to the pooled sample size N. This script reports both, plus the
relative gap and r(Sigma)/N, from the same residualized embeddings the
identified-direction analysis uses.

Run: source .venv/bin/activate && OMP_NUM_THREADS=1 \
     python data/scripts/replications/spectral_gap_check.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import lstsq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from replications.baur_2023 import get_features_and_target, load_analysis_data

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
    Rs = []
    for c in ALL_12:
        loaded = load_analysis_data(c)
        if loaded is None:
            continue
        feats = get_features_and_target(*loaded, drop_mismatched_crime=True)
        if feats is None:
            continue
        T, conf, Y, meta = feats
        B = loc_basis(meta["lat"], meta["lon"], meta["zip_labels"])
        beta, *_ = lstsq(B, T, rcond=None)
        Rs.append(T - B @ beta)
        print(f"  loaded {c:12} N={len(Y):6} p={T.shape[1]}", flush=True)

    R_all = np.vstack(Rs)
    N, p = R_all.shape
    Rc = R_all - R_all.mean(0)
    # eigenvalues of the pooled residualized covariance via SVD
    sv = np.linalg.svd(Rc, compute_uv=False)
    eig = (sv ** 2) / N                      # covariance eigenvalues
    eig = np.sort(eig)[::-1]
    lam1, lam2 = float(eig[0]), float(eig[1])
    trace = float(eig.sum())
    gap = lam1 - lam2
    rel_gap = gap / lam1
    eff_rank = trace / lam1                   # r(Sigma) = tr(Sigma)/||Sigma||
    out = {
        "N_pooled": int(N), "p_dim": int(p),
        "lambda_1": lam1, "lambda_2": lam2,
        "gap_abs": gap, "gap_relative": rel_gap,
        "trace": trace, "effective_rank": eff_rank,
        "eff_rank_over_N": eff_rank / N,
        "top5_eigenvalues": [float(x) for x in eig[:5]],
        "share_leading": lam1 / trace,
    }
    (OUT / "spectral_gap.json").write_text(json.dumps(out, indent=2))
    print("\n=== pooled residualized embedding covariance spectrum ===")
    print(f"N={N}  p={p}")
    print(f"lambda_1={lam1:.5f}  lambda_2={lam2:.5f}  gap={gap:.5f}  "
          f"relative gap={rel_gap:.3f}")
    print(f"leading eigenvalue share of trace = {lam1/trace:.3f}")
    print(f"effective rank r(Sigma)=tr/lambda_1 = {eff_rank:.2f}")
    print(f"r(Sigma)/N = {eff_rank/N:.5f}   (A1 needs this small)")
    print(f"top-5 eigenvalues: {[round(x,4) for x in eig[:5]]}")
    print(f"wrote {OUT}/spectral_gap.json")


if __name__ == "__main__":
    main()

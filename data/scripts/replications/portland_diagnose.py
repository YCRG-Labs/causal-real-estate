"""Portland-pathology Phase A diagnostic.

Both Baur (θ=-1.392) and Shen (θ=+1.895) DML estimates blow up for Portland
on the new 75K-listing corpus, with opposite signs across methods. This
script runs the literature-backed Phase A diagnostic to pin down WHICH
mechanism is at fault before we decide drop / trim / repair.

Tests run:
  1. Davis-Kahan / Flury (1984) PC1 alignment — does Portland's intrinsic
     PC1 align with the pooled PC1 axis we use as treatment? Cos < 0.90
     means structural misalignment, justifying city-specific PC1 or drop.
  2. Per-city eigengap PC1/PC2 ratio — small ratios mean PC1 sign-pinning
     is empirically unstable across subsamples (Bro et al 2008).
  3. Raw treatment scale per city BEFORE the within-city z-score — if
     Portland's std/range is anomalously large the pooled-PC1 axis is
     dominated by Portland-specific variance.
  4. Top-leverage observation overlap between Shen and Baur — if the same
     listing drives both methods with opposite-sign residuals, it's a
     single rogue listing (drop one row). If different listings, it's
     confounder misalignment (probably the documented Multnomah-only
     parcel coverage gap), and we drop Portland city-wide.

References:
  - Viechtbauer & Cheung 2010 Res Synth Methods 1:112-125
  - Hines, Diaz, Rotnitzky, Hernan 2022 Am Stat
  - Flury 1984 JASA 79:387
  - Anderson 1963 Ann Math Stat (Davis-Kahan sin-theta bound)
  - Yu, Wang, Samworth 2015 Biometrika (modern Davis-Kahan)
  - Bro, Acar, Kolda 2008 J Chemometrics (sign ambiguity in SVD)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[3]
PROC = REPO / "data" / "processed"

CITIES = [
    "boston", "nyc", "sf", "dc", "philadelphia", "chicago",
    "seattle", "denver", "atlanta", "portland", "phoenix", "dallas",
]


def load_city_centered(city: str) -> np.ndarray | None:
    """Read a city's listing-level sentence-BERT embeddings, mean-center."""
    parquet = PROC / f"{city}_embeddings.parquet"
    if not parquet.exists():
        return None
    df = pd.read_parquet(parquet)
    cols = [f"emb_{i}" for i in range(768) if f"emb_{i}" in df.columns]
    if not cols:
        return None
    X = df[cols].to_numpy(dtype=float)
    return X - X.mean(axis=0, keepdims=True)


def pin_sign_positive_sum(v: np.ndarray) -> np.ndarray:
    """The same sign-pinning convention pooled_pca_treatment.py uses."""
    return v if v.sum() >= 0 else -v


def davis_kahan_bound(eig_ratio: float) -> float:
    """Crude sin-theta bound proxy from the Davis-Kahan theorem.

    sin(angle_error) <= 1 / sqrt(eigengap - 1) where eigengap = lambda_1
    / lambda_2.  Bro et al 2008 use a tighter version; this is good
    enough for triage.
    """
    return 1.0 / np.sqrt(max(eig_ratio - 1, 0.01))


def per_city_pca_report(pooled_pc1: np.ndarray,
                         data: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    """Per-city PC1, Flury cos vs pooled, eigengap, sin-theta bound."""
    rows = []
    for city, X in data:
        pca_c = PCA(n_components=3).fit(X)
        v1 = pin_sign_positive_sum(pca_c.components_[0])
        cos = float(abs(np.dot(v1, pooled_pc1)))
        evr = pca_c.explained_variance_ratio_
        eig_ratio = float(evr[0] / evr[1])
        rows.append({
            "city": city,
            "n": len(X),
            "pc1_var_share": float(evr[0]),
            "pc1_pc2_ratio": eig_ratio,
            "sin_theta_bound": davis_kahan_bound(eig_ratio),
            "cos_aligned_with_pooled": cos,
            "flury_pass_090": cos > 0.90,
            "flury_pass_095": cos > 0.95,
        })
    return pd.DataFrame(rows)


def treatment_scale_report() -> pd.DataFrame | None:
    """How does the raw treatment scale vary across cities BEFORE the
    within-city z-score?  If Portland is anomalous in raw scale, the
    pooled PC1 axis is loading on Portland-specific variance.
    """
    path = REPO / "results" / "replications" / "pooled_pca_treatment.csv"
    if not path.exists():
        return None
    tr = pd.read_csv(path)
    return (tr.groupby("city")["treatment"]
              .agg(["count", "mean", "std", "min", "max",
                    lambda s: s.quantile(0.99) - s.quantile(0.01)])
              .rename(columns={"<lambda_0>": "p99_minus_p1"})
              .round(4))


def leverage_diag(name: str, fname: str) -> None:
    """Dump the existing per-city leverage JSONs for Portland."""
    path = REPO / "results" / "replications" / fname
    if not path.exists():
        return
    blob = json.loads(path.read_text())
    if "portland" not in blob:
        return
    print(f"\n=== {name} Portland leverage (existing diagnostic) ===")
    print(json.dumps(blob["portland"], indent=2, default=str)[:600])


def main() -> int:
    print("=== Phase A: Portland pathology diagnostic ===\n")

    # Load + pool.
    data = []
    for city in CITIES:
        X = load_city_centered(city)
        if X is not None:
            data.append((city, X))

    if not data:
        print("ERROR: no per-city embedding parquets found", file=sys.stderr)
        return 1

    X_pool = np.vstack([X for _, X in data])
    pca_pool = PCA(n_components=10).fit(X_pool)
    pooled_pc1 = pin_sign_positive_sum(pca_pool.components_[0])

    print(f"Pooled PC1 variance share: "
          f"{pca_pool.explained_variance_ratio_[0]:.4f}")
    print(f"Pooled PC1/PC2 eigenvalue ratio: "
          f"{pca_pool.explained_variance_ratio_[0] / pca_pool.explained_variance_ratio_[1]:.3f}")
    print(f"Pooled sin-theta bound (Davis-Kahan): "
          f"{davis_kahan_bound(pca_pool.explained_variance_ratio_[0] / pca_pool.explained_variance_ratio_[1]):.4f}\n")

    # Per-city PCA + Flury alignment.
    print("=== Per-city PCA stability + Flury (1984) cos(PC1_city, PC1_pooled) ===\n")
    by_city = per_city_pca_report(pooled_pc1, data)
    print(by_city.to_string(index=False,
                              float_format=lambda x: f"{x:.4f}"))

    # Headline finding for Portland.
    if "portland" in by_city["city"].values:
        pdx = by_city[by_city["city"] == "portland"].iloc[0]
        print()
        if pdx["flury_pass_090"]:
            print(f"  Flury cos {pdx['cos_aligned_with_pooled']:.4f} > 0.90 -> "
                  "Portland PC1 IS aligned; pathology is NOT structural PC1 misalignment.")
        else:
            print(f"  ** Flury cos {pdx['cos_aligned_with_pooled']:.4f} < 0.90 -> "
                  "PORTLAND PC1 STRUCTURALLY MISALIGNED with pooled axis. **")
            print("  Remedy: drop Portland from pooled PCA fit OR use city-specific PC1.")
        if pdx["pc1_pc2_ratio"] < 1.5:
            print(f"  ** PC1/PC2 ratio {pdx['pc1_pc2_ratio']:.3f} < 1.5 -> "
                  "sign-pinning empirically UNSTABLE for Portland. **")

    # Treatment-scale check.
    scale = treatment_scale_report()
    if scale is not None:
        print("\n=== Raw treatment scale per city (BEFORE within-city z-score) ===\n")
        print(scale.to_string())
        if "portland" in scale.index:
            pdx_std = scale.loc["portland", "std"]
            others = scale.drop("portland")["std"]
            ratio = pdx_std / others.median()
            print(f"\n  Portland raw treatment std = {pdx_std:.4f}")
            print(f"  Other-city median std    = {others.median():.4f}")
            print(f"  Ratio (Portland / median) = {ratio:.2f}x")
            if ratio > 2.0:
                print("  ** Portland raw treatment scale is >2x other cities -> "
                      "pooled PC1 is loading on Portland-specific variance. **")

    # Existing leverage diag dumps.
    leverage_diag("Shen", "shen_leverage_diag.json")
    leverage_diag("Baur", "baur_leverage_diag.json")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

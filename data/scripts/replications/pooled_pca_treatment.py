"""Cross-market pooled PCA on sentence-BERT embeddings (within-city centered).

Replaces the per-market PC1 treatment that produces sign-flipping DML
estimates across cities with a single global axis defined by the leading
eigenvector of the pooled within-group covariance matrix (Flury 1984;
Veroniki/cross-population PCA tradition; May et al. 2019 SEAT). Within-
city mean centering removes between-city vocabulary differences so PC1
captures the within-market dimension of listing language that the DML
score treats as the treatment.

Output: results/replications/pooled_pca_treatment.csv with columns
[city, listing_id, key, treatment, treatment_z], where treatment is the
projection on the global axis, treatment_z is per-city standardized, and
key is a stable content key (url, falling back to source_html_sha256) that
downstream consumers should merge on instead of the positional listing_id
-- listing_id is an arange() over the CURRENT read of the embeddings
parquet and silently drifts whenever that parquet is rewritten.

The DML estimand then has the interpretation 'per per-city-sigma move
along the common axis', directly comparable across markets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[3]
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _load_city_embeddings(parquet_dir: Path, city: str, suffix: str = ""):
    p = parquet_dir / f"{city}_embeddings{suffix}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        return None
    id_col = "listing_id" if "listing_id" in df.columns else df.index.name
    if id_col is None or id_col not in df.columns:
        df["listing_id"] = np.arange(len(df))
        id_col = "listing_id"
    if "url" in df.columns:
        key_col = "url"
    elif "source_html_sha256" in df.columns:
        key_col = "source_html_sha256"
    else:
        raise AssertionError(
            f"{city}: embeddings parquet has neither 'url' nor "
            "'source_html_sha256' to key the treatment on")
    return df, emb_cols, id_col, key_col


def compute_pooled_pca_treatment(
    cities, parquet_dir, n_components=1, within_city_center=True, seed=42,
    embedding_suffix="",
):
    """Pooled within-group PCA on stacked sentence-BERT embeddings.

    Per Flury (1984) and Veroniki et al. (2016) cross-population PC tradition,
    centering each city at its own mean before pooling produces a leading
    eigenvector identified as the within-city common axis, removing the
    between-city geographic-vocabulary direction that would otherwise dominate
    naive pooled PCA. Sentence-BERT outputs are L2-normalized so we do NOT
    apply per-dim StandardScaler — that would inflate noise dimensions (per
    Ethayarajh 2019 anisotropy guidance).
    """
    parquet_dir = Path(parquet_dir)
    blocks, ids, keys, city_lab = [], [], [], []
    n_per_city = {}
    for c in cities:
        out = _load_city_embeddings(parquet_dir, c, suffix=embedding_suffix)
        if out is None:
            print(f"  [skip] {c}: no embeddings parquet")
            continue
        df, emb_cols, id_col, key_col = out
        X = df[emb_cols].to_numpy(dtype=float)
        if within_city_center:
            X = X - X.mean(axis=0, keepdims=True)
        blocks.append(X)
        ids.append(df[id_col].to_numpy())
        keys.append(df[key_col].astype(str).to_numpy())
        city_lab.extend([c] * len(df))
        n_per_city[c] = len(df)
        print(f"  [{c}] n={len(df)}, embed_dim={len(emb_cols)}, key_col={key_col}")
    X_all = np.vstack(blocks)
    print(f"\n  pooled n={X_all.shape[0]}, dim={X_all.shape[1]}")

    pca = PCA(n_components=n_components, random_state=seed).fit(X_all)
    direction = pca.components_
    if direction[0].sum() < 0:
        direction = -direction

    scores = X_all @ direction.T

    out = pd.DataFrame({"city": city_lab,
                         "listing_id": np.concatenate(ids),
                         "key": np.concatenate(keys)})
    if n_components == 1:
        out["treatment"] = scores[:, 0]
        out["treatment_z"] = out.groupby("city")["treatment"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) or 1.0)
        )
    else:
        for k in range(n_components):
            out[f"treatment_{k}"] = scores[:, k]
            out[f"treatment_z_{k}"] = out.groupby("city")[f"treatment_{k}"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=1) or 1.0)
            )
    print(f"\n  explained variance ratio: "
          f"{pca.explained_variance_ratio_.round(4).tolist()}")
    return out, pca, n_per_city


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=ALL_12)
    ap.add_argument("--parquet_dir", default=str(REPO / "data" / "processed"))
    ap.add_argument("--n_components", type=int, default=1)
    ap.add_argument("--embedding_suffix", default="",
                    help="appended to '{city}_embeddings' before .parquet, "
                         "e.g. '_all_MiniLM_L6_v2' for the MiniLM robustness "
                         "check embeddings")
    ap.add_argument("--out_csv",
                    default=str(REPO / "results" / "replications"
                                  / "pooled_pca_treatment.csv"))
    ap.add_argument("--out_components",
                    default=str(REPO / "results" / "replications"
                                  / "pooled_pca_components.npy"),
                    help="save the global PCA loadings for replication")
    args = ap.parse_args()

    print(f"=== Pooled within-city-centered PCA on {len(args.cities)} cities ===")
    out, pca, n_per = compute_pooled_pca_treatment(
        args.cities, args.parquet_dir, n_components=args.n_components,
        embedding_suffix=args.embedding_suffix,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    np.save(args.out_components, pca.components_)

    print(f"\n  per-city n: {n_per}")
    print(f"\nCSV -> {out_csv}")
    print(f"Components -> {args.out_components}")
    print(f"\n  per-city treatment summary:")
    summ = out.groupby("city")["treatment"].agg(["mean", "std", "min", "max"])
    print(summ.to_string(float_format=lambda x: f"{x:+.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

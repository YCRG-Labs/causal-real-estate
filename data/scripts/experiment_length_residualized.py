"""Does the text price effect survive once the length/HTML artifacts are removed?

Three treatments, all from all-mpnet-base-v2, all pooled-and-within-city-centered
exactly like the published pipeline, differing only in what the embedding was
built from and whether length is partialled out before PCA:

  T0  published pipeline as-is (baseline reproduction)
  T1  descriptions html.unescape'd before cleaning, then re-embedded
  T2  T1 embedding, then each of the 768 dims residualized on
      [1, log_len, log_len^2] (pooled OLS), then PCA -> PC1

For each treatment we run the same partially-linear ridge DML per market, on the
listing-level confounder block (lat, lon, beds, baths, sqft, year_built with
missingness indicators), which is available for all twelve markets, and report the
per-market |theta| and a random-effects pooled estimate.

    python data/scripts/experiment_length_residualized.py --stage embed
    python data/scripts/experiment_length_residualized.py --stage estimate
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from generate_embeddings import clean_description
from replications.compare_to_dml import run_dml
from property_type_confounding import STRUCT, ALL_12

PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "experiment_length"
OUT.mkdir(parents=True, exist_ok=True)
EMB_COLS = [f"emb_{i}" for i in range(768)]


# ---------------------------------------------------------------- stage: embed
def _embed_city(model, df: pd.DataFrame) -> np.ndarray:
    raw = df["description"].astype(str)
    unescaped = raw.map(html.unescape)          # THE FIX: decode &amp; &mdash; &rsquo;
    cleaned = unescaped.map(clean_description)   # same cleaner the pipeline uses
    return model.encode(cleaned.tolist(), batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=False)


def stage_embed(device: str | None = None) -> None:
    """Re-embed each city and write to disk immediately.

    Memory note: on Apple MPS the allocator does not release between
    model.encode calls, so encoding all twelve cities in one process climbs to
    tens of GB and OOMs the machine. We free every intermediate, empty the MPS
    cache after each city, and skip cities already on disk so the job can be run
    in chunks. For a full clean run prefer CPU (device="cpu") or a cloud GPU.
    """
    import gc
    import torch
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "cpu"  # safe default; MPS leaks across cities
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    for city in ALL_12:
        src = PROC / f"{city}_embeddings.parquet"
        dst = OUT / f"{city}_reembed.parquet"
        if dst.exists():
            print(f"  {city}: already done, skip")
            continue
        if not src.exists():
            print(f"  {city}: no source, skip")
            continue
        df = pd.read_parquet(src, columns=["description"])
        E = _embed_city(model, df)
        out = df[["description"]].copy()
        out["log_len"] = np.log1p(df["description"].astype(str).str.len().to_numpy())
        for j, c in enumerate(EMB_COLS):
            out[c] = E[:, j]
        out.to_parquet(dst)
        print(f"  {city}: re-embedded {len(df)} listings", flush=True)
        del df, E, out
        gc.collect()
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ------------------------------------------------------------- treatment build
def _within_city_center(blocks: list[np.ndarray]) -> np.ndarray:
    return np.vstack([b - b.mean(0, keepdims=True) for b in blocks])


def _pc1_scores(stack: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    d = PCA(n_components=1, random_state=0).fit(stack).components_[0]
    if d.sum() < 0:
        d = -d
    return stack @ d


def _residualize_length(X: np.ndarray, logtok: np.ndarray) -> np.ndarray:
    B = np.column_stack([np.ones_like(logtok), logtok, logtok ** 2])
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    return X - B @ coef


def _build_treatment(city_frames: dict, source: str, resid_length: bool) -> dict:
    """Return {city: z-scored PC1} pooled over the twelve markets."""
    blocks, lens, order = [], [], []
    for city, df in city_frames.items():
        X = df[EMB_COLS].to_numpy(float)
        lens.append(df["log_len"].to_numpy(float))
        blocks.append(X)
        order.append((city, len(df)))
    Xc = _within_city_center(blocks)
    if resid_length:
        Xc = _residualize_length(Xc, np.concatenate(lens))
    scores = _pc1_scores(Xc)
    out, i = {}, 0
    for city, n in order:
        s = scores[i:i + n]
        out[city] = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        i += n
    return out


# ---------------------------------------------------------- stage: estimate
def _confounders(df: pd.DataFrame):
    lat = pd.to_numeric(df.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df.longitude, errors="coerce").to_numpy(float)
    S = df[STRUCT].apply(pd.to_numeric, errors="coerce")
    X = np.column_stack([lat, lon, S.fillna(S.median()).to_numpy(float),
                         S.isna().to_numpy(float)])
    y = np.log(pd.to_numeric(df.price, errors="coerce").to_numpy(float))
    ok = np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    return X, y, ok


def _re_pool(theta, se):
    theta, se = np.asarray(theta), np.asarray(se)
    v = se ** 2
    w = 1 / v
    mu = (w * theta).sum() / w.sum()
    Q = (w * (theta - mu) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (len(theta) - 1)) / c)
    w2 = 1 / (v + tau2)
    return float((w2 * theta).sum() / w2.sum()), float((1 / w2.sum()) ** 0.5), float(tau2)


def stage_estimate() -> None:
    listings = {c: pd.read_parquet(PROC / f"{c}_embeddings.parquet") for c in ALL_12}
    reemb = {c: pd.read_parquet(OUT / f"{c}_reembed.parquet") for c in ALL_12}
    for c in ALL_12:
        assert len(listings[c]) == len(reemb[c]), f"row mismatch {c}"

    # published treatment straight from the committed csv, for T0 reproduction
    pooled = pd.read_csv(REPO / "results" / "replications" / "pooled_pca_treatment.csv")

    treatments = {
        "T0_published": {c: (pooled[pooled.city == c]
                             .assign(_i=lambda d: np.arange(len(d)))
                             .set_index("_i")["treatment_z"].to_numpy())
                         for c in ALL_12},
        "T1_html_fixed": _build_treatment(reemb, "reemb", resid_length=False),
        "T2_len_resid": _build_treatment(reemb, "reemb", resid_length=True),
    }

    rows = []
    for name, tmap in treatments.items():
        ths, ses = [], []
        for c in ALL_12:
            df = listings[c]
            X, y, ok = _confounders(df)
            T = np.asarray(tmap[c], float)
            if len(T) != len(df):
                print(f"  {name}/{c}: treatment length {len(T)} != {len(df)}, skip")
                continue
            r = run_dml(T[ok].reshape(-1, 1), X[ok], y[ok], label=f"{name}:{c}",
                        ci_method="if", n_boot=None, use_ridge=True, seed=42, n_pca=1)
            if r is None:
                continue
            rows.append({"treatment": name, "city": c, "n": int(ok.sum()),
                         "theta": float(r.theta), "abs_theta": abs(float(r.theta)),
                         "se": float(r.se), "covers_zero": bool(r.ci_low < 0 < r.ci_high)})
            ths.append(abs(float(r.theta)))
            ses.append(float(r.se))
        mu, mse, tau2 = _re_pool(ths, ses)
        rows.append({"treatment": name, "city": "POOLED", "n": None,
                     "theta": mu, "abs_theta": mu, "se": mse, "tau2": tau2,
                     "k_excl_zero": int(sum(1 for x in rows
                                            if x["treatment"] == name and x["city"] != "POOLED"
                                            and not x["covers_zero"]))})

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'treatment':16s}{'city':14s}{'n':>7s}{'|theta|':>9s}{'se':>8s}{'0?':>4s}")
    print("-" * 58)
    for r in rows:
        z = "" if r["city"] == "POOLED" else ("*" if r["covers_zero"] else " ")
        n = "" if r["n"] is None else f"{r['n']:d}"
        extra = f"  tau2={r['tau2']:.4f} k_excl0={r['k_excl_zero']}/12" if r["city"] == "POOLED" else ""
        print(f"{r['treatment']:16s}{r['city']:14s}{n:>7s}{r['abs_theta']:9.4f}{r['se']:8.4f}{z:>4s}{extra}")
    print("\n* = 95% CI covers zero")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "estimate"], required=True)
    args = ap.parse_args()
    if args.stage == "embed":
        stage_embed()
    else:
        stage_estimate()

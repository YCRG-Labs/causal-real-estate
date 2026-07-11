from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import svd

REPO = Path(__file__).resolve().parents[3]
PANEL = REPO / "results" / "soldprice"
EMB = REPO / "results" / "soldprice" / "emb"
EMB.mkdir(parents=True, exist_ok=True)
CITIES = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]
DIM = 768


def embed_city(model, city):
    dst = EMB / f"{city}_emb.parquet"
    if dst.exists():
        print(f"  {city}: cached", flush=True)
        return
    d = pd.read_parquet(PANEL / f"{city}_panel.parquet", columns=["description"])
    txt = d["description"].fillna("").astype(str).tolist()
    E = model.encode(txt, batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False).astype(np.float32)
    out = pd.DataFrame(E, columns=[f"emb_{i}" for i in range(DIM)])
    out["log_len"] = np.log(np.array([len(t) for t in txt]) + 1.0)
    out.to_parquet(dst)
    print(f"  {city}: embedded {len(out)}", flush=True)
    del E, out, txt
    gc.collect()


def pooled_pc1():
    blocks, sizes = [], []
    for c in CITIES:
        e = pd.read_parquet(EMB / f"{c}_emb.parquet",
                            columns=[f"emb_{i}" for i in range(DIM)])
        blocks.append(e.to_numpy(np.float64))
        sizes.append((c, len(e)))
    Xc = np.vstack([b - b.mean(0, keepdims=True) for b in blocks])
    del blocks
    gc.collect()
    d = svd(Xc, full_matrices=False)[2][0]
    if d.sum() < 0:
        d = -d
    sc = Xc @ d
    del Xc
    gc.collect()
    i = 0
    for c, n in sizes:
        s = sc[i:i + n]
        t = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        pd.DataFrame({"treatment": t}).to_parquet(EMB / f"{c}_treatment.parquet")
        i += n
    print(f"pooled PC1 built over {sum(n for _, n in sizes)} listings, "
          f"per-city standardized -> {EMB}/{{city}}_treatment.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pc1_only", action="store_true")
    args = ap.parse_args()
    if not args.pc1_only:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2", device=args.device)
        for c in CITIES:
            embed_city(model, c)
        del model
        gc.collect()
    pooled_pc1()


if __name__ == "__main__":
    main()

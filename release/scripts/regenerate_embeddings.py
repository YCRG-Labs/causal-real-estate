"""Regenerate embeddings from a description CSV produced by fetch_descriptions.py.

Reproduces the two encoders used in the paper:
    sentence-transformers/all-mpnet-base-v2  (768-dim)
    sentence-transformers/all-MiniLM-L6-v2   (384-dim)

Usage:
    python scripts/regenerate_embeddings.py \\
        --in descriptions.csv --model mpnet --out embeddings.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return WHITESPACE.sub(" ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--model", choices=list(MODELS), default="mpnet")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df["clean_description"] = df["description"].map(clean)
    df = df[df["clean_description"].str.len() > 0].reset_index(drop=True)

    model = SentenceTransformer(MODELS[args.model])
    emb = model.encode(df["clean_description"].tolist(), show_progress_bar=True)

    cols = {f"emb_{i}": emb[:, i] for i in range(emb.shape[1])}
    keep = [c for c in ("zip", "latitude", "longitude", "price") if c in df.columns]
    out = pd.concat([df[keep].reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(out):,} rows, {emb.shape[1]}-dim)")


if __name__ == "__main__":
    main()

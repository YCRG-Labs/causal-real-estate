"""
Modern encoder robustness comparison for the listing-text embedding pipeline.

Replaces all-mpnet-base-v2 with three SOTA-tier sentence encoders and writes
per-city parquets plus a sidecar metadata JSON describing dim / model_tag /
prefix convention. Downstream loaders should read the JSON, not hardcode 768.

Usage:
    python generate_embeddings_modern.py                   # all three models, all three cities
    python generate_embeddings_modern.py --city sf
    python generate_embeddings_modern.py --model bge_large_en
    python generate_embeddings_modern.py --model modernbert_embed_large --matryoshka_dim 256
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _silence
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import RAW_DIR, PROCESSED_DIR, CITIES
from utils import ensure_dirs
from generate_embeddings import clean_description, geocode_from_zip

DESC_DIR = RAW_DIR / "descriptions"

MODEL_REGISTRY = {
    "mpnet_base": {
        "hf_id": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "needs_prefix": False,
        "doc_prefix": "",
        "model_kwargs": None,
        "trust_remote_code": False,
    },
    "bge_large_en": {
        "hf_id": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "needs_prefix": False,
        "doc_prefix": "",
        "model_kwargs": None,
        "trust_remote_code": False,
    },
    "modernbert_embed_large": {
        "hf_id": "lightonai/modernbert-embed-large",
        "dim": 1024,
        "needs_prefix": True,
        "doc_prefix": "search_document: ",
        "model_kwargs": {"attn_implementation": "sdpa"},
        "trust_remote_code": False,
    },
}


def generate_embeddings_modern(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    matryoshka_dim: int | None = None,
    device: str = "cpu",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings under one of the three audited encoder families.

    Parameters
    ----------
    texts : list[str]
        Cleaned listing descriptions (run clean_description first).
    model_name : str
        Either an MODEL_REGISTRY key ("mpnet_base", "bge_large_en",
        "modernbert_embed_large") or a raw HF path (passed through).
    batch_size : int
        32 is the sweet spot for M3 Pro CPU on seq_len <= 256.
    matryoshka_dim : int | None
        Only valid for "modernbert_embed_large" (supports 256). Ignored
        with a warning for the other two. None means full dim.
    device : str
        "cpu" recommended on M3 Pro for n ~ 1k (MPS overhead is not amortised).

    Returns
    -------
    np.ndarray, shape (len(texts), output_dim), unit-normalized rows.
    output_dim equals the model dim unless matryoshka_dim is set.
    """
    if model_name in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[model_name]
        hf_id = spec["hf_id"]
    else:
        hf_id = model_name
        spec = {
            "needs_prefix": False,
            "doc_prefix": "",
            "model_kwargs": None,
            "trust_remote_code": False,
        }

    if matryoshka_dim is not None and model_name != "modernbert_embed_large":
        print(
            f"  WARNING: matryoshka_dim={matryoshka_dim} requested but "
            f"{model_name} does not support Matryoshka truncation; ignoring."
        )
        matryoshka_dim = None

    init_kwargs = {"device": device}
    if spec.get("model_kwargs"):
        init_kwargs["model_kwargs"] = spec["model_kwargs"]
    if spec.get("trust_remote_code"):
        init_kwargs["trust_remote_code"] = True
    if matryoshka_dim is not None:
        init_kwargs["truncate_dim"] = matryoshka_dim

    model = SentenceTransformer(hf_id, **init_kwargs)

    if spec["needs_prefix"] and spec["doc_prefix"]:
        prepared = [spec["doc_prefix"] + t for t in texts]
    else:
        prepared = texts

    embeddings = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return embeddings


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def run_city(city: str, model_tag: str, matryoshka_dim: int | None = None,
             device: str | None = None, batch_size: int = 128) -> dict:
    desc_path = DESC_DIR / f"{city}_descriptions.csv"
    listings_path = PROCESSED_DIR / f"{city}_listings.parquet"
    if desc_path.exists():
        df = pd.read_csv(desc_path)
        source = "descriptions_csv"
    elif listings_path.exists():
        live = pd.read_parquet(listings_path)
        df = live.rename(columns={"address": "address", "description": "description"}).copy()
        if "zip" not in df.columns:
            df["zip"] = ""
        if "url" not in df.columns:
            df["url"] = ""
        source = "listings_parquet"
    else:
        print(f"{city}: no descriptions csv and no listings parquet, skipping")
        return {}

    print(f"\n=== {city} / {model_tag} ===")
    print(f"  Loaded {len(df)} descriptions from {source}")

    if source == "descriptions_csv":
        df = geocode_from_zip(df, city)
    df["clean_description"] = df["description"].fillna("").apply(clean_description)
    df = df[df["clean_description"].str.len() > 20].reset_index(drop=True)
    print(f"  {len(df)} after cleaning")
    if len(df) == 0:
        print(f"  {city}: no clean descriptions, skipping")
        return {}

    texts = df["clean_description"].tolist()
    if device is None:
        device = _auto_device()
    print(f"  encoding on device={device}, batch_size={batch_size}")

    t0 = time.time()
    embeddings = generate_embeddings_modern(
        texts, model_tag, batch_size=batch_size, matryoshka_dim=matryoshka_dim,
        device=device,
    )
    elapsed = time.time() - t0

    out_dim = embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(out_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    result = pd.concat([df, emb_df], axis=1)

    suffix = model_tag if matryoshka_dim is None else f"{model_tag}_d{matryoshka_dim}"
    parquet_path = PROCESSED_DIR / f"{city}_embeddings_{suffix}.parquet"
    result.to_parquet(parquet_path, index=False)

    meta = {
        "city": city,
        "model_tag": model_tag,
        "hf_id": MODEL_REGISTRY.get(model_tag, {}).get("hf_id", model_tag),
        "embedding_dim": int(out_dim),
        "matryoshka_dim": matryoshka_dim,
        "n_listings": int(len(result)),
        "batch_size": batch_size,
        "device": device,
        "normalize_embeddings": True,
        "doc_prefix": MODEL_REGISTRY.get(model_tag, {}).get("doc_prefix", ""),
        "encode_seconds": round(elapsed, 2),
        "sentence_transformers_version": __import__("sentence_transformers").__version__,
    }
    meta_path = PROCESSED_DIR / f"{city}_embeddings_{suffix}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  encoded {len(embeddings)} x {out_dim}d in {elapsed:.1f}s "
          f"({elapsed/len(embeddings)*1000:.1f} ms/listing)")
    print(f"  norm stats: mean={norms.mean():.4f} std={norms.std():.4f}")
    print(f"  wrote {parquet_path.name} + {meta_path.name}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="all", help="city slug or 'all' or comma-separated")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--matryoshka_dim",
        type=int,
        default=None,
        help="Only valid with --model modernbert_embed_large (supports 256)",
    )
    parser.add_argument("--device", default=None, help="cpu|cuda|mps|auto (default auto)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    ensure_dirs()
    if args.city == "all":
        cities = list(CITIES.keys()) + ["dc", "philadelphia", "chicago", "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
    elif "," in args.city:
        cities = [c.strip() for c in args.city.split(",") if c.strip()]
    else:
        cities = [args.city]
    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    device = args.device if args.device and args.device != "auto" else None

    runs = []
    total_t0 = time.time()
    for model_tag in models:
        for city in cities:
            meta = run_city(
                city, model_tag,
                matryoshka_dim=args.matryoshka_dim,
                device=device, batch_size=args.batch_size,
            )
            if meta:
                runs.append(meta)
    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 60)
    print(f"Done. {len(runs)} embedding sets in {total_elapsed:.1f}s total.")
    print("=" * 60)
    if runs:
        summary = pd.DataFrame(runs)[
            ["city", "model_tag", "embedding_dim", "n_listings", "encode_seconds"]
        ]
        print(summary.to_string(index=False))
    else:
        print("(no successful runs — descriptions or model files missing)")


if __name__ == "__main__":
    main()

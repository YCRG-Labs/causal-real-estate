import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import (
    RAW_DIR, PROCESSED_DIR, CITIES,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_ALTERNATIVES,
)
from utils import ensure_dirs

DESC_DIR = RAW_DIR / "descriptions"
BATCH_SIZE = 64


CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}


def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def clean_description(text):
    if not isinstance(text, str):
        return ""
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()

    text = expand_contractions(text)

    boilerplate = [
        "for more information",
        "call today",
        "schedule a showing",
        "contact us",
        "click here",
        "virtual tour",
        "open house",
        "price reduced",
        "must see",
        "won't last",
        "act fast",
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")

    text = re.sub(r"\d+\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|place|pl|court|ct|boulevard|blvd),?\s*\w*,?\s*\w{2}\s*\d{5}", "[ADDRESS]", text)

    return text.strip()


def encode_with_model(texts, model_name, dim, out_path):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    emb_cols = [f"emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    return emb_df, embeddings


def generate_embeddings(city):
    desc_path = DESC_DIR / f"{city}_descriptions.csv"
    if not desc_path.exists():
        print(f"{city}: no descriptions file found, skipping")
        return

    df = pd.read_csv(desc_path)
    print(f"{city}: {len(df)} descriptions loaded")

    df["clean_description"] = df["description"].apply(clean_description)
    df = df[df["clean_description"].str.len() > 20]
    print(f"  {len(df)} after cleaning")

    texts = df["clean_description"].tolist()

    print(f"\n  Primary model: {EMBEDDING_MODEL}")
    emb_df, embeddings = encode_with_model(texts, EMBEDDING_MODEL, EMBEDDING_DIM, None)
    result = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    out_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved {len(result)} embeddings ({EMBEDDING_DIM}d) → {out_path}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  norm stats: mean={norms.mean():.3f} std={norms.std():.3f} min={norms.min():.3f} max={norms.max():.3f}")

    for alt_model, alt_dim in EMBEDDING_ALTERNATIVES.items():
        print(f"\n  Alternative model: {alt_model}")
        alt_emb_df, alt_embeddings = encode_with_model(texts, alt_model, alt_dim, None)
        alt_result = pd.concat([df.reset_index(drop=True), alt_emb_df.reset_index(drop=True)], axis=1)

        safe_name = alt_model.replace("/", "_").replace("-", "_")
        alt_out = PROCESSED_DIR / f"{city}_embeddings_{safe_name}.parquet"
        alt_result.to_parquet(alt_out, index=False)

        alt_norms = np.linalg.norm(alt_embeddings, axis=1)
        print(f"  Saved {len(alt_result)} embeddings ({alt_dim}d) → {alt_out}")
        print(f"  norm stats: mean={alt_norms.mean():.3f} std={alt_norms.std():.3f}")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        generate_embeddings(city)
    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()

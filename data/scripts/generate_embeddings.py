import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import (
    RAW_DIR, PROCESSED_DIR, CITIES,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)
from utils import ensure_dirs

DESC_DIR = RAW_DIR / "descriptions"
BATCH_SIZE = 64


def clean_description(text):
    if not isinstance(text, str):
        return ""
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()

    boilerplate = [
        "for more information",
        "call today",
        "schedule a showing",
        "contact us",
        "click here",
        "virtual tour",
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")

    text = re.sub(r"\d+\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|place|pl|court|ct|boulevard|blvd),?\s*\w*,?\s*\w{2}\s*\d{5}", "[ADDRESS]", text)

    return text.strip()


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

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = df["clean_description"].tolist()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)

    result = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    out_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    result.to_parquet(out_path, index=False)
    print(f"  Saved {len(result)} embeddings ({EMBEDDING_DIM}d) → {out_path}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  norm stats: mean={norms.mean():.3f} std={norms.std():.3f} min={norms.min():.3f} max={norms.max():.3f}")


def main():
    ensure_dirs()
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        generate_embeddings(city)
    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()

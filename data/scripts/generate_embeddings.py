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


def geocode_from_zip(df, city):
    """
    If descriptions lack lat/lon, assign coordinates from zip-code centroids
    using the pgeocode (GeoNames) offline lookup. This enables the spatial
    join to parcels that pulls in the full 34-feature confounder set (census,
    crime, amenity, micro-geo). Accuracy is at the zip-centroid level
    (~0.5-1 km), which is sufficient for block-group-level spatial joins.
    """
    lat = pd.to_numeric(df.get("latitude", pd.Series(dtype=float)), errors="coerce")
    lon = pd.to_numeric(df.get("longitude", pd.Series(dtype=float)), errors="coerce")
    n_missing = lat.isna().sum()

    if n_missing == 0:
        print(f"  All {len(df)} descriptions already have coordinates")
        return df

    if "zip" not in df.columns:
        print(f"  {n_missing}/{len(df)} descriptions lack coordinates and no zip column")
        return df

    print(f"  {n_missing}/{len(df)} descriptions lack coordinates — geocoding via pgeocode...")

    import pgeocode
    nomi = pgeocode.Nominatim("us")

    zips = df["zip"].astype(float).astype(int).astype(str).str.zfill(5)
    unique_zips = zips.unique()
    zip_lookup = {}
    for z in unique_zips:
        result = nomi.query_postal_code(z)
        if pd.notna(result.latitude) and pd.notna(result.longitude):
            zip_lookup[z] = (result.latitude, result.longitude)

    matched = 0
    for idx in df.index:
        if pd.notna(lat.loc[idx]) and pd.notna(lon.loc[idx]):
            continue
        z = zips.loc[idx]
        if z in zip_lookup:
            df.loc[idx, "latitude"] = zip_lookup[z][0]
            df.loc[idx, "longitude"] = zip_lookup[z][1]
            matched += 1

    still_missing = pd.to_numeric(df["latitude"], errors="coerce").isna().sum()
    print(f"  Geocoded {matched}/{n_missing} via zip-centroid "
          f"({len(zip_lookup)}/{len(unique_zips)} zips resolved, "
          f"{still_missing} still missing)")
    return df


def generate_embeddings(city):
    desc_path = DESC_DIR / f"{city}_descriptions.csv"
    if not desc_path.exists():
        print(f"{city}: no descriptions file found, skipping")
        return

    df = pd.read_csv(desc_path)
    print(f"{city}: {len(df)} descriptions loaded")

    df = geocode_from_zip(df, city)

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

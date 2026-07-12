"""Self-contained HTML-fixed re-embedding for a fresh Brev box. No repo imports.

Reproduces the published pipeline's cleaning EXACTLY (clean_description +
expand_contractions, all-mpnet-base-v2, normalize_embeddings=False), with one
change: html.unescape is applied before cleaning, so '&amp;' '&mdash;' '&rsquo;'
stop entering the token stream.

Input : a directory of per-city parquets named <city>_desc.parquet, each with
        columns [row, description] where row is the listing's positional index in
        the original data/processed/<city>_embeddings.parquet (preserve order!).
Output: <city>_reembed.parquet with columns [row, description, log_len,
        emb_0..emb_767], one row per input row, same order.

    python brev_reembed.py --in_dir brev_input --out_dir brev_out

Deps: pip install sentence-transformers pandas pyarrow
"""
from __future__ import annotations

import argparse
import gc
import glob
import html
import os
import re

import numpy as np
import pandas as pd

# ---- pipeline cleaning, copied verbatim from generate_embeddings.py ----------
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}
BOILERPLATE = [
    "for more information", "call today", "schedule a showing", "contact us",
    "click here", "virtual tour", "open house", "price reduced", "must see",
    "won't last", "act fast",
]
_ADDR = re.compile(
    r"\d+\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|place|pl|"
    r"court|ct|boulevard|blvd),?\s*\w*,?\s*\w{2}\s*\d{5}")


def clean_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    for phrase in BOILERPLATE:
        text = text.replace(phrase, "")
    text = _ADDR.sub("[ADDRESS]", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="brev_input")
    ap.add_argument("--out_dir", default="brev_out")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=dev)

    for path in sorted(glob.glob(os.path.join(args.in_dir, "*_desc.parquet"))):
        city = os.path.basename(path)[:-len("_desc.parquet")]
        dst = os.path.join(args.out_dir, f"{city}_reembed.parquet")
        if os.path.exists(dst):
            print(f"  {city}: already done, skip", flush=True)
            continue
        df = pd.read_parquet(path).sort_values("row").reset_index(drop=True)
        raw = df["description"].astype(str)
        cleaned = raw.map(html.unescape).map(clean_description)   # THE FIX
        E = model.encode(cleaned.tolist(), batch_size=args.batch,
                         show_progress_bar=True, convert_to_numpy=True,
                         normalize_embeddings=False)
        out = pd.DataFrame({"row": df["row"].to_numpy(),
                            "description": raw.to_numpy(),
                            "log_len": np.log1p(raw.str.len().to_numpy())})
        for j in range(E.shape[1]):
            out[f"emb_{j}"] = E[:, j]
        out.to_parquet(dst)
        print(f"  {city}: {len(df)} listings -> {dst}", flush=True)
        del df, E, out, cleaned, raw
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("done", flush=True)


if __name__ == "__main__":
    main()

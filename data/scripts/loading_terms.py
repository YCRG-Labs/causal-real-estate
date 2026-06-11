"""What is the text direction made of? Recover the words and phrases that load on
the leading, price-oriented embedding axis used as the treatment in the Baur
channel, so the abstract "semantic content" can be shown concretely.

Method: stack the sentence-BERT embeddings across markets, centered within each
market (the paper's pooled within-city axis), take the leading principal
component, and orient it so it correlates positively with log price. Project every
listing onto that axis, then regress the projection on a TF-IDF vocabulary of the
descriptions. The largest positive coefficients are the language associated with
the high-price end of the axis; the largest negative ones with the low end.

Run where the embeddings + listings live (Brev):
  python3 data/scripts/loading_terms.py
Prints the top loading terms at each end; paste them to build the paper table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def load_market(city):
    ep = PROC / f"{city}_embeddings.parquet"
    if not ep.exists():
        return None
    e = pd.read_parquet(ep)
    emb = [c for c in e.columns if c.startswith("emb_")]
    if "source_html_sha256" not in e.columns or not emb:
        return None
    e["sha16"] = e["source_html_sha256"].astype(str).str[:16]
    L = pd.read_parquet(PROC / f"{city}_listings.parquet")
    L["sha16"] = L["source_html_sha256"].astype(str).str[:16]
    text_col = "clean_description" if "clean_description" in L.columns else "description"
    df = e.merge(L[["sha16", text_col, "price"]].drop_duplicates("sha16"),
                 on="sha16", how="inner")
    df = df[df[text_col].astype(str).str.len() > 20]
    price = pd.to_numeric(df["price"], errors="coerce")
    df = df[(price > 1e4) & price.notna()].reset_index(drop=True)
    if len(df) < 200:
        return None
    X = df[emb].to_numpy(float)
    Xc = X - X.mean(0)  # within-city centering
    return Xc, np.log(pd.to_numeric(df["price"]).to_numpy(float)), df[text_col].astype(str).tolist()


def main():
    parts = [load_market(c) for c in ALL_12]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise SystemExit("no markets with embeddings + descriptions found")
    Xc = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    texts = [t for p in parts for t in p[2]]
    print(f"pooled {len(texts):,} listings across {len(parts)} markets")

    pc1 = PCA(n_components=1, random_state=42).fit_transform(Xc)[:, 0]
    if np.corrcoef(pc1, y)[0, 1] < 0:
        pc1 = -pc1
    pc1 = (pc1 - pc1.mean()) / (pc1.std() or 1.0)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=40, max_df=0.5,
                          stop_words="english", token_pattern=r"[a-zA-Z][a-zA-Z]+")
    T = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    coef = Ridge(alpha=10.0).fit(T, pc1).coef_

    order = np.argsort(coef)
    print(f"\nvocabulary: {len(terms):,} terms (min_df=40, uni+bigram)\n")
    print("=== TOP positive-loading (high-price end of the text axis) ===")
    for i in order[::-1][:25]:
        print(f"  {coef[i]:+.3f}  {terms[i]}")
    print("\n=== TOP negative-loading (low-price end) ===")
    for i in order[:25]:
        print(f"  {coef[i]:+.3f}  {terms[i]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

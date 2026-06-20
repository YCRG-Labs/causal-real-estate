import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

PROC = Path(__file__).resolve().parents[1] / "processed"
OUT = Path(__file__).resolve().parents[2] / "results" / "interpretability"


def axis_terms(city: str, topn: int, min_df: int = 30, alpha: float = 1.0) -> dict:
    d = pd.read_parquet(PROC / f"{city}_embeddings.parquet")
    emb = [c for c in d.columns if c.startswith("emb_")]
    txt = "clean_description" if "clean_description" in d.columns else "description"
    d = d[d[txt].astype(str).str.len() > 20].reset_index(drop=True)

    X = d[emb].to_numpy(float)
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[0]

    price = pd.to_numeric(d["price"], errors="coerce").to_numpy()
    lp = np.log(np.where(price > 0, price, np.nan))
    ok = np.isfinite(lp)
    if np.corrcoef(proj[ok], lp[ok])[0, 1] < 0:
        proj = -proj

    V = TfidfVectorizer(min_df=min_df, ngram_range=(1, 2), stop_words="english")
    T = V.fit_transform(d[txt].astype(str))
    coef = Ridge(alpha=alpha).fit(T, proj).coef_
    vocab = np.array(V.get_feature_names_out())
    order = np.argsort(coef)
    return {
        "city": city, "n": int(len(d)),
        "high_end": list(vocab[order[::-1][:topn]]),
        "low_end": list(vocab[order[:topn]]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="nyc,dallas,phoenix")
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [x.strip() for x in args.cities.split(",") if x.strip()]:
        r = axis_terms(c, args.topn)
        rows.append(r)
        print(f"\n=== {r['city']} (n={r['n']:,}) ===")
        print("  HIGH-price end:", ", ".join(r["high_end"]))
        print("  LOW-price end :", ", ".join(r["low_end"]))

    pd.DataFrame(rows).to_json(OUT / "axis_vocabulary.json", orient="records", indent=2)
    print(f"\nwrote {OUT / 'axis_vocabulary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

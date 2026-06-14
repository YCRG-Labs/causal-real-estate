"""Regenerate the per-market sentence-transformer embeddings locally from the
listings parquets, after the compute box was lost.

Faithful to the paper's pipeline: all-mpnet-base-v2, the same clean_description
text cleaning, unit-normalized 768-dim vectors. Reads data/processed/<city>_listings.parquet
(descriptions live there), renames the centroid coordinates to the latitude/longitude
the downstream loaders and build_release_12.py expect, and writes the canonical
data/processed/<city>_embeddings.parquet (no model suffix) with the structured
columns + emb_0..emb_767.

  python regen_embeddings_local.py --city sf            # pilot
  python regen_embeddings_local.py --all                # all twelve
"""
import sys, os, time, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from generate_embeddings import clean_description

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def run_city(city: str, model: SentenceTransformer, batch_size: int) -> dict:
    lp = PROC / f"{city}_listings.parquet"
    if not lp.exists():
        return {"city": city, "status": "MISSING listings"}
    df = pd.read_parquet(lp)
    df = df.rename(columns={"lat_centroid": "latitude", "lon_centroid": "longitude"})
    df["clean_description"] = df["description"].fillna("").apply(clean_description)
    df = df[df["clean_description"].str.len() > 20].reset_index(drop=True)
    if len(df) == 0:
        return {"city": city, "status": "no clean descriptions"}

    t0 = time.time()
    emb = model.encode(df["clean_description"].tolist(), batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True)
    elapsed = time.time() - t0

    norms = np.linalg.norm(emb, axis=1)
    nan = int(np.isnan(emb).any(axis=1).sum())
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    out = pd.concat([df, emb_df], axis=1)
    out.to_parquet(PROC / f"{city}_embeddings.parquet", index=False)
    return {"city": city, "status": "ok", "n": len(out), "dim": emb.shape[1],
            "nan_rows": nan, "norm_mean": round(float(norms.mean()), 4),
            "secs": round(elapsed, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cities = ALL_12 if args.all else ([args.city] if args.city else [])
    if not cities:
        raise SystemExit("pass --city <slug> or --all")

    print(f"loading {MODEL_ID} on {args.device} ...")
    model = SentenceTransformer(MODEL_ID, device=args.device)

    rows = []
    for c in cities:
        r = run_city(c, model, args.batch_size)
        rows.append(r)
        if r["status"] == "ok":
            print(f"  {c:13s} n={r['n']:>6,}  dim={r['dim']}  nan_rows={r['nan_rows']}  "
                  f"norm={r['norm_mean']}  {r['secs']}s")
        else:
            print(f"  {c:13s} {r['status']}")

    ok = [r for r in rows if r["status"] == "ok"]
    bad_nan = [r for r in ok if r["nan_rows"] > 0]
    print(f"\n{len(ok)}/{len(cities)} markets regenerated -> {PROC}")
    if bad_nan:
        print(f"WARNING: NaN rows in {[r['city'] for r in bad_nan]} -- rerun with --device cpu")
        return 1
    return 0 if len(ok) == len(cities) else 1


if __name__ == "__main__":
    raise SystemExit(main())
